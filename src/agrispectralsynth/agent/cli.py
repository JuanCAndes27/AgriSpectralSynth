"""
agrispectralsynth-agent: build rewards, evaluate policies, train and apply the agent.

    agrispectralsynth-agent rewards  --images data/benchmarks/NeonTreeEvaluation/evaluation/RGB \\
                                     --annotations data/benchmarks/NeonTreeEvaluation/annotations \\
                                     --out results/agent
    agrispectralsynth-agent evaluate --table results/agent --seeds 10
    agrispectralsynth-agent train    --table results/agent --out results/agent/agent.json
    agrispectralsynth-agent apply    --agent results/agent/agent.json --input data/raw --output results/conteo
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path


def cmd_rewards(a) -> int:
    from .evaluation import RewardWeights
    from .rewards import DRONES, RewardJob, build_reward_table

    job = RewardJob(
        sensors=[s.strip() for s in a.sensors.split(",")] if a.sensors else list(DRONES),
        gsd_m=a.gsd, iou=a.iou, weights=RewardWeights(a.count_weight, a.time_weight),
    )
    s = build_reward_table(a.images, a.annotations, a.out, job, workers=a.workers, limit=a.limit)
    print(f"{s['images']} images, {s['contexts']} contexts, {s['rewards']} rewards in {s['seconds']} s -> {s['out_dir']}")
    return 0 if not s["errors"] else 2


def cmd_evaluate(a) -> int:
    from . import report
    from .experiment import cross_validate, summarize
    from .rewards import load_reward_table

    T = load_reward_table(a.table)
    out = Path(a.out or a.table)
    out.mkdir(parents=True, exist_ok=True)
    protocols = ["image", "site"] if a.grouping == "both" else [a.grouping]
    rows, raw = {}, {}
    for proto in protocols:
        raw[proto] = cross_validate(T, k=a.folds, seeds=range(a.seeds), epochs=a.epochs, grouping=proto)
        rows[proto] = summarize(raw[proto])
        print(f"\n== {report.PROTO_LABEL[proto]} ==")
        print(f"{'policy':32s}{'reward':>14s}{'F1':>8s}{'|count|':>9s}{'inventory':>11s}{'gap':>7s}")
        for r in rows[proto]:
            print(f"{r['policy']:32s}{r['test_reward']:8.3f} ±{r['test_reward_sd']:.3f}{r['test_f1']:8.3f}"
                  f"{r['test_count_err']:9.2f}{r['inventory_err']:+11.1%}{r['gap_closed']:+7.2f}")
    report.write_summary_csv(rows, out / "summary.csv")
    if "image" in rows and "site" in rows:
        report.fig_policies(rows, out / "fig_policies.png")
        report.fig_sites(T, raw["image"], out / "fig_sites.png")
        with open(out / "choices_by_site.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["site", "agent_top", "oracle_top"])
            w.writeheader()
            w.writerows(report.choices_by_site(T, raw["image"]))
    report.fig_regret(raw[protocols[0]], out / "fig_regret.png", report.PROTO_LABEL[protocols[0]].lower())
    if a.images and a.annotations and "image" in raw:
        report.fig_examples(T, raw["image"], a.images, a.annotations, out / "fig_examples.png")
    print(f"\nTables and figures in {out}")
    return 0


def cmd_train(a) -> int:
    from .experiment import train_final_agent
    from .rewards import load_reward_table

    T = load_reward_table(a.table)
    agent = train_final_agent(T, a.policy, a.epochs)
    agent.save(a.out)
    print(f"Agent ({a.policy}, {len(T.arms)} actions, {len(T.features)} features) -> {a.out}")
    return 0


def cmd_apply(a) -> int:
    import numpy as np

    from ..indices import NDVI
    from ..pipeline import find_images, read_rgb
    from ..sensors.registry import load_sensor
    from ..spectral.engine import SpectralEngine
    from .actions import ARM_INDEX, ARMS, make_signals, run_arm
    from .bandit import TrainedAgent
    from .features import feature_vector, scene_features

    agent = TrainedAgent.load(a.agent)
    out = Path(a.output)
    (out / "boxes").mkdir(parents=True, exist_ok=True)
    sensor = load_sensor(a.sensor)
    engine = SpectralEngine()
    rows = []
    for p in find_images(Path(a.input), limit=a.limit):
        rgb, _ = read_rgb(p)
        if rgb.dtype != np.uint8:
            rgb = (rgb.astype(np.float32) / np.iinfo(rgb.dtype).max * 255 + 0.5).astype(np.uint8)
        scene = engine.prepare(rgb)
        bands = engine.render(scene, sensor, noise_std=0)
        sig = make_signals(rgb, NDVI().compute(bands[sensor.roles["red"]], bands[sensor.roles["nir"]]),
                           bands[sensor.roles["nir"]])
        f = scene_features(sig, scene.veg_fraction, sensor.id, a.gsd)
        x = np.array([f[k] for k in agent.features])
        arm_name = agent.choose(x)
        boxes = run_arm(ARMS[ARM_INDEX[arm_name]], sig, a.gsd)
        with open(out / "boxes" / f"{p.stem}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["xmin", "ymin", "xmax", "ymax"])
            w.writerows(boxes.astype(int).tolist())
        area_ha = rgb.shape[0] * rgb.shape[1] * a.gsd ** 2 / 1e4
        rows.append({"image": p.name, "sensor": sensor.id, "action": arm_name, "trees": len(boxes),
                     "trees_per_ha": round(len(boxes) / area_ha, 1),
                     "expected_reward": agent.expected_rewards(x)[arm_name]})
        print(f"{p.name:50s} {arm_name:20s} {len(boxes):5d} trees")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["image"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} images -> {out / 'summary.csv'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agrispectralsynth-agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rewards", help="evaluate every action on every annotated image (reward table)")
    r.add_argument("--images", type=Path, required=True)
    r.add_argument("--annotations", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--sensors", default=None, help="comma separated (default: the 4 drones)")
    r.add_argument("--gsd", type=float, default=0.1, help="ground sampling distance of the photos (m)")
    r.add_argument("--iou", type=float, default=0.4)
    r.add_argument("--count-weight", type=float, default=0.25)
    r.add_argument("--time-weight", type=float, default=0.0)
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(func=cmd_rewards)

    e = sub.add_parser("evaluate", help="cross-validate the policies on a reward table")
    e.add_argument("--table", type=Path, required=True)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--grouping", choices=["site", "image", "both"], default="both")
    e.add_argument("--folds", type=int, default=5)
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--epochs", type=int, default=3)
    e.add_argument("--images", type=Path, default=None, help="(optional) RGB folder, for the example figure")
    e.add_argument("--annotations", type=Path, default=None, help="(optional) annotations, for the example figure")
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("train", help="train the final agent on the whole table")
    t.add_argument("--table", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--policy", default="LinUCB")
    t.add_argument("--epochs", type=int, default=3)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("apply", help="choose a method per image and count trees")
    p.add_argument("--agent", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sensor", default="dji_mavic3m")
    p.add_argument("--gsd", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_apply)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
