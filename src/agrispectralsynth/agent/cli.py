"""
agrispectralsynth-agent: build rewards, evaluate policies, train and apply the agent.

    agrispectralsynth-agent rewards  --images data/benchmarks/NeonTreeEvaluation/evaluation/RGB \\
                                     --annotations data/benchmarks/NeonTreeEvaluation/annotations \\
                                     --out results/agent
    agrispectralsynth-agent evaluate --table results/agent --seeds 10
    agrispectralsynth-agent train    --table results/agent --out results/agent/agent.json
    agrispectralsynth-agent apply    --agent results/agent/agent.json --input data/raw --output results/conteo

DeepForest and real hyperspectral data (NEON) as extra actions:

    pip install -e ".[deepforest]"
    agrispectralsynth-agent download-deepforest
    python scripts/download_neontree_benchmark.py --with-hyperspectral
    agrispectralsynth-agent rewards ... --deepforest \
        --hsi-dir data/benchmarks/NeonTreeEvaluation/evaluation/Hyperspectral
    agrispectralsynth-agent evaluate --table results/agent --time-weight 0.05 --actions classical,df
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
        deepforest=a.deepforest, deepforest_weights=str(a.deepforest_weights) if a.deepforest_weights else None,
        hsi_dir=str(a.hsi_dir) if a.hsi_dir else None, hsi_bands_csv=str(a.hsi_bands) if a.hsi_bands else None,
    )
    if a.deepforest:
        from .deepforest import default_weights_path, torch_available

        if not torch_available():
            print('DeepForest needs torch + torchvision: pip install -e ".[deepforest]"')
            return 1
        w = Path(job.deepforest_weights or default_weights_path())
        if not w.exists():
            print(f"DeepForest weights not found at {w}. Run: agrispectralsynth-agent download-deepforest")
            return 1
    s = build_reward_table(a.images, a.annotations, a.out, job, workers=a.workers, limit=a.limit)
    print(f"{s['images']} images, {s['contexts']} contexts, {s['rewards']} rewards in {s['seconds']} s -> {s['out_dir']}")
    return 0 if not s["errors"] else 2


def _default_bands(hsi_dir):
    return Path(hsi_dir).parents[1] / "neon_aop_bands.csv" if hsi_dir else None


def _table_options(a) -> dict:
    """Reward weights and action set requested on the command line."""
    from .actions import arm_names_in
    from .evaluation import RewardWeights

    opts = {}
    if a.time_weight is not None or a.count_weight is not None:
        opts["weights"] = RewardWeights(0.25 if a.count_weight is None else a.count_weight,
                                        0.0 if a.time_weight is None else a.time_weight)
    if a.actions:
        opts["arms"] = arm_names_in([g.strip() for g in a.actions.split(",")])
    return opts


def cmd_download_deepforest(a) -> int:
    from .deepforest import WEIGHTS_URL, download_weights

    print(f"Downloading {WEIGHTS_URL} ...")
    print(f"-> {download_weights(a.out)}")
    return 0


def cmd_evaluate(a) -> int:
    from . import report
    from .experiment import cross_validate, summarize
    from .rewards import load_reward_table

    T = load_reward_table(a.table, **_table_options(a))
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
        report.fig_examples(T, raw["image"], a.images, a.annotations, out / "fig_examples.png",
                            hsi_dir=a.hsi_dir, bands_csv=a.hsi_bands or _default_bands(a.hsi_dir),
                            deepforest_weights=str(a.deepforest_weights) if a.deepforest_weights else None)
    print(f"\nTables and figures in {out}")
    return 0


def cmd_train(a) -> int:
    from .experiment import train_final_agent
    from .rewards import load_reward_table

    T = load_reward_table(a.table, **_table_options(a))
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
    from .actions import ARM_INDEX, ARMS, detect, make_signals, run_arm
    from .bandit import TrainedAgent
    from .features import feature_vector, scene_features

    agent = TrainedAgent.load(a.agent)
    specs = {n: ARMS[ARM_INDEX[n]] for n in agent.arms}
    can_df = False
    if any(sp.is_detector for sp in specs.values()):
        from .deepforest import default_weights_path, torch_available

        can_df = torch_available() and Path(a.deepforest_weights or default_weights_path()).exists()
        if not can_df:
            print("DeepForest (torch or weights) not available: its actions are left out")
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
        hsi_file = None
        if a.hsi_dir:
            from .hyperspectral import hsi_path

            hsi_file = hsi_path(a.hsi_dir, p.stem)
        allowed = [n for n, sp in specs.items()
                   if (not sp.needs_hsi or hsi_file is not None) and (not sp.is_detector or can_df)]
        arm_name = agent.choose(x, allowed)
        spec = specs[arm_name]
        if spec.needs_hsi:
            from .hyperspectral import attach_real_ndvi

            attach_real_ndvi(sig, hsi_file, a.hsi_bands or _default_bands(a.hsi_dir), sensor, rgb.shape[:2])
        if spec.is_detector:
            detect(sig, str(a.deepforest_weights) if a.deepforest_weights else None)
        boxes = run_arm(spec, sig, a.gsd)
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


def _hsi_args(parser) -> None:
    parser.add_argument("--hsi-dir", type=Path, default=None,
                        help="NEON hyperspectral tiles ({stem}_hyperspectral.tif): real-NDVI actions")
    parser.add_argument("--hsi-bands", type=Path, default=None,
                        help="neon_aop_bands.csv (default: two levels above --hsi-dir)")
    parser.add_argument("--deepforest-weights", type=Path, default=None,
                        help="DeepForest NEON.pt (default: models/deepforest/NEON.pt)")


def _selection_args(parser) -> None:
    parser.add_argument("--time-weight", type=float, default=None,
                        help="recompute the reward with this cost per second of processing")
    parser.add_argument("--count-weight", type=float, default=None,
                        help="recompute the reward with this weight of the count error")
    parser.add_argument("--actions", default=None,
                        help="comma separated action groups to keep: classical, df, hsi (default: all in the table)")


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
    _hsi_args(r)
    r.add_argument("--deepforest", action="store_true", help="add the DeepForest actions (needs torch + weights)")
    r.set_defaults(func=cmd_rewards)

    d = sub.add_parser("download-deepforest", help="download the DeepForest 1.0.0 NEON weights (~130 MB)")
    d.add_argument("--out", type=Path, default=None, help="default: models/deepforest/NEON.pt")
    d.set_defaults(func=cmd_download_deepforest)

    e = sub.add_parser("evaluate", help="cross-validate the policies on a reward table")
    e.add_argument("--table", type=Path, required=True)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--grouping", choices=["site", "image", "both"], default="both")
    e.add_argument("--folds", type=int, default=5)
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--epochs", type=int, default=3)
    e.add_argument("--images", type=Path, default=None, help="(optional) RGB folder, for the example figure")
    e.add_argument("--annotations", type=Path, default=None, help="(optional) annotations, for the example figure")
    _selection_args(e)
    _hsi_args(e)
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("train", help="train the final agent on the whole table")
    t.add_argument("--table", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--policy", default="LinUCB")
    t.add_argument("--epochs", type=int, default=3)
    _selection_args(t)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("apply", help="choose a method per image and count trees")
    p.add_argument("--agent", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sensor", default="dji_mavic3m")
    p.add_argument("--gsd", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None)
    _hsi_args(p)
    p.set_defaults(func=cmd_apply)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
