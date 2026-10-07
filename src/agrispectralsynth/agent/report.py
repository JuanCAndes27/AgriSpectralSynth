"""
Tables and figures for the agent experiments.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .experiment import FoldResult
from .rewards import RewardTable

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PROTO_COLORS = {"image": "#2a78d6", "site": "#eb6834"}
PROTO_LABEL = {"image": "Imágenes nuevas (sitios conocidos)", "site": "Sitios nuevos"}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def write_summary_csv(rows_by_protocol: Dict[str, List[dict]], path: Path) -> None:
    fields = ["protocol", "policy", "test_reward", "test_reward_sd", "test_f1", "test_f1_sd",
              "test_count_err", "test_count_err_sd", "inventory_err", "inventory_err_sd", "gap_closed"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for proto, rows in rows_by_protocol.items():
            for r in rows:
                w.writerow({"protocol": proto, **r})


def fig_policies(rows_by_protocol: Dict[str, List[dict]], path: Path) -> None:
    plt = _mpl()
    policies = [r["policy"] for r in rows_by_protocol["image"] if r["policy"] != "Oráculo"]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(policies) + 1.6))
    y = np.arange(len(policies))[::-1]
    for k, (proto, rows) in enumerate(rows_by_protocol.items()):
        d = {r["policy"]: r for r in rows}
        x = [d[p]["test_reward"] for p in policies]
        e = [d[p]["test_reward_sd"] for p in policies]
        off = 0.17 if k == 0 else -0.17
        ax.errorbar(x, y + off, xerr=e, fmt="o", ms=7, color=PROTO_COLORS[proto], ecolor=PROTO_COLORS[proto],
                    elinewidth=1.5, capsize=0, mec=SURFACE, mew=1.5, label=PROTO_LABEL[proto], zorder=3)
        orc = d["Oráculo"]["test_reward"]
        ax.axvline(orc, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(orc, len(policies) - 0.35, " oráculo", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(y, policies)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("Recompensa media en prueba (F1 − 0.25 · error de conteo relativo)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.set_title("Agente vs referencias (media ± d.e. sobre semillas)", loc="left", pad=26, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_regret(results: List[FoldResult], path: Path, protocol_label: str) -> None:
    plt = _mpl()
    curves = defaultdict(list)
    for r in results:
        if r.train_regret.size:
            curves[r.policy].append(r.train_regret)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, cs), col in zip(sorted(curves.items()), colors):
        n = min(len(c) for c in cs)
        m = np.mean([c[:n] for c in cs], axis=0)
        steps = np.arange(1, n + 1)
        ax.plot(steps, m / steps, color=col, lw=2, label=name)
        ax.text(steps[-1], (m / steps)[-1], f" {name}", color=INK2, fontsize=8, va="center")
    ax.set_xlabel("Pasos de entrenamiento (escena → acción → recompensa)")
    ax.set_ylabel("Arrepentimiento medio por paso")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
    ax.set_title(f"Aprendizaje en línea: distancia al oráculo — {protocol_label}", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sites(T: RewardTable, results: List[FoldResult], path: Path, policy: str = "LinUCB", min_images: int = 3) -> None:
    """Per site: reward of best fixed method, the agent, and the oracle (image protocol)."""
    plt = _mpl()
    agent = defaultdict(list)
    fixed = defaultdict(list)
    for r in results:
        if r.policy not in (policy, "Mejor método fijo"):
            continue
        tgt = agent if r.policy == policy else fixed
        for i, a in zip(r.test_idx, r.choices):
            tgt[T.sites[i]].append(T.R[i, a])
    sites = [s for s in np.unique(T.sites) if (T.sites == s).sum() // 4 >= min_images]
    sites = sorted(sites, key=lambda s: T.R[T.sites == s].max(1).mean())
    y = np.arange(len(sites))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(sites) + 1.5))
    for vals, col, lab in (([np.mean(fixed[s]) for s in sites], "#898781", "Mejor método fijo"),
                           ([np.mean(agent[s]) for s in sites], "#2a78d6", f"Agente ({policy})"),
                           ([T.R[T.sites == s].max(1).mean() for s in sites], "#0b0b0b", "Oráculo")):
        if lab == "Oráculo":
            ax.scatter(vals, y, s=160, color=col, linewidth=2, zorder=3, label=lab, marker="|")
        else:
            ax.scatter(vals, y, s=55, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=lab)
    counts = {s: (T.sites == s).sum() // 4 for s in sites}
    ax.set_yticks(y, [f"{s} ({counts[s]})" for s in sites])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("Recompensa media")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.set_title("Por sitio NEON (entre paréntesis: número de imágenes)", loc="left", pad=26, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def choices_by_site(T: RewardTable, results: List[FoldResult], policy: str = "LinUCB", top: int = 2) -> List[dict]:
    by = defaultdict(Counter)
    for r in results:
        if r.policy != policy:
            continue
        for i, a in zip(r.test_idx, r.choices):
            by[T.sites[i]][T.arms[a]] += 1
    rows = []
    for site, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(c.values())
        orc = Counter(T.arms[a] for a in T.R[T.sites == site].argmax(1))
        rows.append({
            "site": site,
            "agent_top": ", ".join(f"{a} ({100 * n / total:.0f} %)" for a, n in c.most_common(top)),
            "oracle_top": ", ".join(f"{a} ({100 * n / sum(orc.values()):.0f} %)" for a, n in orc.most_common(top)),
        })
    return rows


def fig_examples(T: RewardTable, results: List[FoldResult], images_dir: Path, annotations_dir: Path, path: Path,
                 sites=("SJER", "TEAK", "NIWO", "OSBS"), policy: str = "LinUCB", sensor: str = "dji_mavic3m",
                 gsd_m: float = 0.1) -> None:
    """Test-fold examples: annotated crowns (yellow) vs crowns of the method the agent chose (magenta)."""
    import cv2

    from ..indices import NDVI
    from ..pipeline import read_rgb
    from ..sensors.registry import load_sensor
    from ..spectral.engine import SpectralEngine
    from .actions import ARM_INDEX, ARMS, make_signals, run_arm
    from .evaluation import score_detections
    from .groundtruth import load_annotations

    plt = _mpl()
    gts = load_annotations(annotations_dir)
    seed0 = min(r.seed for r in results)
    choice = {}
    for r in results:
        if r.policy == policy and r.seed == seed0:
            for i, a in zip(r.test_idx, r.choices):
                choice[i] = a
    picks = []
    for site in sites:
        idx = [i for i in choice if T.sites[i] == site and T.sensors[i] == sensor]
        if idx:
            # the image closest to the site's median agent reward: typical, not cherry-picked
            vals = np.array([T.R[i, choice[i]] for i in idx])
            picks.append(idx[int(np.argmin(np.abs(vals - np.median(vals))))])
    eng, sen = SpectralEngine(), load_sensor(sensor)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 3.9))
    for ax, i in zip(np.atleast_1d(axes), picks):
        stem = T.images[i]
        img_path = next(Path(images_dir).glob(f"{stem}.*"))
        rgb, _ = read_rgb(img_path)
        sc = eng.prepare(rgb)
        b = eng.render(sc, sen, noise_std=0)
        sig = make_signals(rgb, NDVI().compute(b[sen.roles["red"]], b[sen.roles["nir"]]), b[sen.roles["nir"]])
        arm = T.arms[choice[i]]
        pred = run_arm(ARMS[ARM_INDEX[arm]], sig, gsd_m)
        sc_ = score_detections(pred, gts[stem].boxes)
        im = rgb.copy()
        for x0, y0, x1, y1 in gts[stem].boxes.astype(int):
            cv2.rectangle(im, (x0, y0), (x1, y1), (255, 214, 0), 2)
        for x0, y0, x1, y1 in pred.astype(int):
            cv2.rectangle(im, (x0, y0), (x1, y1), (232, 63, 164), 1)
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{T.sites[i]} · {arm}\nreales {sc_.n_true} · detectadas {sc_.n_pred} · F1 {sc_.f1:.2f}",
                     loc="left", fontsize=8.5)
    fig.suptitle("Elección del agente en imágenes de prueba (amarillo: copas anotadas; magenta: detectadas)",
                 x=0.01, y=0.99, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140)
    plt.close(fig)
