#!/usr/bin/env python3
"""
Counting experiment on NEON (NeonTreeEvaluation) with DeepForest and real
hyperspectral data.

    agrispectralsynth-agent rewards --images .../evaluation/RGB --annotations .../annotations \\
        --out results/agent_neon --deepforest --hsi-dir .../evaluation/Hyperspectral
    python scripts/experimento_neon_hiperespectral.py --table results/agent_neon --out docs/agente/neon_hiperespectral

Questions answered (tables + figures in --out):

1. Counting by family of methods (best method of each family chosen on the
   training folds): RGB-only classical, classical on SYNTHETIC NDVI, classical
   on REAL NDVI (NEON hyperspectral integrated with each drone's bands),
   DeepForest, DeepForest filtered by synthetic NDVI, DeepForest filtered by
   real NDVI.                                              -> familias.csv, fig_familias.png
2. The agent with different action sets (what each kind of data adds).
                                                           -> conjuntos_acciones.csv
3. Time cost: reward = F1 - 0.25 count error - lambda * seconds. How the agent
   trades DeepForest (accurate, slow) for classical methods (fast).
                                                           -> costo_tiempo.csv, fig_costo_tiempo.png
4. Synthetic vs real NDVI, per site and sensor (validation of the simulator).
                                                           -> validacion_ndvi.csv, fig_validacion_ndvi.png
5. Per-site F1 of the families.                            -> sitios.csv, fig_sitios.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from agrispectralsynth.agent.actions import ARM_INDEX, ARMS, arm_names_in
from agrispectralsynth.agent.evaluation import RewardWeights
from agrispectralsynth.agent.experiment import POLICIES, cross_validate, summarize
from agrispectralsynth.agent.report import GRID, INK2, MUTED, SURFACE, _mpl
from agrispectralsynth.agent.rewards import load_reward_table

SENSOR_LABEL = {"dji_mavic3m": "Mavic 3M", "dji_phantom4m": "Phantom 4M",
                "micasense_rededge_mx": "RedEdge-MX", "parrot_sequoia_plus": "Sequoia+"}


def family_of(name: str) -> str:
    a = ARMS[ARM_INDEX[name]]
    if a.is_detector:
        return {"rgb": "DeepForest (RGB)", "ndvi": "DeepForest + NDVI sintético",
                "hsi": "DeepForest + NDVI real"}[a.signal]
    return {"exg": "Clásico RGB", "dark": "Clásico RGB", "ndvi": "Clásico NDVI sintético",
            "hsi": "Clásico NDVI real"}[a.signal]


FAMILIES = ["Clásico RGB", "Clásico NDVI sintético", "Clásico NDVI real",
            "DeepForest (RGB)", "DeepForest + NDVI sintético", "DeepForest + NDVI real"]
FAMILY_COLOR = {"Clásico RGB": "#898781", "Clásico NDVI sintético": "#eb6834", "Clásico NDVI real": "#1baf7a",
                "DeepForest (RGB)": "#2a78d6", "DeepForest + NDVI sintético": "#eda100",
                "DeepForest + NDVI real": "#4a3aa7"}

ACTION_SETS = {
    "Clásicas (v0.4)": ["classical"],
    "Clásicas + NDVI real": ["classical", "hsi"],
    "Clásicas + DeepForest": ["classical", "df"],
    "Todas": ["classical", "df", "hsi"],
}

AGENTS = {k: POLICIES[k] for k in ("LinUCB factorizado", "LinUCB")}


def chosen_stats(T, results, policy):
    """Mean seconds per context and fraction of DeepForest choices of a policy (pooled over folds/seeds)."""
    secs, df = [], []
    for r in results:
        if r.policy == policy:
            secs.append(T.seconds[r.test_idx, r.choices])
            df.append(np.array([T.arms[c].startswith("df_") for c in r.choices]))
    return float(np.concatenate(secs).mean()), float(np.concatenate(df).mean())


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 1. Families
# ---------------------------------------------------------------------------

def families(table, seeds, out):
    rows, best_arms = [], {}
    for fam in FAMILIES:
        arms = [a.name for a in ARMS if family_of(a.name) == fam]
        T = load_reward_table(table, arms=arms)
        for grouping in ("site", "image"):
            res = cross_validate(T, k=5, seeds=range(seeds), epochs=1, policies={}, grouping=grouping)
            s = {r["policy"]: r for r in summarize(res)}
            b = s["Mejor método fijo"]
            secs, _ = chosen_stats(T, res, "Mejor método fijo")
            picks = [T.arms[r.choices[0]] for r in res if r.policy == "Mejor método fijo"]
            top = max(set(picks), key=picks.count)
            rows.append({"familia": fam, "validacion": grouping, "metodo_mas_elegido": top,
                         "F1": b["test_f1"], "F1_sd": b["test_f1_sd"], "error_conteo_rel": b["test_count_err"],
                         "error_inventario": b["inventory_err"], "recompensa": b["test_reward"],
                         "oraculo_familia_F1": s["Oráculo"]["test_f1"], "segundos_img": round(secs, 3)})
            if grouping == "site":
                best_arms[fam] = top
    write_csv(out / "familias.csv", rows)

    plt = _mpl()
    site = [r for r in rows if r["validacion"] == "site"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(site))[::-1]
    for ax, key, lab in ((axes[0], "F1", "F1 (IoU ≥ 0.4) en sitios nuevos"),
                         (axes[1], "error_inventario", "Error de inventario (Σ predichos − Σ reales) / Σ reales")):
        vals = [r[key] for r in site]
        ax.barh(y, vals, color=[FAMILY_COLOR[r["familia"]] for r in site], height=0.62)
        for yi, v, r in zip(y, vals, site):
            pos = max(v, r["oraculo_familia_F1"]) if key == "F1" else v
            ax.text(pos + (0.015 if v >= 0 else -0.015), yi, f"{v:+.0%}" if key != "F1" else f"{v:.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([r["familia"] for r in site] if key == "F1" else [])
        ax.set_xlabel(lab, fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        if key == "F1":
            ax.set_xlim(0, max(vals) * 1.18)
            for yi, r in zip(y, site):
                ax.plot(r["oraculo_familia_F1"], yi, "|", color=INK2, ms=14, mew=1.5)
        else:
            ax.axvline(0, color=MUTED, lw=1)
            lo, hi = min(vals + [0]), max(vals + [0])
            ax.set_xlim(lo - 0.35, hi + 0.35)
    fig.suptitle("Conteo de copas en NEON por familia de métodos (mejor método de cada familia, elegido en entrenamiento)\n"
                 "| = oráculo dentro de la familia", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_familias.png", dpi=150)
    plt.close(fig)
    return rows, best_arms


# ---------------------------------------------------------------------------
# 2. Action sets
# ---------------------------------------------------------------------------

def action_sets(table, seeds, out):
    rows = []
    for label, groups in ACTION_SETS.items():
        T = load_reward_table(table, arms=arm_names_in(groups))
        for grouping in ("site", "image"):
            res = cross_validate(T, k=5, seeds=range(seeds), epochs=3, policies=AGENTS, grouping=grouping)
            for r in summarize(res):
                if r["policy"] in ("Random",):
                    continue
                secs, dff = chosen_stats(T, res, r["policy"])
                rows.append({"acciones": label, "n_acciones": len(T.arms), "validacion": grouping,
                             "politica": r["policy"], "recompensa": r["test_reward"], "F1": r["test_f1"],
                             "error_conteo_rel": r["test_count_err"], "error_inventario": r["inventory_err"],
                             "brecha_cerrada": r["gap_closed"], "segundos_img": round(secs, 3),
                             "frac_deepforest": round(dff, 3)})
    write_csv(out / "conjuntos_acciones.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# 3. Time cost sweep
# ---------------------------------------------------------------------------

def cost_sweep(table, seeds, out, lambdas=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)):
    rows = []
    for lam in lambdas:
        T = load_reward_table(table, weights=RewardWeights(0.25, lam))
        res = cross_validate(T, k=5, seeds=range(seeds), epochs=3, grouping="site",
                             policies={"LinUCB factorizado": POLICIES["LinUCB factorizado"]})
        for r in summarize(res):
            if r["policy"] not in ("LinUCB factorizado", "Mejor método fijo", "Oráculo"):
                continue
            secs, dff = chosen_stats(T, res, r["policy"])
            rows.append({"lambda_s": lam, "politica": r["policy"], "F1": r["test_f1"],
                         "error_conteo_rel": r["test_count_err"], "recompensa": r["test_reward"],
                         "segundos_img": round(secs, 3), "frac_deepforest": round(dff, 3)})
    write_csv(out / "costo_tiempo.csv", rows)
    plot_cost(out)
    return rows


def plot_cost(out):
    """Figure of the time-cost sweep, from costo_tiempo.csv."""
    with open(out / "costo_tiempo.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lambdas = list(dict.fromkeys(float(r["lambda_s"]) for r in rows))
    df_secs = np.median([float(r["segundos_img"]) for r in rows
                         if r["politica"] == "Mejor método fijo" and float(r["frac_deepforest"]) == 1.0])
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    style = {"LinUCB factorizado": ("#2a78d6", "o", "Agente (LinUCB factorizado)"),
             "Mejor método fijo": ("#eb6834", "s", "Mejor método fijo"),
             "Oráculo": ("#898781", "^", "Oráculo")}
    for pol, (c, m, lab) in style.items():
        rs = [r for r in rows if r["politica"] == pol]
        x = np.arange(len(rs))
        for ax, key, k in ((axes[0], "recompensa", 1), (axes[1], "F1", 1), (axes[2], "frac_deepforest", 100)):
            ax.plot(x, [k * float(r[key]) for r in rs], marker=m, color=c, label=lab, lw=1.8, ms=6, mec=SURFACE)
    for ax, lab in zip(axes, ("Recompensa (lo que optimiza el agente)", "F1 en sitios nuevos",
                              "% de imágenes con DeepForest")):
        ax.set_xticks(np.arange(len(lambdas)))
        ax.set_xticklabels([f"{v:g}" for v in lambdas], fontsize=8.5)
        ax.set_xlabel("λ: costo por segundo de proceso")
        ax.set_title(lab, loc="left", fontsize=9.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[2].set_ylim(-3, 103)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(f"Precisión vs. costo (sitios nuevos): DeepForest ≈ {df_secs:.1f} s/img (CPU, 1 hilo) "
                 "frente a ≈ 0.03 s de los métodos clásicos", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_costo_tiempo.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Synthetic vs real NDVI
# ---------------------------------------------------------------------------

def ndvi_validation(table, out):
    import pandas as pd

    v = pd.read_csv(Path(table) / "hsi_validation.csv")
    g = v.groupby(["site", "sensor"]).agg(
        imagenes=("image", "count"), ndvi_real=("real_ndvi_mean", "mean"), ndvi_sintetico=("syn_ndvi_mean", "mean"),
        sesgo=("bias", "mean"), rmse=("rmse", "mean"), r_mediana=("r", "median"),
        rojo_real=("real_red", "mean"), rojo_sint=("syn_red", "mean"),
        nir_real=("real_nir", "mean"), nir_sint=("syn_nir", "mean")).round(3).reset_index()
    g.to_csv(out / "validacion_ndvi.csv", index=False)
    s = v.groupby("sensor").agg(ndvi_real=("real_ndvi_mean", "mean"), ndvi_sintetico=("syn_ndvi_mean", "mean"),
                                sesgo=("bias", "mean"), rmse=("rmse", "mean"), r_mediana=("r", "median"),
                                rojo_real=("real_red", "mean"), rojo_sint=("syn_red", "mean"),
                                nir_real=("real_nir", "mean"), nir_sint=("syn_nir", "mean")).round(3).reset_index()
    px = np.load(Path(table) / "hsi_pixels.npz")
    s["r_pixeles"] = [round(float(np.corrcoef(px[k][0], px[k][1])[0, 1]), 3) for k in s["sensor"]]
    s.to_csv(out / "validacion_ndvi_sensores.csv", index=False)

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    real, syn = px["dji_mavic3m"]
    hb = axes[0].hexbin(real, syn, gridsize=60, extent=(-0.2, 1, -0.2, 1), mincnt=1, bins="log", cmap="Blues")
    axes[0].plot([-0.2, 1], [-0.2, 1], color=MUTED, lw=1, ls="--")
    axes[0].set_xlabel("NDVI real (NEON AOP integrado con bandas del Mavic 3M)")
    axes[0].set_ylabel("NDVI sintético (desde la foto RGB)")
    axes[0].set_title(f"Píxeles de 1 m, n = {real.size:,}, r = {np.corrcoef(real, syn)[0, 1]:.2f}".replace(",", " "),
                      loc="left", fontsize=9)
    fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.02).set_label("píxeles (log)", fontsize=8)

    m = g[g["sensor"] == "dji_mavic3m"].sort_values("ndvi_real")
    y = np.arange(len(m))
    axes[1].hlines(y, m["ndvi_sintetico"], m["ndvi_real"], color=GRID, lw=3)
    axes[1].plot(m["ndvi_real"], y, "o", color="#1baf7a", label="real (hiperespectral)", ms=6)
    axes[1].plot(m["ndvi_sintetico"], y, "o", color="#eb6834", label="sintético", ms=6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{a} (n={b})" for a, b in zip(m["site"], m["imagenes"])], fontsize=8)
    axes[1].set_xlabel("NDVI medio de la parcela (Mavic 3M)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[1].grid(axis="x", color=GRID, lw=0.8)
    axes[1].set_axisbelow(True)
    fig.suptitle("Validación del simulador: NDVI sintético vs NDVI real de NEON", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "fig_validacion_ndvi.png", dpi=150)
    plt.close(fig)
    return g, s


# ---------------------------------------------------------------------------
# 5. Sites
# ---------------------------------------------------------------------------

def sites(table, best_arms, out, min_images=3):
    T = load_reward_table(table)
    rows = []
    for site in np.unique(T.sites):
        m = T.sites == site
        n_img = len(np.unique(T.images[m]))
        if n_img < min_images:
            continue
        row = {"sitio": site, "imagenes": n_img, "arboles": int(T.n_true[m].sum() / len(np.unique(T.sensors[m])))}
        for fam, arm in best_arms.items():
            row[fam] = round(float(T.F1[m, T.arms.index(arm)].mean()), 3)
        rows.append(row)
    rows.sort(key=lambda r: -r["DeepForest (RGB)"])
    write_csv(out / "sitios.csv", rows)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    fams = [f for f in FAMILIES if f in best_arms]
    for k, fam in enumerate(fams):
        off = 0.3 - 0.6 * k / max(1, len(fams) - 1)          # one row per family inside each site
        ax.plot([r[fam] for r in rows], y + off, "o", color=FAMILY_COLOR[fam], label=f"{fam}: {best_arms[fam]}",
                ms=5.5, mec=SURFACE, mew=0.8)
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color=GRID, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['sitio']} ({r['imagenes']} img)" for r in rows], fontsize=8.5)
    ax.set_xlabel("F1 medio del método (los 4 drones)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("F1 por sitio NEON del mejor método de cada familia", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sitios.png", dpi=150)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip", default="", help="comma separated parts to skip: familias,acciones,costo,validacion,sitios")
    ap.add_argument("--plots-only", action="store_true", help="redraw the time-cost figure from its CSV")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plots_only:
        plot_cost(a.out)
        return
    skip = set(a.skip.split(","))
    best = {}
    if "familias" not in skip:
        fr, best = families(a.table, a.seeds, a.out)
        for r in fr:
            print(f"{r['validacion']:5s} {r['familia']:30s} {r['metodo_mas_elegido']:20s} F1 {r['F1']:.3f} "
                  f"inv {r['error_inventario']:+.1%} {r['segundos_img']:.2f} s")
    if "validacion" not in skip:
        _, s = ndvi_validation(a.table, a.out)
        print(s.to_string())
    if "sitios" not in skip and best:
        sites(a.table, best, a.out)
    if "acciones" not in skip:
        for r in action_sets(a.table, a.seeds, a.out):
            print(f"{r['acciones']:24s} {r['validacion']:5s} {r['politica']:30s} R {r['recompensa']:.3f} "
                  f"F1 {r['F1']:.3f} DF {r['frac_deepforest']:.0%} {r['segundos_img']:.2f} s")
    if "costo" not in skip:
        for r in cost_sweep(a.table, a.seeds, a.out):
            print(f"λ={r['lambda_s']:<6g} {r['politica']:22s} F1 {r['F1']:.3f} DF {r['frac_deepforest']:.0%} "
                  f"{r['segundos_img']:.2f} s")


if __name__ == "__main__":
    main()
