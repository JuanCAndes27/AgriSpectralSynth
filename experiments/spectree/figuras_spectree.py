#!/usr/bin/env python3
"""Figures of the SpecTree experiments report (Spanish labels)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

FIG = C.ROOT / "figs"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, YELLOW, PURPLE, GRAY, INK2, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7",
                                                         "#898781", "#52514e", "#e1e0d9")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#a8a79f",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 200,
                     "figure.facecolor": "white"})


def forest(ax, labels, est, lo, hi, colors, ref=0.0, xlabel=""):
    y = np.arange(len(labels))[::-1]
    for yi, e, l, h, c in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(e, yi, "o", color=c, ms=6, mec="white", mew=1)
    ax.axvline(ref, color=GRAY, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_selectores():
    s = pd.read_csv(C.OUT / "e03_resumen.csv")
    s = s[~s.policy.isin(["SBS", "Random"])].sort_values("d_vs_SBS", ascending=False)
    col = {"VBS": GRAY, "Ridge": GREEN, "Trees": GREEN, "Selector-VI": GREEN, "CheapRule": YELLOW}
    names = {"VBS": "Oráculo (VBS)", "Trees": "ExtraTrees (info. completa)", "Ridge": "Ridge (info. completa)",
             "Selector-VI": "Selector elegido en val. interna (P2)", "CheapRule": "Regla barata", "LinUCB": "LinUCB disjunto",
             "LinTS": "Thompson lineal", "LinUCB-F": "LinUCB-F", "LinUCB-F-greedy": "LinUCB-F sin exploración",
             "LinUCB-F (1 pasada)": "LinUCB-F, 1 pasada"}
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    forest(ax, [names.get(p, p) for p in s.policy], s.d_vs_SBS, s.ci_lo, s.ci_hi, [col.get(p, BLUE) for p in s.policy],
           xlabel="Δ recompensa frente a SBS (macro por sitio; IC 95 % por remuestreo de sitios; P2 con 98.75 %)")
    ax.set_title("E03 · 40 acciones RGB, sitios retenidos. Verde: información completa; azul: bandidos",
                 loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e03_selectores.png")
    plt.close(fig)


def fig_contexto():
    c = pd.read_csv(C.OUT / "e01_comparaciones.csv")
    c = c[c.comparacion.isin(["C1 - C0", "C2 - C0", "C3 - C0", "C2 - C3", "C4 - C0"])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    lab = {"C1 - C0": "C1 (+NDVI sint.) − C0", "C2 - C0": "C2 (+9 sintéticos) − C0", "C3 - C0": "C3 (+9 RGB) − C0",
           "C2 - C3": "C2 − C3 (igual dimensión)", "C4 - C0": "C4 (+NDVI real) − C0"}
    for ax, pol in zip(axes, ("LinUCB-F", "Ridge", "LinUCB")):
        g = c[c.selector == pol]
        cols = [PURPLE if x == "C4 - C0" else (BLUE if "C2" in x or "C1" in x else ORANGE) for x in g.comparacion]
        forest(ax, [lab[x] for x in g.comparacion], g.d_recompensa, g.ci_lo, g.ci_hi, cols, xlabel="Δ recompensa")
        ax.set_title(pol + (" — P1: C2−C0 con IC 98.75 %" if pol == "LinUCB-F" else ""), loc="left", fontsize=9)
    fig.suptitle("E01 · Mismas acciones (40), mismas particiones: solo cambia el contexto", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e01_contexto.png")
    plt.close(fig)


def fig_filtro():
    cu = pd.read_csv(C.OUT / "e02_curva.csv")
    er = pd.read_csv(C.OUT / "e02_error_radiometrico.csv")
    auc = pd.read_csv(C.OUT / "e02_auc.csv").set_index("indice").auc_TP_vs_FP
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for idx, col, lab in (("syn", ORANGE, "NDVI sintético"), ("real", GREEN, "NDVI real")):
        g = cu[cu.indice == idx]
        ax.plot(g.FP_rechazados, g.TP_retenidos, "-o", color=col, ms=2.5, lw=1.4, label=f"{lab} (AUC TP/FP = {auc[idx]:.2f})")
    ax.plot([0, 1], [1, 0], color=GRAY, ls="--", lw=1, label="filtro sin información")
    ax.set_xlabel("Fracción de falsas detecciones rechazadas")
    ax.set_ylabel("Fracción de aciertos conservados")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("(a) Cajas de DeepForest (s = 0.3) al variar el umbral", loc="left", fontsize=9)
    ax.grid(color=GRID, lw=0.5)
    ax = axes[1]
    order = ["rojo", "NIR", "NDVI", "NDVI calibrado"]
    er = er.set_index("magnitud").loc[order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, er.sesgo, 0.4, color=ORANGE, label="sesgo")
    ax.bar(x + 0.2, er.rmse, 0.4, color=BLUE, label="RMSE")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Error radiométrico sintético − real (1 m, macro por sitio)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e02_filtro.png")
    plt.close(fig)


def fig_estratos():
    s = pd.read_csv(C.OUT / "e05_estratos.csv")
    s = s[s.estrato.isin(["bajo", "medio", "alto"]) & s.policy.isin(["SBS", "Ridge", "Trees", "VBS", "LinUCB-F"])]
    descs = [("densidad", "Densidad (árb/ha)"), ("tamano", "Tamaño de copa"), ("proximidad", "Separación entre copas"),
             ("sombra_proxy", "Proxy de sombra*")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    col = {"SBS": ORANGE, "Ridge": GREEN, "Trees": "#0d7a55", "VBS": GRAY, "LinUCB-F": BLUE}
    for ax, (d, lab) in zip(axes, descs):
        g = s[s.descriptor == d]
        for k, pol in enumerate(["SBS", "LinUCB-F", "Ridge", "Trees", "VBS"]):
            gg = g[g.policy == pol].set_index("estrato").reindex(["bajo", "medio", "alto"])
            ax.plot(np.arange(3) + (k - 2) * 0.07, gg.recompensa, "o-" if pol != "VBS" else "^--", color=col[pol], ms=4,
                    lw=1.2, label=pol)
        n = g[g.policy == "SBS"].set_index("estrato").reindex(["bajo", "medio", "alto"])
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{e}\n{int(p)} parc./{int(si)} sit." for e, p, si in zip(n.index, n.parcelas, n.sitios)], fontsize=7)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[0].set_ylabel("Recompensa (macro por sitio)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("E05 · Terciles fijados en entrenamiento (cajas anotadas; *proxy no validado)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e05_estratos.png")
    plt.close(fig)


def fig_generador():
    e = pd.read_csv(C.OUT / "e06_error_espectral.csv").set_index("variante")
    u = pd.read_csv(C.OUT / "e06_utilidad_contexto.csv")
    names = {"V0": "V0 motor completo", "V1": "V1 ExG lineal", "V2": "V2 f constante", "V3": "V3 sin cromática",
             "V4": "V4 umbrales ExG laxos", "V5": "V5 fracción por VARI", "V6": "V6 regresión supervisada*"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    for v, nm in names.items():
        if v == "V2":
            continue
        ax.plot(e.loc[v, "sesgo"], e.loc[v, "r"], "o", color=GREEN if v == "V6" else (BLUE if v == "V0" else ORANGE), ms=7)
        off = {"V0": (6, -12), "V1": (6, 8), "V3": (-70, 8), "V4": (6, -10), "V5": (6, 3), "V6": (-120, 6)}.get(v, (5, 3))
        ax.annotate(nm, (e.loc[v, "sesgo"], e.loc[v, "r"]), textcoords="offset points", xytext=off, fontsize=7.5)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("Sesgo de NDVI (sintético − real), macro por sitio")
    ax.set_ylabel("r dentro de cada parcela (media)")
    ax.set_title("(a) Fidelidad espectral a 1 m", loc="left", fontsize=9)
    ax.grid(color=GRID, lw=0.5)
    ax = axes[1]
    g = u[(u.selector == "Ridge") & (u.variante != "C0")].set_index("variante").reindex(list(names))
    forest(ax, [names[v] for v in g.index], g.d_vs_C0, g.ci_lo_C0, g.ci_hi_C0,
           [GREEN if v == "V6" else (BLUE if v == "V0" else ORANGE) for v in g.index],
           xlabel="Δ recompensa frente a C0 (Ridge, acciones fijas)")
    ax.set_title("(b) Utilidad como contexto (exploratorio)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e06_generador.png")
    plt.close(fig)


def fig_sensibilidad():
    g = pd.read_csv(C.OUT / "e08_grilla.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, pol in zip(axes, ("LinUCB-F", "Ridge")):
        p = g[g.policy == pol].pivot(index="kappa", columns="lambda", values="d_vs_SBS")
        lo = g[g.policy == pol].pivot(index="kappa", columns="lambda", values="ci_lo")
        hi = g[g.policy == pol].pivot(index="kappa", columns="lambda", values="ci_hi")
        lim = np.nanmax(np.abs(g[g.policy.isin(["LinUCB-F", "Ridge"])].d_vs_SBS))
        im = ax.imshow(p.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                sig = "*" if (lo.values[i, j] > 0 or hi.values[i, j] < 0) else ""
                ax.text(j, i, f"{p.values[i, j]:+.3f}{sig}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(p.shape[1])); ax.set_xticklabels([f"{x:g}" for x in p.columns])
        ax.set_yticks(range(p.shape[0])); ax.set_yticklabels([f"{x:g}" for x in p.index])
        ax.set_xlabel("λ (s⁻¹)"); ax.set_ylabel("κ")
        ax.set_title(f"{pol} − SBS (* IC 95 % excluye 0)", loc="left", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.suptitle("E08 · Sensibilidad a los pesos de la recompensa (40 acciones, sitios retenidos)", x=0.01, y=1.06, ha="left", fontsize=10)
    fig.savefig(FIG / "e08_sensibilidad.png", bbox_inches="tight")
    plt.close(fig)


def fig_latencia():
    t = pd.read_csv(C.ROOT / "timings.csv")
    t = t[t.rep >= 0]
    comp = t.groupby("policy")[["sintesis", "contexto", "seleccion", "accion", "posproceso"]].mean()
    s = pd.read_csv(C.OUT / "e04_resumen.csv").set_index("policy")
    order = ["Clasico-fijo", "Regla-barata", "LinUCB-F", "DF-fijo"]
    comp = comp.reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    left = np.zeros(len(order))
    cols = {"sintesis": PURPLE, "contexto": YELLOW, "seleccion": GRAY, "accion": BLUE, "posproceso": GREEN}
    labs = {"sintesis": "síntesis", "contexto": "contexto", "seleccion": "selección", "accion": "acción", "posproceso": "posproceso"}
    for k in cols:
        ax.barh(range(len(order)), comp[k], left=left, color=cols[k], label=labs[k], height=0.6)
        left += comp[k].to_numpy()
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("Segundos por parcela (media, 1 hilo CPU)")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right")
    ax.set_title("(a) Latencia total desglosada", loc="left", fontsize=9)
    ax = axes[1]
    for p, c in zip(order, (ORANGE, YELLOW, BLUE, GREEN)):
        ax.errorbar(s.loc[p, "mediana"], s.loc[p, "f1_macro_189"], xerr=[[0.0], [max(0.0, s.loc[p, "p95"] - s.loc[p, "mediana"])]],
                    fmt="o", color=c, ms=7, capsize=3, label=f"{p} (MAE conteo {s.loc[p, 'mae_conteo_189']:.1f})")
    ax.set_xlabel("Latencia total por parcela, s (punto: mediana; barra: hasta p95)")
    ax.set_ylabel("F1 macro por sitio (189 parcelas)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(b) Calidad frente a latencia", loc="left", fontsize=9)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "e04_latencia.png")
    plt.close(fig)


def fig_resolucion():
    b = pd.read_csv(C.OUT / "e10b_resolucion_por_algoritmo.csv").set_index("algoritmo")
    cols = ["NDVI sintético (10 cm)", "NDVI sintético degradado a 1 m", "NDVI real (1 m)"]
    b = b[cols]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(b))
    for k, (c, col) in enumerate(zip(cols, (ORANGE, YELLOW, GREEN))):
        ax.bar(x + (k - 1) * 0.27, b[c], 0.27, color=col, label=c)
    ax.set_xticks(x); ax.set_xticklabels(b.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("F1 macro por sitio")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_ylim(0, 0.42)
    ax.set_title("E10b · Control de resolución: mismos algoritmos sobre las tres señales", loc="left", fontsize=9, pad=24)
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "e10_resolucion.png")
    plt.close(fig)


def fig_externo():
    a = pd.read_csv(C.OUT / "e07_acciones.csv")
    s = pd.read_csv(C.OUT / "e07_resumen.csv")
    fam_c = {"DF": BLUE, "DF+NDVIsint": PURPLE, "Clas-RGB": ORANGE, "Clas-NDVIsint": GREEN}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    for fam, g in a.groupby("familia"):
        ax1.scatter(g.reward_neon_macro, g.reward_ext, s=22, color=fam_c[fam], label=fam, edgecolor="white", lw=0.5)
    lim = [min(a.reward_neon_macro.min(), a.reward_ext.min()) - 0.03, max(a.reward_neon_macro.max(), a.reward_ext.max()) + 0.03]
    ax1.plot(lim, lim, color=GRAY, lw=1, ls="--")
    ax1.set_xlabel("Recompensa en NEON (macro por sitio, Mavic 3M)")
    ax1.set_ylabel("Recompensa externa (Campo Grande)")
    ax1.set_title("Cada acción fija: NEON frente a externo", fontsize=9, loc="left")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax1.grid(color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    keep = ["VBS-ext", "Selector-VI", "Trees", "Ridge", "Ridge·C0", "CheapRule", "LinUCB-F", "SBS", "DF-defecto"]
    s = s.set_index("policy").loc[[k for k in keep if k in s.policy.values]].reset_index()
    lab = {"VBS-ext": "VBS externo (oráculo)", "CheapRule": "Regla barata", "DF-defecto": "DeepForest s=0.1"}
    y = np.arange(len(s))[::-1]
    ax2.barh(y, s.reward, color=[GRAY if p == "VBS-ext" else (INK2 if p == "SBS" else BLUE) for p in s.policy], height=0.6)
    for yi, p, r in zip(y, s.policy, s.reward):
        if p == "SBS" or pd.isna(s.set_index("policy").neon_reward_macro.get(p, np.nan)):
            continue
        ax2.plot(s.set_index("policy").neon_reward_macro[p], yi, "|", color=ORANGE, ms=12, mew=2)
    ax2.plot([], [], "|", color=ORANGE, ms=10, mew=2, label="misma política en NEON (VC por sitio)")
    sbs_neon = s.set_index("policy").neon_reward_macro.get("SBS", np.nan)
    if not pd.isna(sbs_neon):
        ax2.plot(sbs_neon, y[list(s.policy).index("SBS")], "|", color=ORANGE, ms=12, mew=2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([lab.get(p, p) for p in s.policy])
    ax2.set_xlabel("Recompensa media por parche")
    ax2.set_title("Políticas entrenadas solo con NEON", fontsize=9, loc="left")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "e07_externo.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_selectores, fig_contexto, fig_filtro, fig_estratos, fig_generador, fig_sensibilidad, fig_latencia, fig_resolucion,
              fig_externo):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("pendiente", f.__name__, e)
