#!/usr/bin/env python3
"""
E00 — rebuild the tables from per-plot records, reconcile with v0.5, quantify uncertainty by site.

Outputs (results/spectree/out):
  e00_conciliacion.csv      regenerated vs v0.5 rewards.csv (per action)
  e00_resumen_rgb40.csv     main policies, RGB portfolio (40), v0.5 context (25 descriptors)
  e00_resumen_all58.csv     same with the 58-action reference portfolio
  e00_por_sitio.csv         per-site reward and difference vs SBS
  policy_decisions.csv      every test decision (fold, seed, policy, plot, sensor, action, reward)
  splits.json, actions.csv, features.json, config.yaml
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

OLD_REWARDS = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def reconcile():
    if OLD_REWARDS is None or not OLD_REWARDS.exists():
        return None
    old = pd.read_csv(OLD_REWARDS)
    new = C.metrics(0.4, "hung")
    m = old.merge(new, on=["image", "sensor", "action" if "action" in old else "arm"], how="inner",
                  suffixes=("_v05", "_new")) if "action" in old else \
        old.rename(columns={"arm": "action"}).merge(new, on=["image", "sensor", "action"], suffixes=("_v05", "_new"))
    m["d_f1"] = m.f1_new - m.f1_v05
    m["d_npred"] = m.n_pred_new - m.n_pred_v05
    g = m.groupby("action").agg(n=("d_f1", "size"), exact_npred=("d_npred", lambda x: float((x == 0).mean())),
                                max_abs_d_f1=("d_f1", lambda x: float(x.abs().max())),
                                mean_f1_v05=("f1_v05", "mean"), mean_f1_new=("f1_new", "mean")).reset_index()
    g.to_csv(C.OUT / "e00_conciliacion.csv", index=False)
    return g, m


def artefacts(T):
    folds = C.site_folds(T.sites, C.K)
    splits = {"protocol": "5 particiones agrupadas por sitio (site_folds v0.5, codiciosa balanceada)",
              "seeds": list(C.SEEDS), "folds": []}
    for f, test in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(T.sites)), test)
        splits["folds"].append({"fold": f, "test_sites": sorted(set(T.sites[test])),
                                "test_images": sorted(set(T.images[test])),
                                "train_sites": sorted(set(T.sites[tr])), "n_test_plots": len(set(T.images[test]))})
    json.dump(splits, open(C.ROOT / "splits.json", "w"), indent=1)
    rows = []
    for a in C.ARMS:
        rows.append({"id": a.name, "familia": C.family(a.name), "algoritmo": a.algorithm, "senal": a.signal,
                     "diametro_m": a.crown_m, "umbral": a.threshold, "puntaje_min": a.score,
                     "filtro_ndvi": 0.3 if a.is_detector and a.signal != "rgb" else "",
                     "modalidad": "RGB+HSI" if a.needs_hsi else "RGB",
                     "portafolio_rgb40": a.name in C.RGB40})
    pd.DataFrame(rows).to_csv(C.ROOT / "actions.csv", index=False)
    defs = {
        "veg_fraction": ("fracción de vegetación media del motor (ExG suavizado)", "sintético", "0-1"),
        "ndvi_mean": ("NDVI sintético medio", "sintético", "-"), "ndvi_std": ("desv. NDVI sintético", "sintético", "-"),
        "ndvi_p90": ("p90 NDVI sintético", "sintético", "-"), "exg_mean": ("ExG cromático medio", "RGB", "-"),
        "exg_std": ("desv. ExG", "RGB", "-"), "lum_mean": ("luminancia media", "RGB", "0-1"),
        "lum_std": ("desv. luminancia", "RGB", "0-1"), "dark_fraction": ("fracción bajo Otsu de luminancia", "RGB", "0-1"),
        "green_dark_corr": ("correlación ExG vs oscuridad", "RGB", "-"), "texture": ("magnitud media del gradiente", "RGB", "-"),
        "gran_2m": ("fracción verde que sobrevive apertura 2 m", "RGB", "0-1"), "gran_4m": ("idem 4 m", "RGB", "0-1"),
        "gran_7m": ("idem 7 m", "RGB", "0-1"), "otsu_exg": ("umbral de Otsu del ExG", "RGB", "-"),
        "green_blobs_ha": ("objetos verdes por ha", "RGB", "1/ha"), "green_blob_area_m2": ("log10 área mediana objetos verdes", "RGB", "log m2"),
        "green_blob_fill": ("relleno mediano objetos verdes", "RGB", "0-1"), "dark_blobs_ha": ("objetos oscuros por ha", "RGB", "1/ha"),
        "dark_blob_area_m2": ("log10 área mediana objetos oscuros", "RGB", "log m2"), "dark_blob_fill": ("relleno objetos oscuros", "RGB", "0-1"),
        "nir_mean": ("NIR sintético medio", "sintético", "reflectancia"), "nir_std": ("desv. NIR sintético", "sintético", "reflectancia"),
        "ndre_mean": ("NDRE sintético medio", "sintético", "-"), "ndre_std": ("desv. NDRE sintético", "sintético", "-"),
        "ndvi_frac05": ("fracción NDVI sintético > 0.5", "sintético", "0-1"), "vari_mean": ("VARI medio", "RGB", "-"),
        "vari_std": ("desv. VARI", "RGB", "-"), "vari_p90": ("p90 VARI", "RGB", "-"), "gli_mean": ("GLI medio", "RGB", "-"),
        "gli_std": ("desv. GLI", "RGB", "-"), "chrom_r": ("cromaticidad r media", "RGB", "0-1"),
        "chrom_g": ("cromaticidad g media", "RGB", "0-1"), "chrom_b": ("cromaticidad b media", "RGB", "0-1"),
        "exg_p90": ("p90 ExG", "RGB", "-"), "real_ndvi_mean": ("NDVI real NEON medio (1 m)", "HSI", "-"),
        "real_ndvi_std": ("desv. NDVI real", "HSI", "-"), "real_ndvi_p90": ("p90 NDVI real", "HSI", "-"),
    }
    for s in ("dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"):
        defs[f"sensor_{s}"] = ("sensor nominal (one-hot)", "metadato", "0/1")
    feats = {"grupos": {g: C.context_features(g) for g in ("C0", "C1", "C2", "C3", "C4", "V05")},
             "definiciones": {k: {"definicion": v[0], "origen": v[1], "unidad": v[2]} for k, v in defs.items()}}
    json.dump(feats, open(C.ROOT / "features.json", "w"), indent=1, ensure_ascii=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    wpath = Path("/tmp/claude-0/-home-claude-agrispectralsynth/e24bc419-cf00-53d3-96c8-274fdd70d0b3/scratchpad/dfw/NEON.pt")
    wh = hashlib.sha256(wpath.read_bytes()).hexdigest() if wpath.exists() else "n/d"
    import platform

    import sklearn
    import torch

    cfg = f"""objetivo: "SpecTree E00-E09 (ver preregistro.md)"
codigo_commit: "{commit}"
checkpoint: "DeepForest 1.0.0 NEON.pt"
checkpoint_sha256: "{wh}"
recompensa: {{kappa: 0.25, lambda: 0.0, iou: 0.4, emparejamiento: "hungaro IoU total"}}
selectores: {{alpha_linucb: {C.ALPHA_LINUCB}, eta: {C.ETA}, v_thompson: {C.V_TS}, pasadas: {C.EPOCHS}}}
particiones: {{tipo: "sitio", k: {C.K}, semillas: {list(C.SEEDS)}}}
bootstrap: {{replicas: {C.BOOT}, semilla: {C.BOOT_SEED}, unidad: "sitio"}}
versiones: {{python: "{platform.python_version()}", numpy: "{np.__version__}", pandas: "{pd.__version__}", torch: "{torch.__version__}", sklearn: "{sklearn.__version__}"}}
hardware: "{platform.processor() or platform.machine()}, 2 núcleos, 1 hilo por proceso"
"""
    (C.ROOT / "config.yaml").write_text(cfg, encoding="utf-8")


def main():
    rec = reconcile()
    if rec is not None:
        g, m = rec
        print("Conciliación v0.5 vs regenerado: acciones con n_pred idéntico en 100 % de contextos:",
              int((g.exact_npred == 1).sum()), "/", len(g), "; máx |ΔF1|:", round(m.d_f1.abs().max(), 4))
    pols = ("SBS", "VBS", "Random", "Ridge", "LinUCB", "LinUCB-F", "LinTS")
    T = C.build_table(C.RGB40, "V05")
    artefacts(T)
    dec = C.cross_validate(T, pols, tag="E00-rgb40")
    T58 = C.build_table(C.ALL58, "V05")
    dec58 = C.cross_validate(T58, pols, tag="E00-all58")
    pd.concat([dec, dec58]).to_csv(C.ROOT / "policy_decisions.csv", index=False)
    s40, s58 = C.summary(dec), C.summary(dec58)
    s40.to_csv(C.OUT / "e00_resumen_rgb40.csv", index=False)
    s58.to_csv(C.OUT / "e00_resumen_all58.csv", index=False)
    print(C.fmt(s40).to_string(index=False))
    print(C.fmt(s58).to_string(index=False))
    # per site
    pp = C.per_plot(dec).groupby(["policy", "site"]).reward.mean().unstack("policy")
    n = dec.groupby("site").image.nunique()
    pp.insert(0, "parcelas", n)
    for p in pols:
        if p != "SBS":
            pp[f"{p}-SBS"] = pp[p] - pp["SBS"]
    pp.to_csv(C.OUT / "e00_por_sitio.csv")
    pp.to_csv(C.ROOT / "summary_by_site.csv")
    print(pp.round(3).to_string())


if __name__ == "__main__":
    main()
