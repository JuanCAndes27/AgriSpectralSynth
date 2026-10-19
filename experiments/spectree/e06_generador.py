#!/usr/bin/env python3
"""
E06 — What does the physical model add over simple transformations?

Synthesis variants (NDVI maps; evaluated at 1 m against NEON with common valid masks):
  V0 full engine (PROSAIL mixture, fraction from smoothed ExG with darkness gating, chroma, brightness)
  V1 monotone linear map of ExG onto [NDVI_soil, NDVI_veg] (no gating, no engine)
  V2 constant fraction (image-mean f) -> spatially constant NDVI (control)
  V3 V0 without chromatic modulation (c = 1): NDVI identical by Proposition 1; red/NIR levels change
  V4 V0 with more permissive ExG thresholds (-0.02, 0.06)
  V5 V0 with the fraction estimated from VARI instead of ExG
  V6 supervised linear regression RGB(1 m: r, g, b, ExG, luminance, VARI) -> real NDVI, fitted on the
     TRAINING sites of each outer fold (extra spectral supervision; inference stays RGB-only)
Measured: NDVI bias / MAE / RMSE / r per plot (site-macro), between-site stability, red/NIR error (V0, V3),
then the context utility: C0 + 3 statistics (mean, sd, p90 at 1 m) of each variant, actions fixed (RGB 40),
LinUCB-F and Ridge; cost of each variant (s per 400x400 image).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]
VARS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    paths = sorted((C.S01 / "plots").glob("*.npz"))
    Z = {p.stem: np.load(p) for p in paths}
    ctx = C.contexts()
    sites = ctx.groupby("image").site.first()
    images = np.array(sorted(Z))
    folds = C.site_folds(sites.reindex(images).to_numpy(), C.K)
    fold_of = {im: f for f, idx in enumerate(folds) for im in images[idx]}

    # V3 == V0 check (Proposition 1)
    dmax = max(float(np.abs(Z[s][f"px/{sid}/V3"] - Z[s][f"px/{sid}/V0"]).max()) for s in images for sid in DRONES)

    # V6: per-fold linear regression on training sites (per sensor)
    def design(z):
        F = z["px/rgbfeat"].reshape(6, -1).T
        return np.c_[F, np.ones(len(F))]
    v6 = {}            # (fold, image, sensor) -> 1 m map
    coefs = []
    for f, idx in enumerate(folds):
        train = [s for s in images if fold_of[s] != f]
        for sid in DRONES:
            X = np.concatenate([design(Z[s])[Z[s][f"px/{sid}/valid"].ravel()] for s in train])
            y = np.concatenate([Z[s][f"px/{sid}/real_ndvi"][Z[s][f"px/{sid}/valid"]] for s in train])
            w = np.linalg.lstsq(X, y, rcond=None)[0]
            coefs.append({"fold": f, "sensor": sid, **{k: v for k, v in zip(["r", "g", "b", "exg", "lum", "vari", "b0"], w)}})
            for s in images:
                v6[(f, s, sid)] = np.clip(design(Z[s]) @ w, -1, 1).reshape(Z[s][f"px/{sid}/V0"].shape).astype(np.float32)
    pd.DataFrame(coefs).to_csv(C.OUT / "e06_v6_coeficientes.csv", index=False)

    def vmap(v, s, sid, f=None):
        if v == "V6":
            return v6[(fold_of[s] if f is None else f, s, sid)]
        return Z[s][f"px/{sid}/{v}"]

    # ---- spectral error (test predictions only for V6) ----
    rows = []
    for s in images:
        for sid in DRONES:
            val = Z[s][f"px/{sid}/valid"]
            real = Z[s][f"px/{sid}/real_ndvi"][val]
            for v in VARS:
                syn = vmap(v, s, sid)[val]
                e = syn - real
                r = np.corrcoef(syn, real)[0, 1] if syn.std() > 1e-6 and real.std() > 1e-6 else np.nan
                rows.append({"image": s, "site": sites[s], "sensor": sid, "variante": v, "sesgo": e.mean(),
                             "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()), "r": r})
            for v in ("V0", "V3"):
                for band in ("red", "nir"):
                    e = Z[s][f"px/{sid}/{v}_{band}"][val] - Z[s][f"px/{sid}/real_{band}"][val]
                    rows.append({"image": s, "site": sites[s], "sensor": sid, "variante": f"{v}-{band}",
                                 "sesgo": e.mean(), "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()), "r": np.nan})
    err = pd.DataFrame(rows)
    err.to_csv(C.OUT / "e06_error_detalle.csv", index=False)
    site_err = err.groupby(["variante", "image", "site"])[["sesgo", "mae", "rmse", "r"]].mean().groupby(["variante", "site"]).mean()
    out = site_err.groupby("variante").mean()
    out["sd_sesgo_entre_sitios"] = site_err.groupby("variante").sesgo.std()
    out.to_csv(C.OUT / "e06_error_espectral.csv")
    print("max |V3 - V0| NDVI =", dmax)
    print(out.round(4).to_string())

    # ---- cost of each variant (seconds per 400x400 image, 1 thread) ----
    cost = time_variants(images[:: max(1, len(images) // 15)])
    cost.to_csv(C.OUT / "e06_costo.csv", index=False)
    print(cost.round(4).to_string(index=False))

    # ---- context utility: C0 + 3 statistics of each variant (1 m), actions fixed ----
    base = C.build_table(C.RGB40, "C0")
    c0 = list(base.features)
    decs = []
    for v in ["C0"] + VARS:
        for f, test in enumerate(C.site_folds(base.sites, C.K)):
            X = base.X.copy()
            if v != "C0":
                st = np.array([[np.mean(m[val]), np.std(m[val]), np.percentile(m[val], 90)] for m, val in
                               ((vmap(v, s, sid, f), Z[s][f"px/{sid}/valid"]) for s, sid in zip(base.images, base.sensors))])
                X = np.c_[X, st]
            T = C.Table(X, base.R, base.F1, base.CE, base.AE, base.NP, base.TP, base.SEC, base.NT, base.images,
                        base.sites, base.sensors, base.arms, c0 + ([f"{v}_mean", f"{v}_sd", f"{v}_p90"] if v != "C0" else []))
            train = np.setdiff1d(np.arange(len(T.sites)), test)
            for pol in ("LinUCB-F", "Ridge"):
                for seed in (C.SEEDS if pol == "LinUCB-F" else [0]):
                    ch = C._ridge(T, train, test) if pol == "Ridge" else C._bandit(T, train, test, pol, seed)[0]
                    for i, a in zip(test, ch):
                        decs.append({"fold": f, "seed": seed, "policy": f"{pol}|{v}", "image": T.images[i],
                                     "site": T.sites[i], "sensor": T.sensors[i], "action": T.arms[a],
                                     "reward": T.R[i, a], "f1": T.F1[i, a], "abs_count_error": T.AE[i, a]})
        print("contexto", v, flush=True)
    dec = pd.DataFrame(decs)
    dec.to_csv(C.OUT / "e06_decisiones.csv", index=False)
    rows = []
    for pol in ("LinUCB-F", "Ridge"):
        for v in VARS:
            e0, l0, h0, _ = C.paired_boot(dec, f"{pol}|{v}", f"{pol}|C0")
            e1, l1, h1, _ = C.paired_boot(dec, f"{pol}|{v}", f"{pol}|V0") if v != "V0" else (np.nan,) * 4
            rows.append({"selector": pol, "variante": v, "recompensa_macro": C.macro(dec)[f"{pol}|{v}"],
                         "d_vs_C0": e0, "ci_lo_C0": l0, "ci_hi_C0": h0, "d_vs_V0": e1, "ci_lo_V0": l1, "ci_hi_V0": h1})
        rows.append({"selector": pol, "variante": "C0", "recompensa_macro": C.macro(dec)[f"{pol}|C0"]})
    util = pd.DataFrame(rows)
    util.to_csv(C.OUT / "e06_utilidad_contexto.csv", index=False)
    print(C.fmt(util).to_string(index=False))


def time_variants(stems):
    import cv2

    from agrispectralsynth.agent.actions import exg_chromatic
    from agrispectralsynth.indices import NDVI
    from agrispectralsynth.pipeline import read_rgb
    from agrispectralsynth.sensors.registry import load_sensor
    from agrispectralsynth.spectral.engine import SpectralEngine
    from agrispectralsynth.spectral.reflectance import ReflectanceModel

    cv2.setNumThreads(1)
    bench = Path("/tmp/claude-0/-home-claude/e24bc419-cf00-53d3-96c8-274fdd70d0b3/scratchpad/nte")
    sen = load_sensor("dji_mavic3m")
    eng = SpectralEngine()
    rows = []
    for s in stems:
        rgb, _ = read_rgb(bench / "evaluation" / "RGB" / f"{s}.tif")
        for name, fn in (
            ("V0 motor completo", lambda: NDVI().compute(*[eng.render(eng.prepare(rgb), sen, noise_std=0)[k]
                                                          for k in (sen.roles["red"], sen.roles["nir"])])),
            ("V1 ExG lineal", lambda: np.clip((exg_chromatic(rgb) - 0.02) / 0.08, 0, 1)),
            ("V5 VARI", lambda: ReflectanceModel.normalize(rgb)),
        ):
            fn()
            t0 = time.perf_counter()
            for _ in range(3):
                fn()
            rows.append({"variante": name, "image": s, "segundos": (time.perf_counter() - t0) / 3})
    return pd.DataFrame(rows).groupby("variante", as_index=False).segundos.mean()


if __name__ == "__main__":
    main()
