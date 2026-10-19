#!/usr/bin/env python3
"""
E10 — Controls requested by the TGRS-style review (W7, C6, C11, Q4, Q11). Exploratory; added after the
pre-registration (declared as a deviation: extra analyses, none replaces a pre-registered one).

  a) Proposition 1, reproducible check (C11): every plot x sensor, max |NDVI_engine - NDVI(f)|, with the
     saturated-pixel definition (red or NIR clipped at 1, or red + NIR <= 1e-6) and the excluded fraction.
  b) Resolution control (W7): the 14 classical algorithms that ran on REAL 1 m NDVI are re-run on SYNTHETIC
     NDVI aggregated to the same 1 m support and upsampled the same way. If F1 falls to the real-NDVI level,
     the support, not the spectral content, explains the low F1 of the real-NDVI actions.
  c) Registration sensitivity (W7): pixel correlation synthetic vs real NDVI for shifts of -2..+2 m.
  d) Aggregation order (C6): NDVI of 1 m-averaged red/NIR vs 1 m average of 10 cm NDVI.
  e) Loss decomposition of LinUCB-F vs SBS (Q11): same action / other DeepForest threshold /
     DeepForest filter change / switch to a classical action.
  f) Sensor ablation: context without the nominal-sensor one-hot (LinUCB-F, Ridge; RGB 40).
  g) VBS composition at lambda = 0.1 (Q4): actions used, share requiring real NDVI, F1, count MAE, seconds,
     for the 40-action RGB portfolio and the 58-action reference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

BENCH = Path("/tmp/claude-0/-home-claude/e24bc419-cf00-53d3-96c8-274fdd70d0b3/scratchpad/nte")
DRONES = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"]


def prop1_check(stems):
    from agrispectralsynth.indices import NDVI
    from agrispectralsynth.pipeline import read_rgb
    from agrispectralsynth.sensors.registry import load_sensor
    from agrispectralsynth.spectral.engine import SpectralEngine

    eng = SpectralEngine()
    rows = []
    for s in stems:
        rgb, _ = read_rgb(BENCH / "evaluation" / "RGB" / f"{s}.tif")
        sc = eng.prepare(rgb)
        for sid in DRONES + ["sentinel2a_msi", "landsat8_oli"]:
            try:
                sen = load_sensor(sid)
            except Exception:
                continue
            b = eng.render(sc, sen, noise_std=0)
            R, N = b[sen.roles["red"]], b[sen.roles["nir"]]
            nd = NDVI().compute(R, N)
            names = [x.name for x in sen.band_list]
            pl = eng.plan(sen)
            i, j = names.index(sen.roles["red"]), names.index(sen.roles["nir"])
            f = sc.veg_fraction
            r = f * (pl.veg[i] - pl.soil[i]) + pl.soil[i]
            n = f * (pl.veg[j] - pl.soil[j]) + pl.soil[j]
            pred = (n - r) / (n + r)
            sat = (R >= 0.999) | (N >= 0.999) | (R + N <= 1e-6)
            w = pl.rgb_weights[i]
            rows.append({"image": s, "sensor": sid, "pixeles": nd.size, "frac_saturados": float(sat.mean()),
                         "max_abs_dif_no_saturados": float(np.abs(nd - pred)[~sat].max()),
                         "max_abs_dif_todos": float(np.abs(nd - pred).max()),
                         "peso_canal_rojo_banda_roja": float(w.max()), "c_comun_rojo_nir": bool(w.max() > 0.999 and
                                                                                               pl.rgb_weights[j].max() > 0.999)})
    return pd.DataFrame(rows)


def resolution_control():
    from agrispectralsynth.agent.actions import ARMS, Signals, run_arm
    from agrispectralsynth.agent.hyperspectral import upsample
    from s02_metricas import box_iou, tp_counts

    hsi_arms = [a for a in ARMS if a.needs_hsi and not a.is_detector]
    rows = []
    man = pd.read_csv(C.S01 / "data_manifest.csv").set_index("plot")
    for p in sorted((C.S01 / "plots").glob("*.npz")):
        z = np.load(p)
        gt = z["gt/boxes"]
        H, W = int(man.loc[p.stem, "height_px"]), int(man.loc[p.stem, "width_px"])
        for sid in DRONES:
            syn1, nir1 = z[f"px/{sid}/V0"], z[f"px/{sid}/V0_nir"]
            sig = Signals(index={"hsi": upsample(syn1, (H, W))}, peaks={"hsi": upsample(nir1, (H, W))})
            for arm in hsi_arms:
                b = run_arm(arm, sig, 0.1)
                tp = tp_counts(box_iou(b, gt))[("hung", 0.4)]
                rows.append({"image": p.stem, "sensor": sid, "algoritmo": arm.name.replace("hsi", "X"),
                             "f1": 2 * tp / max(len(b) + len(gt), 1), "n_pred": len(b), "n_true": len(gt)})
    d = pd.DataFrame(rows)
    d["site"] = d.image.map(C.contexts().groupby("image").site.first())
    m = C.metrics(0.4, "hung")
    m = m[m.action.str.contains("_hsi|_ndvi") & ~m.action.str.startswith("df_")].copy()
    m["algoritmo"] = m.action.str.replace("hsi", "X").str.replace("ndvi", "X")
    m["senal"] = np.where(m.action.str.contains("_hsi"), "NDVI real (1 m)", "NDVI sintético (10 cm)")
    d["senal"] = "NDVI sintético degradado a 1 m"
    allx = pd.concat([m[["image", "site", "sensor", "algoritmo", "senal", "f1"]], d[["image", "site", "sensor", "algoritmo", "senal", "f1"]]])
    per = allx.groupby(["senal", "algoritmo", "image", "site"]).f1.mean().groupby(["senal", "algoritmo", "site"]).mean() \
        .groupby(["senal", "algoritmo"]).mean().unstack("senal")
    best = per.max()
    return per, best


def main():
    stems = sorted(p.stem for p in (C.S01 / "plots").glob("*.npz"))
    a = prop1_check(stems)
    a.to_csv(C.OUT / "e10a_proposicion1.csv", index=False)
    print("Prop. 1:", a.groupby("sensor")[["frac_saturados", "max_abs_dif_no_saturados", "max_abs_dif_todos"]].max().to_string(),
          "\nc común rojo/NIR en todos:", bool(a.c_comun_rojo_nir.all()))

    per, best = resolution_control()
    per.to_csv(C.OUT / "e10b_resolucion_por_algoritmo.csv")
    best.to_csv(C.OUT / "e10b_resolucion_mejor.csv")
    print(per.round(3).to_string()); print(best.round(3).to_string())

    # c) registration and d) aggregation order
    rows = []
    for p in sorted((C.S01 / "plots").glob("*.npz")):
        z = np.load(p)
        for sid in ("dji_mavic3m",):
            syn, real, v = z[f"px/{sid}/V0"], z[f"px/{sid}/real_ndvi"], z[f"px/{sid}/valid"]
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    s2 = np.roll(np.roll(syn, dy, 0), dx, 1)
                    m = v.copy()
                    m[:2], m[-2:], m[:, :2], m[:, -2:] = False, False, False, False
                    rows.append({"image": p.stem, "dy": dy, "dx": dx, "r": np.corrcoef(s2[m], real[m])[0, 1]})
    reg = pd.DataFrame(rows)
    regm = reg.groupby(["dy", "dx"]).r.mean().unstack()
    best_shift = reg.loc[reg.groupby("image").r.idxmax()][["dy", "dx"]].value_counts().head(5)
    regm.to_csv(C.OUT / "e10c_registro.csv")
    print(regm.round(3).to_string()); print("desplazamiento óptimo por parcela:\n", best_shift.to_string())

    agg = []
    for p in sorted((C.S01 / "plots").glob("*.npz")):
        z = np.load(p)
        for sid in DRONES:
            v = z[f"px/{sid}/valid"]
            real = z[f"px/{sid}/real_ndvi"][v]
            r, n = z[f"px/{sid}/V0_red"][v], z[f"px/{sid}/V0_nir"][v]
            for name, syn in (("media de NDVI 10 cm", z[f"px/{sid}/V0"][v]), ("NDVI de rojo/NIR medios", (n - r) / (n + r + 1e-9))):
                e = syn - real
                agg.append({"image": p.stem, "sensor": sid, "agregacion": name, "sesgo": e.mean(),
                            "rmse": np.sqrt((e ** 2).mean()), "r": np.corrcoef(syn, real)[0, 1]})
    agg = pd.DataFrame(agg)
    agg["site"] = agg.image.map(C.contexts().groupby("image").site.first())
    aggm = agg.groupby(["agregacion", "image", "site"])[["sesgo", "rmse", "r"]].mean().groupby(["agregacion", "site"]).mean().groupby("agregacion").mean()
    aggm.to_csv(C.OUT / "e10d_agregacion.csv")
    print(aggm.round(4).to_string())

    # e) loss decomposition LinUCB-F vs SBS (E03 decisions)
    dec = pd.read_csv(C.OUT / "e03_decisiones.csv")
    sbs = dec[dec.policy == "SBS"].set_index(["fold", "ctx"])
    lf = dec[dec.policy == "LinUCB-F"].join(sbs[["action", "reward"]], on=["fold", "ctx"], rsuffix="_sbs")

    def cat(r):
        if r.action == r.action_sbs:
            return "misma acción"
        a, b = C.ARM[r.action], C.ARM[r.action_sbs]
        if not a.is_detector:
            return "cambio a clásico"
        if a.signal != b.signal:
            return "cambio de filtro DeepForest"
        return "otro umbral DeepForest"
    lf["categoria"] = lf.apply(cat, axis=1)
    lf["perdida"] = lf.reward - lf.reward_sbs
    dec_e = lf.groupby("categoria").agg(frac_decisiones=("perdida", "size"), perdida_media=("perdida", "mean"),
                                        perdida_total=("perdida", "sum"))
    dec_e["frac_decisiones"] /= len(lf)
    dec_e["contribucion_a_perdida_media"] = dec_e.perdida_total / len(lf)
    dec_e.to_csv(C.OUT / "e10e_descomposicion_perdida.csv")
    by_site = lf.groupby(["site", "categoria"]).perdida.sum().unstack(fill_value=0) / lf.groupby("site").size().values[:, None]
    by_site.to_csv(C.OUT / "e10e_descomposicion_por_sitio.csv")
    print(dec_e.round(4).to_string())

    # f) sensor ablation
    T = C.build_table(C.RGB40, "V05")
    keep = [f for f in T.features if not f.startswith("sensor_")]
    Tn = C.build_table(C.RGB40, keep)
    d1 = C.cross_validate(T, ["SBS", "LinUCB-F", "Ridge"], tag="con sensor")
    d2 = C.cross_validate(Tn, ["LinUCB-F", "Ridge"], tag="sin sensor")
    d2["policy"] = d2.policy + " sin sensor"
    d = pd.concat([d1, d2])
    rows = []
    for pol in ("LinUCB-F", "Ridge"):
        est, lo, hi, _ = C.paired_boot(d, f"{pol} sin sensor", pol)
        rows.append({"selector": pol, "con_sensor": C.macro(d)[pol], "sin_sensor": C.macro(d)[f"{pol} sin sensor"],
                     "d": est, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(C.OUT / "e10f_sin_sensor.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # g) VBS composition at lambda = 0.1
    rows = []
    for name, acts in (("RGB 40", C.RGB40), ("58 (con HSI)", C.ALL58)):
        T = C.build_table(acts, "V05", lam=0.1)
        ch = T.R.argmax(axis=1)
        a = np.array(T.arms)[ch]
        fam = pd.Series([C.family(x) for x in a]).value_counts(normalize=True)
        idx = np.arange(len(ch))
        df = pd.DataFrame({"image": T.images, "site": T.sites, "f1": T.F1[idx, ch], "ae": T.AE[idx, ch], "s": T.SEC[idx, ch]})
        pp = df.groupby(["image", "site"]).mean().groupby("site").mean().mean()
        rows.append({"portafolio": name, "f1_macro": pp.f1, "mae_conteo": df.groupby("image").ae.mean().mean(),
                     "segundos": df.groupby("image").s.mean().mean(),
                     "frac_DeepForest": float(np.mean([C.ARM[x].is_detector for x in a])),
                     "frac_requiere_HSI": float(np.mean([C.ARM[x].needs_hsi for x in a])),
                     "familias": "; ".join(f"{k} {v:.0%}" for k, v in fam.items())})
    pd.DataFrame(rows).to_csv(C.OUT / "e10g_vbs_lambda01.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
