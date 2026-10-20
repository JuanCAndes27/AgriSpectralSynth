#!/usr/bin/env python3
"""
E04 — Does the saving survive when the WHOLE system is measured?

Deployment scenario: one drone (Mavic 3M), RGB photo already in memory -> final boxes.
Methods
  DF-fijo        DeepForest, s chosen on training (SBS)
  Clasico-fijo   best classical action on training (RGB signals only; no synthesis)
  Regla-barata   one RGB feature decides classical vs DeepForest (needs RGB context, no synthesis)
  LinUCB-F       learned selector (v0.5 context: synthesis + 25 descriptors + policy)
Operating point (lambda used to TRAIN the policy) chosen per outer fold by site-grouped inner validation:
the largest lambda in {0, 0.05, 0.1, 0.2, 0.5} whose inner-validation decisions meet the pre-registered
tolerances vs DF-fijo (dF1 >= -0.02, count MAE <= 1.10x); lambda = 0 if none does.
Note: adding the (action-independent) context cost to every action's time does not change the argmax of a
selector, so retraining with "total-time" reward leaves its decisions unchanged; the context cost matters only
in the comparison with methods that do not pay it.
Latency: 2 plots per site (seed 7), 1 warm-up + 3 repetitions in alternating method order, 1 thread.
Components: synthesis, context, selection, detector / classical action, post-processing. File loading and
model start-up are reported separately.
"""

import gc
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

BENCH = Path("/tmp/claude-0/-home-claude/e24bc419-cf00-53d3-96c8-274fdd70d0b3/scratchpad/nte")
WEIGHTS = "/tmp/claude-0/-home-claude-agrispectralsynth/e24bc419-cf00-53d3-96c8-274fdd70d0b3/scratchpad/dfw/NEON.pt"
SENSOR = "dji_mavic3m"
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.5)
TOL_F1, TOL_MAE = -0.02, 1.10
RGB_FEATS = None


def fold_models(T_by_lam, folds):
    """Per outer fold: operating point by inner validation; trained LinUCB-F and cheap rule at that point."""
    rgb_feats = [f for f in C.context_features("C0") if not f.startswith("sensor_")]
    T0 = T_by_lam[0.0]
    out = []
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(T0.sites)), test)
        sbs = int(np.argmax(T0.R[train].mean(axis=0)))
        is_df = np.array([C.ARM[a].is_detector for a in T0.arms])
        cls = int(np.flatnonzero(~is_df)[np.argmax(T0.R[train][:, ~is_df].mean(axis=0))])
        picks = {}
        for name in ("LinUCB-F", "CheapRule"):
            best_lam = 0.0
            for lam in LAMBDAS:
                T = T_by_lam[lam]
                ok = True
                f1s, f1d, aes, aed = [], [], [], []
                for inner in C._inner_folds(T, train):
                    tr = np.setdiff1d(train, inner)
                    if name == "LinUCB-F":
                        ch, _ = C._bandit(T, tr, inner, "LinUCB-F", 0)
                    else:
                        ch, _ = C._cheap_rule(T, tr, inner, rgb_feats)
                    s0 = int(np.argmax(T0.R[tr].mean(axis=0)))
                    f1s.append(T.F1[inner, ch].mean()); f1d.append(T.F1[inner, s0].mean())
                    aes.append(T.AE[inner, ch].mean()); aed.append(T.AE[inner, s0].mean())
                if np.mean(f1s) - np.mean(f1d) >= TOL_F1 and np.mean(aes) <= TOL_MAE * np.mean(aed):
                    best_lam = lam
            picks[name] = best_lam
        out.append({"fold": f, "test": test, "train": train, "sbs": sbs, "clasico": cls, **{f"lam_{k}": v for k, v in picks.items()}})
        print("fold", f, picks, flush=True)
    return out


def main():
    T_by_lam = {lam: C.build_table(C.RGB40, "V05", lam=lam) for lam in LAMBDAS}
    T0 = T_by_lam[0.0]
    folds = C.site_folds(T0.sites, C.K)
    fm = fold_models(T_by_lam, folds)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("test", "train")} | {"sbs": T0.arms[r["sbs"]], "clasico": T0.arms[r["clasico"]]}
                  for r in fm]).to_csv(C.OUT / "e04_puntos_operacion.csv", index=False)

    # ---- quality on all plots (Mavic 3M contexts), decisions of seed 0 ----
    rgb_feats = [f for f in C.context_features("C0") if not f.startswith("sensor_")]
    from agrispectralsynth.agent.bandit import FactoredLinUCB, Standardizer

    policies = {}
    rows = []
    for r in fm:
        test = [i for i in r["test"] if T0.sensors[i] == SENSOR]
        # train the deployed objects
        Tl = T_by_lam[r["lam_LinUCB-F"]]
        std = Standardizer().fit(Tl.X[r["train"]])
        pol = FactoredLinUCB(len(Tl.arms), Tl.X.shape[1] + 1, np.random.default_rng(0), Tl.arms, alpha=C.ALPHA_LINUCB, lam=C.ETA)
        rng = np.random.default_rng(0)
        Z = std(Tl.X[r["train"]])
        for _ in range(C.EPOCHS):
            for i in rng.permutation(len(r["train"])):
                a = pol.select(Z[i]); pol.update(Z[i], a, Tl.R[r["train"][i], a])
        Tc = T_by_lam[r["lam_CheapRule"]]
        rule_ch, rule_meta = C._cheap_rule(Tc, r["train"], np.array(test), rgb_feats)
        policies[r["fold"]] = {"std": std, "pol": pol, "rule": rule_meta, "rule_ch": dict(zip(test, rule_ch))}
        for i in test:
            sel = int(pol.greedy(std(T0.X[i])[0]))
            for name, a in (("DF-fijo", r["sbs"]), ("Clasico-fijo", r["clasico"]), ("Regla-barata", policies[r["fold"]]["rule_ch"][i]),
                            ("LinUCB-F", sel)):
                rows.append({"fold": r["fold"], "policy": name, "image": T0.images[i], "site": T0.sites[i],
                             "action": T0.arms[a], "f1": T0.F1[i, a], "abs_count_error": T0.AE[i, a],
                             "reward": T0.R[i, a], "seconds_accion": T0.SEC[i, a], "tp": T0.TP[i, a],
                             "n_pred": T0.NP[i, a], "n_true": T0.NT[i]})
    q = pd.DataFrame(rows)
    q.to_csv(C.OUT / "e04_decisiones_mavic.csv", index=False)

    # ---- instrumented latency on a site-stratified sample ----
    rng = np.random.default_rng(7)
    sample = []
    for site, g in q[q.policy == "DF-fijo"].groupby("site"):
        ims = sorted(g.image.unique())
        sample += list(rng.choice(ims, size=min(2, len(ims)), replace=False))
    timings = measure(sample, q, policies, fm, T0)
    timings.to_csv(C.ROOT / "timings.csv", index=False)

    # ---- summaries ----
    tot = timings[timings.rep >= 0].groupby(["policy", "image"]).total.mean().reset_index()
    lat = tot.groupby("policy").total.agg(media="mean", mediana="median", p95=lambda x: np.quantile(x, 0.95))
    comp = timings[timings.rep >= 0].groupby("policy")[["sintesis", "contexto", "seleccion", "accion", "posproceso"]].mean()
    qq = q.copy()
    qq["seconds"] = qq.seconds_accion
    mac_f1 = C.macro(qq, "f1")
    mae = C.per_plot(qq, "abs_count_error").groupby("policy").abs_count_error.mean()
    summ = lat.join(comp).join(mac_f1.rename("f1_macro_189")).join(mae.rename("mae_conteo_189"))
    frac_df = q.assign(df=q.action.str.startswith("df_")).groupby("policy").df.mean()
    summ = summ.join(frac_df.rename("frac_DeepForest_189"))
    # quality on the timing sample
    qs = q[q.image.isin(sample)]
    summ = summ.join(C.macro(qs.assign(seconds=0), "f1").rename("f1_macro_muestra"))
    summ.to_csv(C.OUT / "e04_resumen.csv")
    print(summ.round(4).to_string())

    # P4: LinUCB-F vs DF-fijo (non-inferiority + lower total latency)
    est, lo, hi, _ = C.paired_boot(qq, "LinUCB-F", "DF-fijo", col="f1", level=1 - 0.05 / 4)
    mae_ratio = mae["LinUCB-F"] / mae["DF-fijo"]
    tl = tot.pivot(index="image", columns="policy", values="total")
    tl["site"] = tl.index.map(q.groupby("image").site.first())
    d = (tl["LinUCB-F"] - tl["DF-fijo"]).groupby(tl.site).mean()
    reps = d.to_numpy()[np.random.default_rng(C.BOOT_SEED).integers(0, len(d), (C.BOOT, len(d)))].mean(axis=1)
    p4 = {"dF1_macro": est, "dF1_lo": lo, "dF1_hi": hi, "ratio_mae_conteo": mae_ratio,
          "d_latencia_s": float(d.mean()), "lat_lo": float(np.quantile(reps, 0.00625)), "lat_hi": float(np.quantile(reps, 0.99375)),
          "no_inferior_F1": lo >= TOL_F1, "no_inferior_MAE": mae_ratio <= TOL_MAE, "menor_latencia": float(np.quantile(reps, 0.99375)) < 0}
    for name in ("Regla-barata", "Clasico-fijo"):
        e2, l2, h2, _ = C.paired_boot(qq, name, "DF-fijo", col="f1")
        p4[f"{name}_dF1"] = e2; p4[f"{name}_dF1_lo"] = l2; p4[f"{name}_dF1_hi"] = h2
        p4[f"{name}_ratio_mae"] = mae[name] / mae["DF-fijo"]
    pd.Series(p4).to_csv(C.OUT / "e04_P4.csv")
    print(pd.Series(p4).to_string())


def measure(sample, q, policies, fm, T0):
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    from agrispectralsynth.agent.actions import ARM_INDEX, ARMS, make_signals, run_arm
    from agrispectralsynth.agent.deepforest import DeepForestDetector
    from agrispectralsynth.agent.features import FEATURE_NAMES, scene_features
    from agrispectralsynth.indices import NDVI
    from agrispectralsynth.pipeline import read_rgb
    from agrispectralsynth.sensors.registry import load_sensor
    from agrispectralsynth.spectral.engine import SpectralEngine

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    det = DeepForestDetector(WEIGHTS, threads=1)
    t_model = time.perf_counter() - t0
    rss_model = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sen = load_sensor(SENSOR)
    eng = SpectralEngine()
    fold_of = q.groupby("image").fold.first()
    rule_feat = {}
    rows = []
    imgs = {}
    for s in sample:
        t0 = time.perf_counter()
        rgb, _ = read_rgb(BENCH / "evaluation" / "RGB" / f"{s}.tif")
        imgs[s] = (rgb, time.perf_counter() - t0)
    methods = ["DF-fijo", "Clasico-fijo", "Regla-barata", "LinUCB-F"]
    C0F = [f for f in C.context_features("C0") if not f.startswith("sensor_")]

    def run(method, s):
        rgb = imgs[s][0]
        f = fold_of[s]
        r = fm[f]
        t = {"sintesis": 0.0, "contexto": 0.0, "seleccion": 0.0, "accion": 0.0, "posproceso": 0.0}
        if method == "DF-fijo":
            arm = ARMS[ARM_INDEX[T0.arms[r["sbs"]]]]
            tt = time.perf_counter(); b, sc, _ = det.predict(rgb); t["accion"] = time.perf_counter() - tt
            tt = time.perf_counter(); b = b[sc >= arm.score]; t["posproceso"] = time.perf_counter() - tt
            return t, arm.name
        if method == "Clasico-fijo":
            arm = ARMS[ARM_INDEX[T0.arms[r["clasico"]]]]
            tt = time.perf_counter()
            sig = make_signals(rgb, np.zeros(rgb.shape[:2], np.float32), np.zeros(rgb.shape[:2], np.float32))
            t["contexto"] = time.perf_counter() - tt          # RGB signals needed by the action
            tt = time.perf_counter(); run_arm(arm, sig, 0.1); t["accion"] = time.perf_counter() - tt
            return t, arm.name
        if method == "Regla-barata":
            meta = policies[f]["rule"]
            tt = time.perf_counter()
            sig = make_signals(rgb, np.zeros(rgb.shape[:2], np.float32), np.zeros(rgb.shape[:2], np.float32))
            feats = scene_features(sig, np.zeros(rgb.shape[:2], np.float32), SENSOR, 0.1)
            t["contexto"] = time.perf_counter() - tt
            tt = time.perf_counter()
            if meta.get("rule", "").startswith("fixed"):
                name = meta["rule"].split(":", 1)[1] if ":" in meta["rule"] else T0.arms[r["sbs"]]
            else:
                fname, op, thr = meta["rule"].split()
                use_df = feats[fname] > float(thr) if op == ">" else feats[fname] < float(thr)
                name = meta["df"] if use_df else meta["classical"]
            t["seleccion"] = time.perf_counter() - tt
        else:  # LinUCB-F
            tt = time.perf_counter()
            sc_ = eng.prepare(rgb)
            bands = eng.render(sc_, sen, noise_std=0)
            ndvi = NDVI().compute(bands[sen.roles["red"]], bands[sen.roles["nir"]])
            t["sintesis"] = time.perf_counter() - tt
            tt = time.perf_counter()
            sig = make_signals(rgb, ndvi, bands[sen.roles["nir"]])
            feats = scene_features(sig, sc_.veg_fraction, SENSOR, 0.1)
            x = np.array([feats[k] for k in FEATURE_NAMES])
            t["contexto"] = time.perf_counter() - tt
            tt = time.perf_counter()
            P = policies[f]
            name = T0.arms[int(P["pol"].greedy(P["std"](x)[0]))]
            t["seleccion"] = time.perf_counter() - tt
        arm = ARMS[ARM_INDEX[name]]
        if arm.is_detector:
            tt = time.perf_counter(); b, sc, _ = det.predict(rgb); t["accion"] = time.perf_counter() - tt
            tt = time.perf_counter()
            keep = sc >= arm.score
            if arm.signal == "ndvi":
                from agrispectralsynth.agent.actions import filter_by_index
                if method != "LinUCB-F":
                    sc_ = eng.prepare(rgb); bands = eng.render(sc_, sen, noise_std=0)
                    ndvi = NDVI().compute(bands[sen.roles["red"]], bands[sen.roles["nir"]])
                filter_by_index(b[keep], ndvi, 0.3)
            t["posproceso"] = time.perf_counter() - tt
        else:
            if arm.signal == "ndvi" and method != "LinUCB-F":
                tt = time.perf_counter()
                sc_ = eng.prepare(rgb); bands = eng.render(sc_, sen, noise_std=0)
                ndvi = NDVI().compute(bands[sen.roles["red"]], bands[sen.roles["nir"]])
                sig = make_signals(rgb, ndvi, bands[sen.roles["nir"]])
                t["sintesis"] += time.perf_counter() - tt
            tt = time.perf_counter(); run_arm(arm, sig, 0.1); t["accion"] = time.perf_counter() - tt
        return t, name

    for m in methods:                                     # warm-up
        run(m, sample[0])
    for rep in range(3):
        order = methods[rep:] + methods[:rep]
        for s in sample:
            for m in order:
                gc.collect()
                tt = time.perf_counter()
                t, name = run(m, s)
                total = time.perf_counter() - tt
                rows.append({"rep": rep, "policy": m, "image": s, "accion_elegida": name, "total": total, **t,
                             "carga_imagen": imgs[s][1]})
        print("rep", rep, flush=True)
    df = pd.DataFrame(rows)
    df["hardware"] = f"{platform.processor() or platform.machine()} | 1 hilo | torch {torch.__version__}"
    df["arranque_modelo_s"] = t_model
    df["rss_modelo_MB"] = (rss_model - rss0) / 1024
    return df


if __name__ == "__main__":
    main()
