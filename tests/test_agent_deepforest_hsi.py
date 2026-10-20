"""
AgriSpectralSynth - DeepForest and real-hyperspectral (NEON) actions of the agent.

Neither torch nor the DeepForest weights nor real NEON tiles are needed:
the detector is replaced by a stand-in with the same interface and the
hyperspectral tiles are synthetic cubes written in the NEON format
(426 bands, int16 reflectance x 10000, band table in micrometres).
"""

import csv

import cv2
import numpy as np
import pytest

from agrispectralsynth.agent import deepforest as dfmod
from agrispectralsynth.agent.actions import (
    ARM_INDEX, ARMS, CLASSICAL_ARMS, DF_ARMS, DFH_ARMS, HSI_ARMS, arm_group, arm_names_in, arms_for,
    filter_by_index, make_signals, run_arm,
)
from agrispectralsynth.agent.bandit import TrainedAgent
from agrispectralsynth.agent.evaluation import RewardWeights
from agrispectralsynth.agent.hyperspectral import (
    RealSpectra, attach_real_ndvi, hsi_path, read_band_table, read_cube, sensor_bands,
)
from agrispectralsynth.agent.rewards import RewardJob, build_reward_table, load_reward_table

from test_agent import forest, write_voc

NEON_WL_UM = np.linspace(0.3827, 2.5122, 426)
NOISY = (NEON_WL_UM > 1.34) & (NEON_WL_UM < 1.445) | (NEON_WL_UM > 1.79) & (NEON_WL_UM < 1.955) | (NEON_WL_UM > 2.48)


def write_band_table(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "nanometer", "noise", "BandName"])
        for i, (v, n) in enumerate(zip(NEON_WL_UM, NOISY), start=1):
            w.writerow([i, f"{v:.9f}", int(n), f"band_{i}"])
    return path


def spectra(veg: np.ndarray) -> np.ndarray:
    """(426, H, W) reflectance: green vegetation (red edge at 715 nm) where veg, flat soil elsewhere."""
    nm = NEON_WL_UM * 1000
    leaf = 0.04 + 0.04 * np.exp(-((nm - 550) / 30) ** 2) + 0.42 / (1 + np.exp(-(nm - 715) / 12))
    soil = 0.12 + 0.0001 * (nm - 400)
    return np.where(veg[None], leaf[:, None, None], soil[:, None, None])


def write_cube(path, veg: np.ndarray, nodata_corner=False):
    import rasterio

    data = np.round(spectra(veg) * 10000).astype(np.int16)
    if nodata_corner:
        data[:, 0, 0] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=veg.shape[1], height=veg.shape[0], count=426,
                       dtype="int16") as dst:
        dst.write(data)
    return path


class FakeDetector:
    """Stand-in for DeepForest: one box per green blob (score 0.5) plus a low-score box on soil."""

    calls = 0

    def predict(self, rgb):
        FakeDetector.calls += 1
        x = rgb.astype(np.float32)
        green = (x[..., 1] > x[..., 0] + 30)
        n, _, stats, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8))
        boxes = [[s[0], s[1], s[0] + s[2], s[1] + s[3]] for s in stats[1:]]
        scores = [0.5] * len(boxes)
        boxes.append([0, 0, 6, 6])
        scores.append(0.15)
        return np.asarray(boxes, np.float32), np.asarray(scores, np.float32), 0.5


@pytest.fixture
def fake_deepforest(monkeypatch):
    FakeDetector.calls = 0
    monkeypatch.setattr(dfmod, "get_detector", lambda weights=None, threads=None: FakeDetector())
    return FakeDetector


@pytest.fixture
def benchmark_hsi(tmp_path):
    """Tiny benchmark with RGB (10 cm), crowns and co-registered 1 m hyperspectral tiles."""
    img_dir, ann_dir, hsi_dir = tmp_path / "evaluation" / "RGB", tmp_path / "annotations", tmp_path / "evaluation" / "Hyperspectral"
    for d in (img_dir, ann_dir, hsi_dir):
        d.mkdir(parents=True)
    write_band_table(tmp_path / "neon_aop_bands.csv")
    for k, (site, n, r) in enumerate([("AAAA", 9, 14), ("AAAA", 4, 25), ("BBBB", 16, 9), ("BBBB", 9, 14)]):
        rgb, boxes = forest(n=n, r=r, seed=k)
        stem = f"{site}_{k:03d}_2020"
        cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        write_voc(ann_dir / f"{stem}.xml", f"{stem}.png", boxes, rgb.shape[1], rgb.shape[0])
        if k < 3:                                    # the last plot has no hyperspectral tile
            green = (rgb[..., 1].astype(int) - rgb[..., 0]) > 30
            veg = cv2.resize(green.astype(np.float32), (24, 24), interpolation=cv2.INTER_AREA) > 0.5
            write_cube(hsi_dir / f"{stem}_hyperspectral.tif", veg)
    return img_dir, ann_dir, hsi_dir, tmp_path


# ---------------------------------------------------------------------------
# Action sets
# ---------------------------------------------------------------------------

def test_action_sets():
    assert len(CLASSICAL_ARMS) == 32 and len(DF_ARMS) == 8 and len(HSI_ARMS) == 14 and len(DFH_ARMS) == 4
    assert len(ARMS) == 58 and len(ARM_INDEX) == 58                        # unique names
    assert arms_for() == CLASSICAL_ARMS
    assert len(arms_for(deepforest=True)) == 40 and len(arms_for(hsi=True)) == 46
    assert arms_for(True, True) == ARMS
    assert {arm_group(a) for a in DF_ARMS} == {"df"} and {arm_group(a) for a in DFH_ARMS} == {"hsi"}
    assert len(arm_names_in(["classical"])) == 32
    assert len(arm_names_in(["classical", "hsi"])) == 46                   # DF + hyperspectral needs "df" too
    assert len(arm_names_in(["classical", "df", "hsi"])) == 58
    with pytest.raises(ValueError):
        arm_names_in(["lidar"])
    a = ARMS[ARM_INDEX["df_ndvi_s0.3"]]
    assert a.is_detector and a.sensor_dependent and a.score == 0.3
    assert not ARMS[ARM_INDEX["df_rgb_s0.1"]].sensor_dependent
    assert ARMS[ARM_INDEX["lmw_hsi-otsu_4m"]].needs_hsi


def test_filter_by_index():
    idx = np.zeros((50, 50), np.float32)
    idx[:20, :20] = 0.8
    boxes = np.array([[0, 0, 20, 20], [30, 30, 45, 45], [10, 10, 30, 30]], np.float32)
    kept = filter_by_index(boxes, idx, 0.3)
    np.testing.assert_array_equal(kept, boxes[:1])
    assert filter_by_index(np.zeros((0, 4), np.float32), idx, 0.3).shape == (0, 4)


def test_detector_arms_use_cached_detections():
    rgb, _ = forest()
    ndvi = np.zeros(rgb.shape[:2], np.float32)
    ndvi[:120] = 0.7                                         # top half is "vegetation"
    sig = make_signals(rgb, ndvi, ndvi)
    boxes = np.array([[10, 10, 30, 30], [10, 150, 30, 170], [50, 50, 70, 70]], np.float32)
    sig.detections = (boxes, np.array([0.9, 0.9, 0.15], np.float32), 1.0)
    assert len(run_arm(ARMS[ARM_INDEX["df_rgb_s0.1"]], sig)) == 3
    assert len(run_arm(ARMS[ARM_INDEX["df_rgb_s0.2"]], sig)) == 2
    assert len(run_arm(ARMS[ARM_INDEX["df_ndvi_s0.2"]], sig)) == 1          # the box on "soil" is removed
    sig.index["hsi"] = np.full_like(ndvi, 0.8)
    assert len(run_arm(ARMS[ARM_INDEX["df_hsi_s0.1"]], sig)) == 3


def test_hsi_arms_run_on_real_ndvi():
    rgb, _ = forest()
    sig = make_signals(rgb, np.zeros(rgb.shape[:2], np.float32), np.zeros(rgb.shape[:2], np.float32))
    green = (rgb[..., 1].astype(int) - rgb[..., 0]) > 30
    sig.index["hsi"] = np.where(green, 0.8, 0.05).astype(np.float32)
    sig.peaks["hsi"] = sig.index["hsi"]
    for arm in HSI_ARMS:
        b = run_arm(arm, sig, 0.1)
        assert b.ndim == 2 and b.shape[1] == 4
    assert len(run_arm(ARMS[ARM_INDEX["cc_hsi"]], sig, 0.1)) == 9


def test_trained_agent_respects_allowed_actions():
    ag = TrainedAgent(["cc_hsi", "cc_exg"], ["f"], np.zeros(1), np.ones(1), np.array([[1.0, 1.0], [0.0, 0.0]]), {})
    assert ag.choose(np.array([1.0])) == "cc_hsi"
    assert ag.choose(np.array([1.0]), allowed=["cc_exg"]) == "cc_exg"


# ---------------------------------------------------------------------------
# Hyperspectral
# ---------------------------------------------------------------------------

def test_band_table_and_cube(tmp_path):
    bands = write_band_table(tmp_path / "bands.csv")
    wl, noise = read_band_table(str(bands))
    assert wl.size == 426 and 380 < wl[0] < 385 and wl[-1] > 2500 and noise.sum() == NOISY.sum()
    veg = np.zeros((6, 8), bool)
    veg[:, :4] = True
    tif = write_cube(tmp_path / "X_hyperspectral.tif", veg, nodata_corner=True)
    cube, wl_sel = read_cube(tif, bands)
    assert cube.shape[1:] == (6, 8) and cube.shape[0] == wl_sel.size
    assert wl_sel.min() >= 390 and wl_sel.max() <= 1010                    # only the 400-1000 nm grid
    assert np.isnan(cube[:, 0, 0]).all()                                   # nodata -> NaN
    assert 0.4 < np.nanmax(cube) < 0.5                                     # int16 / 10000
    assert hsi_path(tmp_path, "X") == tif and hsi_path(tmp_path, "Y") is None


@pytest.mark.parametrize("sensor", ["dji_mavic3m", "micasense_rededge_mx", "sentinel2a_msi"])
def test_real_ndvi_of_vegetation_and_soil(tmp_path, sensor):
    bands = write_band_table(tmp_path / "bands.csv")
    veg = np.zeros((10, 10), bool)
    veg[:, :5] = True
    cube, wl = read_cube(write_cube(tmp_path / "X_hyperspectral.tif", veg, nodata_corner=True), bands)
    b = sensor_bands(cube, wl, sensor)
    assert np.isnan(next(iter(b.values()))[0, 0])                          # nodata stays NaN
    real = RealSpectra(cube, wl, sensor, (100, 100))
    assert real.ndvi.shape == (100, 100) and real.ndvi_1m.shape == (10, 10)
    assert real.ndvi_1m[5, 2] > 0.7 and abs(real.ndvi_1m[5, 8]) < 0.15
    assert real.ndvi[50, 10] > 0.7 and real.ndvi[50, 90] < 0.15


def test_attach_real_ndvi(tmp_path):
    bands = write_band_table(tmp_path / "bands.csv")
    tif = write_cube(tmp_path / "X_hyperspectral.tif", np.ones((4, 4), bool))
    rgb, _ = forest(size=40, n=1, r=5)
    sig = make_signals(rgb, np.zeros((40, 40), np.float32), np.zeros((40, 40), np.float32))
    attach_real_ndvi(sig, tif, bands, "dji_mavic3m", (40, 40))
    assert sig.index["hsi"].shape == (40, 40) and sig.index["hsi"].mean() > 0.7


# ---------------------------------------------------------------------------
# Reward table with DeepForest + hyperspectral
# ---------------------------------------------------------------------------

def test_reward_table_with_deepforest_and_hsi(benchmark_hsi, fake_deepforest):
    img_dir, ann_dir, hsi_dir, root = benchmark_hsi
    out = root / "table"
    job = RewardJob(sensors=["dji_mavic3m", "parrot_sequoia_plus"], deepforest=True, hsi_dir=str(hsi_dir))
    assert job.bands_csv() == root / "neon_aop_bands.csv"
    s = build_reward_table(img_dir, ann_dir, out, job, workers=1, progress=False)
    assert s["images"] == 3 and s["contexts"] == 6 and s["rewards"] == 6 * 58 and not s["errors"]
    assert fake_deepforest.calls == 3                                      # one inference per image

    T = load_reward_table(out)
    assert T.R.shape == (6, 58) and not np.isnan(T.R).any()
    df = T.arms.index("df_rgb_s0.3")
    assert (T.seconds[:, df] >= 0.5).all()                                 # every DF arm pays the inference
    assert T.F1[:, df].mean() > 0.9
    assert (T.F1[:, T.arms.index("cc_hsi")] > 0.5).all()

    with open(out / "hsi_validation.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and float(rows[0]["real_ndvi_mean"]) > 0
    px = np.load(out / "hsi_pixels.npz")
    assert set(px.files) == {"dji_mavic3m", "parrot_sequoia_plus"} and px["dji_mavic3m"].shape == (2, 3 * 24 * 24)

    # time cost and action subsets are applied when loading, without re-running anything
    slow = load_reward_table(out, weights=RewardWeights(0.25, 1.0))
    np.testing.assert_allclose(slow.R[:, df], T.R[:, df] - T.seconds[:, df], atol=1e-4)
    sub = load_reward_table(out, arms=arm_names_in(["classical", "hsi"]))
    assert len(sub.arms) == 46 and not any(a.startswith("df_") for a in sub.arms)


def test_cli_with_deepforest_and_hsi(benchmark_hsi, fake_deepforest, tmp_path, monkeypatch):
    from agrispectralsynth.agent.cli import main

    img_dir, ann_dir, hsi_dir, root = benchmark_hsi
    out = root / "table"
    weights = tmp_path / "w" / "NEON.pt"
    args = ["rewards", "--images", str(img_dir), "--annotations", str(ann_dir), "--out", str(out),
            "--sensors", "dji_mavic3m", "--workers", "1", "--deepforest", "--deepforest-weights", str(weights),
            "--hsi-dir", str(hsi_dir)]
    if not dfmod.torch_available():
        assert main(args) == 1                                             # clear message: install torch
        return
    assert main(args) == 1                                                 # weights missing
    weights.parent.mkdir()
    weights.write_bytes(b"0")
    assert main(args) == 0
    assert main(["evaluate", "--table", str(out), "--out", str(root / "ev"), "--seeds", "1", "--folds", "2",
                 "--epochs", "1", "--grouping", "image", "--time-weight", "0.1", "--actions", "classical,hsi"]) == 0
    assert (root / "ev" / "summary.csv").exists()

    # apply: an agent that prefers the hyperspectral action falls back when there is no tile
    ag = TrainedAgent(["cc_hsi", "wsd_exg-otsu_4m"], ["veg_fraction"], np.zeros(1), np.ones(1),
                      np.array([[1.0, 0.0], [0.0, 0.0]]), {})
    ag.save(root / "agent.json")
    res = root / "count"
    assert main(["apply", "--agent", str(root / "agent.json"), "--input", str(img_dir), "--output", str(res),
                 "--hsi-dir", str(hsi_dir)]) == 0
    with open(res / "summary.csv", encoding="utf-8") as f:
        actions = {r["image"]: r["action"] for r in csv.DictReader(f)}
    assert actions["AAAA_000_2020.png"] == "cc_hsi" and actions["BBBB_003_2020.png"] == "wsd_exg-otsu_4m"


# ---------------------------------------------------------------------------
# DeepForest wrapper (sliding window + NMS), without the real weights
# ---------------------------------------------------------------------------

def test_deepforest_tiling_and_nms():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    det = dfmod.DeepForestDetector.__new__(dfmod.DeepForestDetector)
    det.torch = torch
    seen = []

    def patch(rgb):
        seen.append(rgb.shape[:2])
        return np.array([[100, 100, 150, 150]], np.float32), np.array([0.9], np.float32)

    det._predict_patch = patch
    boxes, scores, secs = det.predict(np.zeros((1000, 1000, 3), np.uint8))
    assert all(s == (dfmod.PATCH, dfmod.PATCH) for s in seen) and len(seen) == 9   # 3 x 3 windows
    assert boxes.shape == (len(seen), 4) and secs >= 0
    seen.clear()
    boxes, _, _ = det.predict(np.zeros((400, 400, 3), np.uint8))
    assert len(seen) == 1 and boxes.shape == (1, 4)


def test_deepforest_missing_weights(tmp_path):
    pytest.importorskip("torchvision")
    with pytest.raises(FileNotFoundError, match="download-deepforest"):
        dfmod.DeepForestDetector(tmp_path / "nope.pt")
