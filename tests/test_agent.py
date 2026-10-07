"""
AgriSpectralSynth - crown delineation agent: ground truth, reward,
actions, features, bandits and the experiment protocol.
"""

import csv

import cv2
import numpy as np
import pytest

from agrispectralsynth.agent.actions import ARM_INDEX, ARMS, make_signals, otsu_threshold, run_arm
from agrispectralsynth.agent.bandit import (
    FactoredLinUCB, LinThompson, LinUCB, Standardizer, TrainedAgent, arm_factors,
)
from agrispectralsynth.agent.evaluation import RewardWeights, box_iou, reward, score_detections
from agrispectralsynth.agent.experiment import cross_validate, site_folds, summarize
from agrispectralsynth.agent.features import FEATURE_NAMES, scene_features
from agrispectralsynth.agent.groundtruth import load_annotations, read_csv_annotations, read_voc_xml, site_of, wkt_bounds
from agrispectralsynth.agent.rewards import RewardJob, build_reward_table, load_reward_table

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def forest(n=9, size=240, r=14, seed=0):
    """RGB scene: soil with n round green crowns. Returns (rgb uint8, boxes)."""
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (170, 150, 120)
    boxes = []
    g = int(np.ceil(np.sqrt(n)))
    step = size // g
    for k in range(n):
        cy, cx = (k // g) * step + step // 2, (k % g) * step + step // 2
        cy += int(rng.integers(-3, 4)); cx += int(rng.integers(-3, 4))
        cv2.circle(img, (cx, cy), r, (70, 140, 60), -1)
        boxes.append([cx - r, cy - r, cx + r + 1, cy + r + 1])
    return img, np.asarray(boxes, np.float32)


def write_voc(path, name, boxes, w, h):
    objs = "".join(
        f"<object><name>Tree</name><bndbox><xmin>{int(b[0])}</xmin><ymin>{int(b[1])}</ymin>"
        f"<xmax>{int(b[2])}</xmax><ymax>{int(b[3])}</ymax></bndbox></object>" for b in boxes)
    path.write_text(f"<annotation><filename>{name}</filename><size><width>{w}</width><height>{h}</height>"
                    f"<depth>3</depth></size>{objs}</annotation>", encoding="utf-8")


def signals(rgb):
    # NDVI stand-in for unit tests: vegetation where green dominates
    from agrispectralsynth.indices import NDVI
    from agrispectralsynth.spectral.engine import SpectralEngine

    eng = SpectralEngine()
    st = eng.prepare(rgb)
    b = eng.render(st, "dji_mavic3m", noise_std=0)
    return make_signals(rgb, NDVI().compute(b["Red"], b["NIR"]), b["NIR"]), st


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def test_voc_and_csv_readers(tmp_path):
    write_voc(tmp_path / "PLOT_001_2019.xml", "PLOT_001_2019.tif", [[1, 2, 11, 12], [20, 20, 40, 45]], 100, 100)
    gt = read_voc_xml(tmp_path / "PLOT_001_2019.xml")
    assert gt.count == 2 and gt.width == 100
    np.testing.assert_array_equal(gt.boxes[1], [20, 20, 40, 45])

    with open(tmp_path / "ann.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
        w.writerow(["a.png", 0, 0, 10, 10, "Tree"])
        w.writerow(["a.png", 5, 5, 25, 25, "Tree"])
        w.writerow(["b.png", 1, 1, 9, 9, "Tree"])
    groups = read_csv_annotations(tmp_path / "ann.csv")
    assert groups["a.png"].count == 2 and groups["b.png"].count == 1
    all_gt = load_annotations(tmp_path)
    assert set(all_gt) == {"PLOT_001_2019", "a", "b"}


def test_wkt_bounds_polygon_and_multipolygon():
    assert wkt_bounds("POLYGON ((1 2, 5 2, 5 9, 1 9, 1 2))") == [1, 2, 5, 9]
    assert wkt_bounds("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((10 10, 12 10, 12 13, 10 10)))") == [0, 0, 12, 13]


@pytest.mark.parametrize("stem,site", [("SJER_002_2018", "SJER"), ("2018_TEAK_3_315000_4096000_image_163", "TEAK"),
                                       ("foo", "UNKNOWN")])
def test_site_of(stem, site):
    assert site_of(stem) == site


# ---------------------------------------------------------------------------
# Evaluation / reward
# ---------------------------------------------------------------------------

def test_iou_basics():
    a = np.array([[0, 0, 10, 10]])
    assert box_iou(a, a)[0, 0] == pytest.approx(1)
    assert box_iou(a, np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)
    assert box_iou(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0


def test_score_perfect_and_partial():
    true = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], float)
    s = score_detections(true, true)
    assert (s.tp, s.fp, s.fn, s.f1, s.count_error) == (3, 0, 0, 1.0, 0)
    pred = np.vstack([true[:2], [[60, 60, 70, 70], [80, 80, 90, 90]]])
    s = score_detections(pred, true)
    assert (s.tp, s.fp, s.fn, s.count_error) == (2, 2, 1, 1)
    assert s.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3), abs=1e-3)


def test_one_to_one_matching():
    """Two predictions over one tree count as one TP and one FP."""
    s = score_detections(np.array([[0, 0, 10, 10], [0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]))
    assert (s.tp, s.fp) == (1, 1)


def test_empty_cases():
    assert score_detections(np.zeros((0, 4)), np.zeros((0, 4))).f1 == 1.0
    assert score_detections(np.zeros((0, 4)), np.array([[0, 0, 5, 5]])).f1 == 0.0


def test_reward_penalises_count_error():
    true = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    exact = score_detections(true, true)
    over = score_detections(np.vstack([true, [[50, 50, 60, 60]] * 4]), true)
    assert reward(exact) == 1.0
    assert reward(over) < over.f1                  # count term subtracts
    assert reward(over, w=RewardWeights(count_weight=0)) == pytest.approx(over.f1)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def test_otsu_separates_two_modes():
    x = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)]).astype(np.float32)
    assert 0.1 <= otsu_threshold(x) < 0.9


def test_every_arm_runs_and_returns_boxes():
    rgb, _ = forest()
    sig, _ = signals(rgb)
    for arm in ARMS:
        b = run_arm(arm, sig, gsd_m=0.1)
        assert b.ndim == 2 and b.shape[1] == 4


def test_good_arm_finds_the_crowns():
    rgb, true = forest(n=9, r=14)          # 2.8 m crowns at 10 cm
    sig, _ = signals(rgb)
    s = score_detections(run_arm(ARMS[ARM_INDEX["wsd_exg-otsu_4m"]], sig, 0.1), true)
    assert s.f1 > 0.8 and abs(s.count_error) <= 1


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_features_vector():
    rgb, _ = forest()
    sig, st = signals(rgb)
    f = scene_features(sig, st.veg_fraction, "dji_phantom4m")
    assert list(f) == FEATURE_NAMES
    assert f["sensor_dji_phantom4m"] == 1.0 and f["sensor_dji_mavic3m"] == 0.0
    assert f["green_blobs_ha"] > 0 and np.isfinite(list(f.values())).all()


# ---------------------------------------------------------------------------
# Bandits
# ---------------------------------------------------------------------------

def toy_problem(n=600, seed=0):
    """Two kinds of scene; arm 0 is best for kind A, arm 2 for kind B."""
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 2, n)
    X = np.c_[kind + rng.normal(0, 0.1, n), rng.normal(0, 1, n)]
    R = np.full((n, 3), 0.2) + rng.normal(0, 0.05, (n, 3))
    R[kind == 0, 0] += 0.5
    R[kind == 1, 2] += 0.5
    return X, R, kind


@pytest.mark.parametrize("cls", [LinUCB, LinThompson])
def test_linear_bandits_learn_context(cls):
    X, R, kind = toy_problem()
    Z = Standardizer().fit(X)(X)
    pol = cls(3, Z.shape[1], np.random.default_rng(0))
    for i in range(len(Z)):
        a = pol.select(Z[i])
        pol.update(Z[i], a, R[i, a])
    choice = np.array([pol.greedy(z) for z in Z])
    assert np.mean(choice == np.where(kind == 0, 0, 2)) > 0.95


def test_factored_bandit_and_factors():
    names = [a.name for a in ARMS]
    G = arm_factors(names)
    assert G.shape == (len(ARMS), 11) and (G[:, 0] == 1).all()
    assert G[names.index("cc_dark"), 3] == 1 and G[names.index("cc_dark"), 10] == 1   # dark, otsu
    pol = FactoredLinUCB(len(ARMS), 4, np.random.default_rng(0), names)
    x = np.ones(4)
    a = pol.select(x)
    pol.update(x, a, 1.0)
    assert pol.theta().shape == (len(ARMS), 4)


def test_trained_agent_roundtrip(tmp_path):
    ag = TrainedAgent(["a", "b"], ["f1", "f2"], np.zeros(2), np.ones(2),
                      np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), {"policy": "test"})
    ag.save(tmp_path / "agent.json")
    ag2 = TrainedAgent.load(tmp_path / "agent.json")
    assert ag2.choose(np.array([2.0, 0.0])) == "a"
    assert ag2.choose(np.array([-2.0, 0.0])) == "b"


# ---------------------------------------------------------------------------
# Reward table + protocol (end to end on a tiny synthetic benchmark)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_benchmark(tmp_path):
    img_dir, ann_dir = tmp_path / "RGB", tmp_path / "annotations"
    img_dir.mkdir(); ann_dir.mkdir()
    for k, (site, n, r) in enumerate([("AAAA", 9, 12), ("AAAA", 4, 25), ("BBBB", 16, 8), ("BBBB", 9, 14),
                                      ("CCCC", 4, 20), ("CCCC", 16, 9)]):
        rgb, boxes = forest(n=n, r=r, seed=k)
        stem = f"{site}_{k:03d}_2020"
        cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        write_voc(ann_dir / f"{stem}.xml", f"{stem}.png", boxes, rgb.shape[1], rgb.shape[0])
    return img_dir, ann_dir, tmp_path / "table"


def test_reward_table_and_protocol(tiny_benchmark):
    img_dir, ann_dir, out = tiny_benchmark
    s = build_reward_table(img_dir, ann_dir, out, RewardJob(sensors=["dji_mavic3m", "micasense_rededge_mx"]),
                           workers=1, progress=False)
    assert s["images"] == 6 and s["contexts"] == 12 and s["rewards"] == 12 * len(ARMS)
    T = load_reward_table(out)
    assert T.R.shape == (12, len(ARMS)) and not np.isnan(T.R).any()
    # RGB-only arms are identical for both sensors of an image
    exg = T.arms.index("wsd_exg-otsu_4m")
    for im in np.unique(T.images):
        np.testing.assert_allclose(T.R[T.images == im, exg], T.R[T.images == im, exg][0])

    folds = site_folds(T.sites, 3)
    assert all(len(set(T.sites[f])) == 1 for f in folds)          # whole sites per fold
    rows = summarize(cross_validate(T, k=3, seeds=range(2), epochs=2))
    names = {r["policy"] for r in rows}
    assert {"Oráculo", "Mejor método fijo", "LinUCB"} <= names
    orc = next(r for r in rows if r["policy"] == "Oráculo")["test_reward"]
    assert all(r["test_reward"] <= orc + 1e-9 for r in rows)


def test_cli_train_and_apply(tiny_benchmark, tmp_path):
    from agrispectralsynth.agent.cli import main

    img_dir, ann_dir, out = tiny_benchmark
    assert main(["rewards", "--images", str(img_dir), "--annotations", str(ann_dir), "--out", str(out),
                 "--sensors", "dji_mavic3m", "--workers", "1"]) == 0
    assert main(["train", "--table", str(out), "--out", str(out / "agent.json")]) == 0
    res = tmp_path / "count"
    assert main(["apply", "--agent", str(out / "agent.json"), "--input", str(img_dir), "--output", str(res)]) == 0
    with open(res / "summary.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and all(int(r["trees"]) > 0 for r in rows)
    assert (res / "boxes" / "AAAA_000_2020.csv").exists()
