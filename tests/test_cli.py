"""
AgriSpectralSynth - main command line (`agrispectralsynth`).
"""

import csv

import cv2
import numpy as np
import pytest

from agrispectralsynth.cli import main
from agrispectralsynth.sensors import available_sensors


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    for i in range(3):
        img = np.full((80, 80, 3), (120, 135, 160), np.uint8)       # BGR soil
        cv2.circle(img, (40, 40), 15, (60, 150, 100), -1)            # crown
        cv2.imwrite(str(d / f"{i}_Cli_Test_2025.png"), img)
    return d


def test_list_sensors(capsys):
    assert main(["--list-sensors"]) == 0
    out = capsys.readouterr().out
    for sid in available_sensors():
        assert sid in out


def test_default_run(raw, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["-i", str(raw), "-o", str(out), "-w", "1", "-q"]) == 0
    assert (out / "dji_mavic3m" / "ndvi_raw" / "0_Cli_Test_2025_NDVI.tif").exists()
    assert "3 processed" in capsys.readouterr().out


def test_sensor_groups_and_indices(raw, tmp_path):
    out = tmp_path / "out"
    assert main(["-i", str(raw), "-o", str(out), "-w", "1", "-q", "-s", "drones", "--all-indices", "-n", "1"]) == 0
    for sid in ("dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus"):
        assert (out / sid / "indices" / "NDRE" / "0_Cli_Test_2025_NDRE.tif").exists()
    with open(out / "manifest.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 4


def test_overwrite_and_skip(raw, tmp_path, capsys):
    out = tmp_path / "out"
    main(["-i", str(raw), "-o", str(out), "-w", "1", "-q"])
    capsys.readouterr()
    main(["-i", str(raw), "-o", str(out), "-w", "1", "-q"])
    assert "3 up to date" in capsys.readouterr().out
    main(["-i", str(raw), "-o", str(out), "-w", "1", "-q", "--overwrite"])
    assert "3 processed" in capsys.readouterr().out


def test_legacy_models_and_cmap(raw, tmp_path):
    for model in ("unmixing", "legacy"):
        out = tmp_path / model
        assert main(["-i", str(raw), "-o", str(out), "-w", "1", "-q", "--model", model, "--cmap", "jet"]) == 0
        assert (out / "dji_mavic3m" / "ndvi_visual" / "0_Cli_Test_2025_NDVI.png").exists()


def test_config_file(raw, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("pipeline:\n  sensors: [landsat_oli]\noutput:\n  save_masks: false\n", encoding="utf-8")
    out = tmp_path / "out"
    assert main(["-c", str(cfg), "-i", str(raw), "-o", str(out), "-w", "1", "-q"]) == 0
    assert (out / "landsat_oli" / "ndvi_raw").is_dir()
    assert not (out / "landsat_oli" / "canopy_mask").exists()


def test_error_exit_codes(raw, tmp_path):
    assert main(["-i", str(tmp_path / "missing"), "-o", str(tmp_path / "o"), "-q"]) == 1     # no input folder
    assert main(["-i", str(raw), "-o", str(tmp_path / "o"), "-q", "-s", "no_such_sensor"]) == 1
    assert main(["-i", str(raw), "-o", str(tmp_path / "o"), "-q", "--model", "unmixing",
                 "-s", "sentinel2a_msi"]) == 1                                            # legacy model + other sensor
    # GSD simulation on PNGs without --source-gsd: images fail, exit code 2
    assert main(["-i", str(raw), "-o", str(tmp_path / "o2"), "-q", "-w", "1", "--simulate-gsd",
                 "-s", "sentinel2a_msi"]) == 2


def test_simulate_gsd_with_source_gsd(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    img = np.full((400, 400, 3), (120, 135, 160), np.uint8)
    for k in range(6):
        cv2.circle(img, (60 + 55 * k, 200), 20, (60, 150, 100), -1)
    cv2.imwrite(str(raw / "big.png"), img)
    out = tmp_path / "out"
    assert main(["-i", str(raw), "-o", str(out), "-q", "-w", "1", "-s", "sentinel2a_msi",
                 "--simulate-gsd", "--source-gsd", "0.1"]) == 0
    import rasterio

    with rasterio.open(out / "sentinel2a_msi" / "ndvi_raw" / "big_NDVI.tif") as src:
        assert (src.width, src.height) == (4, 4)          # 40 m / 10 m


def test_module_entry_point():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "agrispectralsynth", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--sensors" in r.stdout
