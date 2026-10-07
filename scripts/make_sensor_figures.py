#!/usr/bin/env python3
"""
Figures for sensors/README.md.

    python scripts/make_sensor_figures.py --images <folder of RGB tiles> \
        --scene <one RGB tile> --big <large RGB image> --big-gsd 0.1

Writes PNGs to sensors/figuras/:
    srf_sensores.png       response of every band over vegetation and soil spectra
    ndvi_6_sensores.png    the same scene simulated with each sensor
    ndvi_por_sensor.png    median NDVI of canopy and soil per sensor
    efecto_escala.png      drone vs Sentinel-2 vs Landsat ground resolution
Only --scene is required for the first two; the others are optional.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agrispectralsynth.indices import NDVI  # noqa: E402
from agrispectralsynth.pipeline import degrade_to_gsd, find_images, read_rgb  # noqa: E402
from agrispectralsynth.segmentation import canopy_mask  # noqa: E402
from agrispectralsynth.sensors import available_sensors, load_sensor  # noqa: E402
from agrispectralsynth.sensors.srf import WAVELENGTHS  # noqa: E402
from agrispectralsynth.spectral.engine import SpectralEngine  # noqa: E402
from agrispectralsynth.spectral.spectra import load_spectrum  # noqa: E402
from agrispectralsynth.utils.colormaps import colorize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sensors" / "figuras"

# Validated palette (light surface): role -> colour. Red/green are close for
# protanopes, so every band also carries its name as a direct label.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ROLE_COLOR = {"blue": "#2a78d6", "green": "#008300", "red": "#e34948", "red_edge": "#eda100", "nir": "#4a3aa7"}
SERIES = ("#2a78d6", "#eb6834")

ORDER = ["dji_mavic3m", "dji_phantom4m", "micasense_rededge_mx", "parrot_sequoia_plus", "sentinel2a_msi", "landsat_oli"]
SHORT = {
    "dji_mavic3m": "Mavic 3M",
    "dji_phantom4m": "Phantom 4M",
    "micasense_rededge_mx": "RedEdge-MX",
    "parrot_sequoia_plus": "Sequoia+",
    "sentinel2a_msi": "Sentinel-2A",
    "landsat_oli": "Landsat 8/9",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def band_role(sensor, name):
    inv = {v: k for k, v in sensor.roles.items()}
    if name in inv:
        return inv[name]
    c = sensor.band(name).center
    return "red_edge" if 690 <= c < 800 else ("nir" if c >= 800 else "green")


def fig_srf():
    veg = load_spectrum("vegetation/healthy").reflectance
    soil = load_spectrum("soil/soil_mixed").reflectance
    sensors = [s for s in ORDER if s in available_sensors()]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(9, 1.55 * len(sensors) + 0.6), sharex=True)
    for ax, sid in zip(axes, sensors):
        s = load_sensor(sid)
        ax.plot(WAVELENGTHS, veg, color=INK, lw=1.5, label="vegetación sana (PROSAIL)")
        ax.plot(WAVELENGTHS, soil, color=MUTED, lw=1.5, ls="--", label="suelo")
        for i, b in enumerate(s.band_list):
            r = s.srf_matrix()[i] * 0.6
            col = ROLE_COLOR[band_role(s, b.name)]
            ax.fill_between(WAVELENGTHS, 0, r, color=col, alpha=0.18 if b.auxiliary else 0.32, lw=0)
            ax.plot(WAVELENGTHS, r, color=col, lw=1.2, ls=":" if b.auxiliary else "-")
            ax.text(b.center, 0.62, b.name, ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_ylim(0, 0.75)
        ax.set_yticks([0, 0.3, 0.6])
        ax.grid(axis="y", color=GRID, lw=0.6)
        gsd = f"{s.gsd_m:g} m" if s.gsd_m >= 1 else f"{100 * s.gsd_m:.1f} cm"
        ax.set_title(f"{s.name}  ·  GSD {gsd}", loc="left", fontsize=9.5, color=INK, pad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.995, 0.985))
    axes[-1].set_xlabel("Longitud de onda (nm)")
    axes[len(axes) // 2].set_ylabel("Reflectancia  /  respuesta relativa")
    axes[-1].set_xlim(400, 1000)
    fig.suptitle("Respuesta espectral de cada banda sobre los espectros de la librería", x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.005, "Curvas gaussianas con el centro y ancho publicados. Línea punteada: azul tomado de la cámara RGB del dron.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    fig.savefig(OUT / "srf_sensores.png", dpi=150)
    plt.close(fig)


def ndvi_for(engine, state, sid):
    s = load_sensor(sid)
    b = engine.render(state, s, noise_std=0)
    return NDVI().compute(b[s.roles["red"]], b[s.roles["nir"]]), b


def fig_scene(scene: Path):
    rgb, _ = read_rgb(scene)
    eng = SpectralEngine()
    st = eng.prepare(rgb)
    sensors = [s for s in ORDER if s in available_sensors()]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.9))
    axes = axes.ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("Foto RGB (entrada)", fontsize=9.5, loc="left")
    for ax, sid in zip(axes[1:], sensors):
        nd, _ = ndvi_for(eng, st, sid)
        ax.imshow(colorize(nd, "RdYlGn", -0.2, 1.0))
        ax.set_title(f"{SHORT[sid]}  ·  mediana {np.median(nd):.2f}", fontsize=9.5, loc="left")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = axes[-1]
    cax.set_visible(True)
    cax.imshow(np.linspace(1.0, -0.2, 256)[:, None].repeat(12, 1), cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto",
               extent=(0, 1, -0.2, 1.0))
    cax.set_position([cax.get_position().x0 + 0.08, cax.get_position().y0, 0.02, cax.get_position().height])
    cax.yaxis.tick_right(); cax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]); cax.set_title("NDVI", fontsize=9)
    fig.suptitle(f"La misma escena vista por los 6 sensores simulados  ({scene.name})", x=0.01, ha="left", fontsize=11.5)
    fig.savefig(OUT / "ndvi_6_sensores.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_dots(folder: Path):
    eng = SpectralEngine()
    sensors = [s for s in ORDER if s in available_sensors()]
    canopy = {s: [] for s in sensors}
    soil = {s: [] for s in sensors}
    images = find_images(folder)
    for p in images:
        rgb, _ = read_rgb(p)
        st = eng.prepare(rgb)
        ref, _ = ndvi_for(eng, st, "dji_mavic3m")
        m = canopy_mask(ref, veg_fraction=st.veg_fraction).astype(bool)   # same pixels for every sensor
        bare = st.veg_fraction < 0.05
        for sid in sensors:
            nd, _ = ndvi_for(eng, st, sid)
            canopy[sid].append(nd[m]); soil[sid].append(nd[bare])
    med = {k: {s: float(np.median(np.concatenate(v[s]))) for s in sensors} for k, v in (("Copa", canopy), ("Suelo", soil))}

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(sensors))[::-1]
    for (label, vals), col in zip(med.items(), SERIES):
        x = [vals[s] for s in sensors]
        ax.scatter(x, y, s=60, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.22, f"{xi:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [SHORT[s] for s in sensors])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(sensors) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("NDVI mediano")
    ax.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"NDVI de copa y suelo por sensor ({len(images)} imágenes, mismos píxeles)", loc="left", fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(OUT / "ndvi_por_sensor.png", dpi=150)
    plt.close(fig)
    return med


def fig_scale(big: Path, gsd: float):
    rgb, _ = read_rgb(big)
    eng = SpectralEngine()
    st = eng.prepare(rgb)
    panels = [("Mavic 3M · 10 cm (foto)", "dji_mavic3m", False), ("Sentinel-2A · 10 m", "sentinel2a_msi", True),
              ("Landsat 8/9 · 30 m", "landsat_oli", True)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    h, w = rgb.shape[:2]
    for ax, (title, sid, degrade) in zip(axes, panels):
        s = load_sensor(sid)
        b = eng.render(st, s, noise_std=0)
        if degrade:
            b, _, _ = degrade_to_gsd(b, st.veg_fraction, s, gsd, None)
        nd = NDVI().compute(b[s.roles["red"]], b[s.roles["nir"]])
        ax.imshow(colorize(nd, "RdYlGn", -0.2, 1.0), extent=(0, w * gsd, h * gsd, 0), interpolation="nearest")
        ax.set_title(f"{title}  ·  {nd.shape[1]}×{nd.shape[0]} px  ·  NDVI {nd.mean():.2f}", fontsize=9.5, loc="left")
        ax.set_xlabel("m")
    fig.suptitle("Efecto de la resolución espacial: las copas se mezclan con el suelo en el píxel satelital",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "efecto_escala.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--big", type=Path)
    ap.add_argument("--big-gsd", type=float, default=0.1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_srf()
    if a.scene:
        fig_scene(a.scene)
    if a.images:
        for k, v in fig_dots(a.images).items():
            print(k, {s: round(x, 3) for s, x in v.items()})
    if a.big:
        fig_scale(a.big, a.big_gsd)
    print("Figures in", OUT)


if __name__ == "__main__":
    main()
