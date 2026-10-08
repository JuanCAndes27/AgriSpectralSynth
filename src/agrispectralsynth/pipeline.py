"""
Batch pipeline: RGB images -> synthetic multispectral bands for one or
more sensors, vegetation indices, canopy masks and a statistics manifest.

Outputs go to ``<output_dir>/<sensor_id>/<product>/``; ``manifest.csv``
has one row per (image, sensor).

Why it is faster than the old ``tests/test_spectral_synthesis.py``
-----------------------------------------------------------------
* Images are processed in parallel (one process per core).
* Colouring uses a 256-entry LUT instead of calling a matplotlib
  colormap on every pixel (which allocated a float64 RGBA copy).
* PNGs are written by OpenCV with fast compression (level 1).
* Images whose outputs are already newer than the source are skipped,
  so re-running after adding 20 new images only processes those 20.
* GeoTIFFs keep the source georeferencing when the input has it.

Author
------
Juan Carlos Vega

License
-------
MIT
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import os
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from .config import AppConfig
from .constants import SUPPORTED_IMAGE_FORMATS
from .indices import INDEX_REGISTRY, NDVI
from .segmentation.canopy import canopy_mask
from .sensors.registry import load_sensor, resolve_sensor_list
from .sensors.sensor_base import Sensor
from .spectral.reflectance import ReflectanceModel, ReflectanceParams
from .utils.colormaps import colorize_bgr

logger = logging.getLogger("agrispectralsynth")

GEOTIFF_SUFFIXES = (".tif", ".tiff")
GENERATED_SUFFIXES = ("_NDVI", "_NIR", "_canopy", "_MS")
REFLECTANCE_SCALE = 10000
LEGACY_SENSOR = "dji_mavic3m"   # the only sensor the "unmixing"/"legacy" models simulate

MANIFEST_FIELDS = [
    "source", "stem", "dataset", "sensor", "status", "error", "width", "height", "gsd_m", "georeferenced",
    "veg_fraction_mean", "canopy_fraction",
    "ndvi_mean", "ndvi_std", "ndvi_p5", "ndvi_median", "ndvi_p95",
    "seconds",
]


# =============================================================================
# Job description (plain data -> cheap to pickle to worker processes)
# =============================================================================

@dataclass
class JobSettings:
    output_dir: Path
    params: ReflectanceParams
    seed: int = 42
    overwrite: bool = False
    colormap: str = "RdYlGn"
    ndvi_vmin: float = -0.2
    ndvi_vmax: float = 1.0
    preview_ext: str = ".png"
    jpeg_quality: int = 95
    compress: str = "none"
    save_nir: bool = True
    save_ndvi: bool = True
    save_multispectral: bool = True
    save_masks: bool = True
    save_yolo: bool = False
    extra_indices: List[str] = field(default_factory=list)
    canopy_threshold: float = 0.45
    canopy_min_fraction: float = 0.5
    canopy_kernel: int = 5
    canopy_min_area: int = 30
    sensors: List[str] = field(default_factory=lambda: [LEGACY_SENSOR])
    vegetation: str = "vegetation/healthy"
    soil: str = "soil/soil_mixed"
    rgb_camera: str = "rgb_camera"
    simulate_gsd: bool = False
    source_gsd_m: Optional[float] = None

    @classmethod
    def from_config(cls, cfg: AppConfig, output_dir: Path, overwrite: Optional[bool] = None) -> "JobSettings":
        p = ReflectanceParams(**cfg.reflectance.model_dump(), noise_std=cfg.simulation.noise_std)
        return cls(
            output_dir=Path(output_dir),
            params=p,
            seed=cfg.general.random_seed,
            overwrite=cfg.pipeline.overwrite if overwrite is None else overwrite,
            colormap=cfg.pipeline.colormap,
            ndvi_vmin=cfg.pipeline.ndvi_vmin,
            ndvi_vmax=cfg.pipeline.ndvi_vmax,
            preview_ext="." + cfg.output.export_format,
            jpeg_quality=cfg.simulation.jpeg_quality,
            compress=cfg.output.compress,
            save_nir=cfg.simulation.generate_nir,
            save_ndvi=cfg.simulation.generate_ndvi,
            save_multispectral=cfg.output.save_multispectral,
            save_masks=cfg.output.save_masks,
            save_yolo=cfg.output.save_yolo,
            extra_indices=cfg.enabled_indices(),
            canopy_threshold=cfg.canopy.ndvi_threshold,
            canopy_min_fraction=cfg.canopy.min_fraction,
            canopy_kernel=cfg.canopy.kernel_size,
            canopy_min_area=cfg.canopy.min_area,
            sensors=resolve_sensor_list(cfg.pipeline.sensors),
            vegetation=cfg.spectral.vegetation,
            soil=cfg.spectral.soil,
            rgb_camera=cfg.spectral.rgb_camera,
            simulate_gsd=cfg.pipeline.simulate_gsd,
            source_gsd_m=cfg.pipeline.source_gsd_m,
        )

    # -- output paths --------------------------------------------------------

    def outputs_for(self, stem: str, sensor_id: str) -> Dict[str, Path]:
        """Output files of one image for one sensor: <output_dir>/<sensor_id>/<product>/..."""
        o = self.output_dir / sensor_id
        sensor = load_sensor(sensor_id)
        paths: Dict[str, Path] = {}
        if self.save_multispectral:
            paths["multispectral"] = o / "multispectral" / f"{stem}_MS.tif"
        if self.save_ndvi:
            paths["ndvi_raw"] = o / "ndvi_raw" / f"{stem}_NDVI.tif"
            paths["ndvi_visual"] = o / "ndvi_visual" / f"{stem}_NDVI{self.preview_ext}"
        if self.save_nir:
            paths["nir"] = o / "nir" / f"{stem}_NIR.png"
        if self.save_masks:
            paths["canopy_mask"] = o / "canopy_mask" / f"{stem}_canopy.png"
        if self.save_yolo:
            paths["yolo"] = o / "yolo" / f"{stem}.txt"
        for name in self.extra_indices:
            if all(r in sensor.roles for r in INDEX_REGISTRY[name][1]):
                paths[name] = o / "indices" / name / f"{stem}_{name}.tif"
        return paths

    def output_dirs(self) -> List[Path]:
        return sorted({p.parent for sid in self.sensors for p in self.outputs_for("x", sid).values()})


# =============================================================================
# I/O helpers (unicode-safe on Windows: cv2.imread/imwrite fail on non-ASCII paths)
# =============================================================================

def read_rgb(path: Path) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Returns (rgb, geo) where rgb is (H, W, 3) in RGB order with the file's
    native dtype, and geo is {"transform", "crs"} for georeferenced rasters.
    """
    path = Path(path)
    if path.suffix.lower() in GEOTIFF_SUFFIXES:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                if src.count < 3:
                    raise ValueError(f"{path.name}: needs >= 3 bands, has {src.count}")
                rgb = np.moveaxis(src.read((1, 2, 3)), 0, -1)
                geo = None
                if src.crs is not None:
                    geo = {"transform": src.transform, "crs": src.crs}
        return np.ascontiguousarray(rgb), geo

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{path.name}: cannot decode image")
    if img.ndim == 2:
        raise ValueError(f"{path.name}: grayscale image, RGB expected")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None


def write_image(path: Path, img: np.ndarray, jpeg_quality: int = 95) -> None:
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in (".jpg", ".jpeg") else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"cannot encode {path.name}")
    buf.tofile(str(path))


def write_geotiff(
    path: Path,
    bands: np.ndarray,
    geo: Optional[dict],
    descriptions: Iterable[str],
    compress: str = "none",
) -> None:
    """bands: (count, H, W) float32 or uint16. Keeps transform/CRS when ``geo`` is given."""
    import warnings

    is_float = bands.dtype == np.float32
    opts = {}
    if compress and compress != "none":
        opts = {"compress": compress, "predictor": 3 if is_float else 2}
        if compress == "deflate":
            opts["zlevel"] = 1
        elif compress == "zstd":
            opts["zstd_level"] = 1
    with warnings.catch_warnings():
        # Images without georeferencing (PNG/JPG sources) are written with an
        # identity transform on purpose; GDAL's warning about it is just noise.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        try:
            _write(path, bands, geo, descriptions, opts, is_float)
        except Exception:
            if opts.get("compress") != "zstd":
                raise
            # GDAL build without ZSTD -> fall back to DEFLATE
            opts = {"compress": "deflate", "predictor": opts["predictor"], "zlevel": 1}
            _write(path, bands, geo, descriptions, opts, is_float)


def _write(path, bands, geo, descriptions, opts, is_float):
    count, h, w = bands.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=count,
        dtype=bands.dtype.name,
        transform=geo["transform"] if geo else Affine.identity(),
        crs=geo["crs"] if geo else None,
        tiled=w >= 512 and h >= 512,
        **opts,
    ) as dst:
        dst.write(bands)
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)
        if not is_float:
            dst.update_tags(scale_factor=str(1.0 / REFLECTANCE_SCALE), units="reflectance")


# =============================================================================
# Discovery
# =============================================================================

def guess_dataset(stem: str) -> str:
    """'470_Velasquez_Camacho_et_al_2023' -> 'Velasquez_Camacho_et_al_2023'."""
    tokens = stem.split("_")
    for i, tok in enumerate(tokens):
        if tok[:1].isupper() and not tok.isdigit():
            return "_".join(tokens[i:])
    return ""


def find_images(input_dir: Path, recursive: bool = False, limit: Optional[int] = None) -> List[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir.resolve()}")
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
            continue
        if p.stem.endswith(GENERATED_SUFFIXES):
            logger.warning("Skipping %s: looks like a generated product, not an RGB source", p.name)
            continue
        images.append(p)
    images.sort()
    return images[:limit] if limit else images


def is_up_to_date(src: Path, outputs: Dict[str, Path]) -> bool:
    try:
        src_mtime = src.stat().st_mtime
        return all(p.exists() and p.stat().st_mtime >= src_mtime for p in outputs.values())
    except OSError:
        return False


# =============================================================================
# Spatial resolution (optional): emulate the sensor's ground sampling distance
# =============================================================================

def source_gsd(geo: Optional[dict], default: Optional[float]) -> Optional[float]:
    """GSD of the input image in metres: from a projected GeoTIFF, else the configured value."""
    if geo and geo.get("crs") is not None and getattr(geo["crs"], "is_projected", False):
        return abs(float(geo["transform"].a))
    return default


def degrade_to_gsd(
    bands: Dict[str, np.ndarray],
    veg_fraction: np.ndarray,
    sensor: Sensor,
    src_gsd: float,
    geo: Optional[dict],
):
    """
    Area-average the bands to the sensor's GSD (pixel aggregation, the
    mixed-pixel effect). Bands with a coarser native GSD (e.g. Sentinel-2
    red-edge at 20 m) are aggregated to their own GSD and replicated on
    the common grid, as in a resampled L2A product.
    Returns (bands, veg_fraction, geo) unchanged if the sensor is finer.
    """
    target = sensor.gsd_m
    if not target or target <= src_gsd * 1.01:
        return bands, veg_fraction, geo
    h, w = veg_fraction.shape
    nw, nh = max(1, round(w * src_gsd / target)), max(1, round(h * src_gsd / target))
    if nw < 2 or nh < 2:
        raise ValueError(
            f"image covers {w * src_gsd:.0f} x {h * src_gsd:.0f} m: too small for a {target:g} m sensor"
        )

    def area(a, size):
        return cv2.resize(a, size, interpolation=cv2.INTER_AREA)

    out = {}
    for name, img in bands.items():
        g = sensor.band(name).gsd_m
        if g and g > target * 1.01:
            cw, ch = max(1, round(w * src_gsd / g)), max(1, round(h * src_gsd / g))
            out[name] = cv2.resize(area(img, (cw, ch)), (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            out[name] = area(img, (nw, nh))
    if geo:
        t, sx, sy = geo["transform"], w / nw, h / nh
        geo = {"transform": Affine(t.a * sx, t.b * sy, t.c, t.d * sx, t.e * sy, t.f), "crs": geo["crs"]}
    return out, area(veg_fraction, (nw, nh)), geo


# =============================================================================
# Single image
# =============================================================================

def _sensor_seed(job: JobSettings, image_name: str, sensor_id: str) -> int:
    return job.seed + zlib.crc32(f"{image_name}|{sensor_id}".encode("utf-8"))


def _write_products(job, outputs, bands, ndvi, mask, sensor, geo):
    if "multispectral" in outputs:
        names = sensor.band_names()
        stack = np.stack([bands[b] for b in names])
        stack *= REFLECTANCE_SCALE
        stack += 0.5
        write_geotiff(outputs["multispectral"], stack.astype(np.uint16), geo, names, job.compress)

    if "ndvi_raw" in outputs:
        write_geotiff(outputs["ndvi_raw"], ndvi[None], geo, ["NDVI"], job.compress)
        vis = colorize_bgr(ndvi, job.colormap, job.ndvi_vmin, job.ndvi_vmax)
        write_image(outputs["ndvi_visual"], vis, job.jpeg_quality)

    if "nir" in outputs:
        write_image(outputs["nir"], (bands[sensor.roles["nir"]] * 255.0 + 0.5).astype(np.uint8))

    if "canopy_mask" in outputs:
        write_image(outputs["canopy_mask"], mask * np.uint8(255))

    if "yolo" in outputs:
        from .yolo.labels import YOLOLabelGenerator

        YOLOLabelGenerator(min_area=job.canopy_min_area).save(mask, outputs["yolo"])

    for name in job.extra_indices:
        if name in outputs:
            cls, roles = INDEX_REGISTRY[name]
            values = cls().compute(*(bands[sensor.roles[r]] for r in roles))
            write_geotiff(outputs[name], values[None], geo, [name], job.compress)


def process_image(path: Path, job: JobSettings) -> List[dict]:
    """
    Process one RGB image for every sensor in the job. Never raises:
    errors are reported in the returned rows (one row per sensor).
    """
    path = Path(path)
    base = {"source": str(path), "stem": path.stem, "dataset": guess_dataset(path.stem)}
    outputs = {sid: job.outputs_for(path.stem, sid) for sid in job.sensors}
    todo = [sid for sid in job.sensors if job.overwrite or not is_up_to_date(path, outputs[sid])]
    rows = [{**base, "sensor": sid, "status": "skipped"} for sid in job.sensors if sid not in todo]
    if not todo:
        return rows

    t0 = time.perf_counter()
    try:
        rgb, geo0 = read_rgb(path)
        if job.params.model == "spectral":
            from .spectral.engine import SpectralEngine

            engine = SpectralEngine(job.vegetation, job.soil, job.rgb_camera, params=job.params)
            scene = engine.prepare(rgb)
            veg0 = scene.veg_fraction
        else:
            legacy_model = ReflectanceModel(params=job.params)
            veg0 = None
        src_gsd = source_gsd(geo0, job.source_gsd_m) if job.simulate_gsd else None
        if job.simulate_gsd and src_gsd is None:
            raise ValueError("simulate_gsd needs the input GSD: use a projected GeoTIFF or set pipeline.source_gsd_m")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return rows + [{**base, "sensor": sid, "status": "error", "error": err} for sid in todo]
    t_shared = time.perf_counter() - t0

    for sid in todo:
        t1 = time.perf_counter()
        row = {**base, "sensor": sid}
        try:
            sensor = load_sensor(sid)
            rng = np.random.default_rng(_sensor_seed(job, path.name, sid))
            if job.params.model == "spectral":
                bands = engine.render(scene, sensor, rng=rng)
                veg_fraction = veg0
            else:
                bands, veg_fraction = legacy_model.compute(rgb, rng=rng, return_fraction=True)

            geo, gsd = geo0, src_gsd
            if job.simulate_gsd:
                bands, veg_fraction, geo = degrade_to_gsd(bands, veg_fraction, sensor, src_gsd, geo0)
                gsd = max(src_gsd, sensor.gsd_m or src_gsd)

            ndvi = NDVI().compute(bands[sensor.roles["red"]], bands[sensor.roles["nir"]])
            mask = canopy_mask(
                ndvi,
                threshold=job.canopy_threshold,
                veg_fraction=veg_fraction,
                min_fraction=job.canopy_min_fraction,
                kernel_size=job.canopy_kernel if not job.simulate_gsd else 1,
                min_area=job.canopy_min_area if not job.simulate_gsd else 1,
            )
            _write_products(job, outputs[sid], bands, ndvi, mask, sensor, geo)

            p5, p50, p95 = np.percentile(ndvi, [5, 50, 95])
            row.update(
                status="ok",
                width=ndvi.shape[1],
                height=ndvi.shape[0],
                gsd_m=round(gsd, 4) if gsd else "",
                georeferenced=geo is not None,
                veg_fraction_mean=round(float(veg_fraction.mean()), 4),
                canopy_fraction=round(float(mask.mean()), 4),
                ndvi_mean=round(float(ndvi.mean()), 4),
                ndvi_std=round(float(ndvi.std()), 4),
                ndvi_p5=round(float(p5), 4),
                ndvi_median=round(float(p50), 4),
                ndvi_p95=round(float(p95), 4),
            )
        except Exception as exc:  # keep the batch going, report at the end
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.perf_counter() - t1 + t_shared / len(todo), 3)
        rows.append(row)
    return rows


# =============================================================================
# Batch
# =============================================================================

def _init_worker() -> None:
    # Each process is single-threaded; parallelism comes from the pool.
    cv2.setNumThreads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _process_star(args):
    return process_image(*args)


def write_manifest(path: Path, rows: List[dict]) -> None:
    """Merge with an existing manifest so skipped images keep their statistics."""
    merged: Dict[tuple, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                merged[(r["source"], r.get("sensor", LEGACY_SENSOR))] = r
    for r in rows:
        key = (r["source"], r["sensor"])
        if r.get("status") == "skipped" and key in merged:
            continue
        merged[key] = r
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(merged):
            writer.writerow(merged[key])


def run_pipeline(
    cfg: AppConfig,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    workers: Optional[int] = None,
    limit: Optional[int] = None,
    overwrite: Optional[bool] = None,
    progress: bool = True,
) -> dict:
    """
    Run the whole batch. Returns a summary dict with counts, timings and
    the list of failed files.
    """
    pc = cfg.pipeline
    input_dir = Path(input_dir or pc.input_dir)
    output_dir = Path(output_dir or pc.output_dir)
    limit = limit if limit is not None else pc.limit
    workers = workers if workers is not None else pc.workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    job = JobSettings.from_config(cfg, output_dir, overwrite)
    if job.params.model != "spectral" and job.sensors != [LEGACY_SENSOR]:
        raise ValueError(
            f"reflectance.model='{job.params.model}' only simulates {LEGACY_SENSOR}; "
            "use model 'spectral' for other sensors"
        )
    for d in job.output_dirs():
        d.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir, recursive=pc.recursive, limit=limit)
    if not images:
        raise FileNotFoundError(f"No images ({', '.join(SUPPORTED_IMAGE_FORMATS)}) in {input_dir.resolve()}")

    dupes = len(images) - len({p.stem for p in images})
    if dupes:
        logger.warning("%d images share a file name stem; their outputs will overwrite each other", dupes)

    logger.info("Input : %s (%d images)", input_dir.resolve(), len(images))
    logger.info("Output: %s", output_dir.resolve())
    logger.info("Sensors: %s", ", ".join(job.sensors))
    logger.info("Workers: %d | model: %s | overwrite: %s", workers, job.params.model, job.overwrite)

    t0 = time.perf_counter()
    tasks = [(p, job) for p in images]
    if workers == 1:
        results = map(_process_star, tasks)
    else:
        # "spawn" everywhere: it is what Windows uses anyway, and on Linux a
        # forked child can deadlock if the parent already started OpenCV's thread pool.
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            mp_context=multiprocessing.get_context("spawn"),
        )
        chunksize = max(1, min(16, len(tasks) // (workers * 4)))
        results = pool.map(_process_star, tasks, chunksize=chunksize)

    try:
        if progress:
            try:
                from tqdm import tqdm

                results = tqdm(results, total=len(tasks), unit="img", desc="AgriSpectralSynth")
            except ImportError:
                pass
        per_image = list(results)
    finally:
        if workers != 1:
            pool.shutdown()

    elapsed = time.perf_counter() - t0
    rows = [r for img_rows in per_image for r in img_rows]

    def image_status(img_rows):
        st = {r["status"] for r in img_rows}
        return "error" if "error" in st else ("ok" if "ok" in st else "skipped")

    counts = {s: sum(image_status(r) == s for r in per_image) for s in ("ok", "skipped", "error")}

    if cfg.output.save_reports:
        write_manifest(output_dir / "manifest.csv", rows)

    errors = [(r["source"], r["sensor"], r.get("error", "")) for r in rows if r.get("status") == "error"]
    for src, sid, err in errors[:20]:
        logger.error("%s [%s] -> %s", Path(src).name, sid, err)

    return {
        "images": len(images),
        "sensors": job.sensors,
        **counts,
        "errors": errors,
        "seconds": round(elapsed, 2),
        "ms_per_image": round(1000 * elapsed / counts["ok"], 1) if counts["ok"] else 0.0,
        "output_dir": str(output_dir.resolve()),
    }
