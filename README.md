# AgriSpectralSynth

Generador de datos multiespectrales **sintéticos** a partir de imágenes RGB aéreas de árboles (p. ej. MillionTrees). Simula **6 sensores**: 4 drones (DJI Mavic 3M, DJI Phantom 4M, MicaSense RedEdge-MX, Parrot Sequoia+) y 2 satélites (Sentinel-2A, Landsat 8/9). Produce las bandas de cada sensor, NDVI y otros índices, máscaras de copa y un `manifest.csv` con estadísticas por imagen y sensor: la base de datos espectral simulada sobre la que trabajará el agente de análisis.

![La misma escena con los 6 sensores](sensors/figuras/ndvi_6_sensores.png)

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -e ".[dev]"
```

`-e` instala el paquete en modo editable: ya no hace falta `sys.path.insert(...)` en los scripts.
Para descargar MillionTrees: `pip install -e ".[milliontrees]"` (ese paquete exige Python 3.10–3.12).

## Uso rápido

```bash
# Procesa data/raw -> data/processed con configs/default.yaml, en paralelo
python scripts/generate_synthetic.py

# Equivalente con el comando instalado
agrispectralsynth -c configs/default.yaml -i data/raw -o data/processed --workers 8

# Sólo 20 imágenes, regenerando aunque ya existan
agrispectralsynth -i data/raw -n 20 --overwrite

# Todos los sensores, o un grupo
agrispectralsynth -i data/raw -s all
agrispectralsynth -i data/raw -s drones
agrispectralsynth --list-sensors

# Satélites a su resolución real (10 m / 30 m); las fotos NEON tienen 10 cm
agrispectralsynth -i data/raw -s satellites --simulate-gsd --source-gsd 0.1

# Reproducir resultados anteriores (solo Mavic 3M)
agrispectralsynth -i data/raw -o data/processed_v02 --model unmixing
agrispectralsynth -i data/raw -o data/processed_v01 --model legacy --cmap jet
```

Volver a ejecutar sólo procesa las imágenes nuevas o modificadas (las demás se saltan); `--overwrite` fuerza todo.

## Salidas

```
data/processed/
├── manifest.csv                     una fila por imagen y sensor: dataset, NDVI medio/p5/p50/p95, fracción de copa…
└── <sensor>/                        dji_mavic3m, sentinel2a_msi, …
    ├── multispectral/<img>_MS.tif     todas las bandas del sensor, uint16 (reflectancia × 10000)
    ├── ndvi_raw/<img>_NDVI.tif        NDVI float32
    ├── ndvi_visual/<img>_NDVI.png     NDVI coloreado (RdYlGn, rango -0.2…1.0)
    ├── nir/<img>_NIR.png              vista rápida del NIR
    ├── canopy_mask/<img>_canopy.png   máscara de copa (0/255)
    └── indices/<IDX>/<img>_<IDX>.tif  GNDVI, NDRE, SAVI, MSAVI, EVI (opcional, --all-indices)
```

> **Cambio respecto a v0.2:** las salidas ahora van dentro de una carpeta por sensor (`data/processed/dji_mavic3m/ndvi_raw/…`).

Si la entrada es un GeoTIFF georreferenciado, todas las salidas `.tif` conservan su CRS y transformación.

## Motor espectral (v0.3, `spectral`)

Cada píxel se modela como un **espectro continuo** (400–1000 nm), mezcla de vegetación y suelo de la [librería espectral](spectral_library/README.md). La vegetación se genera con PROSAIL. Luego ese espectro se integra con la **respuesta espectral de cada banda** definida en [`sensors/`](sensors/README.md). Agregar un sensor es escribir un YAML.

- La foto aporta la fracción de vegetación, el brillo relativo (sombras) y el detalle de color. El nivel de reflectancia lo pone la librería, porque las fotos aéreas no son radiométricas.
- El NDVI depende solo de los espectros y de las bandas del sensor, no del color de la foto. Las diferencias entre sensores son diferencias reales de banda.
- Resultado típico (Mavic 3M): copa sana pura ≈ 0.91, suelo ≈ 0.15, sombras sin falsos positivos.

👉 **El documento completo de cómo se construyó cada sensor, con fuentes, supuestos, figuras y limitaciones, está en [`sensors/README.md`](sensors/README.md).**

Los modelos anteriores siguen disponibles para comparar: `--model unmixing` (v0.2, empírico) y `--model legacy` (v0.1). Ambos simulan solo el Mavic 3M. Todos los parámetros están en `configs/default.yaml`.

## Agente de delineación y conteo de copas (v0.4)

Un **bandido contextual** elige, para cada escena, cuál de 32 métodos de delineación usar, y aprende de una recompensa calculada contra copas anotadas a mano. La recompensa es F1 con IoU ≥ 0.4 menos una penalización por error de conteo. La verdad de campo es el benchmark **NeonTreeEvaluation** (194 imágenes, 6,633 copas, 22 sitios NEON).

- **En sitios conocidos** supera al mejor método fijo (recompensa 0.154 vs 0.116) y baja el error de inventario de +31 % a +1.3 %.
- **En sitios nunca vistos** todavía no lo supera en recompensa, aunque sobrecuenta mucho menos.

👉 Diseño de la recompensa, protocolo, resultados y limitaciones: [`docs/agente/README.md`](docs/agente/README.md)

```bash
python scripts/download_neontree_benchmark.py
agrispectralsynth-agent rewards  --images data/benchmarks/NeonTreeEvaluation/evaluation/RGB \
                                 --annotations data/benchmarks/NeonTreeEvaluation/annotations --out results/agent
agrispectralsynth-agent evaluate --table results/agent --seeds 10
agrispectralsynth-agent train    --table results/agent --out results/agent/agent.json
agrispectralsynth-agent apply    --agent results/agent/agent.json --input data/raw --output results/conteo
```

## Tests

```bash
pytest
```

## Estructura

```
src/agrispectralsynth/
├── pipeline.py        procesamiento por lotes en paralelo
├── cli.py             línea de comandos
├── config.py          configuración (pydantic + YAML)
├── spectral/          motor espectral (engine.py), lectura de la librería, modelos v0.1/v0.2
├── indices/           NDVI, GNDVI, NDRE, SAVI, MSAVI, EVI
├── segmentation/      máscaras de vegetación y copas
├── sensors/           lectura de los YAML de sensores y respuestas espectrales (SRF)
├── agent/             agente: verdad de campo, recompensa, acciones, contexto, bandidos, experimentos
├── datasets/          gestor de MillionTrees (descarga, polígonos, máscaras reales)
└── yolo/              etiquetas YOLO-seg desde máscaras
sensors/               definición de cada sensor (YAML) + documento explicativo  ← datos
spectral_library/      espectros de los materiales (CSV)                       ← datos
docs/agente/           documento del agente, figuras y resultados
scripts/               generate_synthetic.py, build_spectral_library.py, make_sensor_figures.py,
                       download_neontree_benchmark.py, download_milliontrees.py
configs/default.yaml
assets/samples/        imágenes de muestra (incluye la comparación v0.1 vs v0.2)
```

`data/` está en `.gitignore`: las imágenes de entrada y los productos generados no se suben al repo. Las muestras para mostrar van en `assets/samples/`.

## Licencia

MIT
