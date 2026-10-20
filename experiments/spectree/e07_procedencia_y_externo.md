# E07 — Procedencia del detector y transferencia externa

Protocolo congelado: `e07_protocolo.md` (SHA-256 `9f4f69d9…7a6dd1`, commit `3f31fab`), publicado antes de leer las anotaciones externas.

## A. Procedencia de `NEON.pt`

| Elemento | Qué consta | Fuente |
|---|---|---|
| Archivo | `NEON.pt`, DeepForest 1.0.0 (RetinaNet ResNet-50 FPN), SHA-256 `6d219419…d8f282a` | archivo local |
| Preentrenamiento | «millones» de copas generadas sin supervisión a partir de LiDAR en **24 sitios NEON**, seguido de anotación manual sobre RGB «de sitios seleccionados» | DeepForest 1.0, `docs/getting_started.md`, l. 24 |
| Anotación manual | Teselas de entrenamiento de NeonTreeEvaluation: **SJER, TEAK, NIWO, LENO, OSBS y MLBS** («más de 10 000 copas») | Weinstein et al. 2021, *Training annotations*, tabla 2 |
| Separación declarada | «no se usaron ni se solapan espacialmente con ningún dato de evaluación de este benchmark»; las teselas se eligieron para que «no se solaparan con ninguna parcela de evaluación» | Weinstein et al. 2021, *NeonTreeEvaluation R package* y *Training annotations* |
| Separación en el estudio de 2020 | «se eliminó toda tesela de entrenamiento a menos de 1 km de una tesela de evaluación» (estudio con 4 sitios; no consta si se aplicó igual a la versión publicada del modelo) | Weinstein et al. 2020, *2.2 Deep learning* |
| Dependencia metodológica | los autores advierten que DeepForest «se diseñó en conjunto con estos datos de evaluación» | Weinstein et al. 2021 |
| Lista de teselas o años de entrenamiento | **no publicada** | — |

**Consecuencia para las 189 parcelas.** Los seis sitios con anotación manual de entrenamiento aportan **147 de las 189 parcelas (78 %)**: SJER 61, TEAK 51, OSBS 14, NIWO 11, MLBS 8 y LENO 2. Además, los 24 sitios del preentrenamiento probablemente incluyen los 20 evaluados, pero la lista no se publica. La separación es, en el mejor caso, **espacial dentro del mismo sitio**, no geográfica. Las parcelas no consta que se hayan usado para entrenar, pero las escenas comparten sitio, sensor, campaña y protocolo. A eso se suma que el diseño del detector (umbrales, tamaño de parche) se hizo mirando este benchmark. Por tanto, todo resultado de DeepForest sobre NEON en este estudio es **intra-dominio** para el detector: su ventaja frente a las acciones clásicas probablemente está inflada, y la validación cruzada por sitio del selector no corrige esa dependencia.

La ausencia de una lista de teselas **no** demuestra que no haya solapamiento. Para descartarlo haría falta el listado de teselas del modelo publicado, que no está disponible.

## B. Transferencia externa

*(se completa con los resultados)*

## Fuentes

- Weinstein, B. G. et al. (2021). A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network. *PLoS Computational Biology* 17(7): e1009180. https://pmc.ncbi.nlm.nih.gov/articles/PMC8282040/
- Weinstein, B. G. et al. (2020). Cross-site learning in deep learning RGB tree crown detection. *Ecological Informatics* 56: 101061. https://par.nsf.gov/servlets/purl/10164167
- DeepForest 1.0, `docs/getting_started.md`.
- Zamboni, P. et al. (2021). Benchmarking anchor-based and anchor-free state-of-the-art deep learning methods for individual tree detection in RGB high-resolution images. *Remote Sensing* 13(13): 2482. https://www.mdpi.com/2072-4292/13/13/2482 — datos: https://github.com/pedrozamboni/individual_urban_tree_crown_detection
