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

## B. Transferencia externa (Campo Grande, Brasil)

**Datos.** De los 220 parches, 2 no tienen árboles y 218 entran en la recompensa. Se descartaron 29 cajas degeneradas de ancho o alto < 1 px, todas en el borde de la imagen, de modo que quedan 3353 copas válidas (desviación 12 de `run_notes.md`). DeepForest tarda 3.0 s por parche de 512 px; la síntesis y el contexto suman 0.19 s. Todo con 1 hilo, en el mismo procesador.

**Resumen** (recompensa media por parche, IoU 0.4; IC del 95 % por remuestreo pareado de parches; [`e07_resumen.csv`](../../docs/experimentos_spectree/e07_resumen.csv)):

| Política (entrenada solo con NEON) | Recompensa externa | en NEON (VC) | F1 | MAE conteo | Inventario | Tiempo | Δ frente a SBS [IC 95 %] |
|---|---|---|---|---|---|---|---|
| VBS externo (oráculo) | 0.501 | — | 0.544 | 2.5 | +2 % | 0.70 s | +0.393 [+0.367, +0.419] |
| Ridge·V05 | **0.197** | 0.619 | 0.327 | 8.1 | −23 % | 2.16 s | **+0.090 [+0.061, +0.119]** |
| LinUCB-F·V05 | 0.124 | 0.568 | 0.289 | 12.8 | +3 % | 2.06 s | +0.016 [−0.005, +0.039] |
| Ridge·C0 | 0.120 | 0.624 | 0.271 | 9.4 | −34 % | 2.75 s | +0.012 [−0.011, +0.037] |
| **SBS** (`df_rgb_s0.3`) | **0.107** | 0.607 | 0.236 | 8.3 | −53 % | 3.18 s | — |
| Selector-VI (= ExtraTrees), regla barata, Trees·C0 | 0.107 | 0.607–0.621 | 0.236 | 8.3 | −53 % | 3.18 s | 0 (idéntico al SBS) |
| DeepForest s = 0.1 | 0.080 | — | 0.241 | 9.7 | +61 % | 3.18 s | −0.027 [−0.049, −0.004] |
| LinUCB-F·C0 | 0.078 | — | 0.234 | 9.9 | −29 % | 3.06 s | −0.030 [−0.042, −0.017] |

El remuestreo por bloques de 10 parches consecutivos da intervalos casi iguales; Ridge·V05 queda en [+0.063, +0.118].

### Comparación primaria pre-registrada

**Selector-VI − SBS = 0.000.** La validación interna en NEON vuelve a elegir ExtraTrees, como en 4 de 5 particiones de E03. Sobre los 218 parches externos, ExtraTrees elige **siempre** la acción del SBS (`df_rgb_s0.3`). Por la regla del protocolo, el veredicto formal es «no inferior», pero es un resultado **trivial**: el selector no adapta nada. Los árboles no extrapolan y, ante contextos fuera de su rango, devuelven la media de las hojas, que favorece a la acción dominante en NEON. La regla barata también se queda en DeepForest.

### Lo que muestra el conjunto externo

1. **El detector no transfiere.** La mejor acción fija en NEON pierde 0.50 de recompensa en Campo Grande (0.607 → 0.107), con F1 0.24 a IoU 0.4 y 0.30–0.33 a IoU 0.3. El orden de las acciones apenas se conserva (Spearman 0.35 entre NEON y externo, [`e07_acciones.csv`](../../docs/experimentos_spectree/e07_acciones.csv)).
   - Las mejores acciones externas son **clásicas de cuenca hidrográfica con escala de 7 m** (`wsd_ndvi-otsu_7m` 0.341, `wsd_exg-otsu_7m` 0.334), que en NEON ocupaban los puestos 17–18.
   - El mejor DeepForest externo es `df_ndvi_s0.2` (0.208, puesto 9).
   - Esto es coherente con la auditoría A: la ventaja de DeepForest en NEON es en buena parte intra-dominio.
2. **El margen de selección se multiplica por seis fuera de dominio.** VBS − SBS = +0.393, frente a +0.067 en NEON. Elegir bien la acción importa mucho más cuando la acción por defecto falla.
3. **Casi todos los contextos externos están fuera del rango de entrenamiento.** El 97.7 % de los parches tiene al menos un descriptor fuera del intervalo p1–p99 de NEON (4 de media). Los más desplazados son `gran_4m` (z mediana +3.5), `green_dark_corr` (+2.6), `green_blob_fill` y `exg_std` (+2.5) ([`e07_fuera_de_distribucion.csv`](../../docs/experimentos_spectree/e07_fuera_de_distribucion.csv)).
4. **Solo un selector lineal aprovecha parte del margen (exploratorio).**
   - Ridge·V05 cierra el 23 % de la brecha: +0.090 [+0.061, +0.119]. Elige DeepForest s = 0.3 en 102 parches, cuencas a 7 m en 51, DeepForest s = 0.1 en 38 y cuencas a 4 m en 25. Además ahorra 1 s por parche.
   - La ventaja se mantiene a IoU 0.3 (0.262 frente a 0.174) y 0.5 (0.059 frente a 0.036), en los tres terciles de densidad ([`e07_estratos.csv`](../../docs/experimentos_spectree/e07_estratos.csv)) y en la partición de prueba 0 de los autores (44 parches: 0.115 frente a 0.071).
   - **No es una comparación pre-registrada como primaria.** Proviene de una sola ciudad y el intervalo por parches es optimista.
5. **Fuera de dominio, el contexto V05 sí rinde más que C0 (exploratorio, al revés que en NEON).**
   - Ridge·V05 − Ridge·C0 = +0.077 [+0.055, +0.100]; LinUCB-F·V05 − LinUCB-F·C0 = +0.046 [+0.030, +0.063].
   - V05 añade a C0 la fracción de vegetación y la media, la desviación y el p90 del NDVI sintético. Por la proposición 1 son funciones deterministas de f, es decir, **transformaciones no lineales del RGB**, no información espectral nueva.
   - La lectura prudente: esas transformaciones extrapolan mejor que los descriptores morfológicos de C0 en este dominio. No demuestra que la síntesis espectral aporte información, y contradice P1 solo fuera de NEON.
6. **Conteo.** El SBS subcuenta un 53 % del inventario y DeepForest s = 0.1 sobrecuenta un 61 %. El MAE por parche es 8.3 y 9.7, con 15 árboles por parche de mediana. El VBS baja el MAE a 2.5.
7. **Parches vacíos (2).** El SBS produce 3 y 1 falsas detecciones; DeepForest s = 0.1, 13 y 7. Con dos casos no hay estimación útil.

### Diferencias de dominio que limitan la lectura

- **Ambiente y región:** urbano (calles, techos, vehículos, sombras de edificios) y tropical.
- **Imagen:** otro sensor y otro año (2013).
- **Anotación:** cajas derivadas de polígonos de árboles de calle.
- **Tamaño de parche:** 512 px; DeepForest los procesa en una sola pasada.
- **Contraste:** NeonTreeEvaluation anota copas de bosque natural.

Parte de la caída de F1 puede deberse al protocolo de anotación y no solo al detector. El orden de las políticas, en cambio, no depende de ello: todas se evalúan con las mismas etiquetas.

## Fuentes

- Weinstein, B. G. et al. (2021). A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network. *PLoS Computational Biology* 17(7): e1009180. https://pmc.ncbi.nlm.nih.gov/articles/PMC8282040/
- Weinstein, B. G. et al. (2020). Cross-site learning in deep learning RGB tree crown detection. *Ecological Informatics* 56: 101061. https://par.nsf.gov/servlets/purl/10164167
- DeepForest 1.0, `docs/getting_started.md`.
- Zamboni, P. et al. (2021). Benchmarking anchor-based and anchor-free state-of-the-art deep learning methods for individual tree detection in RGB high-resolution images. *Remote Sensing* 13(13): 2482. https://www.mdpi.com/2072-4292/13/13/2482 — datos: https://github.com/pedrozamboni/individual_urban_tree_crown_detection
