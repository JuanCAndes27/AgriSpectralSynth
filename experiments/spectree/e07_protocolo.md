# E07 — Protocolo congelado de transferencia externa

Este protocolo se escribió y se versionó **antes** de abrir las anotaciones y de ejecutar cualquier acción sobre el conjunto externo. Hasta ese momento solo se habían consultado el README del repositorio externo y los nombres de archivo. Su SHA-256 se registra en `e07_protocolo.sha256` y en el commit que lo introduce.

## 1. Conjunto externo

| Campo | Valor |
|---|---|
| Fuente | Zamboni et al. (2021), *Benchmarking anchor-based and anchor-free state-of-the-art deep learning methods for individual tree detection in RGB high-resolution images*, Remote Sensing 13(13):2482 |
| Repositorio | `github.com/pedrozamboni/individual_urban_tree_crown_detection`, commit `db9988358846dcac4c2f1df245be5d30bec7f2a5` (9 jun. 2021) |
| Escena | Arbolado urbano de Campo Grande, Mato Grosso do Sul, Brasil; ortoimágenes aéreas RGB de 2013 |
| Resolución | GSD 10 cm, igual que NEON |
| Unidades | 220 parches de 512 × 512 px (51.2 m × 51.2 m), sin solape, de 2 ortoimágenes |
| Etiquetas | 3382 copas, polígonos manuales en QGIS convertidos a cajas |
| Licencia | El repositorio no declara licencia. Se usa solo para evaluar; **no se redistribuyen** sus imágenes ni etiquetas. |
| Independencia del detector | `NEON.pt` se entrenó solo con datos NEON de EE. UU.; no puede contener estas escenas. |

Diferencias de dominio declaradas de antemano: ambiente urbano (calles, edificios, vehículos, sombras de edificios), bioma tropical/cerrado, especies distintas, otro sensor y otro año, parches de 512 px en vez de 400 px, y un protocolo de anotación distinto (polígonos convertidos a cajas, árboles de calle).

## 2. Qué se congela

- **Generador:** motor espectral sin cambios, sensor de despliegue `dji_mavic3m` (el de E04). No se calculan las otras tres proyecciones.
- **Acciones:** las 40 del portafolio RGB, con los mismos parámetros en píxeles y `gsd = 0.1`. DeepForest se ejecuta con el mismo envoltorio (umbral 0.1, NMS 0.05). Como 512 ≤ 1.5 × 400, la imagen se procesa en una sola pasada, tal como hace el envoltorio con cualquier imagen de ese tamaño.
- **Contexto:** los 25 descriptores v0.5 (`V05`) como en E00/E03/E04, más `C0` para Ridge|C0.
- **Recompensa:** r = F1 − 0.25·min(1, e), λ = 0, IoU ≥ 0.4, asignación húngara (`hung`); IoU 0.3/0.5 y `card` como sensibilidad.
- **Políticas, entrenadas con las 189 parcelas NEON y las cuatro proyecciones, sin ver nada del conjunto externo:**

| Política | Entrenamiento |
|---|---|
| SBS | acción con mayor recompensa media por contexto en NEON |
| Ridge·V05, Ridge·C0 | información completa |
| Trees·V05 | ExtraTrees; configuración por validación interna agrupada por sitio en NEON |
| LinUCB-F·V05 | 3 pasadas, semillas 0–4, decisión greedy |
| Regla barata | tocón sobre un descriptor RGB entre la mejor clásica y el mejor DeepForest de NEON |
| Selector-VI | el selector aprendido con mejor validación interna agrupada por sitio en NEON (mismo procedimiento que P2 de E03) |
| DF-defecto | `df_rgb_s0.1`, la salida estándar del detector, como referencia sin selección |
| VBS externo | oráculo con las etiquetas externas; **solo cota superior** |

Ningún umbral, peso, calibración ni configuración se ajusta con el conjunto externo. No hay adaptación de dominio.

## 3. Medidas y comparaciones

- Por parche: recompensa, F1, error absoluto de conteo, error relativo, tiempo (acción + síntesis + contexto) en el mismo procesador de 1 hilo.
- Agregados: media por parche (no hay sitios), F1 micro, MAE y RMSE de conteo, sesgo de inventario.
- **Comparación primaria externa (1):** Selector-VI − SBS en recompensa media por parche, IC del 95 %. Interpretación con el margen pre-registrado δ = 0.02: superior si el límite inferior > +0.02; no inferior si el límite inferior > −0.02; inconcluso en otro caso.
- **Descriptivas:** caída por transferencia de cada política (externo − NEON validación cruzada, usando las cifras de E00/E03 ya publicadas); brecha SBS–VBS externa; acciones elegidas por familia; fallos por tipo de escena (densidad de copas por parche, en terciles).
- **Incertidumbre:** remuestreo pareado de parches, B = 5000, semilla 12345. Los parches vienen de solo dos ortoimágenes contiguas y están autocorrelacionados, así que el intervalo es **optimista**. Como sensibilidad, remuestreo por bloques de 10 identificadores consecutivos (supone, sin poder comprobarlo, que la numeración sigue el orden espacial).
- **Parches sin árboles (N = 0):** fuera de la recompensa y de F1, según §3.3 del plan; se informan aparte las falsas detecciones por parche de cada política.
- **Partición de los autores:** se informa además el subconjunto `img_list/0/test.txt` para comparabilidad con su artículo; no se usa para decidir nada.

## 4. Procedencia de `NEON.pt` (auditoría documental)

Se resume en `e07_procedencia_y_externo.md`. Se declara de antemano que la ausencia de un listado de teselas de entrenamiento **no** prueba ausencia de solapamiento con las 189 parcelas.

## 5. Qué no se hará

No se cambiará el protocolo tras ver los resultados. Si algún paso falla (por ejemplo, un formato de anotación inesperado), se documentará como desviación con su motivo, sin tocar las políticas ni los umbrales.
