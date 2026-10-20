# SpecTree — notas de ejecución, exclusiones y desviaciones

Ejecución: 19 de octubre de 2026, Intel Xeon @ 2.10 GHz, 2 núcleos, 7.8 GB de RAM, sin GPU.
Pre-registro: `preregistro.md` (commit `fd339fa`, anterior a toda ejecución).

## Datos y exclusiones

- Se procesaron 189 de las 194 parcelas anotadas, con 0 errores. Hay 5 exclusiones:
  - sin cubo hiperespectral: NIWO_011_2018, UNDE_001_2017, UNDE_006_2017 y UNDE_065_2017;
  - cubo con 425 bandas en lugar de 426: ONAQ_021_2019.
- 187 parcelas miden 400 × 400 px. Dos parcelas miden 370 × 400 y 392 × 400.
- No hay parcelas sin árboles: el mínimo es 1 árbol y 20 parcelas tienen 3 o menos. **El bloque de escenas vacías de E09 no se pudo ejecutar** y no se inventaron casos.

## Reproducción

Las predicciones regeneradas (S01–S02) coinciden con la tabla de la versión 0.5: número de cajas idéntico en las 58 acciones y en los 756 contextos, y $|\Delta\mathrm{F1}|\le 10^{-4}$. Las cifras históricas se conservan; las de este informe salen de la regeneración.

## Desviaciones del pre-registro

1. **Se añadió E10, de controles exploratorios.** Responde a puntos de la revisión TGRS que llegó después del pre-registro: proposición 1, resolución, registro, orden de agregación, descomposición de la pérdida, sensor y composición del VBS. No reemplaza ningún análisis pre-registrado.
2. **E06:** para incluir la variante supervisada V6, que solo existe a 1 m, los estadísticos de contexto de **todas** las variantes se calcularon sobre los mapas agregados a 1 m. Por eso la fila V0 de E06 no es idéntica a C1 de E01, que usa 10 cm.
3. **E06, costo de V5:** la medición inicial cronometró solo la normalización RGB y subestima el costo. Se volvió a medir aparte; ver `e06_costo_corregido.csv`.
4. **E06/V6:** cada partición ajusta la regresión con sus sitios de entrenamiento. La política de esa partición usa predicciones dentro de muestra para sus parcelas de entrenamiento y fuera de muestra para las de prueba.
5. **E04, calidad:** se midió solo sobre las proyecciones Mavic 3M, que representan el despliegue con un dron. Las políticas se entrenan con las cuatro proyecciones nominales, igual que en el resto del estudio.
6. **E04, regla barata con filtro sintético:** si la regla elige una acción de DeepForest con filtro sintético, el posproceso incluye la síntesis que ese filtro necesita.
7. **E05:** el descriptor pre-registrado «proximidad» es la distancia normalizada al vecino más cercano, así que un valor *alto* significa copas *más separadas*. En las tablas y figuras se renombra «separación» para evitar confusiones.
8. **Semillas:** las políticas deterministas (SBS, VBS, Ridge, ExtraTrees y regla barata) se calculan una sola vez (semilla = −1) y no se replican.
9. **Criterio de SBS:** maximiza la recompensa media **por contexto** en entrenamiento (micro), como en la v0.5, no la media macro por sitio. Es el criterio declarado en el pre-registro.
10. **E02, F3a:** la calibración lineal tiene un intercepto de ≈ 0.44 y una pendiente de ≈ 0.5. El NDVI calibrado nunca baja de ≈ 0.4, de modo que el umbral fijo 0.3 conserva todas las cajas. Es una consecuencia mecánica de la escala, no evidencia de que el índice discrimine.

11. **Error de redacción en el pre-registro (§2):** dice que el portafolio de 40 acciones tiene «16 clásicas sobre señales RGB y 16 sobre NDVI sintético». El catálogo real, idéntico al de la v0.5, tiene **18 sobre señales RGB** (14 de ExG y 4 de oscuridad) y **14 sobre NDVI sintético**. El total sigue siendo 32 clásicas + 8 de DeepForest = 40. El análisis usó el catálogo real (`actions.csv`); solo cambia la descripción.

12. **E07, cajas degeneradas:** el protocolo no previó que 29 de las 3382 cajas externas tuvieran ancho o alto < 1 px, todas en el borde de la imagen. Se detectó al validar el formato (xmin, ymin, xmax, ymax, comprobado contra las imágenes `gt/*.png` de los autores) y se descartaron antes de ejecutar ninguna acción. Quedan 3353 copas. Una caja sin área no puede emparejarse, así que mantenerlas solo habría sumado falsos negativos a todas las políticas por igual.
13. **E07, comparación primaria degenerada:** el Selector-VI (ExtraTrees) elige la acción del SBS en los 218 parches externos. La diferencia es exactamente 0 y la regla del protocolo da «no inferior». Se informa como **trivial**, no como evidencia de transferencia. Los resultados de Ridge·V05 y de V05 frente a C0 en el conjunto externo **no** estaban pre-registrados como primarios y se presentan como exploratorios.
14. **E07, unidad de remuestreo:** el conjunto externo no tiene sitios. Los intervalos remuestrean parches, que vienen de dos ortoimágenes contiguas, así que son optimistas. El remuestreo por bloques de 10 identificadores consecutivos da intervalos casi idénticos, pero supone un orden espacial que no se pudo verificar.

## Verificación independiente

Un agente que no participó en el análisis recalculó 140 afirmaciones del informe a partir de los CSV (`verificacion_informe.md`): 124 coincidían, 13 tenían errores de redondeo, de agregación o de redacción, y 3 no eran verificables con los archivos. Las 13 se corrigieron en el informe. Los veredictos P1–P4 no cambian, salvo que el criterio de latencia de P4 pasa de «falla» a «inconcluso».

## Advertencias de interpretación

- Todas las comparaciones sobre NeonTreeEvaluation son análisis de un benchmark ya inspeccionado (ver pre-registro).
- Los intervalos por remuestreo de sitios están condicionados a las corridas calculadas. Con 20 sitios, y SJER y TEAK concentrando el 59 % de las parcelas, la incertidumbre entre sitios es grande.
- El «proxy de sombra» no está validado: no existe una anotación de sombra.
- Las curvas de respuesta espectral son gaussianas nominales, no curvas instrumentales medidas.
