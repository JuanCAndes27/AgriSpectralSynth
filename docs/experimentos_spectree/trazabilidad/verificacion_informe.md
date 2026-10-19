# Verificación independiente de `docs/experimentos_spectree/README.md`

**Fecha:** 2026-10-19. **Alcance:** todas las cifras de tablas y prosa del informe, recalculadas con pandas a partir de los CSV de `docs/experimentos_spectree/` (idénticos a `results/spectree/out/`), `results/spectree/timings.csv`, `results/spectree/s01/data_manifest.csv`, `run_notes_s01.md`, `config.yaml`, `features.json`, `splits.json`, `actions.csv`, los logs de `results/spectree/logs/` y el pre-registro. No se modificó el informe ni ningún archivo de resultados.

**Criterios de verificación.** Una cifra es OK si coincide con el valor recalculado tras redondear a la precisión del informe. Los veredictos P1–P4 se contrastan con el pre-registro (§6): IC de Bonferroni al 98.75 %, δ = 0.02, P4 con ΔF1 ≥ −0.02, MAE ≤ 1.10 × DeepForest fijo y latencia media menor; un IC que contiene 0 se reporta como inconcluso.

## Tabla de verificación

| # | Ubicación | Afirmación | Valor en el informe | Valor recalculado | Fuente | Estado | Comentario |
|---|---|---|---|---|---|---|---|
| 1 | Configuración | Parcelas y sitios | 189 / 20 | 189 / 20 | data_manifest.csv | OK | |
| 2 | Configuración | Particiones y semillas | 5 / 5 | 5 particiones; semillas 0–4 | splits.json, config.yaml | OK | |
| 3 | Configuración | Portafolio RGB | 40 acciones | 40 con `portafolio_rgb40=True` (de 58) | actions.csv | OK | Composición real: 18 Clas-RGB + 14 Clas-NDVIsint + 4 DF + 4 DF+NDVIsint; el pre-registro (§2) dice 16 + 16. No lo afirma el informe, pero conviene revisarlo. |
| 4 | Métrica primaria | Remuestreo | 5 000 réplicas de sitios | replicas 5000, unidad sitio, semilla 12345 | config.yaml | OK | |
| 5 | Veredictos, P1 | C2 − C0, LinUCB-F | −0.010 [−0.037, +0.016] | −0.0105 [−0.0374, +0.0164], nivel 0.9875 | e01_comparaciones.csv | OK | |
| 6 | Veredictos, P1 | Veredicto | No respaldada (inconclusa, estimación negativa) | LI < 0 y estimación < δ; el IC contiene 0 | pre-registro §6 | OK | |
| 7 | Veredictos, P2 | Selector-VI − SBS | +0.015 [−0.007, +0.039] | +0.0149 [−0.0067, +0.0393], nivel 0.9875 | e03_resumen.csv | OK | |
| 8 | Veredictos, P2 | Veredicto | Inconclusa | LI < 0 | pre-registro §6 | OK | |
| 9 | Veredictos, P3 | ΔF1, F3b − F0 | 0.000 [0, 0] | 0.0 [0.0, 0.0] (nivel 98.75 % en e02_filtro.py) | e02_resumen.csv | OK | |
| 10 | Veredictos, P3 | El umbral óptimo es «no filtrar» | — | F3b con umbral −inf en 5/5 particiones | e02_umbrales_elegidos.csv | OK | |
| 11 | Veredictos / E04, P4 | ΔF1 | −0.028 [−0.063, −0.000] | −0.0277 [−0.0632, −0.0003] (98.75 %) | e04_P4.csv | OK | |
| 12 | Veredictos / E04, P4 | Razón de MAE | ×1.13 | 1.134 | e04_P4.csv | OK | |
| 13 | Veredictos / E04, P4 | Δ latencia | −0.04 s [−0.35, +0.14] | −0.0406 [−0.351, +0.140] (cuantiles 0.625 %–99.375 %) | e04_P4.csv, e04_latencia.py | OK | |
| 14 | Veredictos, P4 | «No respaldada en ninguno de los tres criterios» | 3 de 3 criterios fallan | F1: falla (estimación −0.028 < −0.02; LI −0.063). MAE: falla (1.13 > 1.10). Latencia: la estimación es favorable (−0.04 s, 2.84 frente a 2.88 s), pero el IC contiene 0, así que es **inconclusa**, no fallida | e04_P4.csv, pre-registro §6 | DISCREPA | El veredicto global («no respaldada») es correcto. La formulación de la latencia exagera: debe decir «inconclusa». La sección E04 ya lo dice bien («no se demuestra menor latencia»). |
| 15 | Veredictos | VBS − SBS | +0.067 [+0.046, +0.089] | +0.0671 [+0.0463, +0.0886] | e00_resumen_rgb40.csv | OK | |
| 16 | Veredictos | Ninguna de las 4 afirmaciones queda respaldada | 0/4 | 0/4 según los criterios | ítems 6, 8, 10, 14 | OK | |
| 17 | E00 | Conciliación con la v0.5: 58 acciones × 756 contextos, cajas iguales en el 100 %, \|ΔF1\| ≤ 1e-4 | 100 %; ≤ 1e-4 | exact_npred = 1 en 58/58; n = 756; máx \|ΔF1\| = 5.0e-5 | e00_conciliacion.csv | OK | |
| 18 | E00, tabla | Fila VBS | 0.674 / 0.706 / 0.707 / 0.704 / 5.7 / −13 % / +0.067 [+0.046, +0.089] / 100 % | 0.6736 / 0.7060 / 0.7072 / 0.7040 / 5.66 / −13.1 % / ídem / 1.0 | e00_resumen_rgb40.csv | OK | |
| 19 | E00, tabla | Fila Ridge | 0.619 / 0.622 / 0.675 / 0.676 / 9.2 / −20 % / +0.012 [−0.001, +0.028] / 18 % | 0.6186 / 0.6220 / 0.6749 / 0.6762 / 9.24 / −20.2 % / +0.0121 [−0.0014, +0.0276] / 18.0 % | ídem | OK | |
| 20 | E00, tabla | Fila SBS | 0.607 / 0.619 / 0.671 / 0.680 / 9.2 / −16 % | 0.6065 / 0.6187 / 0.6709 / 0.6804 / 9.22 / −16.0 % | ídem | OK | |
| 21 | E00, tabla | Fila LinUCB | 0.579 / 0.574 / 0.650 / 0.650 / 10.9 / −22 % / −0.028 [−0.052, −0.004] / −42 % | 0.5786 / 0.5742 / 0.6505 / 0.6499 / 10.93 / −22.0 % / −0.0280 [−0.0523, −0.0042] / −41.7 % | ídem | OK | |
| 22 | E00, tabla | Fila Thompson (LinTS) | 0.573 / 0.541 / 0.645 / 0.629 / 11.3 / −24 % / −0.033 [−0.057, −0.010] / −49 % | 0.5735 / 0.5415 / 0.6445 / 0.6292 / 11.27 / −23.6 % / −0.0330 [−0.0570, −0.0096] / −49.3 % | ídem | OK | |
| 23 | E00, tabla | Fila LinUCB-F | 0.568 / 0.584 / 0.642 / 0.653 / 10.6 / −21 % / −0.039 [−0.066, −0.015] / −58 % | 0.5676 / 0.5839 / 0.6424 / 0.6534 / 10.57 / −20.6 % / −0.0390 [−0.0655, −0.0153] / −58.1 % | ídem | OK | |
| 24 | E00, tabla | Fila Aleatorio, salvo inventario | 0.169 / 0.076 / 0.310 / 0.283 / 28.6 / −0.437 | 0.1695 / 0.0763 / 0.3104 / 0.2829 / 28.59 / −0.4371 | ídem | OK | |
| 25 | E00, tabla | Inventario de Aleatorio | +13 % | +12.50 % (0.12496), que redondea a **+12 %** | ídem | DISCREPA | Error de redondeo. |
| 26 | E00 | Los 3 bandidos están por debajo del SBS con IC que excluye 0 | Sí | LS = −0.0042, −0.0096, −0.0153 | ídem | OK | |
| 27 | E00 | DE entre semillas de los bandidos | 0.004–0.014 | 0.0039 (LinUCB-F), 0.0111 (LinUCB), 0.0144 (LinTS) | ídem | OK | |
| 28 | E00 | Diferencia por sitio LinUCB-F − SBS | de −0.17 (SERC, SOAP) a +0.04 (DELA, SCBI) | SERC −0.168, SOAP −0.168; SCBI +0.037, DELA +0.035 | e00_por_sitio.csv | OK | DELA +0.035 está en el límite (redondea a +0.03). El signo cambia: LinUCB-F > SBS en 6/20 sitios. |
| 29 | E00 | El portafolio de 58 da «prácticamente lo mismo» | — | VBS 0.674, Ridge 0.620, LinUCB-F 0.566 | e00_resumen_all58.csv | OK | Matiz: con 58 acciones, el IC de Ridge frente al SBS excluye 0 por poco: +0.013 [+0.0002, +0.029]. |
| 30 | E00 | Archivos de trazabilidad | Disponibles en trazabilidad/, salvo 2 grandes | Están los 7 archivos más feature_groups.json y run_notes.md; faltan action_metrics y policy_decisions | ls trazabilidad/ | OK | |
| 31 | E01 | Dimensión de C0–C4 | 21; +3; +9; +9; +3 | C0 = 21, C1 = 24, C2 = 30, C3 = 30, C4 = 24 | features.json | OK | |
| 32 | E01, tabla | Fila LinUCB-F, C0–C4 | 0.592 / 0.578 / 0.582 / 0.576 / 0.582 | 0.5922 / 0.5783 / 0.5817 / 0.5764 / 0.5816 | e01_resumen.csv | OK | |
| 33 | E01, tabla | Fila Ridge, C0–C4 | 0.624 / 0.619 / 0.597 / 0.610 / 0.616 | 0.6243 / 0.6186 / 0.5974 / 0.6103 / 0.6155 | ídem | OK | |
| 34 | E01, tabla | Fila LinUCB, C0–C4 | 0.575 / 0.587 / 0.565 / 0.573 / 0.581 | 0.5747 / 0.5874 / 0.5652 / 0.5735 / 0.5808 | ídem | OK | |
| 35 | E01 | SBS | 0.607 | 0.6065 | ídem | OK | |
| 36 | E01 | «Ningún contexto con atributos sintéticos supera a C0 (solo RGB) con ningún selector» | Ninguno | Con LinUCB, **C1 (0.587) > C0 (0.575)**: C1 − C0 = +0.013 [−0.005, +0.031], 12/20 sitios mejor | e01_resumen.csv, e01_comparaciones.csv | DISCREPA | La estimación puntual contradice la frase. Debe decir «ninguno supera a C0 de forma concluyente». El IC contiene 0. |
| 37 | E01 | Ridge, C2 − C0 | −0.027 [−0.063, −0.002] | −0.0269 [−0.0634, −0.0023] (95 %, exploratorio) | e01_comparaciones.csv | OK | |
| 38 | E01 | C2 − C3 | LinUCB-F +0.005 [−0.013, +0.023]; Ridge −0.013 [−0.045, +0.012] | +0.0053 [−0.0129, +0.0231]; −0.0128 [−0.0452, +0.0121] | ídem | OK | |
| 39 | E01 | C4 − C0, IC que contiene 0 | LinUCB-F −0.011; Ridge −0.009 | −0.0106 [−0.026, +0.005]; −0.0088 [−0.029, +0.008] | ídem | OK | Se omite LinUCB, donde C4 − C0 = +0.006 [−0.007, +0.020]; tampoco es concluyente. |
| 40 | E01 | Ridge C0 frente al SBS | +0.018 [+0.003, +0.036] | +0.0178 [+0.0027, +0.0357] | e01_resumen.csv | OK | |
| 41 | E02, tabla | F0 sin filtro | 0.671 / 0.669 / 0.707 / 9.2 | 0.6709 / 0.6691 / 0.7066 / 9.22 | e02_resumen.csv | OK | |
| 42 | E02, tabla | F1 sintético 0.3 | 0.597 / 0.594 / 0.653 / 14.6 / −0.074 [−0.153, −0.017] | 0.5973 / 0.5939 / 0.6526 / 14.58 / −0.0736 [−0.1527, −0.0168] | ídem | OK | |
| 43 | E02, tabla | F2: 0, umbral «sin filtro» en 5 particiones | 0.671 / 9.2 / 0; 5/5 | idéntico a F0; umbral −inf en 5/5 | e02_resumen.csv, e02_umbrales_elegidos.csv | OK | |
| 44 | E02, tabla | F3a y F3b | 0.671 / 9.2 / 0 [0, 0] | idénticos a F0; IC [0, 0] | e02_resumen.csv | OK | |
| 45 | E02, tabla | F4a NDVI real 0.3 | 0.670 / 0.667 / 0.707 / 9.2 / −0.001 [−0.001, 0.000] | 0.6703 / 0.6671 / 0.7071 / 9.20 / −0.0006 [−0.0015, +0.0001] | e02_resumen.csv | OK | |
| 46 | E02, tabla | F4b | 0.671 / 9.2 / 0 | F1 idéntico a F0 en todas las filas parcela × sensor | e02_resumen.csv, e02_detalle.csv | OK | |
| 47 | E02 | AUC TP/FP | 0.54 sintético; 0.50 real | 0.5395; 0.5042 | e02_auc.csv | OK | |
| 48 | E02 | Descarte con umbral 0.3 | 25 % TP; 27 % FP | 25.3 %; 27.1 % | e02_atribucion.csv | OK | |
| 49 | E02 | Calibración: sesgo y RMSE | −0.188 → −0.039; 0.262 → 0.135 | −0.1880 → −0.0387; 0.2624 → 0.1346 | e02_error_radiometrico.csv | OK | |
| 50 | E02 | Intercepto 0.44 y pendiente 0.5 | 0.44 / 0.5 | a medio 0.448 (0.398–0.520); b medio 0.519 (0.443–0.572) | e02_calibracion.csv | OK | |
| 51 | E02 | «El NDVI calibrado nunca baja de ≈ 0.4» | ≈ 0.4 | No hay CSV con el mínimo del NDVI calibrado. Solo se verifica la consecuencia: F3a = F0 exactamente | — | NO VERIFICABLE | |
| 52 | E02, tabla de errores | Rojo, NIR, NDVI y NDVI calibrado, macro por sitio | +0.076/0.098; +0.117/0.198; −0.188/0.262; −0.039/0.135 | Media sobre sitios: +0.0758/0.0978; +0.1170/0.1979; −0.1880/0.2624; −0.0387/0.1346 | e02_error_radiometrico(_por_sitio).csv | OK | |
| 53 | E02 | RMSE ÷ 2 | ÷ 2 | 0.262 / 0.135 = 1.95 | ídem | OK | |
| 54 | E03 | Contexto de la v0.5 | 25 atributos | V05 = 25 | features.json | OK | |
| 55 | E03, tabla | VBS | 0.674 / +0.067 [+0.046, +0.089] | ídem | e03_resumen.csv | OK | |
| 56 | E03, tabla | Selector-VI (P2) | 0.621 / +0.015 [−0.007, +0.039] (98.75 %) | 0.6214 / +0.0149 [−0.0067, +0.0393], nivel 0.9875 | ídem | OK | |
| 57 | E03, tabla | ExtraTrees | 0.621 / +0.015 [−0.003, +0.034] | 0.6214 / +0.0149 [−0.0028, +0.0341] | ídem | OK | |
| 58 | E03, tabla | Ridge | 0.619 / +0.012 [−0.001, +0.028] | ídem al ítem 19 | ídem | OK | |
| 59 | E03, tabla | Regla barata: 0.607 / 0.000 / siempre DeepForest | — | 0.6065 / 0.0 [0, 0]; frac. DF = 1.0 | e03_resumen.csv, e03_frecuencia_familias.csv | OK | |
| 60 | E03, tabla | LinUCB disjunto | 0.579 / −0.028 [−0.052, −0.004] | ídem al ítem 21 | e03_resumen.csv | OK | |
| 61 | E03, tabla | LinUCB-F, una pasada | 0.570 / −0.036 [−0.057, −0.015] | 0.5705 / −0.0361 [−0.0572, −0.0154] | ídem | OK | |
| 62 | E03, tabla | LinUCB-F, α = 0 | 0.570 / −0.036 [−0.061, −0.014] | 0.5701 / −0.0365 [−0.0607, −0.0137] | ídem | OK | |
| 63 | E03, tabla | LinUCB-F, 3 pasadas | 0.568 / −0.039 [−0.066, −0.015] | ídem al ítem 23 | ídem | OK | |
| 64 | E03 | Todos los bandidos tienen IC que excluye 0 | Sí | LS < 0 en los 5 bandidos (incluido LinTS) | ídem | OK | |
| 65 | E03 | Validación interna | ExtraTrees en 4/5 particiones; regla barata en 1 | Trees en particiones 1–4; CheapRule en la 0 | e03_seleccion_interna.csv | OK | |
| 66 | E03 | Partición 0: consultas y pares | 1 536 consultas; 1 213 pares únicos de 20 480 | 1536 / 1213 / 20480 (512 contextos × 40); folds[0] en el script | e03_curva_aprendizaje.csv, e03_selectores.py | OK | |
| 67 | E03 | Costo de la factorización | 468 coef.; 219 024 frente a 27 040 entradas; 0.53 frente a 0.04 ms | 468; 219024 (= 468²); 27040; 0.528 ms; 0.037 ms | e03_costo_politica.csv | OK | |
| 68 | E03 | Curva de aprendizaje | Ridge 0.529 → 0.619; LinUCB-F 0.531 → 0.568; SBS 0.60–0.62; Ridge alcanza al SBS solo con todos los datos | 0.5285 → 0.6186; 0.5310 → 0.5676; SBS 0.5985–0.6234; con 80 parcelas, Ridge 0.600 < SBS 0.610 | e03_curva_aprendizaje.csv | OK | |
| 69 | E03 (E10e) | Descomposición de la pérdida | −0.035 = −0.017 (25 %) + −0.010 (42 %) + −0.007 (1.9 %, −0.39 cada una) | −0.0348 = −0.0172 (25.1 %) + −0.0104 (41.7 %) + −0.0073 (1.85 %, −0.392) | e10e_descomposicion_perdida.csv | OK | −0.035 es media por contexto (micro), distinta del −0.039 macro de E00, como se indica. |
| 70 | E03 (E10e) | Pérdidas concentradas en SOAP, SERC, WREF y LENO | — | Suma por sitio: SERC −0.168, SOAP −0.168, WREF −0.107, LENO −0.100 (siguiente: JERC −0.061) | e10e_descomposicion_por_sitio.csv | OK | |
| 71 | E04 | Arranque del modelo, RSS y versión de torch | 1.80 s; 405 MB; torch 2.14 | 1.796 s; 404.7 MB; torch 2.14.1 | timings.csv | OK | |
| 72 | E04 | Muestra de latencia | 40 parcelas, 2 por sitio, 3 repeticiones | 40 imágenes, 2 × 20 sitios, rep 0–2 | timings.csv | OK | |
| 73 | E04 | Puntos de operación | LinUCB-F λ = 0 en 5/5; regla barata 0.05–0.1 | 0 × 5; 0.05 × 4 y 0.1 × 1 | e04_puntos_operacion.csv | OK | |
| 74 | E04, tabla | Clásico fijo | 0.027 / **0.021** / 0.046 s; 0.006; 0.021; 0.351; 26.7; 0 % | 0.0271 / **0.0205 (→ 0.020)** / 0.0463; 0.0059; 0.0211; 0.3511; 26.71; 0 | e04_resumen.csv | DISCREPA | La mediana redondea a 0.020. Diferencia menor. |
| 75 | E04, tabla | DeepForest fijo | 2.88 / 2.89 / 3.01; 2.88; 0.671; 9.2; 100 % | 2.8817 / 2.8906 / 3.0141; 2.8816; 0.6709; 9.22; 1.0 | ídem | OK | |
| 76 | E04, tabla | Regla barata | 2.96 / 2.96 / 3.10; 0.050; 0.000; 2.91; 0.671; 9.2; 100 % | 2.9556 / 2.9557 / 3.0955; 0.0496; 5e-6; 2.9059; 0.6709; 9.22; 1.0 | ídem | OK | |
| 77 | E04, tabla | LinUCB-F | 2.84 / 2.99 / 3.11; 0.023; 0.050; 0.0003; 2.77; 0.643; 10.5; 99 % | 2.8411 / 2.9863 / 3.1067; 0.0231; 0.0499; 0.00031; 2.7664; 0.6432; 10.46; 0.989 | ídem | OK | Ver ítem 132. |
| 78 | E04 | Costo de síntesis más atributos | 0.07 s = 2.5 % de DeepForest | 0.0231 + 0.0499 = 0.0730 s; 0.0730 / 2.8816 = 2.53 % | ídem | OK | |
| 79 | E04 / E08 | VBS con λ = 0.05 | F1 0.678; MAE 4.4; 2.54 frente a 3.34 s; ahorro del 24 % | κ = 0.25: 0.6778; 4.40; 2.5376 frente a 3.3364 s; 23.9 % | e08_grilla.csv | OK | 3.34 s sale de los tiempos de S01, no de la medición de E04 (DeepForest = 2.88 s). Conviene decirlo. |
| 80 | E04 / E08 | VBS con λ = 0.1 | F1 0.592; viola la tolerancia | 0.5925; ΔF1 = −0.078 < −0.02 | e08_grilla.csv | OK | |
| 81 | E05, tabla | Densidad: baja / media / alta (SBS / Ridge / VBS) | 0.335/0.364/0.546; 0.612/0.647/0.695; 0.633/0.632/0.670 | idénticos tras redondear | e05_estratos.csv | OK | |
| 82 | E05, tabla | Separación (proximidad) | 0.610/0.614/0.669; 0.586/0.606/0.682; 0.396/0.467/0.561 | idénticos | ídem | OK | Se omite el estrato «sin dato» (8 parcelas, 3 sitios). |
| 83 | E05, tabla | Tamaño de copa | 0.659/0.657/0.683; 0.610/0.642/0.709; 0.544/0.565/0.707 | idénticos | ídem | OK | |
| 84 | E05, tabla | Proxy de sombra | 0.650/0.654/0.712; 0.593/0.606/0.660; 0.506/0.527/0.604 | idénticos | ídem | OK | |
| 85 | E05 | VBS − SBS en escenas abiertas | +0.2 | Densidad baja +0.210; copas grandes +0.164; separación alta +0.165 | ídem | OK | Más preciso: «+0.16 a +0.21». |
| 86 | E05 | Ridge en copas separadas | +0.07 | +0.071 | ídem | OK | |
| 87 | E05 | «En escenas densas o con copas que se tocan… el VBS solo le saca +0.04 y ningún selector lo supera» | VBS +0.04; ningún selector > SBS | Densidad alta: VBS +0.037, pero **ExtraTrees/Selector-VI +0.009 > SBS**. Separación baja (copas que se tocan): **VBS +0.059**, Ridge +0.004, ExtraTrees +0.011. Solapamiento alto: VBS +0.066, Ridge +0.012, ExtraTrees +0.016 | e05_estratos.csv | DISCREPA | El +0.04 solo vale para densidad alta, y en todos esos estratos algún selector supera al SBS por poco. Hay que reformular la frase. |
| 88 | E05 | Estrato de baja densidad | 79 parcelas, 6 sitios, sobre todo SJER | 79 parcelas, 6 sitios: SJER 61, TEAK 10, DSNY 3, OSBS 2, JERC 2, NIWO 1 | e05_estratos.csv; terciles recalculados por partición con e05_descriptores.csv y splits.json | OK | |
| 89 | E05 | TEAK, Ridge | +0.069 en densidad baja; −0.038 en alta | +0.0687; −0.0375 | e05_dentro_de_sitio.csv | OK | |
| 90 | E05 | «En SJER, ningún selector supera al SBS en ningún tercil» | Ninguno | **SJER, tercil alto: LinUCB-F +0.0099 > SBS** | e05_dentro_de_sitio.csv | DISCREPA | Pasa a ser cierto si se restringe a selectores con información completa (Ridge, ExtraTrees, regla barata). |
| 91 | E06, tabla | V0 | −0.188 / 0.206 / 0.262 / 0.15 / −0.007 [−0.023, +0.006] / 0.024 s | −0.1880 / 0.2062 / 0.2624 / 0.1466 / −0.0070 [−0.0228, +0.0059] / 0.0244 (corregido) | e06_error_espectral.csv, e06_utilidad_contexto.csv, e06_costo_corregido.csv | OK | |
| 92 | E06, tabla | V1 | −0.182 / 0.200 / 0.254 / 0.15 / +0.001 / 0.004 | −0.1819 / 0.2000 / 0.2541 / 0.1503 / +0.0013 / 0.0042 | ídem | OK | |
| 93 | E06, tabla | V2 | −0.210 / 0.218 / 0.232 / — / −0.003 | −0.2099 / 0.2182 / 0.2315 / NaN / −0.0031 | ídem | OK | |
| 94 | E06, tabla | V3 | −0.188 / 0.206 / 0.262 / 0.15 / −0.007 | −0.1879 / 0.2062 / 0.2624 / 0.1464 / −0.0068 | ídem | OK | |
| 95 | E06, tabla | V4 | −0.070 / 0.123 / 0.172 / 0.13 / +0.014 [+0.004, +0.027] | −0.0702 / 0.1232 / 0.1720 / 0.1338 / +0.0144 [+0.0045, +0.0274] | ídem | OK | |
| 96 | E06, tabla | V5 | −0.302 / 0.318 / 0.350 / 0.45 / +0.001 / 0.009 | −0.3017 / 0.3185 / 0.3501 / 0.4484 / +0.0007 / 0.0087 | ídem | OK | |
| 97 | E06, tabla | V6 | −0.013 / 0.090 / 0.108 / 0.50 / −0.003 | −0.0135 / 0.0895 / 0.1082 / 0.5025 / −0.0027 | ídem | OK | |
| 98 | E06 | † V4 es «una de 14 comparaciones» | 14 | 7 variantes × 2 selectores frente a C0 = 14 | e06_utilidad_contexto.csv | OK | Con LinUCB-F, V0 y V6 frente a C0 también tienen IC que excluye 0, en sentido negativo. No se mencionan. |
| 99 | E06 | V1 cuesta 1/6 | 1/6 | 0.00424 / 0.02438 = 0.17 (≈ 1/5.8) | e06_costo_corregido.csv | OK | |
| 100 | E06 | V3: diferencia máxima de NDVI | 0.010, solo en píxeles recortados | máx \|V3 − V0\| = 0.0102 | logs/e06.log | OK | No se puede verificar que ocurra «solo en píxeles recortados». |
| 101 | E06 | V4 reduce el sesgo | −0.19 → −0.07 | −0.188 → −0.070 | e06_error_espectral.csv | OK | |
| 102 | E06 | r dentro de la parcela, macro por sitio | 0.15 | 0.147, con los 4 sensores | e06_error_detalle.csv | OK | |
| 103 | E06 | r medio por parcela | 0.36 | **0.370 con los 4 sensores**; 0.360 solo con Mavic 3M | e06_error_detalle.csv | DISCREPA | Mezcla de agregaciones: el 0.15 usa los 4 sensores y el 0.36 solo Mavic. Con un criterio coherente: 0.37 (4 sensores) o 0.13/0.36 (Mavic). |
| 104 | E06 | r negativa en 10 de 20 sitios | 10/20 | 10/20 (con 4 sensores y solo con Mavic) | ídem | OK | |
| 105 | E06 | r por sitio | WREF −0.43, SERC −0.38, LENO −0.32, BART −0.31 | **4 sensores: WREF −0.41, SERC −0.35, LENO −0.31, BART −0.30**. Solo Mavic: −0.43, −0.375, −0.32, −0.31 | ídem | DISCREPA | Son valores de Mavic 3M presentados junto a un macro de 4 sensores. |
| 106 | E06 | V6: r = 0.50, sesgo ≈ 0, contexto −0.003 | — | 0.5025; −0.0135; −0.0027 | e06_error_espectral.csv, e06_utilidad_contexto.csv | OK | |
| 107 | E06 | La correlación del artículo es 0.67 y se calculó con todos los píxeles juntos | 0.67 | No está en los CSV | — | NO VERIFICABLE | |
| 108 | E08 | LinUCB-F pierde en 15/20 combinaciones con IC que excluye 0 y no gana en ninguna | 15/20; 0 | 15/20; 0 victorias significativas (1 positiva puntual: κ = 0, λ = 0.1, +0.004) | e08_grilla.csv | OK | |
| 109 | E08 | Ridge no tiene diferencias concluyentes; mejor celda κ = 0, λ = 0.1 | 0 concluyentes; +0.024 | 0/20; +0.0238 [−0.0029, +0.0512] | ídem | OK | |
| 110 | E08 | Brecha SBS–VBS | +0.04 a +0.22 | 0.0406 a 0.2159 | ídem | OK | |
| 111 | E08 | Con κ = 0, LinUCB-F | −0.020* | −0.0198 [−0.0367, −0.0057] (λ = 0) | ídem | OK | |
| 112 | E08 | VBS con 4 acciones | 0.669 frente a 0.674 | 0.6687 frente a 0.6736 | e08_portafolios.csv | OK | |
| 113 | E08 | LinUCB-F con 40 → 8 → 4 acciones | 0.568 → 0.579 → 0.587 | 0.5676 → 0.5795 → 0.5866 | ídem | OK | |
| 114 | E08 | λ = 0.1, 4 acciones: LinUCB-F frente al SBS | 0.275 frente a 0.275 | 0.27471 frente a 0.27467 | ídem | OK | |
| 115 | E09 | Los dos emparejamientos difieren en ≤ 0.003 de F1 | ≤ 0.003 | **máx. 0.0037** (wsd_exg-otsu_4m, IoU 0.3, F1 macro); en selectores, máx. 0.0016 | e09_metodos_fijos.csv, e09_selectores.csv | DISCREPA | Debe decir «≤ 0.004». |
| 116 | E09 | Orden estable de IoU 0.3 a 0.6 | VBS > Ridge ≈ SBS > bandidos ≫ clásicos | Se cumple en los 4 IoU (p. ej., a 0.6: 0.502 > 0.491 ≈ 0.488 > 0.464–0.473 ≫ 0.146) | ídem | OK | |
| 117 | E09 | F1 de DeepForest con s = 0.3 | 0.724 (IoU 0.3) → 0.498 (IoU 0.6) | 0.7240 (card) / 0.7236 (hung) → 0.4976 | e09_metodos_fijos.csv | OK | |
| 118 | E09 | A IoU 0.6, el SBS recalculado elige s = 0.3 en todas las particiones | Sí | acciones = «df_rgb_s0.3» | e09_selectores.csv | OK | |
| 119 | E09, tabla | DeepForest s = 0.3 | 9.0 / 20.6 / −22 % | 8.98 / 20.58 / −22.0 % | e09_metodos_fijos.csv | OK | |
| 120 | E09, tabla | DeepForest s = 0.1 | 9.7 / 19.3 / −0.2 % | 9.70 / 19.32 / −0.24 % | ídem | OK | |
| 121 | E09, tabla | Mejor clásico RGB | 27.1 / 36.7 / **+38 %** | 27.13 / 36.66 / **+37.48 % (→ +37 %)** (wsd_exg-otsu_4m) | ídem | DISCREPA | Error de redondeo. Además, «mejor» se define por F1. Otro clásico RGB, lmw_exg-otsu_4m, tiene F1 casi igual (0.350 frente a 0.351) con MAE 24.7 e inventario +10 %. |
| 122 | E09 | Escenas vacías | 0 parcelas sin árboles; mínimo 1; 20 con ≤ 3 | 0; 1; 20 | e09_escenas_vacias.csv, data_manifest.csv | OK | |
| 123 | E10a | Muestra | 189 parcelas × 5 sensores | 945 filas, 5 sensores (4 drones + S2A) | e10a_proposicion1.csv | OK | |
| 124 | E10a | Diferencia fuera de los píxeles saturados | ≤ 1.5 × 10⁻⁴ | máx. 1.479 × 10⁻⁴ | ídem | OK | |
| 125 | E10a | Fracción saturada media | 0.46 % | **0.43 %** con las 945 filas; 0.46 % solo con Mavic 3M | ídem | DISCREPA | El texto dice 189 × 5 sensores, pero la cifra corresponde a un solo sensor. |
| 126 | E10a | Mediana de la fracción saturada | < 0.01 % | 0.0013 % | ídem | OK | |
| 127 | E10a | «Llega al 20 % en 4 parcelas de WREF» | 4 parcelas al 20 % | Solo **1** parcela llega al 20 % (WREF_084: 20.5 %). Las 4 parcelas oscuras de WREF tienen 12.2 %, 15.8 %, 18.8 % y 20.5 % (Mavic) | ídem | DISCREPA | Reformular: «entre 12 % y 20 % en 4 parcelas de WREF». |
| 128 | E10a | El factor cromático común se cumple en los 5 sensores | Sí | c_comun = True en 945/945 | ídem | OK | |
| 129 | E10b | Mejor F1 con 14 algoritmos | 0.346 / 0.319 / 0.234 | 0.3463 / 0.3187 / 0.2344; 14 algoritmos | e10b_resolucion_mejor.csv, e10b_resolucion_por_algoritmo.csv | OK | |
| 130 | E10b | La degradación explica ≈ 25 % | ≈ 25 % | (0.3463 − 0.3187) / (0.3463 − 0.2344) = 24.7 % | ídem | OK | |
| 131 | E10c | Correlación máxima sin desplazamiento; ±1 m; 102/189 parcelas en (0, 0) | 0.36; 0.29–0.31; 102 | 0.3596; vecinos axiales 0.289–0.312 (diagonales 0.241–0.280); 102 | e10c_registro.csv, logs/e10.log | OK | Precisar que el rango 0.29–0.31 corresponde a desplazamientos axiales. |
| 132 | Consistencia E00 / E04 | F1 y MAE de LinUCB-F | E00: 0.642 / 10.6; E04: 0.643 / 10.5 | E00 usa los 4 sensores. E04 usa solo Mavic 3M (desviación 5), aunque la columna se titula «F1 macro (189)» | e00_resumen_rgb40.csv, e04_resumen.csv, run_notes.md | DISCREPA | La misma política aparece con dos valores sin explicación en el informe. Hay que anotar en la tabla de E04 que la calidad es de Mavic 3M. |
| 133 | E10d | Orden de agregación | −0.205, r 0.144; −0.188, r 0.147 | −0.2054, 0.1441; −0.1880, 0.1466 | e10d_agregacion.csv | OK | |
| 134 | E10f | Sin codificación del sensor | Ridge −0.001; LinUCB-F +0.016 [−0.001, +0.037] | −0.0012; +0.0157 [−0.0011, +0.0366] | e10f_sin_sensor.csv | OK | |
| 135 | E10g | VBS con λ = 0.1 | RGB 40: 0.592 / 4.7 / 1.61 s / 49 %; 58: 1.51 s, 5.6 % HSI | 0.5925 / 4.70 / 1.610 / 48.8 %; 1.510 s, 5.56 % | e10g_vbs_lambda01.csv | OK | |
| 136 | Qué cambia | Margen ≈ 24 % (λ = 0.05); AUC 0.50–0.54; contexto 0.07 s; Ridge ≥ LinUCB-F; 4 ≈ 40 acciones | — | Coherente con los ítems 79, 47, 78, 63/58 y 112 | — | OK | |
| 137 | Reproducir | Tiempo de S01 | 739 s (12.3 min), 2 procesos | 739 s | run_notes_s01.md, config.yaml | OK | |
| 138 | Reproducir | action_metrics.csv | 45 MB; 350 784 filas | 46.8 MB (44.6 MiB); 350 784 filas (= 189 × 4 × 58 × 4 IoU × 2) | wc -l, du | OK | La unidad es MiB, no MB. |
| 139 | Reproducir | policy_decisions.csv y npz | 6 MB; 71 MB | 6.4 MB; 70.5 MB (189 archivos) | du -sb | OK | |
| 140 | Reproducir | E08 ≈ 100 min; E04 ≈ 25 min; el resto < 15 min | — | No hay registro de los tiempos de inicio | — | NO VERIFICABLE | |

## Resumen

- **Afirmaciones verificadas:** 140 filas. Cada fila de tabla cuenta como una afirmación, aunque tenga varias cifras. En total se compararon más de 400 valores numéricos.
- **OK:** 124. **DISCREPA:** 13. **NO VERIFICABLE:** 3.
- **Veredictos P1–P3:** los tres se derivan correctamente del pre-registro.
- **Veredicto P4:** el veredicto global es correcto, pero el criterio de latencia está mal calificado (ítem 14).

## Correcciones requeridas, por orden de importancia

1. **E01 (ítem 36).** Cambiar «Ningún contexto con atributos sintéticos supera a C0 con ningún selector» por «ninguno lo supera de forma concluyente». Con LinUCB, C1 (0.587) > C0 (0.575): +0.013 [−0.005, +0.031].
2. **E05 (ítems 87 y 90).** Corregir dos frases:
   - «En escenas densas o con copas que se tocan… VBS solo +0.04 y ningún selector lo supera». El +0.04 solo vale para densidad alta. Con copas que se tocan (separación baja), el VBS saca +0.059 y con solapamiento alto +0.066. En los tres estratos, ExtraTrees o Ridge superan al SBS por poco (+0.004 a +0.016).
   - «En SJER ningún selector supera al SBS en ningún tercil». En el tercil alto, LinUCB-F lo supera por +0.010. O se corrige la frase o se restringe a los selectores con información completa.
3. **E06 (ítems 103 y 105).** Usar una sola agregación. Con los 4 sensores: r medio por parcela 0.37 (no 0.36); WREF −0.41, SERC −0.35, LENO −0.31 y BART −0.30. Los valores del informe son de Mavic 3M solamente, mientras que el 0.15 macro usa los 4 sensores.
4. **Veredictos P4 (ítem 14).** Cambiar «No respaldada en ninguno de los tres criterios» por: «F1 y MAE fuera de tolerancia; latencia inconclusa (−0.04 s, IC 98.75 % contiene 0)». El veredicto global no cambia.
5. **E10a (ítems 125 y 127).**
   - La fracción saturada media con 189 × 5 es 0.43 %; el 0.46 % es solo Mavic.
   - Cambiar «llega al 20 % en 4 parcelas de WREF» por «entre 12 % y 20 % en 4 parcelas de WREF (máximo 20.5 %)».
6. **E04 frente a E00 (ítem 132).** Indicar en la tabla de E04 que F1 y MAE se evaluaron solo con Mavic 3M (desviación 5). Así se explica 0.643 / 10.5 frente a 0.642 / 10.6. Indicar también que el «2.54 frente a 3.34 s» del VBS usa los tiempos de S01, no la medición de E04 (ítem 79).
7. **Correcciones menores de redondeo:**
   - E09, emparejamientos: «≤ 0.004», no «≤ 0.003» (ítem 115).
   - E09, sesgo del mejor clásico RGB: +37 %, no +38 % (ítem 121).
   - E00, inventario de Aleatorio: +12 %, no +13 % (ítem 25).
   - E04, mediana del clásico fijo: 0.020 s, no 0.021 s (ítem 74).
8. **Precisiones opcionales:**
   - E05: escribir «+0.16 a +0.21» en lugar de «+0.2» (ítem 85).
   - E10c: aclarar que el rango 0.29–0.31 corresponde a desplazamientos axiales (ítem 131).
   - E00: señalar que, con 58 acciones, el IC de Ridge excluye 0 por poco (ítem 29).
   - Revisar la composición del portafolio de 40 acciones (18 + 14 en actions.csv frente a 16 + 16 en el pre-registro) (ítem 3).
