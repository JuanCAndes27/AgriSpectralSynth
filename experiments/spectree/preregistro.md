# SpecTree — pre-registro de los experimentos E00–E09

**Fecha de congelación:** 2026-10-19, antes de ejecutar cualquiera de las pruebas de este documento.
**Plan de referencia:** `EXPERIMENTOS_FALTANTES_SPECTREE.md`.

**Advertencia de origen.** Los resultados de prueba de NeonTreeEvaluation ya se inspeccionaron en la versión 0.5 (artículo actual), y esa inspección motivó este plan. Por eso, todas las ablaciones sobre estos datos se reportan como **análisis sobre un benchmark previamente inspeccionado**, no como confirmación independiente. La única confirmación independiente posible es E07, si se consigue un conjunto externo.

## 1. Datos y unidades

- **Parcelas:** 189 de NeonTreeEvaluation con RGB de 10 cm, cajas anotadas e hiperespectral NEON válido, en 20 sitios.
  - Se excluyen 5 de las 194 parcelas del benchmark: tres de UNDE y una de NIWO no tienen cubo, y ONAQ_021 tiene 425 bandas.
- **Unidad de análisis:** la parcela. Los cuatro sensores nominales (Mavic 3M, Phantom 4M, RedEdge-MX, Sequoia+) son proyecciones de la misma parcela y no se cuentan como réplicas. Tampoco cuentan como réplicas las semillas ni los píxeles.
- **Unidad de remuestreo:** el sitio NEON. No se conoce otra dependencia entre parcelas (unidad de adquisición, fechas repetidas) más allá del sitio. Las parcelas de un sitio comparten vuelo y año, así que agrupar por sitio cubre esa dependencia.

## 2. Portafolios

- **Principal (despliegue solo RGB), 40 acciones:**
  - 16 clásicas sobre señales RGB (ExG y oscuridad);
  - 16 clásicas sobre NDVI sintético;
  - 4 de DeepForest sin filtro (puntaje $s\in\{0.1,0.2,0.3,0.4\}$);
  - 4 de DeepForest con filtro de NDVI sintético.
- **Referencia con modalidad adicional, 58 acciones:** las 40 anteriores más 14 clásicas sobre NDVI real y 4 de DeepForest con filtro de NDVI real. Este portafolio no representa un sistema RGB.

## 3. Objetivo y métricas

- **Métrica primaria:** la recompensa $r=\mathrm{F1}-0.25\min(1,|\hat N-N|/N)$, con IoU $\geq 0.4$, emparejamiento húngaro que maximiza el IoU total y $\lambda=0$.
- **Métricas secundarias:**
  - F1 por parcela;
  - F1 micro (TP, FP y FN sumados);
  - MAE y RMSE del conteo por parcela;
  - error de inventario agregado $(\sum\hat N-\sum N)/\sum N$;
  - tiempo.
- **Agregación primaria (macro por sitio):** promedio sobre semillas y sensores dentro de cada parcela, luego sobre las parcelas de cada sitio y finalmente sobre los sitios.
- **Agregación secundaria (micro por parcela):** promedio sobre las 189 parcelas. Se reportan ambas.
- **SBS:** se elige maximizando la recompensa **media por contexto** en los datos de entrenamiento, que es el criterio de la v0.5.

## 4. Particiones y ajuste

- **Validación externa:** las 5 particiones por sitio de la v0.5 (`site_folds`, asignación codiciosa balanceada y determinista). Se guardan en `splits.json`.
- **Semillas:** 0–4. Afectan el orden de entrenamiento y la exploración de los bandidos.
- **Hiperparámetros fijados a priori** (heredados de la v0.4 y no ajustados aquí):
  - LinUCB: $\alpha=0.5$, $\eta=1$;
  - Thompson: $v=0.2$;
  - 3 pasadas;
  - Ridge: $\eta=1$.
- **Ajuste interno:** lo que sí se ajusta usa validación interna agrupada por sitio dentro del entrenamiento de cada partición (4 subparticiones por sitio):
  - umbrales de filtro y calibración (E02);
  - regla barata (E03);
  - modelo no lineal, con una grilla de 4 configuraciones (E03);
  - portafolio reducido (E08);
  - punto de operación (E04).

  La estandarización se ajusta solo con el entrenamiento. Nada se ajusta con el conjunto de prueba de la partición.

## 5. Incertidumbre

- **Intervalos:** 95 % por remuestreo pareado de **sitios** (5 000 réplicas, semilla 12345). Cada réplica conserva todas las parcelas del sitio y calcula la diferencia macro por sitio.
- **Alcance de los intervalos:** son condicionales a las corridas ya calculadas. No incorporan la variación por reentrenar con otras muestras.
- **Dispersión entre semillas:** se reporta aparte.

## 6. Comparaciones primarias y márgenes

Margen de relevancia práctica: $\delta=0.02$ de recompensa macro por sitio. Hay cuatro comparaciones primarias, cada una con intervalo ajustado por Bonferroni (98.75 %). El resto es exploratorio.

| # | Comparación | Afirmación que se considera respaldada |
|---|---|---|
| P1 (E01) | Contexto C2 (C0 + atributos sintéticos) frente a C0, con LinUCB-F y portafolio de 40 | Límite inferior > 0 y estimación ≥ δ |
| P2 (E03) | Mejor selector aprendido (elegido en validación interna) frente a SBS, con 40 acciones | Límite inferior > 0 |
| P3 (E02) | Filtro sintético calibrado y con umbral ajustado frente a DeepForest sin filtro (F1 macro) | Límite inferior > 0 |
| P4 (E04) | Selector frente a DeepForest fijo: no inferioridad de calidad y menor latencia total | ΔF1 ≥ −0.02, MAE de conteo ≤ 1.10 × el de DeepForest fijo y latencia total media menor |

Un intervalo que contiene 0 se reporta como **inconcluso**. No se afirma equivalencia sin un margen definido.

## 7. Grillas fijadas

- **E01:**
  - C0: 21 atributos RGB, incluido el sensor nominal.
  - C1: C0 más 3 estadísticos de NDVI sintético.
  - C2: C0 más 9 atributos sintéticos:
    - NDVI (media, desviación, p90);
    - fracción de vegetación $f$;
    - NIR (media, desviación);
    - NDRE (media, desviación);
    - fracción de píxeles con NDVI sintético > 0.5.
  - C3: C0 más 9 transformaciones RGB simples:
    - VARI (media, desviación, p90);
    - GLI (media, desviación);
    - cromaticidad r, g, b media;
    - p90 de ExG.
  - C4 (diagnóstica): C0 más 3 estadísticos del NDVI real.
- **E02:**
  - Grilla de umbrales de NDVI: −0.2 a 0.8 en pasos de 0.05.
  - Calibración: lineal $\mathrm{NDVI}_{\mathrm{real}}\approx a+b\,\mathrm{NDVI}_{\mathrm{sint}}$, ajustada con píxeles de 1 m de los sitios de entrenamiento.
- **E05:** estratos por terciles calculados en el entrenamiento de cada partición, para:
  - densidad (árboles/ha);
  - tamaño (mediana del área de las cajas, m²);
  - proximidad (mediana de la distancia al vecino más cercano entre centros, normalizada por el diámetro medio de las dos cajas);
  - solapamiento (fracción de cajas con IoU > 0 con otra caja);
  - proxy de sombra (fracción de píxeles dentro de las cajas por debajo del umbral de Otsu de luminancia de la parcela). Es un **proxy no validado**: no hay anotación de sombra.
- **E08:** $\kappa\in\{0,0.1,0.25,0.5\}$ y $\lambda\in\{0,0.01,0.05,0.1,0.2\}$ s$^{-1}$. Portafolio reducido de $k\in\{4,8\}$ acciones, elegido con selección codiciosa que maximiza el VBS de entrenamiento.
- **E09:**
  - IoU $\in\{0.3,0.4,0.5,0.6\}$;
  - dos emparejamientos: (a) húngaro sobre el IoU total y luego umbral, que es el actual; (b) cardinalidad máxima de pares con IoU $\geq\tau$, con desempate por IoU.
- **E04:**
  - Tolerancias de P4 (§6).
  - Muestra de latencia: 2 parcelas por sitio, elegidas con la semilla 7 (40 parcelas en total, o todas las del sitio si tiene menos de 2).
  - 3 repeticiones en orden alternado, 1 hilo y una ejecución de calentamiento.

## 8. Lo que se reportará aunque contradiga la hipótesis

- Todas las variantes ejecutadas, incluidos los resultados negativos.
- Las exclusiones y fallos, en `run_notes.md`.
- Toda desviación de este plan, marcada como **desviación**, con su motivo.
