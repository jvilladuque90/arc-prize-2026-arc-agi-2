# Prerregistro — `strat`: el mapa del terreno que no hemos mirado

Escrito **antes** de correr. 2026-09-10. ~2-2,3 h-cuota (L4×4, 4 workers, un solo brazo).

## Por qué

La auditoría del 10-sep encontró que **toda la evidencia por tarea de la campaña viene del decil
más barato del eval** (24 tareas, coste mediano 781 frente a 1.586, percentil 11, máximo 2.132
frente a 7.200). Sobre ese decil descansan tres conclusiones que gobiernan las decisiones:
*"la selección es sin pérdidas"*, *"16 vistas = 1 hipótesis"* y el **75%/11%** de la envolvente de
salida (3 de 4 tareas). El oculto son 240 tareas de toda la distribución.

## Diseño

- **Producción exacta, un solo brazo**: 4B `sft139`, r=256, α=32, 128 secuencias de TTT
  (calendario de producción), 16 vistas de decode, 45 s de DFS por vista, semilla md5 por tarea.
- **20 tareas estratificadas por coste entre los percentiles 5 y 95** del eval limpio
  (600 → 3.300), repartidas en 4 workers por **LPT** (la más cara al worker menos cargado) con el
  modelo de tiempo `t ≈ 161 + 0,348·coste` ajustado sobre 25 puntos ya medidos.
- **Deadline dimensionado sobre la cola** (4.200 s por worker, estimado ~3.500) y comprobado
  **dentro de la tarea** (antes de cada vista de decode), con bandera `truncado` — las dos
  lecciones de `rbuyer`.
- Por tarea: `vistas_con_gt` (0-16), **pools con scores** (para re-jugar la selección offline),
  `gt_nll_celda_pre` (base, antes del TTT) y `gt_nll_celda_post` (tras el TTT), coste, tiempos.

## Preguntas y predicciones registradas

- **P1 — envolvente de salida en la dificultad real.** Tasa de GT-en-pool con la salida DENTRO de la
  envolvente de las salidas demo > FUERA. Predicción: se replica (Fisher una cola p<0,10 con n=20;
  la potencia es la que es). Si la diferencia desaparece en tareas medianas y grandes, el 75%/11%
  era un artefacto del decil barato.
- **P2 — el proxy.** Spearman(`gt_nll_celda_post`, `vistas_con_gt`) < **−0,4**. Si se cumple, todo
  banco futuro de capacidad puede usar la NLL (informa en el 100% de las tareas) en vez del evento
  raro GT-en-pool: ~10× de potencia por hora-cuota. Se registra también `pre` para ver si el TTT
  es necesario para que el proxy funcione.
- **P3 — "la selección es sin pérdidas".** Re-jugando la selección top-2 sobre los pools guardados:
  fracción de tareas con GT en el pool en las que la selección lo elige. Predicción: >80%. Si sale
  bajo en tareas grandes, hay puntos gratis en la selección que el decil barato ocultaba.
- **P4 — GT-en-pool por tramo de coste.** Se registra la tasa en los tercios barato / medio / caro.
  Sin predicción numérica: es el mapa.

## Qué NO es

No es un test de ningún injerto. Es infraestructura de medida: el resultado son **hechos sobre el
terreno** de los que dependen las siguientes decisiones (dosis vs. no, comprador para el reloj,
Qwen3.5-2B). Nada se adopta ni se revierte con esta corrida.

---

# RESULTADOS (2026-09-11) — 75 min de pared ≈ 2,5 h-cuota, 20/20 tareas, 2 truncadas en vistas

| tarea | coste | gt/16 | nll pre | nll post | d_out |
|---|---|---|---|---|---|
| 28a6681f | 600 | 0 | 0,0728 | 0,0367 | 1,00 |
| dd6b8c4b | 726 | 0 | 0,1052 | 0,1015 | 0,00 |
| faa9f03d | 864 | 0 | 0,0383 | 0,0227 | 10,22 |
| f931b4a8 | 960 | 0 | 0,1249 | 0,3849 | 1,00 |
| 21897d95 | 1036 | 0 | 0,0239 | 0,0233 | 6,00 |
| 3a25b0d8 | 1180 | **1** | 0,0544 | 0,0063 | 5,04 |
| 53fb4810 | 1270 | 0 | 0,0183 | 0,0147 | 4,00 |
| 7b3084d4 | 1352 | 0 | 0,0730 | 0,0880 | 0,67 |
| b6f77b65 | 1440 | **2** | 0,0535 | 0,1104 | 0,00 |
| d59b0160 | 1536 | 0 | 0,0190 | 0,0160 | 0,00 |
| 58f5dbd5 | 1594 | **14** | 0,1166 | 0,0000 | 2,00 |
| 89565ca0 | 1720 | 0 | 0,5267 | 0,6180 | 2,00 |
| 36a08778 | 1892 | **7** | **0,0000** | 0,0000 | 0,05 |
| 4a21e3da | 1944 | 0 | 0,0447 | 0,0288 | 3,08 |
| 67e490f4 | 2090 | 0 | 0,0524 | 0,0659 | 8,50 |
| cb2d8a2c | 2318 | 0 | 0,0487 | 0,0510 | 1,17 |
| 221dfab4 | 2552 | 0 (15 vistas) | 0,0118 | 0,1125 | 4,00 |
| a47bf94d | 2772 | 0 (**0 vistas**, truncada) | 0,0074 | 0,0122 | 7,00 |
| b10624e5 | 2916 | 0 | 0,0385 | 0,0373 | 0,00 |
| 195c6913 | 3300 | 0 | 0,0376 | 0,0321 | 2,00 |

GT en algún pool: **4/20 = 20%** (`a47bf94d` no informa: el deadline cayó tras el TTT, 0 vistas).

## P1 — REFUTADA: la envolvente de salida era un artefacto del decil barato

| régimen | GT en pool |
|---|---|
| salida DENTRO de la envolvente | **1/4 = 25%** |
| salida FUERA | **3/16 = 19%** |

Fisher (una cola) **p = 0,624**; Spearman `d_out` vs gt/16 = −0,176. La predicción registrada era
p<0,10. **El 75%/11% no se replica en dificultad real.** Eran 3 de 4 tareas del percentil 11.

Consecuencias en cascada: (1) el mecanismo que motivó `ttford` era falso; la única lección que
sobrevive de `ttford` es *"fijar el orden de las demos cuesta 5 puntos"*, sin la lectura de "dañó
el régimen del 18%"; (2) **"dosis, no sustitución" queda muerta** antes de gastar su slot; (3) la
palanca de la envolvente de salida desaparece del plan.

## P2 — VALIDADA: el proxy NLL/celda post-TTT predice GT-en-pool

| proxy | n | Spearman | nll mediana con GT | sin GT | separación |
|---|---|---|---|---|---|
| **post-TTT** | 20 | **−0,430** | **0,0032** | 0,0370 | **×11,6** |
| pre-TTT | 20 | +0,086 | 0,0539 | 0,0416 | — |

Predicción registrada: < −0,4. **Se cumple en post-TTT y falla en pre-TTT**: el proxy necesita la
adaptación. Es la ganancia de infraestructura de la campaña: **la NLL post-TTT informa en el 100%
de las tareas** mientras GT-en-pool solo dispara en el 20% ⇒ ~10× de potencia por hora-cuota en
todo banco futuro de rango, dosis o modelo. La pregunta del comprador para el reloj (rama
Qwen3.5-2B) pasa de *inasequible* a *asequible*.

## P3 — (aproximado) la selección sigue pareciendo sin pérdidas

Re-jugado con el `beam_score` guardado (sin la mitad `score_aug` de producción, que el banco no
almacenó): el GT queda en top-2 en **3 de las 4** tareas donde está en el pool. n=4; compatible
con lo anterior, no lo refuerza.

## P4 — el mapa por coste: plano

| tercio | coste | GT en pool |
|---|---|---|
| barato | 600-1180 | 1/6 = 17% |
| medio | 1270-1720 | 2/6 = 33% |
| caro | 1892-3300 | 1/8 = 12% |

Sin tendencia monótona: **el coste tampoco es la historia**. n por tercio de 6-8, orientativo.

## Hallazgo colateral: contaminación medida

`36a08778` tiene **NLL/celda pre-TTT = 0,0000**: el modelo base ya sabe la respuesta antes de
adaptarse. Evidencia directa de contaminación eval→`sft139` en al menos 1/20 (5%). Encaja con la
advertencia de la memoria y obliga a **excluir del eval de medida las tareas con `pre` ≈ 0**.
