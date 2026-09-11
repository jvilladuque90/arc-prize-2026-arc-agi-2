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
