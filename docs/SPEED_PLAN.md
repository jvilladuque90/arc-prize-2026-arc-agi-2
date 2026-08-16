# Familia VELOCIDAD — estrategia, expectativas y reglas de decisión

> Documento de estrategia de la palanca actual (2026-08-11). Complementa
> [TACTICAS.md](TACTICAS.md) (fundamentos), [RESULTS.md](RESULTS.md) (mediciones) y
> [COLAB.md](COLAB.md) (el banco de pruebas proxy). Registro vivo: se actualiza con cada
> resultado de la cadena.

---

## 1. Por qué velocidad, y por qué ahora

Todo lo demás está **medido y cerrado**:

- El campeón (30.14%) es **coverage-limited**: a ~900 s/tarea (rango 600–1250) con 4 workers y
  12 h, cubre ~**190 de 240 tareas (~79%)**; el resto queda en `[[0]]` y vale cero.
- Las palancas de *cap* de tiempo están agotadas (más tiempo/tarea → −2.08; menos → pérdida
  severa de calidad). Los ensembles con partners débiles restan (impuesto de cobertura). La
  selección es neutra. El SFT de inducción dio dos negativos.
- **Lo único nunca medido: la *eficiencia* por tarea.** Todas nuestras perillas cambiaban
  *cuánto tiempo permitimos*; ninguna cambió *cuánto trabajo hace el pipeline por segundo*.

La aritmética que lo gobierna (ver TACTICAS §3):

```
score ≈ cobertura × calidad          cobertura ∝ 1 / tiempo_por_tarea
```

**Hacer cada tarea más barata sin degradar la calidad = más cobertura = score.** Es la misma
lógica que hizo funcionar `cheap-first` (+1.67, nuestro único injerto exitoso): más tareas
drenadas antes del muro, sin tocar el cómputo por tarea.

## 2. Dónde puede estar el ahorro (anatomía del coste por tarea)

Cada tarea del campeón paga tres fases:

| Fase | Qué hace | Coste estimado* | Perilla de ahorro candidata |
|---|---|---|---|
| **TTT** | LoRA r=256, **16 augmentaciones** de los train pairs, 1 época | ~15–30% | `n_aug 16→8` (halva los ejemplos de TTT) |
| **Decode** | turbo-DFS sobre 16 vistas (subkeys), en 4 lotes | ~50–70% | fusionar lotes (4×4→2×8); menos subkeys |
| **Scoring** | re-scoring de cada candidato bajo 8 augmentaciones | ~10–25% | augs de scoring 8→4 |

\* Estimaciones indirectas — **nunca lo medimos con precisión**. Por eso el primer experimento
es el smoke instrumentado: sin el split real, no se puede elegir perilla racionalmente.

## 3. La cadena de experimentos (de gratis a caro)

Tres etapas, cada una decide si la siguiente se ejecuta:

### Etapa A — Ablación proxy en Colab (GRATIS, en curso)
`scripts/colab_ttt_ablation.py` — NVARC 2B en T4, 12 tareas holdout, **diseño pareado**
n_aug=16 vs n_aug=8, métricas densas (ΔNLL teacher-forced de la respuesta verdadera,
cell-accuracy, tiempo de TTT).

**Qué responde:** ¿cuánta *calidad de adaptación* pierde el TTT con la mitad de augmentaciones?
Es el riesgo del injerto `ttt8`: el ahorro de tiempo es seguro; la pérdida de calidad es la
incógnita.

**Lectura del resultado:**
- `ΔNLL(n8) ≥ ~85% de ΔNLL(n8=16)` y cell-acc similar → **riesgo de calidad BAJO** → luz verde.
- `ΔNLL(n8)` claramente inferior (adaptación a medias) → el ahorro no compensa → descartar
  `ttt8` sin gastar Kaggle, considerar perillas de decode/scoring.

*(Caveat honesto: es un proxy — 2B, LoRA r=64, TTT simplificado. Señal relativa, no absoluta.)*

### Etapa B — Smoke instrumentado en Kaggle (~40 min de L4, viernes tras el reset)
`lb33_timing` — el pipeline campeón exacto, 12 tareas repartidas por el espectro de coste, con
el **split TTT/decode/scoring real por tarea**.

**Qué responde:** dónde se van de verdad los ~900 s. Decide la perilla:

| Si el split muestra… | Entonces… |
|---|---|
| TTT ≥ ~20% del tiempo | `ttt8` tiene palanca: EV ≈ +1.5–2.5 pt → **A/B en LB** |
| TTT < ~10% | `ttt8` no paga (EV < +1 pt) → diseñar perilla de decode (fusión de lotes) o scoring (8→4 augs) con los números en la mano |

### Etapa C — A/B en el leaderboard (1 submission)
El injerto elegido (un solo número cambiado, disciplina `cheap-first`) contra el campeón.

**La aritmética de la expectativa para `ttt8`** (si TTT ≈ 25% del tiempo por tarea):
- Halvar el TTT ahorra ~12.5% del tiempo por tarea → cobertura ×1.14 → ~79% → ~90%.
- Las ~26 tareas marginales son las caras/difíciles (calidad esperada en ellas ~15–25%, menor
  que el ~38% medio de las atacadas) → ganancia ≈ **+1.5 a +2.5 pt** → **~31.5–32.5%**.
- Si la ablación proxy mostró pérdida de adaptación, restarle el coste de calidad sobre las
  ~190 tareas ya cubiertas — por eso la Etapa A va primero.

**Lectura del A/B:**
- `> 31%` → nuevo campeón; iterar la familia (¿n_aug 4? ¿scoring 8→4 encima?).
- `≈ 30%` → neutro; la perilla no era la que pensábamos → volver al split de la Etapa B.
- `< 29.5%` → revertir; la calidad pagó más de lo que la cobertura ganó (mismo patrón que el
  cap de 600 s).

## 4. Expectativa honesta de la familia completa

Si **todo** sale bien (ttt8 + quizá una perilla de scoring encima): **~32–33%** — se acerca al
corte del top-8 (~33.5 hoy, probablemente más alto en noviembre) **pero probablemente no lo
cruza sola**. Se persigue igual porque:

1. Es la **única palanca con EV positivo medible** que queda en el mapa.
2. **Compone con cualquier mejora futura de calidad**: cobertura extra multiplica cualquier
   subida de calidad por tarea que aparezca después.
3. El coste experimental es mínimo (proxy gratis + ~40 min de L4 + 1 submission) y el patrón de
   decisión está pre-registrado (este documento) — sin caza de fantasmas.

Lo que esta familia NO es: un camino al 47–67% de la cabeza. Ese salto exige un método
cualitativamente distinto que nadie ha publicado y que no hemos encontrado en dos rondas de
research. Con ~12 semanas al deadline (2 nov), la postura es: **exprimir el EV medible restante,
mantener el campeón estable, y documentar todo al nivel de writeup** (el Grand Prize y el Paper
Track evalúan el writeup de los ganadores — y nuestra documentación es también material del
paper track, donde un método/análisis novedoso puede premiar sin top score, como el TRM 2025).

## 4-bis. Runbook del viernes (2026-08-14, reset 19:00) — pre-registrado

Presupuesto: ~2.5 h de L4 + 1 submission. Orden exacto:

1. **19:05 — `lb33-timing` smoke** (~40 min): split real TTT/decode/scoring en 12 tareas.
2. **Leer el split** y decidir por las reglas de §3-B:
   - decode ≥ ~50% → `lb33-dfsbatch` (ya construido: fusión 4×4→2×8 + cap 540→1080 que preserva
     el presupuesto de exploración; único riesgo = memoria KV ~19GB/24GB).
   - TTT ≥ ~35% → reconsiderar `ttt8` pese a la luz roja del proxy.
   - Nada domina → registrar y no gastar la submission en velocidad.
3. **Smoke del injerto elegido** (~30-40 min) — para `dfsbatch`, verificar en el log: memoria pico
   (`allocated MB for inference`) y tiempo por tarea vs el smoke del campeón.
4. **Submit A/B** vs 30.14 (slot del día) con las lecturas pre-registradas de §3-C.

## 5. Registro de resultados de la cadena

| Fecha | Etapa | Resultado | Decisión |
|---|---|---|---|
| 2026-08-11 | A (proxy Colab) | **ΔNLL(n8) = 48% de ΔNLL(n16)** (0.061 vs 0.128); cell-acc 0.502 vs 0.519; exactas 2 vs 3; solo 3/12 tareas donde n8 adapta igual o mejor. (2 gates del carril resueltos: `--timeout` 30s del CLI; torchao 0.10 de la imagen → pip install torchao>=0.16.) | ❌ **Luz roja para `ttt8`** por la regla pre-registrada (umbral ≥85%): halvar augs halva la adaptación. Matiz: el efecto aguas abajo fue más suave (−1.7pp cell-acc) y el pipeline real tiene redundancia de consenso — `ttt8` queda DEGRADADO a "solo si la Etapa B muestra TTT ≥ 35% del tiempo", no descartado del todo. **Se promueve la fusión de lotes del DFS** (4×4→2×8): única perilla con CERO riesgo de calidad (mismas vistas, solo mejor utilización); su EV depende del share de decode (Etapa B). Riesgo a vigilar: memoria del KV cache con 8 secuencias en una L4. |
| 2026-08-11 | A-bis (bench dfsbatch, Colab, 4B real) | El bench cazó primero un **bug real**: la fusión ingenua [0,4,2,6] mezcla formas (h,w)/(w,h) en tareas no-cuadradas → tensor irregular → habría crasheado en producción (y reveló que los 4 lotes del campeón están agrupados por forma a propósito). Corregido a fusión por forma y medido: **speedup 1.05× medio (0.77–1.37; 0.77× en la tarea con trabajo DFS real)** y **memoria proyectada en L4 = 39.3 GB (no cabe en 22)**. | ❌ **`dfsbatch` ELIMINADO** (doble negativo: sin ganancia + memoria infeasible). Tercera entrega del carril gratuito: dos luces rojas y un bug cazado, sin gastar Kaggle. Candidatas restantes de la familia: `ttt8` (solo si TTT ≥ 35% del tiempo) y **`scoring4`** (augs de re-scoring 8→4; riesgo = calidad de selección; **pre-testeable en Colab** con rank-correlation antes del viernes). |
| 2026-08-12 | A-ter (proxy scoring4, Colab, 4B real) | **PASA la regla pre-registrada.** En las 5 tareas (de 12) con pool de candidatos suficiente: Spearman(S8,S4) = 0.995, acuerdo top-2 = **5/5 (100%)**, GT en top-2 idéntica bajo S8 y S4 (4 y 4, **0 perdidas**). Hallazgo lateral: 7/12 tareas dieron pools de ≤2 candidatos incluso con umbral ensanchado (-log 0.02; el de producción -log 0.2 deja 1-2) → en muchas tareas la selección es trivial y el re-scoring tiene poco que ordenar. Caveats: n=5, pools generados con umbral ensanchado. | ✅ **`scoring4` HABILITADO** para el viernes: si el timing muestra scoring ≥ ~15% del tiempo → graft de un número (augs 8→4) con riesgo de selección medido como bajo. Si scoring < 15% → ahorro insuficiente, no gastar la submission en velocidad (el slot pasa al A/B de la familia ESCALA, ver SCALE_PLAN). |
| 2026-08-13 | A-quater (ttt8 sobre BASE NUEVA, Colab) | Pareado n16 vs n8 con el 2B fusionado (ARC-Heavy n8000), 12 tareas nuevas: **exactas 5/12 = 5/12 (idénticas), cell-acc 0.931 vs 0.921, NLL-post 0.026 vs 0.032**. En la base vieja n8 halvaba la adaptación; con mejor prior el TTT necesita menos augs. | 🟢 **`ttt8` REHABILITADO para el campeón NUEVO** (post-D4): riesgo de calidad medido ≈ neutro sobre la base fusionada. Sigue gated por el split de timing (paga si TTT ≥ ~20-25% del tiempo). Candidato natural al A/B del domingo si el sábado hay campeón nuevo. |
| 2026-08-13 | A-quinquies (barrido umbral DFS, 4B, Colab) | 12 tareas × 3 umbrales (budget 150s): producción -log(0.2) → 0.8 cands medios, **GT en pool 5/12**, 24.6s; -log(0.05) → 2.0 cands, **6/12**, 43.2s (+75%); -log(0.01) → 5.2 cands, **6/12**, 105.7s (+330%). El techo de GT-alcanzable se estanca en 6/12: en la otra mitad el modelo no puede producir la respuesta a ningún umbral. | 🟡 **`dfswide` = palanca DÉBIL**: +1/12 de disponibilidad de GT a +75% de tiempo de decode (la fase dominante) → en neto REGRESIVO por cobertura. Variante adaptativa (re-pass ancho solo si pool ≤1) posible pero EV ≈ +0.5-1pt incierto — a la cola de baja prioridad. El umbral del campeón está bien tuneado. Mapa de velocidad COMPLETO: ttt8 (verde en base nueva), scoring4 (verde), dfsbatch (muerto), dfswide (débil). |
| 2026-08-14 | **B (timing smoke, Kaggle L4×4)** | Split REAL del campeón por primera vez (12 tareas, 6141 s; gates: cambio de layout de montaje de Kaggle mató la v1 — fix dual-layout —, y límite de 2 sesiones GPU simultáneas): **TTT 34.8% · decode 57.2% · scoring 7.9%**. Enorme varianza por tarea (decode 27→790 s; totales 247→1224 s; 2 tareas con scoring=0 = pool vacío). | ❌ **`scoring4` NO paga** (7.9% < gate 15%; ahorro ~4% del tiempo) — cerrada sin gastar submission, para eso era el smoke. ✅ **`ttt8` sobre base nueva CONFIRMADA con palanca**: 34.8% ≥ gate 20-25% → halvar augs ahorra ~17%/tarea → cobertura ×~1.2, riesgo neutro medido → **A/B del domingo** si el sábado hay campeón nuevo. Decode domina (57%) pero sus perillas están medidas muertas (dfsbatch, dfswide); la reasignación adaptativa del slack (27-790 s de dispersión) queda como candidata futura. |
| — | C (A/B en LB) | *(domingo: ttt8-base-nueva, condicional al campeón del sábado)* | — |
| 2026-08-16 | **D — familia nueva: reducir PASOS SECUENCIALES sin tocar el modelo** (a raíz de la pregunta del usuario sobre tokenización por nibbles) | Contexto del mapa de [ARCHITECTURE.md](ARCHITECTURE.md): el decode es 57,2% y es secuencial (1 forward por celda). Empaquetar 2 celdas/token halvaría los pasos, pero exige reentrenar sin arranque en caliente y **rompe la aritmética 2D en el 38% de las salidas (ancho impar)** → descartado sin gastar. La versión **lossless** de la misma idea es especulación con borrador gratis: medido hoy en el eval, **72% de las salidas tienen las dimensiones de su entrada y, de esas, mediana 85% de celdas idénticas** → el borrador puede ser la propia grid de entrada. Bench `colab_specdraft_bench.py` lanzado (4B real, 14 tareas nuevas): mide aceptación vs argmax, densidad de bifurcaciones al umbral de producción, forwards simulados con ventana k, y **valida el modelo de coste** (forward de 16 tokens vs de 1). | **Pre-registrado antes de ver el resultado:** aceptación ≥0,7 **y** coste(16)/coste(1) ≤1,3 → **VERDE**, injerto candidato (especulación dentro de `turbo_dfs`); aceptación 0,4-0,7 → **AMARILLO**, decide el coste de ingeniería; aceptación <0,4 **o** coste(16)/coste(1) >2 → **ROJO**, cerrar con dato como `dfsbatch`. Nota: el bench corre **sin TTT** (el harness fp16 de TTT no es fiel) → la aceptación real en producción puede diferir; se declara como limitación, no se corrige a posteriori. |
| 2026-08-16 | **D — resultado del bench specdraft** (4B real, 14 tareas nuevas, 4525 pasos, 548 s) | **1) El modelo de coste PASA con holgura**: ms por forward según ancho = {1: 56,8 · 4: 64,5 · 8: 75,0 · 16: **55,6** · 32: 61,3} → un forward de 16 tokens cuesta **lo mismo** que uno de 1 (ratio 0,98 vs gate 1,3) ⇒ el decode está limitado por latencia/ancho de banda por paso, no por cómputo, y **reducir forwards se traduce 1:1 en tiempo**. **2) Aceptación media 0,569** (rango 0,06-0,99; techo si el modelo produjera la GT: 0,732). **3) Densidad de bifurcaciones al umbral de producción = 0,007** — el DFS es casi una línea recta: solo el 0,7% de las posiciones tiene ≥2 tokens bajo el umbral (por eso los pools son de 0,8-1,6 candidatos). Variante conservadora ≡ agresiva. **4) Forwards simulados: 4525 → 2407 con k=16 = ×1,88** (×1,69 con k=4; ×1,92 con k=32). | 🟡 **AMARILLO por la letra de la regla** (aceptación 0,569 cae en 0,4-0,7) → decide el coste de ingeniería. **Recomendación: SEGUIR.** La métrica que importa no es la aceptación cruda sino la reducción de forwards (×1,88), y el gate de coste —el que podía matar la palanca— pasó con holgura. Aritmética: decode 57,2% → 30,4% ⇒ **−27% de tiempo por tarea** ⇒ cobertura 0,80 → satura en 1,00 (la cola se vacía por primera vez) ⇒ **+2 a +7,5 pts** según cómo resuelvan las tareas de la cola (las más caras; estimación central +3-5). Dos hechos abaratan el riesgo: (a) con bifurcaciones ~0 se puede implementar como **camino rápido + fallback al código actual** en el 0,7% de posiciones que ramifican, sin reescribir `turbo_dfs`; (b) el sesgo del "sin TTT" probablemente juega **a favor** — en las tareas grandes el techo GT (0,77-0,90) supera con mucho la aceptación medida (0,27-0,37), o sea que el modelo sin adaptar se aleja de la entrada más que la respuesta real. Coste real: las 4 filas del lote se **desincronizan** al aceptar longitudes distintas → hace falta `position_ids` por fila y caché ragged. **Gate duro antes de Kaggle: test de EXACTITUD** (el pool de candidatos debe ser idéntico al del baseline). |
| 2026-08-16 | **D-bis — test de exactitud v1: VACÍO (no verde)** | Los tres brazos (referencia con caché recortada / especulativo / turbo_dfs del campeón) dieron **0 hojas en las 8 tareas** → el `todas_iguales=true` comparaba conjuntos vacíos: **la comparación no vale**. Causa: **sin TTT y al umbral de producción, el 4B no completa NINGUNA grid en tareas nuevas** — la NLL acumulada supera `-log(0,2)` mucho antes de terminar (2-159 forwards, ninguna hoja). Esto **corrige la lectura del bench anterior**: el 0,007 de "bifurcaciones" no significaba "el árbol es casi recto" sino que en el 99% de las posiciones no sobrevivía *ningún* token. Corolario con valor propio: **el TTT no mejora el pool, es lo que hace que el pool exista.** | 🔁 **v2 relanzada** con: tareas de salida corta (árbol completo en ambos brazos), barrido de umbrales `-log(0,2)/-log(0,01)/-log(0,001)` (la exactitud es algorítmica; umbrales anchos = árboles ricos = test más estricto), y marcado explícito de comparación inválida si hay truncamiento o cero hojas. **Dato sólido del mismo run — modelo de coste a batch 4 (el de producción): {w1: 60,9 · w8: 63,9 · w16: 78,2 · w32: 109,4} ms** → a batch 4 el coste **ya no es plano**: k=16 cuesta 1,28× y k=32 cuesta 1,80×. Con k=8 el coste sube solo 5% y la reducción de forwards era ×1,81 ⇒ **óptimo real ~×1,7 en decode con ventana 8** (no 16) ⇒ −23% t/tarea, aún suficiente para vaciar la cola. Ventana corregida en el script. |
