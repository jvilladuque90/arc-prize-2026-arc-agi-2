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
| — | B (timing Kaggle) | *(pendiente, viernes tras reset — decide entre ttt8 [si TTT≥35%] y scoring4 [si scoring≥15% y el proxy de rank-correlation lo avala]; si nada domina, NO gastar la submission)* | — |
| — | C (A/B en LB) | *(pendiente)* | — |
