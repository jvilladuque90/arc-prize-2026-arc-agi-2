# PLAN — nuevo baseline y campaña final (desde 2026-09-29)

> **Estado a 17-oct y panorama completo (qué hace el sistema, diagramas, probado/no probado, próximos pasos): [docs/PANORAMA.md](docs/PANORAMA.md).** Este fichero queda como bitácora.

Deadline de submissions: **2026-11-02** (quedan ~5 semanas). Paper Track: 2026-11-09 (inscritos).
Campeón vigente: **`lb33-det` 30,69 ± 0,415** (n=3). Reglas de cuota y reloj: ver `CLAUDE.md`.

---

## 1. Qué hay ahí fuera (indagación del 29-sep)

### Leaderboard público (29-sep)

| # | equipo | score | cuándo |
|---|---|---|---|
| 1 | Tufa Labs | **83,06** | 29-sep |
| 2 | rabbithole | **79,72** | 28-sep |
| 3 | nvbanana (sucesor de NVARC, ganador 2025) | **75,42** | 29-sep |
| 4 | Yi-Chia Chen | 55,14 | 15-sep |
| 5-13 | pelotón | 34,0 – 37,5 | 28/29-sep |
| — | **nosotros** | 30,69 | — |

### Lo que NO es un baseline nuevo

- **Los tres de arriba no han publicado nada.** El track AGI-2 no tiene hitos de open-source (los hitos
  del 30-jun y 30-sep son solo de AGI-3); sus writeups llegan 7 días *después* del 2-nov, tarde para nosotros.
- **Todo lo público sigue siendo NVARC / `sft139`** — el mismo modelo que el nuestro. Revisados los ~490
  kernels de la competencia: ninguno trae un modelo distinto que puntúe más.
- `quarkliu/...qwen35-9b-best-replay` **no es un solver**: es un replay de respuestas precomputadas
  para el eval público (vale 0 en el oculto).
- **RTX Pro 6000** aparece en algún kernel, pero está **reservada a AGI-3**.

### Lo que SÍ es nuevo y aprovechable

1. **NVARC+ v2** (`luxluxshan/arc2-nvarc-v1`, 81 votos) — el baseline público más serio. Adoptado
   como candidato (§2).
2. **`nisshokuitsuki/qwen3-8b-nemotron77k-grid-direct`** — Qwen3-8B con SFT completo sobre 77.272
   trazas verificadas del dataset **NVIDIA Nemotron-SFT-ARC-AGI-v1** (eval-120 excluido), Apache 2.0.
   Razona (`<think>`) y emite la rejilla, **sin TTT**. Es un **generador de hipótesis distinto** al
   nuestro, que es exactamente lo que el §48 de `TTT_PLAN.md` concluyó que falta (§4, apuesta C).
3. El salto de nvbanana (24% en 2025 → 75% hoy) viene del equipo que hizo `sft139`: lo que cambió es
   **modelo y datos**, no trucos de inferencia. Confirma que el techo se mueve entrenando, no puliendo.

---

## 2. El baseline candidato: NVARC+ v2 — EMPATE en nivel con `lb33-det`, pero σ≈2,6× mayor ⇒ NO se adopta (29,44 y 30,97; 3-oct)

**Modelo** (idéntico al nuestro): `sorokin/qwen3_4b_grids15_sft139` — Qwen3-4B con vocabulario de
16 tokens de rejilla, SFT de NVARC sobre sus datos sintéticos. Misma imagen Docker que el campeón.

**Pipeline por tarea**: TTT con LoRA por tarea → decode DFS por lotes sobre los tokens ARC con umbral
de NLL acumulada → re-scoring con 8 augmentaciones → selección `score_kgmon`.

**Lo que cambia frente a nuestro campeón** (solo orquestación):

| | `lb33-det` (nuestro) | NVARC+ v2 |
|---|---|---|
| TTT | 128 secuencias | **64** (8 aug × 8 geometrías) |
| vistas de decode | 16 | **8** |
| tope por tarea | ninguno | **800 s** (A) / 700 s (B) |
| pasadas | 1 | **A + catch-up + B con semillas nuevas** (LoRA 137, aug 17/29, scoring +7) |
| selección | kgmon | kgmon sobre el pool **fusionado A+B**; el top-1 de A siempre sobrevive |
| sin candidatos | `[[0]]` | attempt_1 = **copia de la entrada** |
| semilla de scoring | md5 estable | `stable_seed` + `PYTHONHASHSEED=0` (equivalente) |

**Su hipótesis**: media carga puntúa igual que carga completa (sus 5 reruns de NVARC tal cual dan
26,9–32,2; afirman 35,97 en 5h44 con media carga) y el tiempo liberado paga una segunda opinión.

**Lo que dice nuestra evidencia, por delante**: el banco `ttt-2hip` (8-sep) midió **+0 tareas** al
unir dos TTT independientes. Pero fue n=12 y sobre el decil barato del eval (sesgo que descubrimos
después), así que **no es concluyente**. El LB es n=240: la pregunta se contesta subiéndolo.

**Submit inicial** (kernel `nvplus`, sin cambios): prerregistro contra `lb33-det`
(banda de 2σ = 29,86–31,52):
- **> 31,52** → nuevo baseline; los injertos de §4 se hacen sobre él.
- **29,86–31,52** → empate; una segunda muestra (rerun gratis) antes de decidir.
- **< 29,86** → se descarta y seguimos sobre `lb33-det`.

---

## 3. Las palancas que nos funcionaron (medidas en el LB)

| palanca | efecto | ¿ya está en NVARC+? |
|---|---|---|
| semilla de scoring estable (md5) | σ 1,27 → 0,415 | **sí** (`stable_seed`) |
| quitar el tope de reloj por tarea **+ semilla md5** (un solo submit, dos injertos) | +4 tareas (29,03 → 30,69), **no atribuible a cada uno** | tope: **NO** (800/700 s); semilla: sí |
| orden cheap-first | incluido en el campeón | sí |
| perfpatch (logits en GPU) | throughput | sí |
| snapshot del adaptador en RAM | −1 GB de GPU, 0 puntos | no (robustez, no se injerta) |

**Las que NO se injertan** (medidas negativas): r512 (−15,3 tareas por reloj), r64, ttford (−5,0),
tope de longitud (stack), SFT fusionado de escala, ensembles 2B/TRM, nottt, ttt05, selens,
modelos base 3.5-4B/MoE/27B, DSL de rejilla y de objetos, haz como respaldo (+29% de reloj).

⚠️ Quitar el tope en NVARC+ **no es trasladable tal cual**: allí el tope es lo que protege el tiempo
de la pasada B. El injerto correcto es **subir el tope de A solo para la cola cara** — ver §4, injerto A3.

---

## 4. Injertos planeados, en orden (un injerto por submit, 1 slot/día)

### Sobre el baseline ganador (si NVARC+ se adopta)

**A1. Pasada B DIRIGIDA** (coste de reloj 0: reasigna el tiempo de B).
La auditoría (`AUDITORIA_perdida.md`) midió que **el 33% de las tareas no emite ningún candidato** y
vale cero seguro. NVARC+ gasta la pasada B re-ejecutando *todas* las tareas en orden barato; lo
coherente con la auditoría es gastarla **primero en las tareas con pool vacío o de un solo candidato**
(como hace `yusuketogashi` Program079). Prerreg: > baseline + 2σ adoptar.

**Encontrado leyendo su código (29-sep): NVARC+ tiene un bug que hace esto peor de lo que parece.**
El worker solo escribe shard si hay candidatos (`if len(decoded_result):`), así que una tarea
**procesada con pool vacío** es indistinguible de una **no alcanzada**: el *catch-up* la repite con
**las mismas semillas** (mismo adaptador ⇒ casi seguro el mismo vacío). Con ~33% de pools vacíos, esa
repetición se come buena parte del tiempo de B. **Preparado** como `nvplus-a1`
(`notebooks/nvplus_a1.ipynb`, 25 líneas de diff, solo orquestación): marcador de tarea procesada,
catch-up solo para las no alcanzadas, y B en orden vacías → un candidato → resto (cheap-first en
cada grupo). Se lanza cuando `nvplus` tenga veredicto.

**A2. En la pasada B, respaldo por haz solo para pools vacíos.** El haz-8 recuperó ~2/18 tareas
fallidas (`haz`, `haz2`); murió porque costaba +29% del reloj de **una** pasada. Dentro de la B
el tiempo ya está pagado: solo cambia en qué se gasta. Instrumento de decisión ya validado:
`déficit → ancho ≈ exp(déficit)`, un forward por tarea.

**A3. Tope de A más alto para la cola cara** (nuestra palanca de +4 tareas, adaptada): subir 800 s →
1200 s solo en las tareas que el tope trunca con vistas a medio decodificar. Presupuestar contra B.

### Apuesta de salto (en paralelo, es la única con techo alto)

**C. Segundo generador: Qwen3-8B-Nemotron77K.** Otro modelo, otros datos, razona antes de responder.
1. **Puerta gratis / barata**: medir en eval-120 (excluido de su SFT) cuántas tareas resuelve y,
   sobre todo, **cuántas resuelve que `sft139` falla** (tenemos los pools del campeón por tarea).
   Si la unión no aporta ≥ 8-10 tareas sobre 120, muere aquí.
2. **Presupuesto de reloj antes de diseñar**: decode con razonamiento es largo. Solo cabe con vLLM
   (wheels como dataset, como `tf52-wheels`) y probablemente solo en el tiempo de la pasada B, o
   sustituyéndola. Medir tokens de traza por tarea en la puerta.
3. Integración: sus rejillas entran al pool y la selección kgmon decide; o attempt_2 cuando discrepa.

**Aritmética medida el 29-sep (0 h-cuota, 478 filas del dataset Nemotron):** las respuestas tienen
**mediana ~18.000 tokens** de razonamiento (p10 ~10K, p90 ~37K). 240 tareas × ~1,1 salidas ≈
**4,8 M tokens** de decode de un 8B. Con `generate` de HF en L4 (~20 tok/s por flujo) no cabe ni la
puerta de 120 tareas; con vLLM + pesos FP8 (L4 es Ada, lo soporta) el KV deja ~5 flujos por GPU ⇒
~700 tok/s agregados ⇒ **~2 h de pared solo para el 8B**, que es el tiempo entero de la pasada B.
Además los modelos de razonamiento de 8B puntúan bajo en AGI-2. ⇒ **Prior bajo y coste de ingeniería
alto (wheels de vLLM para sm89)**. Se ejecuta solo si A1-A3 se agotan, y la puerta se diseña
empezando por medir tokens/traza y tasa en 20 tareas, no en 120.

### Plan A (SFT propio) — solo si C muestra que los datos Nemotron transfieren

Ese modelo es, de hecho, la medición barata de si el dataset Nemotron vale para AGI-2. Si vale,
SFT de `sft139` (formato de 16 tokens, compatible con nuestro TTT) sobre esos datos, con el
detector de vacuidad primero (lección `sft-outenv`).

---

## 5. Presupuesto de cuota (semana 25-sep → 2-oct: 10 h-cuota)

| paso | h-cuota |
|---|---|
| commit `nvplus` (29-sep) | **1,2** (medido) |
| puerta C (8B en eval, 4 GPUs) | ~1,5–2 |
| commit del primer injerto (A1) | ~2 |
| reserva para una submission | 2 |

## 6. Registro

| fecha | qué | resultado |
|---|---|---|
| 29-sep | indagación de baseline (LB, 490 kernels, modelos, web) | NVARC+ v2 + Qwen3-8B-Nemotron77K |
| 29-sep | commit `nvplus` v1 (sin cambios): 36 min de pared ≈ 1,2 h-cuota; submission 120/120 válida | submit `56688393` PENDING (rerun ~12 h) |
| 30-sep | **`nvplus` = 29,44** < 29,86 | ❌ **DESCARTADO** por prerreg (−3 tareas vs 30,69). A1-A3 sin objeto; campeón sigue `lb33-det` |
| 30-sep | apuesta C: aritmética de reloj | ❌ no cabe: ≥15% del reloj (≈ −9 tareas) para un 8B sin medir; queda solo como fuente de DATOS |
| 30-sep | puerta gratis Nemotron (`scripts/nemotron_puerta.py`) | 89% puzzles nuevos, 0 eval; escala AGI-1 (mediana 88 celdas), 15,8% ≥400 |
| 30-sep | `nemvac` (detector de vacuidad, ~0,3 h-cuota) | ❌ grandes r=0,08: sft139 ya los resuelve sin TTT (74% greedy exacto); resto r=0,43 pero pequeños. Fuente Nemotron muerta para el déficit de escala |
| 30-sep | `rearcvac` (re-ARC grande vs chico, 119 pares, ~0,2 h-cuota) | escala ×5,75 sobre regla conocida: greedy 78→67%; regla nueva: 10%. **El déficit es la NOVEDAD de la regla, no la escala** ⇒ ningún dato fabricable lo cubre; Plan A cerrado |
| 30-sep | `nvplus` muestra 2 de varianza (rerun v1, 0 h-cuota), submit `56742827` | **30,97** (≥30,69 ⇒ la muestra 1 era en parte ruido; lectura en la fila siguiente) |
| 3-oct | **`nvplus` n=2: 29,44 · 30,97** (media 30,21, s 1,08, rango 1,53 = 3,7 tareas) frente a `lb33-det` n=3 (media 30,69, s 0,415, rango 0,83 = 2,0 tareas) | **Empate en nivel** (−0,49 pt = −1,2 tareas = 0,6 errores estándar). **σ ≈2,6× mayor**: NVARC+ conserva el tope por tarea (`task_cap` 800/700 s, evaluado al inicio de cada lote) cuya eliminación estrechó el canal del campeón el 2-sep (hipótesis respaldada por el código y por esa medición; NO medida en NVARC+). **No se adopta**: mismo nivel, menos consistencia y, sobre todo, un canal ancho **ciega** los A/B (con σ≈1,1, estimación gruesa de n=2, un injerto tendría que valer ≥2,2 pt para leerse con una sola muestra; con el campeón basta ~1 pt). A1-A3 se aparcan. **Tercera muestra: no** (el prerreg decía «tercera muestra»; con esta σ ya no cambia la decisión: para que la media de 3 superase al campeón en +1 pt la 3ª tendría que dar 34,7, >4σ) |
