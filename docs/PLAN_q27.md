# PLAN q27 — montar Qwen3.8-27B como base y trasladar nuestras palancas (26-oct-2026)

Decisión del usuario (26-oct): *montar el 27B sobre el agente público, auditar cada palanca que tenemos,
medir en las 18 de `strat` (U y tok/s), un submit, y ver qué TTT o técnica similar cabe en el reloj. Sin
defensas de posición: la mejor forma de subir el score.*

Estado: `q27strat` medido el 26-oct (`docs/PREREG_q27strat.md` §Resultado): infra OK a la primera (escalera 0: MTP+KV FP8),
**241 tok/s de media y ~440 sostenidos** en L4×4, U inconclusa (17/18 cortadas a 41K de razonamiento). Submit `q27sub` en cola.

## 1. Las dos bases, con números

| | `lb33-det` (nuestro campeón) | agente 27B (`madarshbb`, público) |
|---|---|---|
| modelo | Qwen3-4B, vocab 16 tokens, SFT NVARC (`sft139`) | Qwen3.8-27B, FP8, sin SFT para ARC |
| adaptación por tarea | **TTT** LoRA r=256, 128 secuencias (34,8% del reloj) | **ninguna**: razonamiento largo en contexto |
| generación | DFS sobre 12 tokens, 16 vistas, umbral p>0,2 | 1 pasada de texto: `<think>` hasta 40K tokens + rejilla |
| verificación | re-scoring con 8 augmentaciones + votos (`kgmon`) | modo *directo*: ninguna; modo *REPL*: ejecuta el programa contra las demos |
| LB (n=240) | 30,69 ± 0,415 | 21,53 (reporte de terceros) con ~12K tokens/intento |
| techo conocido del modelo | ~34 (pelotón público, mismo modelo) | **42,4** verificado por ARC Prize a *xhigh* sin límite de tiempo |
| reloj | 95% de 11,83 h | 11,4 h; ~140 tok/s asumidos ⇒ 5,8 M tokens por corrida |

Lectura: el 27B tiene un techo 8–12 pt por encima del nuestro **sin ninguna palanca**, y su LB bajo es por
**throughput**, no por capacidad. Todo lo que sigue es cómo comprar tokens y cómo gastarlos.

## 2. Auditoría palanca por palanca: ¿se traslada al 27B?

| nuestra palanca (4B) | evidencia que la respalda | ¿traslada? | cómo, y coste |
|---|---|---|---|
| **TTT por tarea** (LoRA r=256) | +20,6 pt sobre el 4B (nottt 9,58 vs 30,14) | **NO, literal** | 27B en bf16 son 54 GB (no caben en 4×24 GB con activaciones); el FP8 de vLLM no es entrenable con peft; y el coste escala ×7 en parámetros: 63 s/tarea en el 4B ⇒ ≥7 min/tarea ⇒ 28 h para 240. Muerto por memoria y por reloj. |
| **sustituto del TTT: escalado en tiempo de test** | ARC Prize: 1,5% → 22,8% → 42,4% según esfuerzo de razonamiento | **SÍ, es la palanca principal** | tokens por intento. Hoy el agente da ~12K (2 muestras, 240 tareas): régimen *low*. Con **1 muestra** ⇒ ~24K; con orden por coste y sin rescate ⇒ más. Medido en `q27strat`: **~440 tok/s sostenidos** (241 de media con cola), ×3 lo que el autor asumía. |
| **quitar el tope por tarea / presupuesto global** | 29,03 → 30,69 (con la semilla md5) | **SÍ** | el agente ya asigna `cap = oferta/n_tareas × 1,3` y respeta un deadline global: misma filosofía. Mejora: cap **por tarea según tamaño del prompt** (las grandes necesitan más tokens de salida solo para escribir la rejilla). |
| **cheap-first** | en el campeón | **SÍ** (ya está) | `run_submit` ordena por longitud del prompt. |
| **re-scoring con 8 augmentaciones** (hace la selección sin pérdidas) | medido en `strat` | **parcial** | con vLLM se puede pedir `logprobs` y puntuar una rejilla candidata bajo vistas transformadas, pero cada vista es un prompt de 2–8K tokens: 8 vistas × N candidatos ⇒ caro. Alternativa gratis: **verificación exacta del programa contra las demos (REPL)**, que es más fuerte que el consenso. |
| **votación entre vistas (`kgmon`)** | selección sin pérdidas | **SÍ, como votación entre muestras** | si hay ≥2 muestras, `attempt_1` = la rejilla más votada / programa verificado. Ya lo hace `ranked_answers` en modo REPL. |
| **semilla estable** | σ 1,27 → 0,415 | **no aplica** | T=1,0 con muestreo; la varianza del 27B será la de las muestras. Se mide con 2 submits (cuesta ~0 h-cuota aquí: el agente corre en el rerun). |
| **instrumento déficit→ancho** | validado en `haz` | **no** | requiere teacher-forcing sobre nuestro vocabulario; el 27B trabaja en texto. |
| **autopsia: 33% pools vacíos, 17% forma mal** | `strat` | **SÍ, como guía de reparto** | las tareas donde el 4B no emite nada son las candidatas a recibir más tokens del 27B (híbrido, §4). |
| **proxy NLL post-TTT** | validado | **no** | específico del 4B+TTT. |
| **sft139 memoriza training/Nemotron** | `nemvac` | **irrelevante** | el 27B no vio nuestro formato; su contaminación la gestiona ARC Prize (semi-privado 42,4). |
| **fallback `[[0]]` vs copia de la entrada** | NVARC+ | **SÍ** | el agente ya copia la entrada en los dos intentos cuando no responde. |

## 3. «TTT o técnicas similares» que SÍ caben en 12 h con el 27B

Ordenadas por (evidencia pública) ÷ (coste en tokens):

1. **Más razonamiento por intento, 1 muestra** (gratis en ingeniería). Es la única con número verificado
   detrás (42,4 a *xhigh*). Riesgo: sin segunda muestra, `attempt_2` = copia de la entrada (vale 0, como hoy).
2. **REPL con verificación exacta** (el otro modo del mismo agente): el modelo escribe `transform()`, lo
   ejecuta contra las demos y solo entrega programas que reproducen todas. Es inducción con verificador, lo que
   nuestras mediciones dicen que falta (`rearcvac`: el déficit es inducir una regla **nueva**). Coste: varios
   turnos por tarea (`max_turns` 40, `context_budget` 24K). Sin número público; se mide como segundo brazo.
3. **Augmentación en contexto (AIRV)**: presentar la tarea bajo 2–4 transformaciones D4/color, resolver cada
   una y des-transformar; votar. Es el equivalente «sin gradiente» de nuestras vistas. Coste ×k tokens; con
   prefix caching no se amortiza (el prompt cambia). Solo si sobran tokens tras 1.
4. **Reparto adaptativo**: dar el presupuesto a las tareas con más probabilidad de resolverse (prompt corto,
   pocas demos, salida pequeña) y recortar a las largas. Nuestro cheap-first a nivel de tokens.
5. **Híbrido 4B+27B** (§4): solo si el 27B no cabe para 240 tareas.

Lo que NO cabe y queda cerrado con aritmética: TTT de pesos sobre el 27B; SFT propio del 27B (ni GPU ni días);
ensembles de varios 27B.

## 4. El reloj: tres repartos posibles para el submit

Medido: `T` ≈ 400 tok/s sostenidos (441 en el log de vLLM con 16 peticiones; 241 de media por la cola). Oferta ≈ 400 × 40.000 s ≈ **16 M tokens** en el rerun (11,4 h − ~15 min de instalación y arranque).

| reparto | tokens/intento | lo que pierde |
|---|---|---|
| A. 27B solo, 2 muestras (autor) | oferta/480 | lo que hoy da 21,53 |
| B. **27B solo, 1 muestra** | oferta/240 | `attempt_2` |
| C. híbrido: 4B TTT en las ~160 tareas «baratas» (donde el pool no sale vacío) + 27B en las ~80 restantes | 27B: oferta×(fracción de reloj)/80 | hay que recortar el 4B (hoy usa el 95%): el 4B a media carga pierde ~1 pt (NVARC+ 30,21) |

**Decisión: reparto B.** Con T ≈ 400, una muestra de ~57K por tarea cuesta 13,7 M ⇒ cabe, y los ~2 M sobrantes pagan
segundas muestras en orden sample-major (el agente ya lo hace). `q27sub` = agente verbatim con `max_tokens` 57.344,
`thinking_token_budget` 49.152, `tok_per_s` 400. Prerreg: > 30,69 + 2σ (31,52) ⇒ nueva base; 21,5–31,5 ⇒ el presupuesto
sube el score pero no basta (siguiente: REPL / más tokens); ≤ 21,5 ⇒ el presupuesto no era el cuello en Kaggle.

## 5. Riesgos, dichos antes

- Infra ajena: modelo FP8 de `foysalemonshanto`, wheels de `mirzamilanfarabi`/`keithtyser`. Si la vía pasa,
  **copiar a datasets propios ese mismo día** (lección `nem8b`). La escalera de arranque existe porque
  MTP/`--reasoning-config` fallan a veces en L4: se registra qué opción arrancó.
- El 21,53 público es un reporte de terceros; nuestro primer submit lo mide (n=240, ~0 h-cuota).
- El semi-privado de ARC Prize no es el oculto de Kaggle: el 42,4 es techo orientativo, no promesa.
- Calendario: 7 días. Un submit por día. Cada submit del agente cuesta ~1 min de cuota (corre en el rerun).

## 6. Secuencia

1. `q27strat` → U, tok/s, infra. **[hecho 26-oct]**
2. Según §4: submit B (1 muestra) o C (híbrido). Prerreg: > 30,69 + 2σ ⇒ nueva base.
3. Segundo brazo en `strat`: modo REPL (verificación exacta). Si U_repl > U_directo ⇒ cambiar modo.
4. Copia de artefactos a datasets propios en cuanto haya un resultado positivo.
