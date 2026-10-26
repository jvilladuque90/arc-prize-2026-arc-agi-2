# PREREG `q27strat` (2026-10-26) — Qwen3.8-27B FP8 (vLLM, 4×L4) sobre las 18 tareas de `strat`

## Por qué este modelo

ARC Prize verificó **Qwen3.8-27B** (abierto) en **42,4%** sobre ARC-AGI-2 semi-privado con `reasoning_effort=xhigh`
($0,45/tarea); 22,8% en *low*, 13,2% en *medium*, 1,5% sin razonar. El score depende casi por entero del
presupuesto de razonamiento. Nuestro campeón (`lb33-det`, Qwen3-4B + TTT) está en 30,69.

## Qué se corre (auditoría del código base)

Agente público `madarshbb/arc-qwen-3-8-27b-agent` (41 votos, LB **21,53** según terceros), **código verbatim**
(10 ficheros: `grids/sandbox/backend/agent/repl/client/repl_agent/run_direct/run_submit/submit_main`).
Solo cambia el orquestador: en vez de saltar el agente en el commit y correrlo en el rerun oculto, aquí
se corre **en el commit** sobre eval, donde hay soluciones.

| pieza | qué hace | valor |
|---|---|---|
| servidor | vLLM OpenAI API, TP=4, `--enable-prefix-caching`, `gpu-memory-utilization 0,90` | `max_model_len` 81.920, `max_num_seqs` 16 |
| escalera de arranque | 6 juegos de opciones, de más a menos ambicioso: MTP especulativo + KV FP8 + `--language-model-only` → … → plano | se registra **cuál arranca** y cuánto tarda |
| modo | **directo**: el modelo razona y escribe la rejilla (es como ARC Prize midió el 42,4%) | `template_kwargs.reasoning_effort = xhigh` |
| presupuesto por intento | `thinking_token_budget` 40.960 + `max_tokens` 49.152 | **1 muestra** por tarea (el autor usa 2 con ~12K de tope efectivo) |
| rescate | si la respuesta se corta sin rejilla: una segunda llamada sin razonar con las notas | 8.192 tokens |
| muestreo | T=1,0, top_p 0,95, top_k 20 | del autor |
| tareas | las 18 de `strat` (eval, producción exacta del campeón): **15 sin GT en el pool del campeón** | concurrencia 16 |
| ventana | 80 min desde el arranque: instalación + arranque + generación con lo que quede (−2 min) | timeout duro 100 min |

Qué NO es: no es el modo REPL (síntesis de programas con verificación contra demos) del mismo agente; ese
queda como segundo brazo si este pasa.

## Medidas

1. **U** = tareas que el campeón falla y el 27B resuelve exactamente (1 muestra).
2. **tok/s agregados** reales en L4×4 = tokens generados / segundos de generación, con 16 en paralelo.
3. Infra: segundos de instalación, segundos de arranque, índice de la escalera que arrancó.
4. Por tarea: tokens, `finish` (`length` = cortada), estado (`answered/rescued/no grid`), pared.

## Lecturas (fijadas antes)

| resultado | lectura | siguiente paso |
|---|---|---|
| **U ≥ 3** y **tok/s ≥ 120** | vía de salto viable en el reloj | submit al LB con 1 muestra y orden por coste (~0 h-cuota: el agente corre en el rerun) |
| **U ≥ 3** y tok/s < 120 | el modelo aporta; el cuello es throughput | optimizar servidor (MTP, `max_num_seqs`, prompts más cortos) antes del submit |
| **U ≤ 2** con ≥ 12 tareas respondidas | a este nivel de razonamiento no complementa al 4B | cerrar directo; probar REPL una vez |
| ≥ 7 tareas cortadas o sin respuesta | **inconclusa** por presupuesto | repetir con menos tareas y más ventana |
| el servidor no arranca en ninguna opción | infra | leer `vllm-server.log`; sin lectura de modelo |

Sanidad: U se mide contra la GT exacta; las 3 tareas con GT en el pool del campeón sirven de control de que el
parseo funciona (se espera que alguna salga correcta).

**Coste previsto:** 80–100 min de pared en L4×4 ≈ **2,7–3,3 h-cuota**. Fuentes ajenas (modelo FP8 de
`foysalemonshanto`, wheels de `mirzamilanfarabi` y `keithtyser`): **comprobar `model_sources` en la metadata
committeada antes de esperar**, y copiarlas a datasets propios si la vía pasa.

---

## RESULTADO (26-oct, 77 min de pared ≈ 2,6 h-cuota)

**Infra: funciona a la primera.** Instalación 210 s, arranque del servidor 522 s, y arrancó la **opción 0 de la
escalera** (la más ambiciosa: MTP especulativo 2 tokens + KV FP8 + `--language-model-only`). KV cache 361.600
tokens ⇒ 14 peticiones de 81.920 en paralelo. Aceptación del borrador MTP ~80% en régimen estable.

**Throughput (la medida que decide):** 941.045 tokens generados en 3.895 s ⇒ **241,6 tok/s agregados de
media**, y el log de vLLM muestra **~440 tok/s sostenidos** con las 16 peticiones activas (la media baja por la
cola: al final quedan pocas peticiones vivas). El autor del agente asumía 140 tok/s: **la L4×4 rinde ×3 más**
de lo que su presupuesto suponía, y por eso su submit cortaba a ~12K tokens.

**U = 0/15, pero inconclusa por presupuesto (fila 4 del prerreg):** 17 de 18 intentos se cortaron en
`max_tokens` (`finish=length`), con **~41K tokens de razonamiento** cada uno (el `thinking_token_budget` forzó el
cierre) y sin rejilla parseable en la respuesta; el rescate sin razonamiento produjo rejillas en 15/18, todas
incorrectas salvo:
- `58f5dbd5` (control, el campeón también la tiene): única que terminó sola (`stop`, 41.850 tokens) y **correcta**.
- `f931b4a8`: 1 de 2 salidas de test correcta tras el rescate (vale 0,5 en Kaggle; el campeón la falla).

| tarea | campeón | tokens | finish | correcta |
|---|---|---|---|---|
| 58f5dbd5 | GT | 41.850 | stop | **[True]** |
| f931b4a8 | falla | 56.650 | length→rescate | [True, False] |
| otras 16 | 13 falla / 2 GT | 49–57K | length | no |

**Lectura:** a 41K tokens de razonamiento el modelo **no había convergido** en 17/18 tareas de `strat`
(estratificadas p5–p95 por coste: son la dificultad real, no el decil barato). ARC Prize midió el 42,4% a *xhigh*
sin tope de tiempo. El cuello sigue siendo el presupuesto de razonamiento, pero ahora con el dato de que hay
**~3× más tokens disponibles** de los que el agente público usa.

**Aritmética para el submit (n=240, rerun de 11,4 h menos ~15 min de instalación+arranque ≈ 11,1 h):**
oferta ≈ 400 tok/s × 40.000 s ≈ **16 M tokens**. Con 1 muestra de ~55K por tarea: 13,2 M ⇒ cabe, y el resto
(~3 M) paga segundas muestras en orden sample-major (el agente ya lo hace así). El autor, con 2 muestras y
49K/intento, pedía 23,6 M sobre una oferta que creía de 5,7 M: por eso 21,53.

**Lo que no se puede leer de aquí:** si con 57K de razonamiento el 27B resuelve tareas que el 4B falla. Eso lo
mide el LB con n=240 a ~0 h-cuota (el agente corre en el rerun). Es el siguiente paso.
