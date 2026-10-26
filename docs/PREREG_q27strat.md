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
