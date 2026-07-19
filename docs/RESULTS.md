# Resultados medidos

Registro obligatorio de cada corrida de evaluación (disciplina experimental: sin número
medido no se prioriza nada). El score es **pass@2** sobre el split `evaluation`
(120 tareas públicas de ARC-AGI-2), salvo que se indique subconjunto.

Referencia externa (set privado, Kaggle 2025): NVARC 24.03% · ARChitects 16.53% ·
MindsAI 12.64%. La baseline sin ML ronda ~0%.

**Leaderboard 2026 (observado 2026-07-07):** líder 55.14%; pelotón enorme en ~33%
(aparentemente un notebook público compartido, "LB33"). El listón de "competitivo" este
año es mucho más alto que el SOTA 2025.

## Leaderboard (submissions reales)

| Fecha | Versión | Score LB | Notas |
|---|---|---|---|
| 2026-07-07 | submit v3 (Vía B: 3B v10, TTT+scoring, budget 1h) | 0.83% | ~2/240 outputs exactos. Primera submission propia; valida el camino completo. |
| 2026-07-11 | **LB33 baseline** (Vía A: checkpoint NVARC, TTT r=256 + DFS) | **28.47%** | Control establecido. Auditado limpio ([LB33_AUDIT.md](LB33_AUDIT.md)); smoke 2.5/4. Es la línea base sobre la que injertar (tiers/cobertura, selección). Esperar caída público→privado de varios puntos (24.03% fue el privado de NVARC en 2025). |
| 2026-07-13 | LB33 + **cheap-first** (injerto de cobertura, id 54657220) | **30.14%** | ✅ **+1.67 vs 28.47%**. Único cambio: cola ordenada barata→cara. Confirma que la **cobertura era el cuello** (el baseline dejaba tareas en `[[0]]`). Nuevo control. |
| 2026-07-14 | LB33 + cheap-first + **presupuesto decode adaptativo** (id 54698030) | **28.06%** | ❌ **REGRESIÓN -2.08 vs 30.14%**. Causa: medir el presupuesto desde el *fin del TTT* (no desde el inicio de la tarea) dio a cada tarea MÁS tiempo total (TTT + hasta 1200s decode) que el baseline (1200s totales) → menos tareas completadas → menos cobertura. Lección: **el tiempo por tarea es la perilla de throughput; más tiempo = menos cobertura**. Revertido; campeón sigue cheap-first (30.14%). |
| 2026-07-15 | LB33 + cheap-first + **cap 600s** | **NO enviado** | El smoke bajó a 1.5/3 (vs 2.5/4) con la mitad de subkeys (14 vs 36): el cap de 600s, combinado con el cap interno del DFS (540s/llamada), deja pasar solo ~1 batch de decodificación por tarea (baseline: ~2) → pérdida de calidad severa. Una tarea produjo cero salida. Descartado sin gastar submission. |

| 2026-07-15 | **NVARC Qwen3-2B solo** (cheap-first, id 54733478) | **18.33%** | Mucho más débil que el 4B (30.14%). **Lección: el smoke (4.0/4) es INÚTIL para juzgar fuerza de modelo** — las 4 tareas de smoke están contaminadas (ambos modelos las memorizaron), así que dan ~perfecto y no discriminan. Solo el LB oculto revela la verdad: 4B (30.14) >> 2B (18.33). El 2B queda como partner de ensemble DÉBIL. |

| 2026-07-16 | Ensemble 4B+2B neutro (4B campeón + 2B sobrante) | **NO enviado** | Smoke OK (4.0/4, **90 subkeys** unidos vs 36 del 4B / 57 del 2B — la unión funciona). PERO el timing revela el 4B en ~600–1250s/tarea → 60 tareas/worker × ~900s ≈ 15h > 12h → el 4B está **coverage-limited**, nunca drena su cola → el 2B nunca corre en sobrante → ensemble = 30.14% garantizado. No se gasta submission. |

**Conclusión ensemble 4B+2B: no puede superar 30.14%.** El 4B es coverage-limited (no tiene
holgura para el 2B en diseño neutro); y en diseño compartido, quitarle cobertura al 4B fuerte
por el 2B débil es ~break-even. Para climbing real hace falta un **base más fuerte a velocidad
similar** (Qwen3.5-4B `raist321/qwen3-5-4b`, misma talla → sin penalización de cobertura) o un
**partner de arquitectura distinta** (ARChitects difusión, semanas). Bigger Qwen (8B/14B) =
más lento = riesgo de cobertura (la razón por la que NVARC eligió 4B).

**Conclusión — la perilla de tiempo por tarea está agotada.** Tres puntos: 1200+TTT (injerto 2)
→ 28.06% (menos cobertura); 1200 (cheap-first) → **30.14%** [óptimo]; 600 (injerto 3) → smoke
muestra pérdida severa de calidad. Ambas direcciones desde 1200 empeoran → **cheap-first
(30.14%) es el punto de operación óptimo del presupuesto de tiempo**. Siguiente frontera:
**calidad** (selección de attempts, sin costo de tiempo). Ver regla C×q en STRATEGY.md.

| Fecha | Modelo | LoRA (datos) | Selección | TTT | n_aug | Tareas | Score | Kernel/notas |
|---|---|---|---|---|---|---|---|---|
| 2026-07-05 | Qwen2.5-0.5B | adapter viejo 0.5B | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v4 (T4×2): **inválido** — OOM en `score_completions` (logits fp32 de vocab completo) dejó todas las tareas en baseline. Fix en commit `30577ba`; relanzado como eval v5. |
| 2026-07-06 | Qwen2.5-0.5B | adapter viejo 0.5B | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v5 (T4×2): **pipeline OK** (sin OOM; 45/74 attempts fueron predicciones reales del modelo, ~10 min/tarea). El 0.5B no acierta ninguna exacta pero hay near-misses (98.9%, 90%, 89% de celdas correctas con shape OK) → el 3B debería convertir varias. |
| 2026-07-06 | Qwen2.5-3B | v10: real aug ×4 + synth 8k, 2000 steps, loss 0.94→0.08 | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v6: **inválido** — transformers ignora `torch_dtype` (deprecado) y cargó el 3B en fp32 (14.3GB → OOM total en T4). Fix en `837e716`; relanzado como v7 con fp16. |
| 2026-07-06 | Qwen2.5-3B | v10 (ídem) | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v7/v8: **inválidos** — OOM en TTT (backward sin gradient checkpointing no cabe en T4 + memoria retenida). Fix `041d225`. |
| 2026-07-06 | Qwen2.5-3B | v10 (ídem) | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v9 (**L4×4, 4 workers, bf16, 1.4h**): pipeline limpio, sin OOM. 62/74 attempts = predicciones reales (vs 45 del 0.5B); near-misses 90–95% de celdas en ~8 tareas pero ninguna exacta. Diagnóstico: el cuello es el **entrenamiento** (0.56 epochs, sintético débil), no la inferencia. |
| 2026-07-11 | Qwen2.5-3B | **v11 ckpt-3900** (loss 0.043, epoch 1.09) | score | ✔ steps=16 | 4 | 48 | 0.000 | eval v10 (L4×4, 2.1h): 115/163 attempts reales; shape_ok ~51% (antes ~40%); near-misses 95–97% en ~8 tareas; **cero exactas**. +1900 steps del mismo corpus mejoran celdas pero no convierten exactas → **retornos decrecientes: el cuello es la calidad/diversidad de datos y el método, no más steps**. Refuerza la vía LB33/NVARC como baseline. |

## Ensemble — TRM (partner diverso)

| Fecha | Versión | Score LB | Notas |
|---|---|---|---|
| 2026-07-17 | **TRM solo** (NVARC, recursivo 7M) | **8.33%** | Coincide con el ~8% del paper. Débil solo, pero **arquitectónicamente diverso** (recursivo, no LLM) → sus aciertos probablemente NO se solapan con los del Qwen (a diferencia del 2B correlacionado). **Dato clave: el run completo (epochs=4000, 240 tareas) CUPO en ≤12h** → timing viable. Pipeline validado offline. |

**Timing del TRM (eager BYOD, 48 tareas):** epochs=1000 → 30.8 min → extrapolado a 240
tareas: **~2.6h (epochs=1000), ~10.3h (epochs=4000)**. El full TTT no cabe con el Qwen;
epochs reducidas sí. Split viable: Qwen ~10h (cobertura ~85%, ~28-29%) + TRM epochs~500
(~1.5h, diversidad). EV: break-even a positivo si la diversidad del TRM supera la pérdida
de cobertura del Qwen. Palanca clave (idea del usuario): minimizar épocas/augs del TRM.

**Gate de entorno combinado: VERDE (2026-07-18).** El TRM corre en la imagen BYOD del Qwen
(convivencia en un kernel confirmada). 7 iteraciones de integración: pydantic_settings (wheels
propios `trm-extra-whls`), torch.compile incompat → **eager + strip del prefijo `_orig_mod.`
del checkpoint** (desacople de raíz), stub de numba (BYOD lo tiene roto). El conflicto
estructural de torch está resuelto; el resto fueron deps faltantes.

**Kernel combinado ensamblado (2026-07-18).** `notebooks/lb33_ensemble_trm.ipynb` (22 celdas):
Fase 1 Qwen (LB33 cheap-first, cap `QWEN_H`) → `submission_qwen.json`; **red de seguridad**
(`submission.json = Qwen` por si la fase 2 no termina); Fase 2 TRM (epochs=500) →
`submission_trm.json`; Fase 3 merge pass@2 (a1=Qwen o TRM si no cubre; a2=diversidad del TRM).
Aislamiento por subproceso: el Qwen corre vía `python starter.py` (libera GPU al salir) → el
TRM arranca con GPUs limpias. **Smoke v1: la fase Qwen funcionó en el kernel combinado**
(120 tareas del eval, 4 con attempt real); el ERROR fue el reload de `submission.json` de la
rama no-rerun (relee un fichero ya renombrado) — arreglado en v2. Deps TRM (wheels) y unsloth
**coexisten** sin romperse. Pendiente: v2 valida fases TRM+merge; luego submit A/B vs 30.14%.

**Análisis del ensemble Qwen(30.14%)+TRM(8.33%):** el techo (unión si fueran disjuntos) = 38.5%.
La pregunta es cuántas tareas resuelve el TRM que el Qwen NO (diversidad real por arquitectura).
Requiere UN kernel con ambos modelos → split de las 12h. El Qwen es coverage-limited, PERO
cheap-first hace que las tareas que perdería al recortar tiempo sean la cola cara/de bajo
rendimiento → su degradación es sub-lineal. Net: **incierto pero plausiblemente positivo**
(a diferencia del 2B claramente negativo). Es el último camino con techo real.

## Entrenamientos

| Fecha | Run | Config | Resultado |
|---|---|---|---|
| 2026-07-05/06 | train v10 (T4, ~7h) | 3B, real aug×4 + synth 8k (57k muestras), 2000 steps | loss 0.94→0.083; adaptador publicado |
| 2026-07-06/07 | train v11 (L4, ~4.9h de pared, 2× quota) | ídem, **desde cero** (el resume falló en silencio: ckpt no montado), llegó a 3900 steps (epoch 1.09) antes de que la quota matara el kernel | loss 1.00→**0.043**; checkpoint-3900 rescatado y publicado como `arc-agi2-model`. Lecciones → `REQUIRE_RESUME` + `TimeLimitCallback` en train.ipynb |

## Cómo registrar

1. Correr `notebooks/evaluate.ipynb` en Kaggle (L4×4, offline) o `scripts/evaluate.py` en kernel GPU.
2. Copiar aquí el contenido de `/kaggle/working/eval_score.json` como fila de la tabla.
3. Si el cambio mejora el mejor score conocido → correr `submit.ipynb` y anotar el
   score del leaderboard.

## Ablations pendientes (en orden de valor)

1. `selection=score` vs `selection=vote` (mismo modelo/datos).
2. ±TTT (`USE_TTT`) y sensibilidad a `TTT_STEPS` (8/16/24).
3. ±datos sintéticos en el SFT; luego ±corpus externos (RE-ARC/BARC).
4. `n_aug` (2/4/8) y `n_cands` (1/2/3).
5. Tamaño de modelo: 1.5B vs 3B vs 4B.
