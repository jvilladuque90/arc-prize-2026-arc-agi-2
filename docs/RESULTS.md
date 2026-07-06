# Resultados medidos

Registro obligatorio de cada corrida de evaluación (disciplina experimental: sin número
medido no se prioriza nada). El score es **pass@2** sobre el split `evaluation`
(120 tareas públicas de ARC-AGI-2), salvo que se indique subconjunto.

Referencia externa (set privado, Kaggle 2025): NVARC 24.03% · ARChitects 16.53% ·
MindsAI 12.64%. La baseline sin ML ronda ~0%.

| Fecha | Modelo | LoRA (datos) | Selección | TTT | n_aug | Tareas | Score | Kernel/notas |
|---|---|---|---|---|---|---|---|---|
| 2026-07-05 | Qwen2.5-0.5B | adapter viejo 0.5B | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v4 (T4×2): **inválido** — OOM en `score_completions` (logits fp32 de vocab completo) dejó todas las tareas en baseline. Fix en commit `30577ba`; relanzado como eval v5. |
| 2026-07-06 | Qwen2.5-0.5B | adapter viejo 0.5B | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v5 (T4×2): **pipeline OK** (sin OOM; 45/74 attempts fueron predicciones reales del modelo, ~10 min/tarea). El 0.5B no acierta ninguna exacta pero hay near-misses (98.9%, 90%, 89% de celdas correctas con shape OK) → el 3B debería convertir varias. |
| 2026-07-06 | Qwen2.5-3B | v10: real aug ×4 + synth 8k, 2000 steps, loss 0.94→0.08 | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v6: **inválido** — transformers ignora `torch_dtype` (deprecado) y cargó el 3B en fp32 (14.3GB → OOM total en T4). Fix en `837e716`; relanzado como v7 con fp16. |
| 2026-07-06 | Qwen2.5-3B | v10 (ídem) | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v7/v8: **inválidos** — OOM en TTT (backward sin gradient checkpointing no cabe en T4 + memoria retenida). Fix `041d225`. |
| 2026-07-06 | Qwen2.5-3B | v10 (ídem) | score | ✔ steps=16 | 4 | 24 | 0.000 | eval v9 (**L4×4, 4 workers, bf16, 1.4h**): pipeline limpio, sin OOM. 62/74 attempts = predicciones reales (vs 45 del 0.5B); near-misses 90–95% de celdas en ~8 tareas pero ninguna exacta. Diagnóstico: el cuello es el **entrenamiento** (0.56 epochs, sintético débil), no la inferencia. |

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
