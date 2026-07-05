# Resultados medidos

Registro obligatorio de cada corrida de evaluación (disciplina experimental: sin número
medido no se prioriza nada). El score es **pass@2** sobre el split `evaluation`
(120 tareas públicas de ARC-AGI-2), salvo que se indique subconjunto.

Referencia externa (set privado, Kaggle 2025): NVARC 24.03% · ARChitects 16.53% ·
MindsAI 12.64%. La baseline sin ML ronda ~0%.

| Fecha | Modelo | LoRA (datos) | Selección | TTT | n_aug | Tareas | Score | Kernel/notas |
|---|---|---|---|---|---|---|---|---|
| _pendiente_ | Qwen2.5-3B | real aug + synth 8k | score | ✔ steps=16 | 4 | 48 | — | primera medición del pipeline nuevo |

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
