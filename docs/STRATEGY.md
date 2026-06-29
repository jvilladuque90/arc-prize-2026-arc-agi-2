# Estrategia — ARC Prize 2026 (ARC-AGI-2)

## Contexto y restricciones que moldean la solución

- **Sin internet en evaluación** → el modelo y sus pesos viajan dentro del notebook / como
  Kaggle Dataset adjunto. Nada de llamar APIs (eso es ARC-AGI-3, otra pista).
- **~12 h de runtime** sobre GPU de Kaggle → presupuesto de cómputo por tarea muy ajustado.
  Esto favorece **modelos pequeños** (0.5B–7B) con adaptación rápida frente a modelos gigantes.
- **pass@2 exacto** → no hay crédito parcial; el grid debe coincidir celda a celda.
- **ARC-AGI-2 es deliberadamente anti-memorización** → soluciones que solo memorizan patrones
  del train fallan. Hace falta *adaptación por tarea* (test-time training) y razonamiento composicional.

## Hardware: Kaggle L4×4 (96GB, offline)

Esta competencia da acceso a **L4×4 = 96GB de VRAM**, pero con condiciones:
- **Internet OFF obligatorio** en toda sesión L4 (también entrenamiento) → modelo base y deps
  desde Kaggle Models/Datasets adjuntos; sin descargas HF ni `pip install` con red.
- Consume **quota al doble** que T4×2/P100; solo para notebooks adjuntos a esta competencia.

No hay GPU local → todo el cómputo va en kernels de Kaggle (`scripts/push_kernels.py`).

## Por qué Qwen2.5 y no Llama 4

| Criterio | Qwen2.5 (hasta 7B–14B en 96GB) | Llama 4 (MoE) |
|---|---|---|
| Cabe offline en L4×4 | ✅ | ❌ (cientos de GB) |
| TTT por-tarea en 12 h | ✅ | ❌ inviable |
| Antecedente ganador ARC | ✅ (familia Qwen usada por top teams 2024) | sin antecedente |

Con 96GB el sweet spot sube: **iterar con Qwen2.5-0.5B/1.5B** (rápido y barato en quota) y
**escalar a 7B–14B con LoRA** para las corridas finales, manteniendo TTT viable.

## Pipeline propuesto

### Fase 0 — Pipeline de entrega (sin ML)  ✅ implementado
`scripts/make_submission.py` produce `submission.json` válido (identidad + output frecuente).
Objetivo: confirmar formato, IDs, y que el notebook corre end-to-end.

### Fase 1 — Datos sintéticos + Fine-tune con LoRA (PEFT)  ✅ implementado
1. Serializar tareas a texto (`src/arc/grid.py::format_task_prompt`).
2. **Augmentar masivamente** (`src/arc/augment.py`): 8 simetrías D4 × N permutaciones de color
   × reordenamiento de ejemplos. 1000 tareas → cientos de miles de muestras.
3. **Datos sintéticos** (`src/arc/synth.py`): generador de tareas ARC-style con regla consistente
   por tarea — 13 reglas (recolor, rotate, flip, transpose, tile, mirror, upscale, border,
   gravity, crop_bbox, most_frequent, symmetrize). Fue el *edge* del ganador 2025. Escalable a
   decenas de miles de tareas (`scripts/build_synth_dataset.py`).
4. Entrenar LoRA sobre Qwen2.5 (`scripts/train_lora.py`, SFT con enmascarado completion-only)
   prediciendo el grid de salida token a token. Mezcla real (augmentado) + sintético.
5. Validar en `evaluation` (120 tareas) con `scripts/evaluate.py`.

### Fase 2 — Test-Time Training (TTT)  ✅ implementado
`src/arc/ttt.py`. En inferencia, por cada tarea de test:
1. Tomar sus pares `train`, augmentarlos (D4+color, leave-one-out) → `task_ttt_samples`.
2. Fine-tune ligero del LoRA sobre esos pares (`TTT.adapt`, pocos pasos), restaurando un
   **snapshot** del adaptador entre tareas (sin fuga de info de una tarea a otra).
3. Predecir con AIRV (`predict_with_ttt`) y restaurar.
Es la técnica que separó a los ganadores del resto. Activable en `submit.ipynb` (`ENABLE_TTT`)
y en `scripts/evaluate.py --ttt`.

### Fase 3 — Inferencia con augmentación + votación (AIRV)
1. Para cada test input, generar predicciones sobre varias vistas augmentadas (D4 + color).
2. Des-augmentar cada predicción al espacio original (las transformaciones son invertibles).
3. Votar; elegir `attempt_1` y `attempt_2` como los 2 candidatos más consensuados.

## Métricas objetivo (realistas)

- 85% es el premio bonus — extremadamente difícil en ARC-AGI-2 (SOTA público 2025 era muy bajo).
- Meta táctica: subir en el **Progress leaderboard** y producir un **writeup** sólido (Grand Prize).

## Backlog técnico

- [x] `scripts/build_finetune_dataset.py` — dataset augmentado (D4+color+LOO) a JSONL.
- [x] `src/arc/synth.py` + `scripts/build_synth_dataset.py` — generador de tareas sintéticas.
- [x] `src/arc/model.py` — carga de Qwen + LoRA, generación con restricción a dígitos/grid.
- [x] `scripts/train_lora.py` — SFT con LoRA (enmascarado completion-only).
- [x] `src/arc/infer.py` — AIRV (augment → predict → des-augment → voto).
- [x] `scripts/evaluate.py` — validación en `evaluation` con pass@2.
- [x] `notebooks/{train,submit}.ipynb` + `scripts/push_kernels.py` — kernels Kaggle (L4×4 offline).
- [x] `src/arc/ttt.py` — TTT real por tarea (snapshot/restore del LoRA) + `evaluate.py --ttt`.
- [ ] Más reglas sintéticas (composición de 3+ ops, simetría 4-fold, patrones periódicos).
- [ ] Ablations: con/sin sintético, con/sin TTT, tamaño de modelo, nº augmentaciones.
