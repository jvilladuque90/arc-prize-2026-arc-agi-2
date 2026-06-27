# Estrategia — ARC Prize 2026 (ARC-AGI-2)

## Contexto y restricciones que moldean la solución

- **Sin internet en evaluación** → el modelo y sus pesos viajan dentro del notebook / como
  Kaggle Dataset adjunto. Nada de llamar APIs (eso es ARC-AGI-3, otra pista).
- **~12 h de runtime** sobre GPU de Kaggle → presupuesto de cómputo por tarea muy ajustado.
  Esto favorece **modelos pequeños** (0.5B–7B) con adaptación rápida frente a modelos gigantes.
- **pass@2 exacto** → no hay crédito parcial; el grid debe coincidir celda a celda.
- **ARC-AGI-2 es deliberadamente anti-memorización** → soluciones que solo memorizan patrones
  del train fallan. Hace falta *adaptación por tarea* (test-time training) y razonamiento composicional.

## Por qué Qwen2.5 pequeño y no Llama 4

| Criterio | Qwen2.5 0.5B–7B | Llama 4 (MoE) |
|---|---|---|
| Cabe offline en notebook | ✅ | ❌ (decenas/cientos de GB) |
| TTT por-tarea en 12 h | ✅ muchas iteraciones | ❌ inviable |
| Antecedente ganador ARC | ✅ (familia Qwen usada por top teams 2024) | sin antecedente |

Recomendación: empezar con **Qwen2.5-0.5B-Instruct** para iterar barato; escalar a 1.5B/3B/7B
según presupuesto de tiempo de inferencia.

## Pipeline propuesto

### Fase 0 — Pipeline de entrega (sin ML)  ✅ implementado
`scripts/make_submission.py` produce `submission.json` válido (identidad + output frecuente).
Objetivo: confirmar formato, IDs, y que el notebook corre end-to-end.

### Fase 1 — Fine-tune con LoRA (PEFT)
1. Serializar tareas a texto (`src/arc/grid.py::format_task_prompt`).
2. **Augmentar masivamente** (`src/arc/augment.py`): 8 simetrías D4 × N permutaciones de color
   × reordenamiento de ejemplos. 1000 tareas → cientos de miles de muestras.
3. Entrenar LoRA sobre Qwen2.5 con `trl` (SFT) prediciendo el grid de salida token a token.
4. Validar en `evaluation` (120 tareas) con `src/arc/metrics.py`.

### Fase 2 — Test-Time Training (TTT)
En inferencia, por cada tarea de test:
1. Tomar sus pares `train`, augmentarlos.
2. Fine-tune ligero de un LoRA *específico de la tarea* (pocos pasos) sobre esos pares.
3. Predecir el/los test outputs.
Es la técnica que separó a los ganadores 2024 del resto.

### Fase 3 — Inferencia con augmentación + votación (AIRV)
1. Para cada test input, generar predicciones sobre varias vistas augmentadas (D4 + color).
2. Des-augmentar cada predicción al espacio original (las transformaciones son invertibles).
3. Votar; elegir `attempt_1` y `attempt_2` como los 2 candidatos más consensuados.

## Métricas objetivo (realistas)

- 85% es el premio bonus — extremadamente difícil en ARC-AGI-2 (SOTA público 2025 era muy bajo).
- Meta táctica: subir en el **Progress leaderboard** y producir un **writeup** sólido (Grand Prize).

## Backlog técnico

- [ ] `scripts/build_finetune_dataset.py` — materializar dataset augmentado a JSONL.
- [ ] `src/arc/model.py` — carga de Qwen + LoRA, generación con restricción a dígitos/grid.
- [ ] `src/arc/ttt.py` — bucle de test-time training por tarea.
- [ ] `src/arc/infer.py` — AIRV (augment → predict → des-augment → voto).
- [ ] `notebooks/submit.ipynb` — notebook de entrega que orquesta todo offline.
- [ ] Validación en `evaluation` y ablations (con/sin TTT, tamaño de modelo, nº augmentaciones).
