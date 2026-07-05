# Estrategia — ARC Prize 2026 (ARC-AGI-2)

## Contexto y restricciones que moldean la solución

- **Sin internet en evaluación** → el modelo y sus pesos viajan dentro del notebook / como
  Kaggle Dataset adjunto. Nada de llamar APIs (eso es ARC-AGI-3, otra pista).
- **~12 h de runtime** sobre GPU de Kaggle → presupuesto de cómputo por tarea muy ajustado.
  Esto favorece **modelos pequeños (~0.5B–7B)** con adaptación rápida frente a modelos gigantes.
- **pass@2 exacto** → no hay crédito parcial; el grid debe coincidir celda a celda.
- **ARC-AGI-2 es deliberadamente anti-memorización** → soluciones que solo memorizan patrones
  del train fallan. Hace falta *adaptación por tarea* (test-time training) y razonamiento composicional.

## Qué enseñó el ARC Prize 2025 (calibración contra el SOTA real)

Resultados del Kaggle 2025 sobre el set privado de ARC-AGI-2:

| Puesto | Equipo | Score | Receta |
|---|---|---|---|
| 1º | **NVARC** (NVIDIA) | **24.03%** | Ensemble de un modelo **~4B** estilo-ARChitects con **TTT** + componentes **TRM**, alimentado por un **corpus sintético masivo y diverso** (generación por etapas, descomposición de conceptos, modelos Qwen). ~$0.20/tarea. |
| 2º | ARChitects | 16.53% | LLM de difusión enmascarada **2D-aware** con auto-refinamiento recursivo y **scoring por perspectivas** (puntuar candidatos bajo vistas augmentadas). |
| 3º | MindsAI | 12.64% | Pipeline TTT muy ingenierizado: TTFT, ensembles de augmentación, tokenizer dropout. |

Papers premiados: **TRM** (7M params, ~8% ARC-AGI-2, refinamiento recursivo de la respuesta) y
CompressARC (76K params, MDL puro por tarea). Tema transversal 2025: **refinement loops**
(generar candidatos + señal de verificación + iterar).

**Conclusiones para nosotros:**
1. Nuestra receta base (LLM pequeño + LoRA + sintético + TTT + augmentación) **es la correcta** —
   es exactamente la familia del 1º y 3º puesto. La organización dice que la brecha de accuracy
   está "bottlenecked by engineering": ganan los detalles, no un cambio de paradigma.
2. El *edge* del ganador fue la **escala y diversidad del corpus sintético** ("mover el
   razonamiento complejo offline, al pipeline de datos"), no un modelo más grande.
3. La **selección de candidatos por verosimilitud bajo vistas augmentadas** (scoring, no voto
   simple) es un componente probado de los dos primeros puestos.
4. 85% (bonus) sigue lejísimos del SOTA (24%). Objetivo realista: **top-8 del Progress
   leaderboard** (todos cobran premio) + **writeup sólido** para el Grand Prize.

## Hardware: Kaggle L4×4 (96GB, offline)

Esta competencia da acceso a **L4×4 = 96GB de VRAM**, pero con condiciones:
- **Internet OFF obligatorio** en toda sesión L4 (también entrenamiento) → modelo base y deps
  desde Kaggle Models/Datasets adjuntos; sin descargas HF ni `pip install` con red.
- Consume **quota al doble** que T4×2/P100; solo para notebooks adjuntos a esta competencia.
- No hay GPU local → todo el cómputo va en kernels de Kaggle (`scripts/push_kernels.py`),
  en **sesiones ≤12h con checkpoint-resume** (el adapter se versiona en el dataset
  `arc-agi2-model` y se retoma en la siguiente sesión).

⚠️ Estado actual: `train.ipynb` y `submit.ipynb` usan **1 sola GPU** (el intento de DDP se
revirtió por cuelgues). En inferencia esto regala un **~4× de throughput**: TTT es
independiente por tarea → la solución correcta es **4 procesos worker, uno por GPU, con
sharding de tareas** (sin DDP, sin comunicación entre GPUs). Ver Fase 4.

## Modelo

- **Iterar** con Qwen2.5-0.5B/1.5B (barato en quota, ciclos rápidos).
- **Final**: apuntar a la clase **~4B** (lo que usó NVARC). Candidatos: **Qwen3-4B**
  (thinking desactivado) vs Qwen2.5-3B/7B — decidir con una ablation en las 120 de evaluación.
- Llama 4 y MoEs grandes descartados: no caben offline ni permiten TTT por-tarea en 12h.

## Pipeline por fases (re-priorizado tras 2025)

### Fase 0 — Pipeline de entrega (sin ML)  ✅ implementado
`scripts/make_submission.py` produce `submission.json` válido (identidad + output frecuente).

### Fase 1 — Datos + Fine-tune con LoRA  ✅ base implementada → **escalar diversidad (P1)**
1. Serializar tareas a texto (`src/arc/grid.py::format_task_prompt`).
2. **Augmentar masivamente** (`src/arc/augment.py`): 8 simetrías D4 × permutaciones de color
   × leave-one-out. 1000 tareas → cientos de miles de muestras.
3. **Datos sintéticos** (`src/arc/synth.py`): 22 reglas procedurales (incl. `compose` de 2 ops).
   Escalable con `scripts/build_synth_dataset.py`.
4. **[NUEVO] Corpus externos** — el edge de NVARC fue la diversidad, y hay corpus abiertos
   que superan por mucho nuestras 22 reglas:
   - **RE-ARC** (Hodel): generadores procedurales de las 400 tareas de ARC-AGI-1 → muestras
     ilimitadas de la distribución original.
   - **BARC / ARC-Heavy** (~400k tareas generadas por LLM, HuggingFace): diversidad conceptual.
   - **ARC-AGI-1 completo** (800 tareas train+eval): datos de entrenamiento legítimos para
     ARC-AGI-2 (sets disjuntos).
   Descargar localmente → filtrar/deduplicar → subir como Kaggle Dataset (offline-friendly).
5. Entrenar LoRA (`scripts/train_lora.py`, completion-only masking); mezcla real + sintético
   + externos. Validar en `evaluation` (120 tareas) con `scripts/evaluate.py`.

### Fase 2 — Test-Time Training (TTT)  ✅ implementado → **activarlo (P0)**
`src/arc/ttt.py`: fine-tune ligero del LoRA por tarea sobre sus pares `train` augmentados
(leave-one-out), con snapshot/restore entre tareas. Es la técnica común a los 3 ganadores 2025.
⚠️ `submit.ipynb` tiene `ENABLE_TTT=False` — encenderlo por defecto una vez medido su costo
por tarea, y calibrar `steps`/`n_aug` contra el presupuesto de 12h.

### Fase 3 — Inferencia: de "voto simple" a **generación multi-candidato + scoring por verosimilitud (P0)**
Hoy `src/arc/infer.py` genera 1 predicción greedy por vista y vota con un `Counter`. Los
ganadores hacen algo más fuerte, en dos pasos:
1. **Generar candidatos diversos**: por vista augmentada, extraer varios candidatos — sampling
   con temperatura baja o **DFS con umbral de probabilidad acumulada** (técnica ARChitects:
   recorre el árbol de tokens y devuelve *todas* las salidas cuya probabilidad supere el umbral,
   más barato que muestrear N veces).
2. **Puntuar, no contar**: para cada candidato único (des-augmentado), calcular su
   **log-verosimilitud promedio bajo K vistas augmentadas** (un forward pass por vista, sin
   generación). `attempt_1`/`attempt_2` = los 2 candidatos distintos con mayor score.
   El voto por mayoría queda como desempate.
Esto convierte pass@2 en lo que realmente es: un problema de *ranking* de candidatos.

### Fase 4 — Ingeniería de runtime (P0; "bottlenecked by engineering")
1. **4 workers × 1 GPU**: proceso independiente por GPU con shard de tareas (TTT + inferencia
   son por-tarea). Sin DDP → sin cuelgues. ~4× más steps de TTT y augmentaciones por tarea.
2. **Presupuesto por tarea**: estimar costo (nº celdas/tokens), ordenar tareas de barata a cara,
   y degradar con elegancia (menos `n_aug`, menos steps TTT, greedy puro) si el reloj aprieta.
3. **Escritura incremental** de `submission.json` tras cada tarea + fallback baseline por tarea
   (ya existe el fallback global; hacerlo granular).

### Fase 5 — Ensemble (stretch, P2)
NVARC ensambló su LLM-TTT con componentes **TRM** (7M params, open source, barato en
inferencia). Opción: entrenar un TRM en sesiones Kaggle y usarlo donde el LLM tiene baja
confianza (score bajo del top-candidato). Solo abordar si las Fases 1–4 están saturadas.

## Disciplina experimental (hoy no hay ningún número medido)

- **Registrar resultados en `docs/RESULTS.md`** (commiteado): fecha, modelo, dataset, flags,
  score en las 120 de evaluación. Sin esto no se puede priorizar nada.
- Nunca entrenar con el split `evaluation`.
- Ablations mínimas (en este orden de valor): scoring-vs-voto, ±TTT, ±sintético/±externos,
  n_aug, tamaño de modelo (1.5B vs 3B vs 4B).

## Hitos (deadline estimado ~nov 2026 — confirmar en las reglas)

| Mes | Objetivo |
|---|---|
| Jul | Fase 3 (scoring) + TTT activo + primer score real medido en eval y en el leaderboard |
| Ago | Fase 4 (4×GPU + presupuesto de tiempo) + corpus RE-ARC/BARC integrados |
| Sep–Oct | Escalar a ~4B (Qwen3-4B vs Qwen2.5) + ablations + iterar sintético propio |
| Nov | Writeup para el Grand Prize (6 criterios) + submissions finales |

## Backlog priorizado

**P0 — impacto directo en score, bajo riesgo:**
- [x] `src/arc/infer.py` + `src/arc/model.py`: generación multi-candidato (greedy + sampling)
      y **scoring por log-verosimilitud augmentada** (`predict_task_scored`); voto como
      desempate. Pendiente: DFS con umbral de probabilidad como generador de candidatos.
- [x] `submit.ipynb`: `ENABLE_TTT=True`, presupuesto de tiempo con degradación por tiers,
      escritura incremental de `submission.json`, fallback granular (via `arc/runner.py`).
- [x] Runner multi-proceso 4×L4 (`src/arc/runner.py` + `scripts/infer_worker.py`;
      usado por `submit.ipynb` y `evaluate.ipynb`).
- [ ] `docs/RESULTS.md`: medir y commitear el score actual en las 120 de evaluación (baseline
      de todo lo demás) y hacer la primera submission real al leaderboard.

**P1 — datos y modelo:**
- [ ] Integrar RE-ARC + BARC/ARC-Heavy + ARC-AGI-1 como Kaggle Datasets y mezclarlos en el SFT.
- [ ] Checkpoint-resume de entrenamiento entre sesiones Kaggle (quota L4 = 2×).
- [ ] Ablation Qwen3-4B vs Qwen2.5-3B/7B; elegir el modelo de las corridas finales.
- [ ] Más reglas sintéticas: composición de 3+ ops, simetría 4-fold, patrones periódicos,
      conteo/aritmética de objetos (debilidades típicas de ARC-AGI-2).

**P2 — stretch:**
- [ ] Ensemble con TRM para tareas de baja confianza.
- [ ] Trucos MindsAI: tokenizer dropout, TTFT.
- [ ] Ablations completas y material del writeup.

**Hecho:**
- [x] Pipeline de entrega, dataset augmentado, generador sintético (22 reglas), LoRA SFT,
      AIRV, TTT con snapshot/restore, kernels Kaggle offline, evaluación pass@2.
