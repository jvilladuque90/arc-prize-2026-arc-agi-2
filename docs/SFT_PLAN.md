# Paso 3 — SFT de la inducción en AGI-2 (plan)

Objetivo: subir el verifiable-hit rate de la inducción en ARC-AGI-2 de ~6% (BARC congelado,
medido en FASE 0) a un nivel competitivo, vía **data sintética + SFT** — la apuesta diferenciada
(pocos hacen inducción bien en AGI-2). Ver [[arc-program-synthesis]], `working_note_es.md`.

## Lo que reutilizamos de BARC (pipeline completo, repo `xu3kev/BARC@master`)

**Data-gen de BARC (cómo hicieron 100K+ ejemplos):**
1. `seeds/*.py` (~100+): cada seed es un **generador ejecutable** de una familia de tareas —
   `# concepts:` + `# description:` + `def main(input_grid)` (solver, usa `common.py`) +
   `def generate_input()` (genera inputs aleatorios). Ejecutar `generate_input`→`main` produce
   pares (input, output) reales.
2. `generate_descriptions.py` (GPT-4): remix de conceptos → 100K descripciones NL.
3. `generate_code.py` (GPT-4o-mini, RAG de seeds): descripción → código (`main`+`generate_input`).
4. `generate_problems.py`: **ejecuta** el código → grids reales, valida.
5. `make_dataset.py` + `finetune/alignment-handbook/` (config `config_fft_llama3_induction_all_data_mixed.yaml`):
   ensambla (task grids → programa) y hace el SFT.

**Data ya publicada:** `seungpilleeee/barc-arc-potpourri-ds` (Kaggle) = data de entrenamiento de
BARC. Re-SFT sobre ella NO aporta (el modelo ya la vio). Necesitamos data NUEVA y MÁS DURA (AGI-2).

## El reto y la vía elegida (offline, sin API)

BARC generó su data con GPT-4 (caro, y AGI-1). Para AGI-2 necesitamos tareas **más compositivas**
(profundidad 1.3→2.7 pasos). Vía offline diferenciada: **componer seeds programáticamente** —
encadenar la transformación de la seed A con la de la seed B (o aplicar B a la salida de A) para
sintetizar tareas de 2 pasos, con su **programa compuesto** como target del SFT. Sin API: solo
ejecutar programas compuestos y validar (consistencia en train pairs, no-trivialidad, no-degenerado).

## Fases (de barata/de-risk a cara)

- **A. Infra de SFT (de-risk primero).** Adjuntar `barc-arc-potpourri-ds` + el modelo BARC (copia
  propia). Reproducir el formato de training example (task grids → programa) de `make_dataset.py`.
  Montar un SFT LoRA del BARC-8B en Kaggle GPU (offline; unsloth/PEFT como el TTT del Qwen).
  **Validar con un slice pequeño** que el pipeline entrena y el checkpoint carga — ANTES de generar
  data. Métrica: loss baja + el checkpoint corre en el probe.
- **B. Data-gen compositiva (el trabajo largo).** Vendorizar los seeds. Escribir el compositor:
  muestrear pares de seeds compatibles (shapes/colores), componer (A luego B), ejecutar
  `generate_input`→`main_A`→`main_B`, validar (train-consistente, no-trivial). Generar N (10K→50K→…)
  ejemplos (task, programa_compuesto). Empezar chico, medir calidad.
- **C. SFT + eval iterativo.** SFT (continuar desde BARC) sobre la data compositiva. Re-correr el
  probe (verifiable-hit rate en AGI-2). Iterar data/SFT hasta que el hit rate suba de ~6%.
- **D. Integración.** Cuando la inducción SFT'd sea fuerte, medir solapamiento con el Qwen (una
  submission) y construir el híbrido inducción+transducción (+TTT) — la receta ganadora completa.

## Riesgos / disciplina

- Es nuestra área más débil (train v10/v11 → "cero exactas" con 8K). Mitigar: partir de un
  checkpoint ya-SFT'd (BARC), añadir data, no de cero; medir exactas por iteración; no encadenar
  data-gen sin señal.
- GPU: planificar alrededor del reset del viernes 19:00 y la regla de 2h de holgura para
  submissions. El SFT es Save&Run (gasta quota); los probes también. Presupuestar por fase.
- Calidad de la composición: no toda composición da tarea válida/sensata. Validar duro; empezar con
  composiciones de 2 pasos de seeds compatibles.

## Estado

- **Fase A — COMPLETA (2026-07-24).** Infra de SFT LoRA del BARC-8B validada offline (loss
  1.65→1.08 en 40 steps sobre 4338 ejemplos). Config que FUNCIONA (reusar en Fase C):
  `notebooks/barc_sft.ipynb` + kernel `barc-sft`.
  - **Imagen ESTANDAR** (docker="") — captura logs; la BYOD los vacía. transformers+peft (sin
    unsloth/trl → portable).
  - **Sin bitsandbytes** (no está en las imágenes): NO se puede 4-bit/QLoRA ni adamw_8bit. Solución:
    **`device_map="auto"`** reparte el 8B bf16 en las 4 L4s (88GB) → sin OOM, sin necesidad de
    gradient-checkpointing. `attn_implementation="sdpa"`.
  - **`DevTrainer`** (subclase de `Trainer`) con `compute_loss` que mueve las `labels` al device de
    los `logits` → arregla el mismatch cuda:0/cuda:3 del pipeline paralelo. `optim="adamw_torch"`.
  - Formato: chat `messages` [system,user,assistant] → `tok.apply_chat_template`; masking manual del
    prompt (labels=-100 hasta el header del assistant). LoRA r=16 en q/k/v/o/gate/up/down.
  - Data slice: `barc-data-dl` baja N de `barc0/induction_heavy_100k_jsonl` (HF, abierto) a un kernel
    output, adjuntado como kernel_source.
- **Fase B — EN CURSO.** Compositor implementado en `src/arc/seedgen.py` (333 líneas) + 163 seeds
  vendorizados en `vendor/barc/seeds/`. Validado LOCALMENTE (sin GPU):
  - **138-142 de 163 seeds ejecutan** y generan pares válidos (el resto crashea: ValueError,
    IndexError… se descartan).
  - Componer A→B: `input ← genA()`, `x ← mainA(input)`, `output ← mainB(x)`. El programa objetivo
    concatena ambos cuerpos (renombrados `_step_a`/`_step_b`) + un `transform` que los encadena.
  - **~36% de pares aleatorios de seeds dan tarea útil**, y **100% de las generadas VERIFICAN** (el
    programa compuesto reproduce todos los pares) → **data correcta por construcción**, sin LLM.
  - Filtros de calidad: no-identidad, output no-degenerado (≥2 colores), y **ambos pasos deben
    aportar** en la mayoría de pares (si no, es tarea de 1 paso disfrazada).
  - Diversidad: 53 conceptos distintos en 29 tareas de muestra.
  - Formato de salida = chat `messages` de BARC (system/user/assistant con ```python) → entra
    directo al SFT de la Fase A.
  - Ejecución a escala: kernel `barc-datagen` (**CPU + internet, NO gasta quota GPU**), objetivo
    12K tareas verificadas → jsonl en el output, que se adjunta al SFT de la Fase C.
- **Siguiente: Fase C** (SFT sobre la data compositiva + re-probe del hit rate).
