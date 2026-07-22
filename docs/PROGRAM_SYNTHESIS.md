# Síntesis de programas verificable (apuesta híbrida)

Tras agotar el paradigma de transducción a ~30% (ver `RESULTS.md`), el usuario eligió la vía
**síntesis de programas verificable**: un modelo genera **programas Python** que resuelven la
tarea; los ejecutamos en los train pairs y **nos quedamos solo con los que reproducen la salida
exacta** (verificable → alta precisión); los aplicamos al test. Como es verificable, sus aciertos
son fiables → híbrido que AÑADE cobertura sin sacrificar la transducción (a diferencia del
ensemble con modelos débiles, que regresó).

Por qué founded: el survey 2026 (ver [[arc-sota-map]]) sitúa la síntesis de programas en 26-79%
y el híbrido (inducción+transducción) como dominante. La transducción pura (lo nuestro) capa ~40%.

## Recursos (todos públicos, offline-compatibles)

- **Modelo de inducción:** `ironbar/barc0llama-3.1-arc-potpourri-induction-8b` (Kaggle Models) =
  copia de `barc0/Llama-3.1-ARC-Potpourri-Induction-8B` (HF). Llama-3.1-8B SFT'd para emitir
  programas ARC. Hay también transducción MARC: `ekinakyurek/Llama-3.1-ARC-Potpourri-Transduction-8B`.
- **Librería común:** `vendor/barc/common.py` (1551 líneas, solo `numpy`+`random`, self-contained).
  Color enum + helpers (flood_fill, crop, blit, find_connected_components, symmetry...). Los
  programas generados la importan (`from common import *`). Vendorizada desde `xu3kev/BARC@master:seeds/common.py`.
- **Referencia (Modal, NO offline):** `xu3kev/ensemble-induction-and-transduction` — su lógica de
  inducción está en el gist `modal_induction.py` (receta abajo) y su ensemble en la celda 17.

## Receta de inducción (de `modal_induction.py`)

**Color map:** `{0:Black,1:Blue,2:Red,3:Green,4:Yellow,5:Gray,6:Pink,7:Orange,8:Purple,9:Brown}`.
`grid_to_input(grid)` = filas unidas por `\n`, celdas (nombres de color) por espacios, `+ "\n"`.

**Prompt (template Llama-3.1):**
- system: `"...You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."`
- user: instrucciones + `Here are the input and output grids for the reference examples:\n{ejemplos}` donde cada ejemplo = `"Example {i+1}\nInput:\n{x}\nOutput:\n{y}"`, luego `Here is the input grid for the test example:\n{input}\n` y `Write a Python function \`transform\` that can convert any given input grid to its corresponding output grid...`
- termina en `<|start_header_id|>assistant<|end_header_id|>`.

**Sampling:** `temperature` (p.ej. 0.8), `max_tokens=1536`, `n` = muestras/tarea (batch).

**Extracción de código:** regex `` ```python(.*?)``` `` (fallback `` ```(.*?)``` ``), `re.DOTALL`.

**Ejecución/verificación:** anteponer `common.py`, `exec` el código, llamar `transform(input_grid)`
(numpy) con timeout (~1s, en subproceso/worker). Comparar con la salida esperada por
`np.array_equal`. Un programa es **train-correct** si pasa TODOS los train pairs.

**Selección:** de los programas train-correct, aplicar al test input → candidatos. attempt_1/2 =
los outputs de test más frecuentes entre los programas train-correct (voto).

**Ensemble con transducción (celda 17 de la referencia):** por tarea, juntar candidatos de
inducción (attempt_1/2) + transducción (attempt_1/2), dedup, tomar top-2. La inducción va PRIMERO
(sus aciertos son verificados → más fiables).

## FASE 0 — probe de viabilidad (antes de construir el sistema completo)

**Pregunta:** ¿el BARC-8B offline genera programas que pasan los train pairs en ARC-AGI-2, y
generalizan al test? (Caveat: BARC es de 2024/AGI-1; el survey nota "cliff" en AGI-2.)

**Diseño:** kernel GPU offline, N tareas del set `evaluation` (tenemos soluciones). Por tarea:
prompt BARC, muestrear K programas (p.ej. K=32, temp 0.8), extraer, ejecutar+verificar en train.
Medir: (a) **verifiable-hit rate** = % tareas con ≥1 programa que pasa TODOS los train pairs;
(b) **generalización** = de esas, % donde el output de test es correcto (pass@2 por voto).

**Umbral de decisión:** si verifiable-hit + generalización rinde ≳15-20% de tareas → construir el
sistema completo + merge híbrido con la transducción NVARC (techo real por encima de ~30%). Si
~0 (modelo demasiado débil en AGI-2) → pivotar.

**Nota de validación:** el set `evaluation` está CONTAMINADO para el modelo NVARC (transducción),
pero **NO para BARC** (BARC no lo entrenó) → el eval SÍ es válido para medir la inducción BARC.
