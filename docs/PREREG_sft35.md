# PREREG `sft35` — ¿convierte la generación 3.5 el SFT de NVARC en habilidad de rejilla? (31-oct-2026)

## Por qué aquí

`gate35b` cerró la pregunta del **prior** (Base vs Base, kernels FLA, control de texto): Qwen3.5-4B-Base
empata a Qwen3-4B-Base (ratio 1,044, gana 12/35). El usuario pidió seguir ("adelante") por la única vía que
queda en el eje del modelo base: el **SFT** completo. Antes de pagar 10–20 h-cuota por un SFT de 266K
secuencias y portar el pipeline (unsloth no carga Gated DeltaNet), se mide con un SFT corto y PAREADO si
la 3.5 aprende la tarea de rejilla **mejor** que la 3 a datos y receta iguales. Si no lo hace en 9K
secuencias, no hay mecanismo por el que lo haga en 266K.

## Diseño

- 4 GPUs = 4 corridas independientes: `q3_s1`, `q35_s1`, `q3_s2`, `q35_s2` (brazo × semilla).
- Datos: muestra única de puzzles NVARC (`sft35-data`, ~40 por grupo-semilla, seed 0). Misma lista de
  secuencias por semilla en ambos brazos (D4 × permutación de 10 colores, corte frontal a ≤2048 tokens,
  última pareja como test, loss solo sobre salidas + `<|im_end|>`, exactamente el formato de producción).
- Receta idéntica: vocabulario recortado a 16 filas (ids de cada tokenizador para 0-9, `\n`, user, assistant,
  endoftext, im_start, im_end), embeddings y cabeza completas entrenables, LoRA r=64 α=32 rsLoRA sobre
  **todas** las lineales del decodificador, AdamW lr 1e-4 coseno (3 % warmup), accum 8, 1 época, tope 3 h.
- Instrumento (población de `gate35b`: 24 baratas + 12 medias del eval público, no vista por ningún brazo):
  NLL/celda de la GT (vista canónica y mediana de 3 vistas fijas) y exactas greedy, **pre y post** SFT;
  curva de loss por tokens; tok/s y memoria.
- Banda de ruido: `q3_s1` vs `q3_s2` y `q35_s1` vs `q35_s2` (mismo modelo, distinto orden/aumentación).

## Predicciones pre-registradas

- **Reabre la vía 3.5**: ratio post-SFT q35/q3 de NLL/celda ≤ 0,85 en AMBAS semillas, gana ≥ 2/3 de tareas
  (test de signo p < 0,05) y la banda semilla-a-semilla es más estrecha que el efecto. Entonces: SFT largo
  (≥50K seq) + port del TTT a transformers+peft (fuera del alcance del 2-nov).
- **Empate** (0,85 < ratio < 1,15 o signo no significativo): vía cerrada definitivamente; el eje del modelo
  base queda agotado con dos mediciones limpias (prior y aprendizaje).
- **Peor** (≥1,15): idem, y confirma que la arquitectura híbrida no compra nada en rejilla a esta talla.
- Se reporta pero no decide: tok/s (la 3.5 con FLA puede ser más lenta; el reloj es el otro cuello).

## Coste

~3,7 h de pared en L4×4 ≈ **7,5 h-cuota** (tope de entrenamiento 3 h + carga + 2 evaluaciones).
Ledger semanal antes de lanzar: ~1,5 h-cuota gastadas desde el reset. `sft35-data` es CPU (0 cuota).
