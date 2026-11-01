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

## RESULTADO (1-nov-2026, kernels v2 + v3, ~18 h-cuota en total)

| brazo | pre-SFT med / media | post-SFT med / media | loss @600 pasos (tokens iguales) | tok/s | exactas |
|---|---|---|---|---|---|
| q3_s1 (Qwen3-4B-Base) | 0,275 / 0,389 | **0,176 / 0,237** | 0,206 | 849 | 0/36 → 0/36 |
| q3_s2 | 0,275 / 0,389 | **0,177 / 0,237** | 0,216 | 852 | 0 → 0 |
| q35_s1 (Qwen3.5-4B-Base) | 0,262 / 0,347 | 0,177 / 0,249 | 0,209 | 772 | 0 → 0 |
| q35_s2 | 0,262 / 0,347 | 0,179 / 0,246 | 0,217 | 786 | 0 → 0 |
| q35_s3 | 0,262 / 0,347 | 0,160 / 0,241 | 0,223 | 784 | 0 → 0 |
| q35_s4 | 0,262 / 0,347 | 0,170 / 0,242 | 0,212 | 780 | 0 → 0 |

NLL/celda de la GT sobre las 36 tareas (24 baratas + 12 medias), vista canónica. Ratios post q35/q3 por
mediana: 1,00 · 1,01 · 0,91 · 0,97; por media: 1,05 · 1,04 · 1,02 · 1,02. Banda semilla-a-semilla del q3:
ratio 0,996 (12/24 baratas). Loss de entrenamiento a tokens iguales: indistinguible (0,21–0,22 en ambos).
Prior pre-SFT: q35 0,95× (mediana) / 0,89× (media) — coherente con el empate de `gate35b`.

**Veredicto: EMPATE ⇒ la vía 3.5 queda cerrada definitivamente.** La prerreg exigía ratio ≤ 0,85 en TODAS
las semillas; ninguna lo cumple y las medias van 2–5 % en contra. La generación 3.5 ni parte mejor (gate35b) ni
aprende la rejilla más rápido a datos y receta iguales. Además es un 8 % más lenta (780 vs 850 tok/s con FLA
y checkpointing en ambos) y exigiría portar el TTT fuera de unsloth.

Limitación honesta: el test de signo por tarea NO se pudo calcular. Los 4 brazos q35 murieron al guardar
(`get_peft_model_state_dict` lee `config.vocab_size`, que `Qwen3_5Config` no expone en la raíz) DESPUÉS de
imprimir los resúmenes y ANTES de escribir `res.json`. El veredicto se apoya en medianas y medias, que están
lejos del umbral en las 4 semillas; no se relanza (≈9 h-cuota) para recuperar un test que no puede cambiarlo.

Lecciones de fontanería pagadas: (1) peft 0.20 aborta la inyección de LoRA con torchao 0.10 (desinstalar);
(2) el OOM de los kernels FLA en el backward es `RuntimeError` de Triton, no `torch.cuda.OutOfMemoryError`;
(3) escribir el JSON de resultados ANTES de cualquier paso de guardado de pesos; (4) la evaluación greedy
de 36 tareas cuesta ~25–30 min por brazo: presupuestarla.

## RESULTADO v4 (1-nov-2026, relanzado a petición del usuario, 8,5 h-cuota): pareado por tarea recuperado

`res.json` se escribió antes del guardado de pesos (que volvió a fallar por `vocab_size`; irrelevante). Ratios
q35/q3 de NLL/celda post-SFT, 36 tareas, vista canónica (test de signo, nº de tareas que gana el 3.5):

| par | ratio mediano | gana 3.5 | p signo |
|---|---|---|---|
| q3_s1 vs q35_s1 | 0,997 | 19/36 | 0,43 |
| q3_s2 vs q35_s2 | 1,031 | 12/36 | 0,99 |
| q3_s1 vs q35_s3 | 1,008 | 17/36 | 0,69 |
| q3_s2 vs q35_s4 | 0,984 | 22/36 | 0,12 |
| mediana de 3 vistas (s1 / s2) | 1,032 / 1,030 | 12 / 14 | 0,99 / 0,93 |
| banda q3_s1 vs q3_s2 | 1,010 | 15/36 | 0,88 |
| banda q35 s1-s2 / s3-s4 | 1,007 / 0,997 | 17 / 21 | 0,69 / 0,20 |

Pre-SFT (enlace con gate35b): 0,945, gana 28/36, p=0,0006 — el prior del 3.5-Base ES ligeramente mejor
(5 %, real) pero **desaparece por completo tras 9K secuencias de SFT**: todos los pares post quedan dentro de
la banda semilla-a-semilla, y con 3 vistas el 3.5 va un 3 % en contra. Cero exactas greedy en los 6 brazos.

**Veredicto definitivo: EMPATE.** Prerreg (≤0,85 en todas las semillas, ≥2/3): no se cumple en ninguna.
Eje del modelo base agotado con prior y aprendizaje medidos limpios y pareados. Datos: `scripts/_sft35_v4.json`.
