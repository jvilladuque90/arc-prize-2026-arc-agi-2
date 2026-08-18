# Plan TTT: alinear el entrenamiento por tarea con el objetivo real

Línea abierta el 2026-08-17 por indicación del usuario, tras cerrar la familia de cobertura:
*"si el TTT, como cualquier entrenamiento, se hace sin estar alineado con el objetivo, no va a
dar avance"*. Todo el trabajo vive en **Colab (gratis)** hasta que muestre mejoras grandes en
GT-en-pool sobre la base de train; solo entonces se lleva a submit.

## 1. Por qué aquí y no en velocidad

Descomposición medida del campeón ([ARCHITECTURE.md](ARCHITECTURE.md) §5):

```
30,1%  ≈  cobertura 0,80  ×  GT-en-pool 0,42  ×  selección 0,90
```

- Velocidad/cobertura: techo total **+7,5 pt**. Seis palancas medidas (`ttt8`, `scoring4`,
  `dfsbatch`, `dfswide`, especulación, tope de longitud); las supervivientes rinden ×1,10-1,25
  y la única que llegó a submit **costó −2,5**. Familia agotada.
- Selección: ~0,90, techo +3,3 pt.
- **GT-en-pool: 0,42 → duplicarlo lleva el score a ~60.** Es el único sitio donde queda un ×2.

## 2. La desalineación, en un número

El DFS acepta un candidato solo si la grid **completa** cumple NLL acumulada `< -log(0,2)`,
es decir **p(secuencia) > 0,2**. Para una salida de `L` tokens eso exige una probabilidad
media por token de `0,2^(1/L)`:

| L (tokens de salida) | prob. media por token requerida |
|---|---|
| 100 | 98,4% |
| 372 (mediana del eval) | **99,57%** |
| 929 (grid 30×30) | **99,83%** |

El TTT se entrena con pérdida de token siguiente, que optimiza **NLL media**; el criterio de
producción es **masa de probabilidad de la secuencia entera**, que se endurece exponencialmente
con el tamaño. Un modelo puede tener excelente precisión por token y no cruzar nunca el umbral.

Esto ya explica dos resultados que teníamos sueltos:
- `dfswide` (abrir el umbral a `-log(0,01)`) solo sumó +1/12 de GT: el problema no es el corte,
  es que la masa no se acumula.
- Sin TTT el pool es **vacío** al umbral de producción (medido 16-ago): el TTT no refina el
  pool, es lo que hace que exista.

## 3. Hipótesis pre-registradas

- **H1 (tamaño)**: GT-en-pool cae monótonamente con `L`. Predicción: casi todas las tareas con
  GT en el pool tienen salidas pequeñas; por encima de ~400 tokens la tasa colapsa.
- **H2 (criterio alineado)**: el TTT del campeón **no** consigue que el modelo reproduzca sus
  propias salidas de train bajo el umbral. Medible sin tocar el test: tras el TTT, calcular
  `p(salida_train | entrada_train)` para cada par de la tarea. Si `< 0,2`, el DFS no tenía
  ninguna posibilidad en esa tarea.
- **H3 (entrenar hasta el criterio)**: seguir entrenando el TTT hasta que los pares de train
  crucen el umbral (más épocas / lr / early-stop por criterio, no por pasos) sube GT-en-pool
  en tareas nuevas. Es el experimento central: convierte un entrenamiento a ciegas en uno con
  objetivo medible.
- **H4 (umbral normalizado por tamaño)**: sustituir el umbral absoluto por uno por token
  (`NLL_media < c`) reabre específicamente las tareas de salida grande, que hoy son imposibles
  por construcción. Riesgo: explota el pool en tareas pequeñas → medir coste antes.

## 4. Gate previo: un harness de TTT FIEL en T4

Bloqueante conocido: el port de TTT a Colab en fp16 **empeoraba** la NLL incluso sobre el 4B
original (carril cerrado el 13-ago). Sospecha principal: producción entrena en **bf16**
(`bf16=True, fp16=False`) y la T4 (SM75) no tiene bf16 por hardware; el port usaba fp16 con
GradScaler sobre LoRA r=256 con rsLoRA (escala 2,0), combinación numéricamente frágil.

Fix a probar: **pesos base en fp16, parámetros LoRA en fp32** (master weights), autocast
estándar, sin GradScaler sobre los LoRA.

**Criterio de fidelidad, falsable**: sin TTT el pool al umbral de producción es *vacío*
(medido). Un harness fiel debe producir **pool no vacío** en tareas nuevas tras el TTT. Si no
lo consigue, no se interpreta ningún resultado posterior.

## 5. Orden de trabajo (todo en Colab)

1. **T1 — harness fiel** (gate del §4). Sin esto, nada más es interpretable.
2. **T2 — auditoría**: sobre 12-24 tareas nuevas, tras el TTT del campeón medir (a) NLL total
   de la GT, (b) `p(salida_train)` de los propios pares de train, (c) pool al umbral real, y
   (d) todo lo anterior contra `L`. Resuelve H1 y H2 de una vez.
3. **T3 — TTT con criterio**: entrenar hasta que los pares de train crucen el umbral (tope de
   pasos por presupuesto) y re-medir GT-en-pool. Resuelve H3.
4. **T4 — umbral normalizado**: barrido de `NLL_media` vs pool y GT. Resuelve H4.

Regla del usuario para esta línea: **no se sube nada a Kaggle por unos pocos puntos
porcentuales**. Se busca una línea de desempeño alta y sostenida en el banco de train antes de
gastar un slot. Cada avance confirmado puede ir a submit cuando convenga, no antes.
