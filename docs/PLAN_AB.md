# Plan A / B — la campaña de saltos grandes (desde 2026-09-09)

## Marco (decisión del usuario, 9-sep)

> *"no pienses en tiempo, estamos muy atrás; si eso abre la posibilidad de saltos de score
> grandes vale la pena"*

Estamos en **30,69** contra un líder en **57%**. Un +1 o +2 no cambia nada: solo consume
calendario. **Queda descartado todo lo que viva en la banda de ruido**: el Qwen3.5-2B por
velocidad (vale 0, no hay comprador para el reloj liberado), el ajuste de rango, el de vistas.

Deadline: **2026-11-02**. Presupuesto autorizado para esta campaña: **10 h-cuota**
(el tope semanal del proyecto sigue siendo 10 h y el pool se comparte con agi-3).

## El diagnóstico sobre el que se construye todo

- `score ≈ P(GT en el pool) ≈ 31%` ⇒ **la selección ya es sin pérdidas**. Mejor scoring vale 0,
  mejor verificación *de lo que ya generamos* vale 0. Solo cuenta **cambiar lo que se genera**.
- El modelo acierta la **forma en el 71%** y el contenido en el 7%: aplica *una* transformación
  plausible con seguridad; falla en **cuál**.
- Fase 0 (0 h-cuota): la entrada de test cae fuera de la envolvente de las demos y ese hueco
  **crece con el nivel del benchmark** (AGI-1 0,500 → AGI-2 train 0,750 → AGI-2 eval 1,161,
  p=2e-8). El exceso es **contenido, no vacío**: área ×1,43, tinta ×1,45, **componentes ×1,00**
  (mismos objetos, más grandes). Y la **salida** exigida cae fuera de la envolvente de las salidas
  demo en el **94,7%** de las tareas.

## Plan A — entrenar contra el déficit de escala

Parchear la transducción para que extrapole escala. SFT continuado sobre `sft139` con datos
sintéticos donde la regla se demuestra a escala pequeña y el objetivo va a escala grande,
calibrado al ×1,43 de área y ×1,45 de tinta medidos.

- Techo estimado **+10-15**. Riesgo medio. Coste ~7-10 h-cuota.
- Virtud: apoyado en **nuestra propia** medición.
- Riesgo principal: que los datos sintéticos enseñen el **atajo** ("si la entrada es grande,
  agranda la salida") en vez de la generalización. Sería la tercera lección de data sintética.

## Plan B — inducción aprendida a un DSL compacto + verificación exacta

El modelo emite un **programa**; el programa se ejecuta sobre las entradas de las demostraciones
y solo entran al pool los que **reproducen todas** las salidas demo. Es la mitad de "inducción"
del híbrido que gana arriba.

**Las dos razones por las que lo habíamos cerrado son falsas para esta versión:**

1. *"El modelo no tiene palabras (16 tokens)"* → **no hace falta modelo nuevo**: se extiende la
   matriz de embeddings de 16 a ~64 filas con tokens de DSL y se entrenan solo esas filas en un
   SFT continuado sobre `sft139`. Se conserva su conocimiento de ARC, el formato de roles y el TTT.
2. *"El thinking cuesta ×8-116 de reloj"* → **emitir un programa es más barato que emitir la
   rejilla**: 30-60 tokens frente a 100-931. Si el programa se ejecuta para producir la salida, el
   decode de la rejilla desaparece. **El techo del 4,8% no bloquea esto**; puede ser negativo.

- Techo **+25 o más** si el DSL cubre. Riesgo **alto**. Coste ~20-40 h-cuota.
- Precedente honesto: tenemos dos negativos en la vecindad, pero fueron **composición mecánica**
  (búsqueda sobre un DSL). Esto es **emisión aprendida**: el modelo propone, la búsqueda solo
  verifica. Es una diferencia real, no una garantía.
- Riesgo que manda: **la expresividad del DSL es el techo**. Si no expresa AGI-2, no hay
  entrenamiento que lo abra.

## Relación entre A y B: NO son excluyentes

- **B se traga el mecanismo de A.** El problema de escala es un problema de *transducción*: existe
  porque hay que emitir token a token una rejilla mayor que ninguna vista. **Un programa es
  indiferente a la escala.** Eso sube el valor de B y baja el de A.
- **Son complementarias en cobertura**: B resuelve lo que el DSL expresa, la transducción cubre el
  resto. La unión **es** la arquitectura híbrida que gana. Ninguna sola llega.
- **Pueden compartir inferencia**: una sola adaptación de TTT (34,8% del reloj) y dos modos de
  decode.
- **Restricción real 1**: el LB no atribuye cambios agrupados (lección `lb33-stack`) ⇒ **medir por
  separado aunque se desplieguen juntas**.
- **Restricción real 2**: cada SFT es una tirada del dado sobre el 71% de formas correctas que ya
  tenemos. Dos SFT encadenados son dos tiradas.

## El orden lo decide una puerta que cuesta 0 h-cuota

**Puerta de B** (primero, porque decide cuál de las dos es el plan principal): medir la
**cobertura y la precisión de verificación** de un DSL de juguete por enumeración, en CPU, sobre
las tareas etiquetadas. Prerregistro en `docs/PREREG_dsl.md`.

- Si el DSL cubre bien ⇒ **B es el plan principal**, A pasa a respaldo para lo que el DSL no exprese.
- Si cubre mal ⇒ **B muere en la puerta** y A es el plan principal, sin haber gastado GPU.

**Puerta de A**: validar el proxy `nll/celda de la GT` con datos ya pagados (`augdose` + `rbuyer`)
y generar los datos de escala. También CPU.

## Regla de instrumentación que cambia la campaña

El **leaderboard es nuestro instrumento más barato y potente**: n=240 por ~2 h-cuota del commit.
El banco `rbuyer` costó 1,6 h-cuota para 7 pares de los que solo **2** informaron. El LB es del
orden de **100× más informativo por hora-cuota** para efectos de capacidad.

⇒ **Dejar de gastar cuota en comprar mediciones y gastarla en comprar capacidad.** Lo caro es
entrenar; medirlo no lo es. Límites: un injerto por brazo, 1 slot/día, y su banda de ruido (±0,4)
resuelve ~1 punto ≈ 2,4 tareas — un arreglo que ataque el 69% de las tareas debe ser mucho mayor.
