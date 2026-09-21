# Prerregistro — puerta de OBJETOS: ¿un DSL por objeto expresa ARC-AGI-2?

Escrito **antes** de correr sobre datos reales. 2026-09-20. Coste: **0 h-cuota** (CPU, sin modelo).

## Qué decide

La puerta del DSL (9-sep) trataba la rejilla como **un bloque**: 5,1% de cobertura en AGI-1 y
**0 de 353** tareas AGI-2. Esta la trata como una **escena de objetos**. Si tampoco cubre, la
familia inducción queda enterrada por ambos lados y con fundamento; si cubre, es la primera señal
estructural de la campaña.

## Motivación medida (`scripts/objetos_premisa.py`, ya ejecutado)

En eval AGI-2: **58,8%** conserva el número de objetos, **30,7%** conserva las formas cambiando
solo el color, **28,9%** tiene como salida uno de los objetos de la entrada. Y cruzando con los
resultados del campeón en las 38 tareas medidas: **0 de 9** con estructura de objetos frente a
**8 de 29** del resto (Fisher una cola p≈0,09) — las tareas con estructura de objetos son
justamente las que fallamos.

## Método

7 políticas de segmentación (`cc4`, `cc8`, `cc4col`, `cc8col`, `color`, `celdas`, `todo`) × 3
familias de regla: **recolorear** (color de cada objeto en función de una propiedad),
**selección** (la salida es un objeto elegido por un criterio) y **filtrar** (borrar los objetos
que no cumplen un predicado). 17 propiedades y 12 criterios. **Verificación exacta**: el programa
debe reproducir celda a celda **todas** las salidas demo.

**Control positivo: 4/4** sobre tareas sintéticas de regla conocida (recolorear por rango de área,
recolorear por forma, seleccionar el mayor, filtrar no-rectángulos), ejecutado antes de este
prerregistro.

## Métricas

- **cobertura** = % de tareas con ≥1 programa que reproduce todas las demos.
- **precisión** = entre esas, % en las que el programa **también acierta el test**.
- **score_obj** = cobertura × precisión.
- **complementariedad** = de las tareas cubiertas, cuántas **falla hoy el campeón** (sobre las 38
  de eval con resultado medido). Un DSL que resuelve lo que ya resolvemos no suma nada.

## Predicciones registradas

- **P1.** cobertura AGI-1 > cobertura AGI-2 (como en la puerta anterior).
- **P2.** precisión > **80%** (en la puerta de rejilla fue 87,2%: la verificación exacta filtra).
- **P3.** cobertura en eval AGI-2 **≥ 10%**, frente al **0%** del DSL de rejilla.

## Regla de decisión (registrada)

| `score_obj` en eval AGI-2 | veredicto |
|---|---|
| **≥ 10%** | señal estructural ⇒ **el proyecto arranca** (SFT de emisión de programas con re-ARC/Nemotron) |
| **3-10%** | zona gris ⇒ decide la **complementariedad**: si cubre tareas que el campeón falla, sigue |
| **< 3%** | **la familia inducción muere definitivamente**, por rejilla y por objeto |

**Excepción que manda**: precisión < 60% ⇒ los programas ajustan las demos y fallan el test ⇒ la
premisa del método se rompe aunque la cobertura sea buena.

⚠️ **Cota inferior**: catálogo enumerado a mano, 3 familias, sin composición. Un modelo aprendido
iría más lejos. Un número bajo desanima; no demuestra. Lo que pesa es el **contraste con el 0% del
DSL de rejilla** sobre las mismas tareas.
