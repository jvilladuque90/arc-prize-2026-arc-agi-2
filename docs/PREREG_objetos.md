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

---

# RESULTADOS (2026-09-20) — 0 h-cuota, segundos de CPU

| población | n | cobertura | precisión | `score_obj` |
|---|---|---|---|---|
| train **AGI-1** | 767 | **4,0%** (31) | 64,5% | 2,6% |
| train **AGI-2** | 233 | **1,7%** (4) | **0,0%** | 0,0% |
| eval **AGI-2** | 114 | **0,0%** (0) | — | **0,0%** |

**AGI-2 completo: 4 tareas cubiertas de 347, y ninguna acierta el test.**

- **P1 ✅** cobertura AGI-1 (4,0%) > AGI-2 (0,0-1,7%).
- **P2 ❌** precisión **64,5%** en AGI-1, frente al 87,2% del DSL de rejilla y al >80% predicho.
  Los programas **por objeto sobreajustan las demos más** que los de rejilla: es más fácil fabricar
  una regla que encaje 3-5 pares y falle el test. En AGI-2 la precisión es **0/4**.
- **P3 ❌** cobertura en eval AGI-2 = 0%, frente al ≥10% predicho.

## VEREDICTO: `score_obj` = 0,0% < 3% ⇒ **la familia inducción muere definitivamente**

Por rejilla (0 de 353) **y** por objeto (4 de 347, 0 aciertos). Coste total de enterrarla: **0 h-cuota**.

## El diagnóstico, que vale más que el número

El 0% **no** se debe a que AGI-2 carezca de estructura de objetos. Medido sobre las 114 de eval:

| precondición estructural | eval AGI-2 |
|---|---|
| `cc4col` segmenta en ≥2 objetos | **99%** |
| misma forma de rejilla entrada→salida | 68% |
| … **y además** las mismas celdas ocupadas (forma de "recolorear") | **32%** |
| la salida es un objeto de la entrada **con sus colores exactos** | **0%** |

La estructura está ahí y la segmentación funciona. Lo que no existe es la **regla**: en el 32% de
tareas con forma de recoloreo, **ninguna de las 17 propiedades intrínsecas del objeto** (área,
rango de área, forma, color, huecos, simetría, repeticiones…) determina el color de salida de
forma consistente. Y la "selección" cae de 28,9% a **0%** al exigir los colores exactos: la salida
toma la **forma** de un objeto de la entrada pero **recoloreada** — o sea, tampoco es selección pura.

⇒ **Las reglas de ARC-AGI-2 no son funciones de propiedades intrínsecas de un objeto aislado.**
Dependen de **relaciones entre objetos y del contexto**, que es exactamente lo que el benchmark
dice buscar. Un catálogo de reglas por objeto no las alcanza, y ampliarlo con más propiedades
intrínsecas tampoco lo haría: el eje es equivocado, no corto.

## Aviso metodológico que deja

Cota inferior, catálogo a mano, 3 familias, sin composición ni relaciones. Lo que pesa no es el 0%
aislado sino el **contraste doble**: el mismo motor saca 4,0% en AGI-1 y ~0% en 347 tareas de
AGI-2, igual que el DSL de rejilla sacó 5,1% / 0%. **Dos familias de inducción, dos acantilados en
el mismo sitio.**
