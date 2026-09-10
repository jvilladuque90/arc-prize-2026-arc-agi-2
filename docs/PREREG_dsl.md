# Prerregistro — puerta del Plan B: ¿un DSL puede expresar ARC-AGI-2?

Escrito **antes** de ejecutar. 2026-09-09. Coste: **0 h-cuota** (CPU local, sin modelo).

## Qué decide

El Plan B (inducción aprendida a un DSL + verificación exacta contra las demos) tiene un riesgo
que manda sobre todos: **la expresividad del DSL es el techo**. Si el DSL no expresa las
transformaciones de ARC-AGI-2, ningún entrenamiento lo abre. Esta puerta lo acota **antes** de
gastar 20-40 h-cuota.

## Método

Enumeración exhaustiva de programas de **profundidad ≤2** sobre un DSL de juguete (~25 operadores
geométricos, de recorte, de escala, de estructura), **más un mapa de color ajustado a las demos**
al final. Un programa "resuelve" una tarea si reproduce **exactamente todas** las salidas de las
demostraciones.

## Métricas

- **cobertura** = % de tareas con ≥1 programa que reproduce todas las demos.
- **precisión de verificación** = entre esas, % en las que el programa **también acierta el test**.
- **score_dsl** = cobertura × precisión = % de tareas realmente resueltas.
- **complementariedad** = distribución de `d_test` (la distancia OOD de la Fase 0) en las tareas
  que el DSL resuelve frente a las que no.

## Predicciones registradas

- **P1.** La cobertura será **mayor en AGI-1 que en AGI-2** (AGI-2 se diseñó para romper DSLs).
- **P2.** La precisión de verificación será **alta (>80%)**: un programa que reproduce 3-5 pares
  exactos rara vez es espurio.
- **P3.** El DSL resolverá preferentemente tareas de **`d_test` bajo** — es decir, solapará con lo
  que la transducción ya hace bien.

## Regla de decisión (registrada)

| `score_dsl` en eval AGI-2 | veredicto |
|---|---|
| **≥ 10%** | B es el **plan principal** (un modelo aprendido va más profundo que una enumeración de profundidad 2); A pasa a respaldo |
| **3-10%** | zona gris ⇒ decide la **complementariedad**: si resuelve tareas de `d_test` alto, sigue; si solo las fáciles, no |
| **< 3%** | **B muere en la puerta**; A es el plan principal, sin haber gastado GPU |

**Excepción que manda sobre la tabla**: si la cobertura es alta pero la **precisión < 60%**, la
premisa de B se rompe aunque el score salga bien — significaría que la verificación exacta contra
las demos **no filtra**, y B se apoya entera en que sí lo haga.

⚠️ **Esto es una COTA INFERIOR**: un modelo aprendido puede componer más profundo y con operadores
parametrizados que aquí no se enumeran. Un número bajo desanima, no demuestra. Operativamente da
igual: con un prior de 0-de-2 en esta familia, la carga de la prueba recae sobre la rama cara.

---

# RESULTADOS (2026-09-09) — 0 h-cuota, CPU, sin modelo

## Control positivo primero

8/8 tareas sintéticas cuya solución se conocía de antemano (`rot90`, `fliph`, `up2`, `tile22`,
`rot90>up2`, `espejo_h`, `transp>tile_h2`, y una con mapa de color): el buscador encuentra el
programa **y** acierta el test en todas. *(El primer intento dio 0/8 por un fallo del control, no
del buscador: le pasaba la solución con una dimensión de más.)*

## Cobertura

| población | n | cobertura | precisión | `score_dsl` |
|---|---|---|---|---|
| train **AGI-1** | 767 | **5,1%** (39) | **87,2%** | 4,4% |
| train **AGI-2** | 233 | **0,0%** (0) | — | 0,0% |
| eval **ARC-AGI-2** | 120 | **0,0%** (0) | — | **0,0%** |

- **P1 confirmada, y no es un gradiente: es un acantilado.** El mismo DSL saca 5,1% en AGI-1 y
  **cero en 353 tareas de AGI-2**. Si la tasa real en AGI-2 fuese 2,5% (la mitad que en AGI-1),
  P(0 de 353) ≈ 1,4e-4. Cota superior al 95%: **la cobertura de AGI-2 con este DSL está por debajo
  del 0,85%**, frente al 5,1% de AGI-1 — una caída de **≥6×**.
- **P2 confirmada**: precisión **87,2%**. Un programa que reproduce 3-5 pares exactos casi nunca es
  espurio ⇒ **la verificación exacta contra las demos SÍ filtra**. La premisa de B era buena; lo que
  falla es que no hay nada que verificar.
- **P3 no evaluable**: cero tareas resueltas en eval.
- Sin confundido trivial: la mediana de demostraciones es **3,0** en las tres poblaciones y la
  fracción con forma entrada=salida es 66-75% en todas. Lo que cambia es el **tamaño**: área de
  entrada mediana **324** en eval AGI-2 frente a **100** en AGI-1 (×3,24).

## VEREDICTO: `score_dsl` = 0,0% < 3% ⇒ **el Plan B muere en la puerta**

Coste: **0 h-cuota**. El Plan A pasa a ser el plan principal.

⚠️ Como estaba registrado, esto es una **cota inferior**: un modelo aprendido compone más profundo
y con operadores parametrizados. Un DSL de juguete de profundidad 2 sacando 0% no *demuestra* que
la familia esté muerta. Pero lo que pesa es el **contraste**: un DSL débil degradaría de forma
proporcional, no caería a cero exacto en 353 tareas. **ARC-AGI-2 parece construido específicamente
para que las transformaciones globales de rejilla no apliquen** — lo que encaja con todo lo medido:
mismos objetos, más grandes y más detallados, exigiendo razonar sobre el **contenido**.
