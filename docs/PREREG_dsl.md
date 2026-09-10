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
