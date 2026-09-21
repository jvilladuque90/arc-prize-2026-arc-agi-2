# Auditoría: dónde se pierde el score (2026-09-21)

Autopsia sobre los **pools completos con scores** que guardó `strat` (producción exacta: 4B
`sft139`, TTT de 128 secuencias, 16 vistas, DFS a 45 s, 18 tareas de eval estratificadas p5-p95).
**0 h-cuota.**

## 1. Dónde se pierde, etapa por etapa

| diagnóstico | tareas | % |
|---|---|---|
| **el decode no produce NINGÚN candidato** | **6/18** | **33,3%** |
| forma incorrecta siempre | 3/18 | 16,7% |
| forma correcta, casi (≥95% de celdas) | 3/18 | 16,7% |
| forma correcta, cerca (80-95%) | 2/18 | 11,1% |
| contenido lejos | 1/18 | 5,6% |
| **GT en el pool** (lo que puntuamos) | 3/18 | 16,7% |
| pérdida por **selección** | — | **0%** (medido en §19) |

- mejor candidato: **98,0% de celdas correctas** (mediana). En las falladas, el mejor candidato
  está a **15 celdas** de la verdad (mediana), **mínimo 2**.
- candidatos por tarea: mediana **5** (de 16 vistas × hasta 8). Redundancia ×1,2.

## 2. El mecanismo

La aceptación del DFS compara la **NLL acumulada de la rejilla entera** contra `1,609`
(`max_score = -log(0,2)`), tanto en el banco como en producción (`turbo_dfs` pasa `scores`
recursivamente). Es decir: **p(rejilla completa) > 0,2**. Impuesto exponencial al tamaño.

Y producción **no tiene respaldo**: si `dfs_result` viene vacío, la tarea cae al placeholder
`[[0]]` — **cero garantizado**.

## 3. El dato que lo convierte en escándalo

Las 6 tareas sin ningún candidato, con su confianza **post-TTT**:

| tarea | celdas | p media/celda | NLL total | veces por encima del umbral |
|---|---|---|---|---|
| 53fb4810 | 450 | **98,5%** | 6,6 | ×4 |
| 4a21e3da | 441 | **97,2%** | 12,7 | ×8 |
| 21897d95 | 900 | **97,7%** | 21,0 | ×13 |
| b10624e5 | 729 | **96,3%** | 27,2 | ×17 |
| 195c6913 | 900 | **96,8%** | 28,9 | ×18 |
| 221dfab4 | 625 | 89,4% | 70,3 | ×44 |

Contra las que **sí** producen candidatos: `89565ca0` (30 celdas, **53,9%**/celda) produce 13
candidatos; `f931b4a8` (80 celdas, **68,1%**) produce 3.

⇒ **El criterio de aceptación está anti-correlacionado con la competencia real del modelo.**
No selecciona por calidad, selecciona por brevedad: descarta tareas donde acierta 98 de cada 100
celdas y admite tareas donde acierta 54.

## 4. El error: el cierre fue prematuro, y es mío

Esta anatomía se descubrió el **17-ago** (§2-§3) y la familia de reglas de aceptación se declaró
**CERRADA el 22-ago** (§14-bis) con tres puntos: umbral fijo 3/40, lineal 3/40, descuento de picos
3/40, duplicando coste cada vez.

**Pero los tres se midieron sin TTT y con UNA sola vista.** Verificado hoy sobre
`notebooks/dfs_budget.ipynb`: no contiene `LoraConfig`, ni `get_peft_model`, ni `entrenar_ttt`,
ni augmentaciones. Y el propio documento lo dejó escrito: *"Pendiente: re-medir con TTT, donde
todas las magnitudes se mueven"* — y **esa re-medición nunca se hizo**.

Las magnitudes sí se movieron: NLL/celda del modelo base **0,088** → con TTT **0,015-0,037**
(5× más plana). El argumento que cerró la familia («los picos son de 2-10 nats, p ≤ 8%, ninguna
cirugía rescata eso») se midió sobre el **modelo base**. Con TTT los picos están mucho más
aplastados, y el impuesto al tamaño pasa a ser el cuello dominante.

**Conclusión de la auditoría: no hay ningún bug. Hay una rama cerrada con evidencia no
representativa, y luego cinco semanas decidiendo sobre ese cierre.**

## 5. Lo que además se cierra hoy (0 h-cuota)

**Consenso por celda entre las 16 vistas: NEGATIVO.** Cuando fallamos, las vistas producen 10-13
rejillas distintas, cada una ~98% correcta. Votar celda a celda **nunca** recupera una tarea nueva
(0 de 12) y casi siempre empeora al mejor candidato individual (98→97%, 99→96%, 92→**68%**).

⇒ **Los errores están CORRELACIONADOS entre augmentaciones.** Esto refuta el supuesto que quedó
en pie desde agosto: *"una celda dudosa en la vista identidad puede ser obvia en la rotada; las
augmentaciones SON el mata-picos"*. Matan algunos picos, pero en las celdas de decisión fallan
todas juntas y hacia el mismo sitio.
