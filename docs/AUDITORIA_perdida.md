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

---

# 6. RESULTADO `haz` (2026-09-21, ~1,5 h-cuota): la re-medición CON TTT

## El instrumento analítico queda VALIDADO

Teacher forcing de la GT → rango del token correcto por posición → `déficit` frente al camino
greedy → `ancho_implicado = exp(déficit)`. Predice el resultado del haz real **sin excepción**:

| déficit | ancho implicado | haz 8 encuentra la GT |
|---|---|---|
| 0,0 (`58f5dbd5`) | 1 | ✅ (ya con ancho 1) |
| 2,0 (`53fb4810`, `3a25b0d8`) | 7,4 | ✅ |
| 3,5 (`d59b0160`) | 33 | ✅ |
| ≥ 4,5 (las otras 14) | ≥ 90 | ❌ |

Un forward por tarea sustituye a un barrido de anchos. Herramienta reutilizable.

## El mecanismo de agosto se CONFIRMA, ahora con TTT y cuantificado

| | pool vacío (n=6) | con pool (n=12) |
|---|---|---|
| tokens donde la GT **no** es el argmax | **4 de 703 = 0,50%** | 4 de 214 = 1,87% |
| déficit mediano | **14,75 nats** | 8,25 nats |
| ancho implicado mediano | **~2,5 millones** | ~3.800 |

Con TTT el modelo acierta el argmax en el **99,5%** de los tokens — y aun así la verdad queda a
14,75 nats del camino greedy, concentrados en ~4 celdas de decisión (≈3,7 nats cada una: el token
correcto a ~2,5% de la masa del argmax). **Los picos siguen siendo profundos después del TTT.**
El veredicto de agosto era correcto en el mecanismo; lo que no era representativo eran las
condiciones.

## Pero el cierre dejaba fuera un caso real

El haz de ancho 8 sobre **una sola vista** recupera **2 tareas de 18 (11%)** que producción falla
hoy: `53fb4810` (pool vacío) y `d59b0160` (pool con 20 candidatos, mejor a 20 celdas). Son
exactamente los casos de **déficit bajo** (2,0 y 3,5): la verdad está a tiro, y el umbral absoluto
la rechaza **solo por longitud**.

⚠️ **n=18 con 2 éxitos**: el intervalo de confianza al 95% va de ~1,4% a ~34%. Podría valer
+1 punto o +20. No se puede estrechar sin más tareas.

## El coste medido es fatal — pero es artefacto de mi implementación

| | valor |
|---|---|
| haz-8, una vista, medido | **59 s/tarea** (13 s a 181 s según longitud) |
| extrapolado a 240 tareas | 3,96 h = **+33,5% de reloj** |

Eso mata cualquier versión general (`r512` cayó del acantilado con +11,9%). **Pero el bucle de
candidatos hace 96 llamadas a `.item()` por paso de token** — 96 sincronizaciones GPU→CPU, hasta
89.000 por tarea larga. El suelo real del forward con lote 8 y caché es ~20 ms/paso ⇒ **~19 s para
930 tokens**, no 181. Vectorizando el `topk` en GPU el coste debería caer ~5-9×, a **~+5% de
reloj** general o **~+3%** si solo se aplica donde el pool sale vacío.

## Estado

**El arreglo no es gratis ni está probado, pero es la primera palanca en semanas con efecto
positivo medido.** Lo que falta antes de tocar al campeón: (1) vectorizar el haz y confirmar que
el coste baja y que las 2 tareas se conservan; (2) comprobar que añadir 8 candidatos no rompe la
selección, que hoy es sin pérdidas.

## 6-bis. Restricción encontrada gratis mientras corre `haz2`: el re-scoring

Margen de la selección hoy (aproximado: solo `beam_score`, sin la mitad `score_aug`):

| tarea | GT/16 | nº únicos | rank de la GT | punt. GT | punt. 2º rival |
|---|---|---|---|---|---|
| 3a25b0d8 | 1 | 13 | **6** | 2,28 | 3,97 |
| b6f77b65 | 2 | 5 | 2 | 4,62 | 4,16 |
| 58f5dbd5 | 14 | 2 | 1 | 39,80 | 12,61 |

⇒ **Es el re-scoring aumentado el que hace la selección sin pérdidas**, no el `beam_score`: por
`beam_score` solo, la GT de `3a25b0d8` sale en el puesto 6 de 13 y aun así producción la elige.

Y eso tiene precio: el re-scoring evalúa cada candidato bajo **8 augmentaciones**. Hoy el scoring
es el **7,9%** del reloj para ~5 candidatos por tarea ⇒ **~1,6% de reloj por candidato**.

**Y en las 2 tareas nuevas la GT está en el haz pero NO es el candidato top** (`mejor_ok=False`).
O sea que no basta con añadir uno. Si hubiera que añadir los 8:

| dónde se aplique | decode del haz | re-scoring (8 cands) | total |
|---|---|---|---|
| todas las tareas | ~+5% | **+12,6%** | **+17,6%** ❌ |
| solo donde el pool sale vacío (33%) | ~+3-5% | +4,2% | **+7-9%** ❌ |

Ambos por encima del techo de +4,8% (`r512` murió con +11,9%). **El re-scoring, no el decode, es
el coste dominante** — y no lo había contado.

La salida posible: en las tareas de pool vacío **saltarse el re-scoring** y enviar directamente
los 2 mejores del haz por `beam_score`. Hoy esas tareas valen cero garantizado, así que cualquier
orden es ≥ 0. Queda por saber **en qué puesto del haz cae la GT**: si es el 2º, el top-2 la captura
gratis. `haz2` guarda los candidatos con su score, así que eso se resuelve offline.
