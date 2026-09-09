# Prerregistro — Fase 0: ¿la entrada de test está fuera de la distribución de las demos?

Escrito **antes** de ejecutar nada. Coste: 0 h-cuota (CPU local; los outputs de banco se
descargan gratis). Fecha: 2026-09-09.

## Hipótesis

**H.** En ARC-AGI-2 la entrada de test está *sistemáticamente fuera* de la envolvente de las
entradas de las demostraciones —por diseño del benchmark—, y esa distancia es la que explica el
hallazgo de `ttt-regla` (5/12 en entradas de la distribución de las demos, 0/12 en test).

Si H es cierta, el método propuesto (descubrir la equivarianza propia de la tarea y usarla para
**meter la entrada de test dentro de la distribución de las demos**) ataca la causa. Si H es
falsa, el método no tiene sobre qué actuar y muere aquí, gratis.

## Métrica

Rasgos por rejilla de entrada: `H`, `W`, `log(H*W)`, `n_colores`, `densidad_no_fondo`,
`n_componentes` (4-conectividad, sin fondo), `frac_componente_mayor`.

Distancia **por envolvente** (robusta con 2-5 demos, sin suponer normalidad): para cada rasgo
`f`, con `lo=min(demos)`, `hi=max(demos)`, `R=hi-lo`:

- si `lo <= test <= hi` ⇒ `e_f = 0`
- si no ⇒ `e_f = dist_al_borde / max(R, piso)`

`d_test = max_f e_f`. Lectura directa: *"el test cae `d` rangos-de-demo fuera de la envolvente"*.
`d = 0` ⇒ el test está dentro del rango de las demos en todos los rasgos.

## Predicciones (registradas)

- **A1.** Mediana de `d_test` en **eval de ARC-AGI-2** > mediana en el **training set**.
  Contraste: Mann-Whitney, una cola.
- **A2.** **> 50%** de las tareas de eval tienen `d_test > 0` (el test se sale de la envolvente
  en al menos un rasgo).
- **B.** Sobre las 16 tareas del banco `augdose` (con `vistas_con_gt` de 0 a 16): correlación
  **negativa** entre `d_test` y `vistas_con_gt` (Spearman ρ < 0).
  ⚠️ `n=16` es baja potencia: esto se declara *consistente / inconsistente*, **nunca "probado"**.
  Además el eval público está bajo sospecha de contaminación para este checkpoint, lo que sesga
  el resultado hacia arriba; sesga el nivel, no necesariamente la pendiente.
- **C.** Catálogo de **re-representaciones** `T` aplicadas por igual a demos y test (invertibles o
  recuperables en la salida, así que la tarea es equivalente): recorte a caja envolvente, quitar
  marco uniforme, reducir por factor de bloque entero, desteselar, canonicalizar color por rango
  de frecuencia, transponer a `H<=W`.
  Predicción: **>= 20%** de las tareas de eval con `d_test > 0` admiten alguna `T` que baje
  `d_test` a 0.

## Regla de decisión (registrada)

| resultado | veredicto |
|---|---|
| **A1 falla** (el gap no es mayor en eval) | la premisa es falsa ⇒ **método muerto**, coste 0 |
| A1 pasa, **C < 10%** | la causa es real pero no tenemos herramienta ⇒ **muerto** |
| A1 pasa y **C >= 20%** | seguir a Fase 1 (banco con GPU: medir GT-en-pool bajo `T`) |
| zona intermedia (10-20%) | decidir con el detalle de qué `T` disparan y sobre qué tareas |

`B` **no** es criterio de decisión (potencia insuficiente): entra como evidencia de apoyo.

---

# RESULTADOS (2026-09-09, 0 h-cuota, CPU local, sin cargar ningún modelo)

## A — la premisa se confirma, y con una escalera monótona

| población | n | mediana `d` | `d>0` | `d>1` |
|---|---|---|---|---|
| **eval ARC-AGI-2** | 120 | **1,161** | 78,3% | 54,2% |
| train completo | 1000 | 0,500 | 66,5% | 25,5% |
| → subset AGI-1 | 767 | 0,500 | 66,9% | 23,3% |
| → subset AGI-2 | 233 | 0,750 | 65,2% | 32,6% |

- **A1 ✅** Mann-Whitney eval > train: z = 5,493, **p = 2,0e-08**. Contra AGI-1: z = 5,744, p = 4,6e-09.
- **A2 ✅** 78,3% de eval con `d>0` (predicción: >50%).

La escalera **AGI-1 (0,500) → AGI-2 train (0,750) → AGI-2 eval (1,161)** es el hallazgo: la
distancia del test a la envolvente de las demos **crece monótonamente con el nivel de dificultad
del benchmark**. La dificultad de ARC-AGI-2 *es*, en buena parte, ese hueco.

## B — consistente en signo, sin potencia (como estaba registrado)

Spearman **ρ = −0,190** (n=16). El signo es el predicho, pero solo 3 de 16 tareas tenían el GT en
algún pool y una de ellas (`e8686506`, `d`=2,86, 5 vistas con GT) va en contra. Se declara
**consistente, no probado**. No entra en la decisión.

## C — FALLA: 3,2% frente al 20% prerregistrado

94 de 120 tareas de eval tienen `d>0`. Una sola `T` la cierra en **3** (`crop_bbox` 1,
`strip_frame` 1, `transpose_canon` 1). Añadir composiciones de dos: **+0**.
`block_downscale`, `detile` y `color_canon` no cierran ni reducen ninguna.

## Por qué falla — y esto es lo que vale la fase

Sobre las 94 tareas con hueco, el test frente a la mediana de las demos:

| magnitud | factor |
|---|---|
| área total | **×1,43** |
| celdas no-fondo (tinta) | **×1,45** |
| componentes conexas | **×1,00** |

**El exceso es contenido, no vacío**: 65 tareas crecen en área *y* en tinta; solo 4 crecen en área
sin crecer en tinta. Todo el catálogo propuesto (recortar, quitar marco, reducir bloques,
desteselar) solo sabe **quitar vacío**. No hay vacío que quitar.

Y el detalle que lo afina: **el número de objetos no cambia (×1,00)** mientras la tinta sube 45%.
No hay *más* objetos — los mismos objetos son **más grandes**. La entrada de test es la misma
escena a mayor escala y con más detalle, no una escena más poblada.

## El hallazgo colateral, y una corrección a mi propia lectura previa

- **La SALIDA del test también cae fuera de la envolvente de las salidas de las demos en el 94,7%
  de las tareas** (mediana `d` = 1,500). Durante el TTT, *todos* los gradientes que recibe el
  modelo apuntan a salidas más pequeñas que la que después debe producir.
- Yo había escrito en el script la lectura "la razón salida/entrada no es constante, luego no
  basta reescalar". **Es falso**: la razón área-salida/área-entrada es constante (±10%) en
  **74,5%** de las demos. La forma **sí** es deducible — lo que concuerda con el 71% de candidatos
  con forma correcta del banco `augdose`. **El fallo no es la forma, es el contenido a escala.**

## VEREDICTO por la regla registrada

> `A1 pasa, C < 10%` ⇒ *la causa es real pero no tenemos herramienta* ⇒ **MUERTO**

**El método de "descubrir la equivarianza y normalizar la entrada" queda muerto**, por 0 h-cuota.
La causa queda **confirmada y mucho más afilada** que antes.
