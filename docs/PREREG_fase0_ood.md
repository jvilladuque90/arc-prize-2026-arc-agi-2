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
