# Prerregistro — `lb33-ttford`: que el TTT ensaye la extrapolación de salida

Escrito **antes** de que corra. 2026-09-10. Un solo injerto sobre el campeón `lb33-det`
(30,69 ± 0,415, n=3).

## Mecanismo (medido, no supuesto)

Sobre las 23 tareas de eval con resultado del campeón (`augdose` + `rbuyer`):

| | Spearman con `vistas_con_gt` |
|---|---|
| `d` de la **ENTRADA** de test | **+0,004** — no predice nada |
| `d` de la **SALIDA** de test | **−0,234** |

| régimen | GT en algún pool | población eval |
|---|---|---|
| salida **DENTRO** de la envolvente de las salidas demo | **3/4 = 75%** | 18% |
| salida **FUERA** | **2/19 = 11%** | 82% |

Y las salidas de las demostraciones son **más pequeñas que la exigida en el 94,7%** de las tareas:
hoy **todos los gradientes del TTT apuntan en la dirección contraria** a la que hace falta.

## El injerto (uno solo)

En el dataset de **TTT** (`train_ds`), la demostración de **salida mayor** queda **siempre la
última**; el orden del resto sigue siendo aleatorio. Así cada una de las 128 secuencias del TTT
predice la salida más grande con solo las pequeñas de contexto: un ensayo de extrapolación.

- El dataset de **decode** (`eval_ds`) **no se toca**.
- **Coste de reloj: CERO.** Mismos tokens, mismas 128 secuencias. Es reordenación, no gasto — lo
  único admisible con el campeón al 95% de su deadline (techo de +4,8%).
- Verificado en local antes de lanzar: ambos ficheros compilan; la demostración mayor queda última
  en 200/200 semillas; el contexto conserva los 6 órdenes posibles con 4 demos (no se pierde
  diversidad de augmentación); ninguna demo se pierde.
- Diferencia frente al campeón: **solo las celdas 0 (markdown), 3 (`arc_loader.py`) y 5
  (`arc_solver.py`)**.

## Predicción y regla de decisión (registradas)

Banda de ±2σ alrededor del campeón (σ = 0,415): **[29,86 – 31,52]**.

| resultado | veredicto |
|---|---|
| **> 31,52** | fuera de banda por arriba ⇒ **adoptar**, el mecanismo es real |
| **29,86 – 31,52** | dentro del ruido ⇒ **no concluyente**, no adoptar ni revertir con una muestra |
| **< 29,86** | fuera de banda por abajo ⇒ **revertir** |

## Riesgo que asumo, dicho antes

Fijar la última posición reduce la variedad de órdenes que ve el TTT (antes 128 órdenes
aleatorios, ahora 128 con la última posición fija). Si la variedad de orden aportaba más de lo
que aporta el ensayo de extrapolación, esto sale negativo. Es el precio de un injerto de coste
cero, y por eso la regla de decisión contempla revertir.

---

## Enviado (2026-09-10 04:35 UTC, ref `56137279`)

Verificación del kernel antes de enviar (todo gratis):

- El injerto **corrió de verdad**: `ultima_mayor` aparece 4 veces en el `arc_loader.py` de salida
  y la llamada `augment(n=16, shfl_keys=True, seed=1, ultima_mayor=True)` está en la línea 366 del
  `arc_solver.py` que ejecutó el kernel.
- `submission.json` bien formado: 120 tareas, 344 intentos.
- ⚠️ **335 de 344 intentos son placeholders `[[0]]` (97%)** — y eso **NO es un defecto**: el
  campeón `lb33-det` produce **exactamente los mismos 335/344** en modo *Save & Run*. El trabajo
  real ocurre en el rerun de la competición sobre el oculto. Comparar contra el campeón antes de
  alarmarse costó cero y evitó descartar un kernel sano.

Detalle operativo: `kaggle competitions submit` en competición de código exige **`-v <version>` y
`-f submission.json`**; sin ellos falla con *"Code competition submissions require both the output
file name and the version number"*.

---

# RESULTADO: **25,69** ⇒ REVERTIR (fuera de banda por abajo)

`30,69 → 25,69` = **−5,00 puntos = −12,0 tareas exactas** de 240 (otra confirmación del quantum
de 1/240). La banda registrada era [29,86 – 31,52]; el resultado cae **4,17 puntos por debajo**.

## Qué queda refutado y qué NO

**Refutado**: esta forma barata de actuar sobre el mecanismo. Fijar la última posición del TTT
**daña**, y mucho.

**NO refutado**: la teoría de la envolvente de salida. Se midió de forma **independiente de este
injerto** (75% de GT-en-pool dentro de la envolvente frente a 11% fuera, sobre 23 tareas con
resultado del campeón). Un injerto fallido no invalida la medición que lo motivó.

## La lectura más probable, y es un hallazgo con signo positivo

El régimen "dentro de la envolvente" es solo el **18%** de las tareas — **pero ahí acertamos el
75%**. Son ~43 tareas del oculto de las que resolvemos ~32. Este injerto **no arregló el 82%
difícil y sí estropeó el 18% que funcionaba**: −12 tareas encaja con haber dañado justo ese
régimen, sesgando la adaptación hacia salidas grandes en tareas cuya salida debía ser pequeña.

Y de rebote deja medido algo que nadie había medido: **la aleatorización del orden de las
demostraciones en el TTT vale ≥5 puntos**. Es el segundo negativo más grande de toda la campaña,
solo por detrás de quitar el TTT entero:

| ablación | Δ puntos | Δ tareas |
|---|---|---|
| quitar el TTT entero (`nottt`) | −20,56 | −49,3 |
| **`r512`** | −6,38 | −15,3 |
| **`ttford` (orden fijo)** | **−5,00** | **−12,0** |
| `ttt05` (media época) | −3,75 | −9,0 |
| `stack` | −1,25 | −3,0 |
| `r64` | −0,97 | −2,3 |

La diversidad de orden del TTT es **carga estructural**, no decoración. Cualquier injerto futuro
que toque la composición de las 128 secuencias parte de aquí.

## Consecuencia de diseño para el siguiente intento

**Restricción nueva y dura**: una intervención sobre el TTT **no puede degradar el régimen del 18%
donde acertamos el 75%**. Todo lo que reestructure las 128 secuencias enteras arriesga ese 18% para
perseguir un 82% que quizá no sea alcanzable así.

⇒ La forma correcta sería **dosis, no sustitución**: dejar la mayoría de las 128 secuencias como
están y convertir solo una fracción (p. ej. 16 de 128) en ensayos de extrapolación. Sigue costando
cero de reloj y no sacrifica la diversidad de orden. **Pero eso es una hipótesis nueva, y el prior
de esta rama acaba de empeorar mucho.**
