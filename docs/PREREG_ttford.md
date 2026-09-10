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
