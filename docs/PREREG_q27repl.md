# q27sub = 16,94 y PREREG `q27repl` (27-oct) — lecciones de RSNA aplicadas

## 1. Lectura del submit `57683185`: 16,94

Prerreg (26-oct): ≤ 21,5 ⇒ «el presupuesto no era el cuello en Kaggle». Se aplica tal cual. Pero antes de
cerrar nada, dos hechos y un error mío:

1. **σ de este sistema en el LB ≈ 2,4–2,6 pt** (una muestra por tarea a T=1,0: binomial con p≈0,17–0,21 sobre
   240). El 4B tenía σ 0,415. Una lectura sola distingue ~5 pt, no 1: 16,94 frente a 21,53 es ~1,8σ. Es exactamente
   la lección 2 de RSNA: no juzgar con un instrumento cuyo ruido supera el efecto.
2. **Error mío, con mecanismo en el código:** en el rerun el modo directo usa `concurrency` del cfg, que el autor
   deja en **10**; en mi banco medí los 440 tok/s con **16**. Subí el presupuesto por intento un 17% (49K→57K) sin
   subir la concurrencia ⇒ menos oferta y más demanda ⇒ tareas saltadas por el deadline (`skipped: deadline` →
   fallback, vale 0). No es observable (el rerun no deja logs), pero es la explicación más simple y la causé yo.
3. Lo que el 16,94 **no** dice: que el 27B no sirva. Dice que **el protocolo del autor en modo directo, tocado
   sin cuidado, rinde entre 17 y 22** con esta σ.

## 2. Qué errores de RSNA aplican aquí (post-mortem del 27-oct)

| lección RSNA | cómo se repite aquí | qué se hace distinto desde hoy |
|---|---|---|
| 1. *Cerramos la ruta correcta tras un piloto débil; un primer piloto mide la implementación, no la idea* | el 27B lleva **un** piloto (directo, protocolo del autor); el ganador de RSNA iteró protocolo, modelo y navegación con herramientas | **no cerrar el 27B con 16,94**; iterar el protocolo: REPL con herramientas y verificación (= «agente que navega»), reparto de tokens, formato |
| 2. *Instrumento cuyo ruido supera el efecto* | LB del 27B σ≈2,5; `strat` n=18 da U con σ≈1,2 tareas | leer el LB con ≥2 muestras por configuración (cada una ≈1 min de cuota); en bancos, contar **verificadas** (determinista) además de U |
| 3. *Mejora el modelo fuerte, no añadas miembros débiles* | tentación de híbrido 4B+27B a medio reloj | el híbrido se aparca: a medio reloj ambos pierden (NVARC+ 30,2; 27B a 12K/intento 21,5) |
| 4. *Del LLM extrae toda la supervisión* (cajas, probabilidades, no solo la etiqueta) | el modo directo tira el razonamiento y se queda con la rejilla | REPL: el **programa** es la supervisión verificable; votos entre programas verificados |
| 5. *Ingeniería verificada antes de gastar GPU; guardas* | cambié cfg sin re-verificar sus consecuencias (concurrencia); el rerun no tiene observabilidad | regla: **cualquier cambio de cfg del agente pasa por el banco `strat` con la MISMA concurrencia que el rerun**; nunca dos cambios a la vez |
| 6. *Timidez: defender el fork mientras quedan rutas fuertes* | «dejar `lb33-det` y Paper Track» | la ruta fuerte es el 27B; `lb33-det` es el seguro, no el plan |

## 3. PREREG `q27repl` — modo REPL sobre las 18 de `strat`

Mismo agente verbatim, modo REPL: el modelo escribe `transform()`, lo ejecuta contra las demos (`verify`), ve
los diffs, corrige, y hace `submit` cuando reproduce todas. Config: 1 intento, ≤40 turnos, **8K de razonamiento
por turno** (12K max por turno), 60K tokens por intento, contexto 24K, escalera de esfuerzo `xhigh→medium→low`
tras un corte, `retain_functions`, `degenerate_guard`, concurrencia 16 (= la que irá al rerun).

Medidas: verificadas contra demos (determinista), correctas en test top-1/top-2 contra la GT, U (campeón falla y
REPL acierta), turnos, llamadas a Python, tokens, tok/s.

| resultado | lectura | siguiente |
|---|---|---|
| **U ≥ 3** | REPL es la vía | submit REPL (misma concurrencia), 2 lecturas del LB |
| U 1–2 | empate con directo (1,5/18) | decide **verificadas**: si ≥5, REPL (es más estable que directo); si no, directo con 16 de concurrencia |
| U = 0 y verificadas ≥ 3 | verifica demos pero no generaliza (el déficit de `rearcvac`) | combinar: programa verificado como attempt_1, directo como attempt_2 |
| U = 0 y verificadas ≤ 1 | REPL no aporta a este presupuesto | directo + 2 lecturas del LB; 27B queda como «21±3» |

Coste: ~60–80 min ≈ 2–2,7 h-cuota. En paralelo (0 cuota): **`q27sub2`** = autor verbatim + `concurrency` 16
(un cambio), para medir el mecanismo del §1.2 con n=240. Prerreg q27sub2: **> 24** ⇒ la concurrencia explica el
16,94 y el directo bien configurado vale ≥ 24; 19–24 ⇒ dentro del ruido del autor; < 19 ⇒ hay otro factor.
