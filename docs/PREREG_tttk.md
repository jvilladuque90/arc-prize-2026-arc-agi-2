# PREREG `tttk` — perillas del calendario del TTT (30-oct-2026)

## Por qué aquí

El TTT vale +20,6 pt (2/3 del score) y solo se han tocado dos perillas en el LB: el rango (64 y 512,
ambos pierden) y la duración (media época, −4,3). Nunca se ha barrido el **lr**, el **nº de épocas** hacia
arriba ni el **nº de permutaciones de color** (128 secuencias = 8 D4 × 16 perms). La loss del TTT está
saturada y no sirve; el instrumento que sí predice es el **déficit de la GT frente al camino greedy por
teacher forcing** (validado en `haz`: predijo sin excepción qué tareas recuperaba el haz-8) y el
**NLL/celda post-TTT** (Spearman −0,43 con GT-en-pool). Un forward por vista, sin decode.

La autopsia dice que 14/18 fallos tienen déficit ≥ 4,5 (ancho ≥ 90): el modelo está a 2–20 celdas pero
no **sabe** la respuesta. Si un calendario del TTT baja ese déficit, es la primera palanca sobre el motor.

## Diseño

- Pares exactos: cada tarea ve **todas** las variantes en el mismo worker/GPU, partiendo del mismo adaptador
  inicial. Medida tras cada TTT: déficit en la vista canónica + mínimo/mediana sobre 3 vistas aumentadas fijas,
  nº de vistas donde el greedy ya da la GT, NLL total.
- `base_s2` = producción con otra semilla de aumentación: **calibra el ruido del propio instrumento**. Una
  variante solo cuenta si su delta pareado sale de esa banda en la mayoría de tareas (test de signo).
- Piloto (hoy, ~1 h-cuota, lo que queda de la semana): 4 tareas, 1 por GPU, a lo largo de la distribución
  (`28a6681f` casi/2 celdas, `faa9f03d` casi/3, `d59b0160` cerca/20, `4a21e3da` vacía); variantes `base`,
  `base_s2`, `lr1e-4`. Objetivo del piloto: fontanería + banda de ruido + primera lectura del lr.
- Completo (tras el reset del viernes 31-oct 19:00; ~6–7 h-cuota): 18 tareas de `strat`, variantes `base`,
  `base_s2`, `lr1e-4`, `ep2` (2 épocas, +35 % de reloj del TTT), `np32` (256 secuencias, mismo coste que ep2).

## Predicciones (antes de ver nada)

| lectura | qué significa | siguiente |
|---|---|---|
| ruido `base_s2`: mediana \|Δ\| ≥ 2 | el déficit por vista es demasiado ruidoso para n=4 | leer solo el mínimo de 4 vistas; decidir con las 18 |
| `lr1e-4` mejora en ≥ 3/4 y fuera de la banda | el lr de producción está corto | entra en el banco de 18 y, si repite, es injerto de coste 0 en el reloj |
| ninguna variante sale de la banda (en el banco de 18, signo p>0,1) | el calendario del TTT no es la perilla; el déficit es de capacidad del modelo | cerrar; final = `lb33-det` |
| `ep2`/`np32` mejoran claro (≥ 12/18) | hay más adaptación comprable con reloj | diseñar cómo pagarlo (scoring 8→4 augs −4 %; vistas 16→12) |

Coste 0 en el reloj si gana el lr; `ep2`/`np32` cuestan +35 % del TTT ≈ +12 % del reloj total, que hay
que pagar con otra cosa (techo +4,8 %).
