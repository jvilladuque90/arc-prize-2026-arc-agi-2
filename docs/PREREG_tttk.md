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

## Resultado del piloto (30-oct, 1605 s de pared ≈ 0,9 h-cuota; 4 tareas × 3 variantes, 0 errores)

Déficit canónico / mínimo de 4 vistas / nº de vistas donde el greedy ya da la GT:

| tarea | base | base_s2 (otra semilla) | lr1e-4 |
|---|---|---|---|
| 28a6681f (casi, 2 celdas) | 5,0 / 0,0 / **2 de 4** | 9,0 / 1,0 / 0 | 2,5 / 2,5 / 0 |
| faa9f03d (casi, 3 celdas) | 10,5 / 10,0 / 0 | 6,0 / 6,0 / 0 | 13,0 / 13,0 / 0 |
| d59b0160 (cerca, 20) | 4,5 / 2,5 / 0 | 5,5 / 4,0 / 0 | 2,0 / 2,0 / 0 |
| 4a21e3da (vacía) | 3,0 / 3,0 / 0 | 7,5 / 7,0 / 0 | 10,0 / 10,0 / 0 |

Lecturas:

1. **La semilla del TTT mueve el déficit tanto como cualquier perilla**: base vs base_s2 difiere en −4,5 / +4,5 /
   +1 / +4 (NLL de la GT 7→13, 12,6→7,8, 5,5→9,8). La banda de ruido del instrumento, por tarea, es ≈ ±4.
   Es la misma señal de `ttt-2hip` (dos TTT diversos y ambos equivocados) y de `ttford` (orden = −5): el
   resultado del TTT por tarea es una variable aleatoria ancha. Solo las 18 tareas con test de signo pueden
   separar una perilla de ese ruido; con n=4 nada se separa.
2. `lr1e-4`: mejora 2/4 en canónica, 1/4 en mínimo; **dentro de la banda**. Sin lectura; sigue en el banco de 18.
3. Dato nuevo y barato: en `28a6681f` el **greedy produce la GT en 2 de 4 vistas** (déficit 0) y aun así
   producción tuvo 0/16 GT-en-pool con 15 candidatos: el camino greedy existe pero el umbral p>0,2 sobre la
   rejilla entera (NLL 5,5 ⇒ p≈0,004) lo rechaza. En las otras 3 tareas, 0/4. El banco de 18 mide esto gratis
   (`n_vistas_greedy` en `base`): si ≥ 3/18 tareas falladas tienen greedy = GT en alguna vista, la palanca
   «greedy sin umbral en 2 vistas» (coste ≈ +3–4 % de reloj, 2 candidatos más de re-scoring) tiene comprador.
4. Solo 1–5 tokens por vista no son argmax: el modelo está a un puñado de celdas en todas, como decía la autopsia.

Siguiente: banco completo tras el reset (18 × {base, base_s2, lr1e-4, ep2, np32} ≈ 7 unidades de TTT ≈ 3,5 h
de pared ≈ 7 h-cuota). Lanzar (el notebook ya tiene `completo` como modo por defecto; Kaggle no propaga el env local):
`PYTHONUTF8=1 python scripts/push_kernels.py tttk --accelerator NvidiaL4`
