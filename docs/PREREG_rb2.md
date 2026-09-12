# Prerregistro — `rb2`: la pregunta del comprador, ahora con el proxy

Escrito **antes** de correr. 2026-09-12. ~2,5 h-cuota (L4×4: 2 GPUs por brazo, sin decode).

## La pregunta

¿El reloj liberado tiene comprador? Concretamente: **sin presión de reloj, ¿`r512` acerca más al
modelo a la respuesta correcta que `r256`?** Es la pregunta que bloquea la rama Qwen3.5-2B (empata
el prior del titular con 0,378× de tronco): si el rango paga, liberar ~50% de reloj vale ~+12 pt;
si no paga, vale 0.

`rbuyer` (9-sep) no pudo contestarla: GT-en-pool solo dispara en ~20% de tareas y 5 de 7 pares
salieron 0-0. `strat` (11-sep) validó el proxy que lo hace asequible.

## Diseño

- Mismas **19 tareas** de `strat` (estratificadas p5-p95 por coste; excluida `36a08778`,
  contaminada con NLL pre-TTT = 0,0000).
- **Brazo A** `r=256, α=32` (campeón) en GPUs 0-1 · **Brazo B** `r=512, α=45` (rsLoRA a 2,0
  idéntico) en GPUs 2-3. Un modelo por worker, cero fuga de estado. Reparto LPT por tiempo de TTT
  medido en `strat` (`ttt ≈ −43 + 0,272·coste`), ×1,15 para r512.
- **Sin decode** (`N_VISTAS=0`): la primaria es la NLL, que solo necesita el TTT y un forward.
- Deadline 5.400 s por worker (estimado ~4.600 en r512), comprobado entre tareas.
- Misma semilla md5 por tarea ⇒ mismas 128 secuencias de TTT en ambos brazos.

## Medida primaria (continua, todas las tareas informan)

`gt_nll_celda` **post-TTT** de la GT, pareada por tarea: `ratio = r512 / r256`. Validada en `strat`:
Spearman −0,430 con GT-en-pool, separación ×11,6. Se reporta **mediana del ratio, ganadas/perdidas
y test de signo** (la media sola ya nos engañó tres veces). Secundaria: `pre` (debe ser idéntica en
ambos brazos: es el modelo base) como control de que los brazos parten del mismo sitio.

## Regla de decisión (registrada) — n=19 pareado

| resultado | lectura | veredicto |
|---|---|---|
| ratio mediano **≤ 0,85** y signo p < 0,10 a favor | el rango acerca a la GT | **HAY comprador** ⇒ rama 2B viva ⇒ siguiente: SFT del 2B con datos no memorizados |
| ratio mediano **≥ 1,15** y signo p < 0,10 en contra | más rango **aleja** (sobreajuste a las demos) | **no hay comprador** ⇒ rama 2B muerta |
| lo demás | el rango no paga | **no hay comprador** ⇒ rama 2B bloqueada (el empate se registra como negativo, como en `rbuyer`) |

Calibración para leer la magnitud: el ancla de `gate-base` dio **+20% de NLL ≈ −2,5 pt de LB**.
Un ratio de 0,85 (−15%) valdría del orden de +2 pt por sí solo; lo que importa es que abre la
puerta a gastar los ~50% de reloj del 2B en rango.

## Lo que NO mide

La velocidad (banco con *gradient checkpointing*, producción no) ni la selección. Solo capacidad.

---

# RESULTADOS (2026-09-12) — 75 min ≈ 2,5 h-cuota, **15 pares de 19**

El brazo r512 tardó **×1,32** en TTT (yo había presupuestado ×1,15) y el deadline entre tareas
dejó fuera **sus 4 tareas más caras** (`221dfab4`, `a47bf94d`, `b10624e5`, `195c6913`, coste
2.552-3.300). Otra vez la cola, aunque menos que en `rbuyer`. Control pre-TTT idéntico **15/15**:
los dos brazos parten exactamente del mismo modelo base.

| tarea | coste | post r256 | post r512 | ratio |
|---|---|---|---|---|
| 28a6681f | 600 | 0,1163 | 0,0541 | **0,465** |
| dd6b8c4b | 726 | 0,1040 | 0,0873 | 0,839 |
| faa9f03d | 864 | 0,0154 | 0,0315 | 2,045 |
| f931b4a8 | 960 | 0,2615 | 0,2742 | 1,049 |
| 21897d95 | 1036 | 0,0292 | 0,0298 | 1,021 |
| 3a25b0d8 | 1180 | 0,0116 | 0,0056 | **0,483** |
| 53fb4810 | 1270 | 0,0087 | 0,0014 | **0,161** |
| 7b3084d4 | 1352 | 0,0994 | 0,0834 | 0,839 |
| b6f77b65 | 1440 | 0,1039 | 0,0766 | 0,737 |
| d59b0160 | 1536 | 0,0182 | 0,0153 | 0,841 |
| 58f5dbd5 | 1594 | **0,0000** | 0,0040 | (÷0: 4000) |
| 89565ca0 | 1720 | 2,2822 | 0,7424 | **0,325** |
| 4a21e3da | 1944 | 0,0365 | 0,0244 | 0,668 |
| 67e490f4 | 2090 | 0,0941 | 0,1386 | 1,473 |
| cb2d8a2c | 2318 | 0,0522 | 0,0501 | 0,960 |

| | valor | listón |
|---|---|---|
| ratio **mediano** r512/r256 | **0,839** | ≤ 0,85 ✅ |
| r512 mejor / peor | **10 / 5** | — |
| test de signo a favor (una cola) | **p = 0,151** | < 0,10 ❌ |

## VEREDICTO por la regla registrada: EMPATE ⇒ el rango no paga ⇒ NO hay comprador ⇒ rama 2B BLOQUEADA

Pasa la magnitud y falla la significancia. La regla exigía las dos, y se mantiene.

## Lo que hay que decir además, sin relitigar

- **`58f5dbd5` tiene post-TTT = 0,0000 en r256**: el TTT la memoriza del todo, y el ratio es una
  división por cero (4000). La mediana lo absorbe; la media (267) es basura. **Sensibilidad,
  etiquetada como tal y NO como veredicto**: excluyendo esa tarea quedan 10 mejor / 4 peor, signo
  p = 0,090 — pasaría el listón. El prerreg no contemplaba excluir denominadores nulos, así que no
  cuenta; pero dice que el resultado está **en el filo**, no en el vacío.
- Las 4 tareas perdidas son las más caras. Con ellas n=19 y el test tendría más potencia en la
  dirección que fuera. El deadline volvió a comerse la cola porque **subestimé el sobrecoste de
  r512 (×1,32 real frente a ×1,15 estimado)**.
- Lectura honesta: **más rango acerca a la GT en 2 de cada 3 tareas, con una mediana de −16%,
  pero no lo suficiente para superar el listón registrado con n=15.** Es "sugerente", no "nulo".

## Qué compra esto

- **Por regla**: la rama Qwen3.5-2B sigue bloqueada. No se gasta 10-20 h-cuota en su SFT sobre
  una señal que no supera su propio prerreg.
- **Por magnitud**: si el efecto real fuera −16% de NLL, la calibración (+20% ≈ −2,5 pt) lo sitúa
  en ~+2 pt — que a su vez no justifica el SFT completo por sí solo. El caso alcista del 2B
  necesitaba que el rango pagara **mucho**, y paga poco o nada.
