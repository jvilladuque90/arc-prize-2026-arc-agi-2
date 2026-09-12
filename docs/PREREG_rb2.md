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
