# Prerregistro — `r-buyer`: ¿el acantilado de `r512` fue el RELOJ o la CAPACIDAD?

Escrito **antes** de correr. Fecha: 2026-09-09. Presupuesto: ~1,0-1,2 h-cuota (L4×4, shard ×4).

## Por qué esta pregunta bloquea todo

`gate-base` dejó viva una rama con upside de dos dígitos: **Qwen3.5-2B empata el prior del
titular** (ratio 0,989) con **0,378×** de tronco. Liberar ~50% de reloj a 0,6 tareas por 1%
serían **~30 tareas ⇒ +12,4 puntos**.

Pero eso exige que **el reloj liberado tenga en qué gastarse**, y las dos formas medidas de
gastarlo dan **+0**: más vistas (16 vistas = 1 hipótesis, redundancia 1,8×) y más hipótesis
(`ttt-2hip`: dos adaptaciones divergen, Jaccard 0,26, unión suma cero). El único candidato que
queda es el **rango del adaptador**, y ahí hay un confundido sin resolver: `r512` perdió 15,3
tareas en el leaderboard, pero también añadió +11,9% de coste por tarea sobre un pipeline que
corre al 95% de su deadline. **¿Perdió por el reloj o porque un adaptador sobre-parametrizado
memoriza aún más las demos?** La loss del TTT ya está saturada en ~1e-3, así que la segunda
explicación es perfectamente plausible y nunca la separamos.

## Diseño

Pareado, **sin presión de reloj**: ningún deadline global obliga a saltarse tareas; cada tarea
recibe su TTT completo (128 secuencias, calendario exacto de producción) y sus 16 vistas de
decode en **ambos** brazos.

- **Brazo A**: `r=256, lora_alpha=32` — la configuración del campeón.
- **Brazo B**: `r=512, lora_alpha=45` — mantiene la escala rsLoRA idéntica (`alpha/sqrt(r)=2,0`),
  que es exactamente el brazo que corrió en el leaderboard.
- Modelo de producción (4B `sft139`), formato con marcadores de rol, mismas semillas por tarea.
- **Población**: 12 tareas de eval limpias de AGI-1 **estratificadas por coste a lo largo de toda
  la distribución** — no las 12 más baratas (lección `lb33-cap`, −2,5 pt).
- 2 GPUs por brazo, mismo orden de tareas en ambos: si el deadline del worker trunca la cola,
  trunca igual en los dos y se emparejan por intersección.
- Presupuesto de DFS por vista: **45 s en ambos brazos** (el de producción). Subirlo confundiría
  búsqueda con capacidad.

## Medidas

- **Primaria**: `vistas_con_gt` (0-16 por tarea) — el GT en el pool. Es el resultado, no un proxy.
- **Secundaria**: NLL de la GT bajo el modelo ya adaptado (continua, más potencia; mecanismo).
- **Instrumental**: `ttt_s` y `decode_s` por brazo. ⚠️ El banco usa *gradient checkpointing* y
  producción no, así que este ratio **no** es el de producción (ese ya lo tenemos del LB: +11,9%).

## Regla de decisión (registrada)

| resultado en la primaria | lectura | veredicto |
|---|---|---|
| `r512` mejor (Δ mediano > 0, y gana más tareas de las que pierde) | el acantilado fue el **reloj** | **hay comprador** ⇒ rama 2B viva |
| `r512` peor con test de signo p<0,05 | el rango está **saturado**, sobreajusta las demos | **no hay comprador** ⇒ rama 2B muerta |
| empate (Δ mediano = 0, sin diferencia) | el rango **no paga** | **no hay comprador** ⇒ rama 2B bloqueada |

**El empate cuenta como negativo, y queda registrado así antes de ver el dato**: la tesis no
necesita que el rango *no estorbe*, necesita que **pague**. Comprar reloj para gastarlo en algo
que empata no compra puntos.

⚠️ **Potencia**: n=12 pareado con salida graduada 0-16 detecta efectos moderados, no pequeños. Un
resultado nulo es *"sin evidencia de comprador"*, no *"prueba de que no lo hay"*. Operativamente
da igual: la carga de la prueba recae sobre la rama cara (10-20 h-cuota de SFT, prior 0-de-2).
La secundaria solo explica mecanismo; **no cambia el veredicto**.
