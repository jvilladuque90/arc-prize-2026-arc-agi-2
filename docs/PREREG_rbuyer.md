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

---

# RESULTADOS (2026-09-09) — ~1,6 h-cuota (48 min de pared, más de lo presupuestado)

## Primaria: empate

| | valor |
|---|---|
| pares logrados | **7** (de 12 planeados) |
| Δ `vistas_con_gt` mediana | **+0,0** (media −0,29) |
| mejora / empata / empeora | **1 / 5 / 1** |
| test de signo | **p = 1,000** |
| vistas con GT totales | r256 **17** → r512 **15** |
| tareas con GT en algún pool | 2 → 2 de 7 |

| tarea | coste | gt r256 | gt r512 | Δ | nll/celda r256 | nll/celda r512 |
|---|---|---|---|---|---|---|
| e8686506 | 403 | 5 | 8 | **+3** | 0,1662 | 0,0046 |
| dd6b8c4b | 726 | 0 | 0 | 0 | 0,0847 | 0,0996 |
| f931b4a8 | 960 | 0 | 0 | 0 | 0,2548 | 0,3353 |
| 38007db0 | 1102 | 12 | 7 | **−5** | 0,0208 | 0,6244 |
| 4e34c42c | 1345 | 0 | 0 | 0 | 0,1883 | 0,1816 |
| 8698868d | 1528 | 0 | 0 | 0 | 0,0737 | 0,0779 |
| 9bbf930d | 1620 | 0 | 0 | 0 | 0,0443 | 0,0438 |

**Secundaria** (NLL/celda de la GT, mecanismo): ratio mediano r512/r256 = **1,057**, mejor en 3 y
peor en 4. Tampoco hay dirección.
**Instrumental**: ×1,15 en el banco (con *gradient checkpointing*; producción midió +11,9%).

## VEREDICTO por la regla registrada

> `empate` ⇒ el rango **no paga** ⇒ **no hay comprador** ⇒ rama 2B **bloqueada**

Se registró el empate como negativo antes de ver el dato, y se mantiene.

## Pero la evidencia es MUCHO más débil de lo planeado, y hay que decirlo

**5 de los 7 pares tienen 0 vistas con GT en los dos brazos**: no aportan ninguna información
sobre la primaria. La medición real son **2 observaciones**, una a favor (+3) y otra en contra
(−5). Eso no es una medida, es ruido.

Y los dos casos informativos se mueven en direcciones opuestas con magnitudes enormes en la
secundaria: `e8686506` mejora ×36 (0,1662 → 0,0046) y `38007db0` empeora ×30 (0,0208 → 0,6244).
Es **compatible** con que `r512` sea un estimador de mayor varianza más que uno peor — pero con
n=7 eso es una hipótesis, no un hallazgo.

## El fallo de diseño, y la conclusión estructural que sí vale

Los cuatro workers agotaron `WALL_MAX` tras 3-5 de sus 6 tareas (y lo rebasaron, 2.123-2.799 s,
porque la comprobación va entre tareas). **La estratificación y el deadline se pelean**: estratificar
mete la cola cara, y el deadline se come exactamente las tareas que la estratificación añadió.
Hay que dimensionar el deadline *sobre el coste de la cola*, no sobre la media.

Y lo más importante: **`GT-en-pool` dispara en ~30% de las tareas, así que la mayoría de los pares
sale 0-0 y no informa.** Para detectar un efecto moderado del rango harían falta **n ≈ 40-60 pares
≈ 8-10 h-cuota** con este arnés — más que el presupuesto semanal entero.

**La pregunta del comprador no es contestable a un coste que podamos pagar con esta métrica.** La
rama 2B no queda bloqueada por un negativo sólido, sino por **medición inasequible**.

## Lo único que abarataría el futuro: validar el proxy

La NLL/celda de la GT bajo el modelo adaptado informa en **todas** las tareas, no solo en el 30%.
Evidencia acumulada aquí (n=16 observaciones, los dos brazos juntos):

- Spearman(nll/celda, `vistas_con_gt`) = **−0,350** — el signo correcto, débil.
- nll/celda mediano: **0,0262** con GT en algún pool (n=5) frente a **0,0847** sin GT (n=11): una
  separación de **×3,2**.

Direccionalmente sirve; no está validado. Validarlo con datos ya pagados (los de `augdose` más
estos) es gratis y multiplicaría por ~10 la potencia por hora-cuota de todo banco futuro de rango
o de modelo. **Ese es el siguiente paso barato, y no gasta GPU.**
