# Plan TTT: alinear el entrenamiento por tarea con el objetivo real

Línea abierta el 2026-08-17 por indicación del usuario, tras cerrar la familia de cobertura:
*"si el TTT, como cualquier entrenamiento, se hace sin estar alineado con el objetivo, no va a
dar avance"*. Todo el trabajo vive en **Colab (gratis)** hasta que muestre mejoras grandes en
GT-en-pool sobre la base de train; solo entonces se lleva a submit.

## 1. Por qué aquí y no en velocidad

Descomposición medida del campeón ([ARCHITECTURE.md](ARCHITECTURE.md) §5):

```
30,1%  ≈  cobertura 0,80  ×  GT-en-pool 0,42  ×  selección 0,90
```

- Velocidad/cobertura: techo total **+7,5 pt**. Seis palancas medidas (`ttt8`, `scoring4`,
  `dfsbatch`, `dfswide`, especulación, tope de longitud); las supervivientes rinden ×1,10-1,25
  y la única que llegó a submit **costó −2,5**. Familia agotada.
- Selección: ~0,90, techo +3,3 pt.
- **GT-en-pool: 0,42 → duplicarlo lleva el score a ~60.** Es el único sitio donde queda un ×2.

## 2. La desalineación, en un número

El DFS acepta un candidato solo si la grid **completa** cumple NLL acumulada `< -log(0,2)`,
es decir **p(secuencia) > 0,2**. Para una salida de `L` tokens eso exige una probabilidad
media por token de `0,2^(1/L)`:

| L (tokens de salida) | prob. media por token requerida |
|---|---|
| 100 | 98,4% |
| 372 (mediana del eval) | **99,57%** |
| 929 (grid 30×30) | **99,83%** |

El TTT se entrena con pérdida de token siguiente, que optimiza **NLL media**; el criterio de
producción es **masa de probabilidad de la secuencia entera**, que se endurece exponencialmente
con el tamaño. Un modelo puede tener excelente precisión por token y no cruzar nunca el umbral.

Esto ya explica dos resultados que teníamos sueltos:
- `dfswide` (abrir el umbral a `-log(0,01)`) solo sumó +1/12 de GT: el problema no es el corte,
  es que la masa no se acumula.
- Sin TTT el pool es **vacío** al umbral de producción (medido 16-ago): el TTT no refina el
  pool, es lo que hace que exista.

## 3. Hipótesis pre-registradas

- **H1 (tamaño)**: GT-en-pool cae monótonamente con `L`. Predicción: casi todas las tareas con
  GT en el pool tienen salidas pequeñas; por encima de ~400 tokens la tasa colapsa.
- **H2 (criterio alineado)**: el TTT del campeón **no** consigue que el modelo reproduzca sus
  propias salidas de train bajo el umbral. Medible sin tocar el test: tras el TTT, calcular
  `p(salida_train | entrada_train)` para cada par de la tarea. Si `< 0,2`, el DFS no tenía
  ninguna posibilidad en esa tarea.
- **H3 (entrenar hasta el criterio)**: seguir entrenando el TTT hasta que los pares de train
  crucen el umbral (más épocas / lr / early-stop por criterio, no por pasos) sube GT-en-pool
  en tareas nuevas. Es el experimento central: convierte un entrenamiento a ciegas en uno con
  objetivo medible.
- **H4 (umbral normalizado por tamaño)**: sustituir el umbral absoluto por uno por token
  (`NLL_media < c`) reabre específicamente las tareas de salida grande, que hoy son imposibles
  por construcción. Riesgo: explota el pool en tareas pequeñas → medir coste antes.

## 4. Gate previo: un harness de TTT FIEL en T4

Bloqueante conocido: el port de TTT a Colab en fp16 **empeoraba** la NLL incluso sobre el 4B
original (carril cerrado el 13-ago). Sospecha principal: producción entrena en **bf16**
(`bf16=True, fp16=False`) y la T4 (SM75) no tiene bf16 por hardware; el port usaba fp16 con
GradScaler sobre LoRA r=256 con rsLoRA (escala 2,0), combinación numéricamente frágil.

Fix a probar: **pesos base en fp16, parámetros LoRA en fp32** (master weights), autocast
estándar, sin GradScaler sobre los LoRA.

**Criterio de fidelidad, falsable**: sin TTT el pool al umbral de producción es *vacío*
(medido). Un harness fiel debe producir **pool no vacío** en tareas nuevas tras el TTT. Si no
lo consigue, no se interpreta ningún resultado posterior.

## 4-bis. T2a EJECUTADO (17-ago): la línea base, sin TTT

38 tareas (24 nuevas del holdout + 14 del eval), teacher forcing, sin entrenar nada:

| Métrica | Valor |
|---|---|
| GT bajo el umbral | **0 / 38** |
| Pares de train bajo el umbral (leave-one-out) | **2 / 118** |
| NLL total mediana | **16,79** (umbral 1,609) → hace falta **×10,4** de reducción |
| NLL/token mediana | 0,088 → prob. media por token 0,9155 |

**H2 confirmada en su forma más dura**: sin TTT el modelo no reproduce ni sus *propias*
demostraciones bajo el criterio (2/118). El tamaño del salto que el TTT debe cubrir es de un
orden de magnitud, no un ajuste fino. Confirma desde otro ángulo que *el TTT no refina el pool,
es lo que hace que el pool exista*.

**H1 sale al revés de lo que suponía, y es el hallazgo del día**: la correlación NLL/token vs L
es **−0,612**. Las salidas grandes se modelan MEJOR por token (mediana 0,029 con L>300 vs 0,146
con L≤300, 5× mejor), pero el umbral **absoluto** suma sobre más tokens y las excluye. O sea:
el criterio de producción está sesgado en contra justo de las tareas que el modelo entiende
mejor. Simulación con umbral por token sobre estos mismos datos:

| criterio | pasan | L medio de las que pasan |
|---|---|---|
| absoluto `-log(0,2)` (producción) | 0/38 | — |
| `NLL/token < 0,02` | 6/38 | **614** |
| `NLL/token < 0,03` | 9/38 | 512 |
| `NLL/token < 0,05` | 12/38 | 532 |

⇒ **H4 asciende a palanca principal**, y no es de velocidad: ataca GT-en-pool. Implementación:
sustituir la constante `max_score` de `turbo_dfs` por un **presupuesto lineal**
`max_score(t) = c·t + holgura` — aprieta las salidas cortas y afloja las largas, así que puede
resultar neutro o favorable en tiempo (a diferencia de `dfswide`, que aflojaba uniformemente y
costaba +75%). Pendiente: re-medir con TTT, donde todas las magnitudes se mueven.

## 5. Orden de trabajo (todo en Colab)

1. **T1 — harness fiel** (gate del §4). Sin esto, nada más es interpretable.
2. **T2 — auditoría**: sobre 12-24 tareas nuevas, tras el TTT del campeón medir (a) NLL total
   de la GT, (b) `p(salida_train)` de los propios pares de train, (c) pool al umbral real, y
   (d) todo lo anterior contra `L`. Resuelve H1 y H2 de una vez.
3. **T3 — TTT con criterio**: entrenar hasta que los pares de train crucen el umbral (tope de
   pasos por presupuesto) y re-medir GT-en-pool. Resuelve H3.
4. **T4 — umbral normalizado**: barrido de `NLL_media` vs pool y GT. Resuelve H4.

Regla del usuario para esta línea: **no se sube nada a Kaggle por unos pocos puntos
porcentuales**. Se busca una línea de desempeño alta y sostenida en el banco de train antes de
gastar un slot. Cada avance confirmado puede ir a submit cuando convenga, no antes.

## 6. HALLAZGO (18-ago): el pipeline alimenta al modelo con el formato equivocado

El tokenizador del campeon es **WordLevel, 13 entradas, 0 merges**. Sobre el texto exacto del
formateador:

```
tk('<|im_start|>user\n')  -> [14, 10]      (NO [14, 11, 10])
tk('user') -> []          tk('assistant') -> []
```

Los ids 11 (`user`) y 12 (`assistant`) **existen en el vocabulario pero el tokenizador nunca
los emite**. Eso rompe DOS cosas a la vez:

**(a) El prompt esta fuera de distribucion.** A/B limpio sobre el modelo original, 14 tareas
nuevas, mismo proceso (`scripts/colab_format_ab.py`):

| | produccion (sin roles) | con roles inyectados |
|---|---|---|
| NLL de la GT (mediana) | **18,605** | **0,037** |
| GT bajo el umbral | **0 / 14** | **14 / 14** |
| pool del DFS | **0** | **16 candidatos** |

Factores por tarea entre ×13 y ×590.000. El modelo fue entrenado CON los marcadores de rol
(por eso estan en el vocabulario y por eso el collator los busca), y la inferencia se los quita.

**(b) El TTT no supervisa nada.** `QwenDataCollatorForCompletionOnlyLM` localiza los tramos de
assistant buscando los ids 11/12; al no aparecer, `start_idx` queda vacio y **todas** las
etiquetas quedan en -100. Replicado localmente sobre una secuencia real: 0 de 32 tokens con
perdida. La CE con todo ignorado devuelve **NaN** (medido en GPU).

Esto explica de golpe varios resultados sueltos de la campaña: por que `ttt8` (mitad de
augmentaciones) salio neutro, por que el 34,8% del presupuesto en TTT no compraba calidad
medible, y por que GT-en-pool estaba clavado en 0,42.

**Caveat honesto**: las 14 tareas son del holdout del set de *training*, que el modelo vio en
su SFT. Con el formato correcto las reproduce casi perfectamente (NLL 0,037) — eso es en parte
memorizacion, asi que los valores ABSOLUTOS no se trasladan al test oculto. Lo que si es
estructural es el **desajuste de formato**: el modelo recibe en inferencia algo distinto de lo
que vio en entrenamiento, y eso no depende de la poblacion.

**Fix minimo y quirurgico**: tras `tokenizer.encode(texto)`, inyectar 11/12 despues de cada
`<|im_start|>` alternando user/assistant (validado en el A/B). Afecta al prompt del decode y al
de `calc_scores`. El DFS no puede generarlos (no estan en `ARC_TOKENS`), asi que la salida no
cambia de forma. Arreglar el collator (marcar los tramos por construccion, no por busqueda de
ids) revive el TTT como efecto aparte.

## 7. CONTRASTE EN EVAL (18-ago): el efecto del formato era memorizacion

Mismo A/B, poblacion del **eval publico** (14 tareas, `colab_format_ab.py ... eval`):

| | produccion | con roles | holdout de TRAINING (referencia) |
|---|---|---|---|
| NLL de la GT (mediana) | 16,77 | **13,03** (x1,3) | 18,6 -> 0,037 (x500) |
| GT bajo el umbral | 0/14 | **0/14** | 0/14 -> 14/14 |
| pool del DFS | 0 | **8** | 0 -> 16 |

**Veredicto: NO se sube.** El efecto espectacular del holdout de training era el modelo
recitando su propio SFT con el formato correcto; en poblacion no memorizada el formato mejora
la NLL solo x1,3 y **ninguna tarea cruza el umbral**. Peor: el pool sube de 0 a 8 candidatos
**sin que ninguno sea correcto** — exactamente el mecanismo que costo -3 pt con el 4B fusionado
(pools mas ricos encarecen decode+scoring y hunden la cobertura). Habria sido el cuarto submit
fallido de la semana.

**Lo que SI queda, y es mayor**: el bug del collator no depende de la poblacion. El TTT de
produccion **no supervisa ni un token** (etiquetas todas en -100, CE = NaN medido en GPU), y
eso consume el **34,8% del presupuesto**. Explica `ttt8` neutro, GT-en-pool clavado en 0,42, y
que "con TTT" y "sin TTT" dieran resultados parecidos en nuestros benches.

⇒ Pregunta decisiva, ya en ejecucion: **¿un TTT que SI entrena mueve GT-en-pool en tareas no
memorizadas?** (`colab_ttt_faithful.py ... eval` — el harness construye la mascara a mano, asi
que entrena de verdad). Si sube -> arreglar el collator es la palanca del proyecto. Si no ->
estamos en el techo de la transduccion pura y la unica salida es la rama de induccion.

## 8. CORRECCION (18-ago): el bug de formato era de MI entorno, no de produccion

El log de produccion del campeon (`kaggle kernels output arc-agi2-lb33-graft`, con
`PYTHONUTF8=1` para que baje completo) muestra:

```
[Rank 1] training stats ... training_loss=0.005190  (global_step=128)
[Rank 0] training stats ... training_loss=0.005742
[Rank 2] training stats ... training_loss=0.002902
[Rank 3] training stats ... training_loss=0.000156
```

Perdidas **reales, no NaN**. Si la mascara del collator estuviera vacia la CE seria NaN (lo
medimos en GPU). Por tanto **en Kaggle el tokenizador SI emite los ids 11/12** y el TTT de
produccion supervisa normalmente. La afirmacion de §6(b) es **falsa para produccion**: el
descarte de 'user'/'assistant' es un artefacto del `tokenizers` mas nuevo que trae Colab.

**Consecuencia sobre nuestras propias mediciones**: todos los benches de Colab que tokenizaron
por TEXTO alimentaron al modelo **sin marcadores de rol**, es decir fuera de distribucion. Eso
invalida sus valores ABSOLUTOS:

| medicion | estado |
|---|---|
| T2a: "GT 0/38, brecha x10.4, train 2/118" | ❌ invalida (prompt sin roles) |
| "sin TTT el pool es vacio" | ❌ invalida (era el formato, no la falta de TTT) |
| "el criterio penaliza salidas grandes" (corr -0,612) | ⚠️ a re-medir con formato correcto |
| aceptacion del borrador / especulacion x1.25-1.39 | ⚠️ medido OOD; la conclusion estructural (el lote sincronizado mata la ganancia) probablemente aguanta |
| `lb33-cap` -2,5 pt | ✅ valida (medida en Kaggle) |
| **TTT sobreajusta: train 24/24, GT 0/8 en eval** | ✅ **valida** — ese harness construye los tokens a mano CON roles |

**Lo que queda en pie tras la correccion**: con el formato correcto y un TTT que entrena de
verdad, sobre 8 tareas de eval no memorizadas, el TTT lleva los pares de train a NLL ~0
(24/24 bajo el umbral) y la GT del test **no cruza en ninguna** (mediana empeora x0,56). Es
sobreajuste puro: aprende las demostraciones, no la regla. ⇒ **H3 muerta**: entrenar "hasta
cumplir el criterio en los pares de train" se logra al 100% y no compra nada en test.

**Leccion de metodo**: un entorno de banco que difiere del de produccion en una libreria puede
invertir por completo la lectura. Antes de declarar un hallazgo sobre produccion hay que
verificarlo CONTRA produccion — aqui bastaba leer el log del campeon, que era gratis y estuvo
disponible todo el tiempo.

## 9. El 2x2 (19-ago): LoRA global y TTT, cruzados, con formato correcto

Diseño del usuario ("el tercer camino"): 2B como **vehiculo de medida**, dos bases (original y
fusionado con el SFT de ARC-Heavy) x dos brazos (sin/con TTT), 8 tareas nuevas de holdout,
tokens construidos a mano CON marcadores de rol. `scripts/colab_2b_lora_ttt.py`, 2181s en T4,
resultados en el dataset `arcscale-2b-lora-ttt`.

| base | sin TTT | con TTT | NLL mediana | coste TTT |
|---|---|---|---|---|
| original | **5/8** | **5/8** | 0,029 → 0,095 | ~103 s/tarea |
| fusionado | 3/8 | 4/8 | 2,774 → 1,972 | ~103 s/tarea |

Lectura, tarea por tarea (brazo `original`, que es el del campeon): el TTT **gana** `9caf5b84`
(NLL 2,625 → 0,105, cruza el umbral) y **pierde** `b2bc3ffd` (0,022 → 1,866, lo saca del pool).
No es que no haga nada: mueve tareas en las dos direcciones y el neto es cero. La NLL mediana
**empeora** (0,029 → 0,095): el TTT desplaza la distribucion incluso donde el modelo ya acertaba.

Sobre el fusionado el TTT si suma (3 → 4, NLL mediana 2,774 → 1,972), coherente con §8: cuando
el prior es peor, adaptar compra mas. Pero el fusionado parte tan por debajo del original que
la suma no lo alcanza — misma conclusion que cerro la rama de escala en el LB.

**Tercera muestra independiente que apunta al mismo sitio.** Ninguna de las tres (T2a corregido,
TTT fiel sobre eval 0/10 = 0/10, este 2x2 5/8 = 5/8) encuentra un aporte neto del TTT, y cuesta
**34,8% del presupuesto de produccion**. De ahi que `lb33-nottt` sea la submission de mayor
valor esperado en cuanto vuelva la quota.

Cautela honesta: n=8 y n=10 son muestras pequeñas; ninguna sola excluye un efecto de +1-2 pt.
Lo que excluyen las tres juntas es el efecto GRANDE que justificaria un tercio del presupuesto.

## 10. La corrida de EPOCAS del LoRA (19-ago) y el fallo de infra que costo 4h

Pregunta del usuario: entrenar el LoRA global, ver que no acierta, **seguir entrenando** — ¿con
mas pasadas se llega al objetivo, o solo satura? Nunca se midio: D1/D2 barrieron el VOLUMEN de
datos (N = 2000 → 4000 → 8000: exactas 7 → 8 → 8, saturan), no las PASADAS sobre un corpus fijo.

`scripts/colab_lora_epochs.py`: 2B + LoRA r=64 (config del SFT) sobre 2000 ejemplos fijos de
ARC-Heavy, 3 epocas, midiendo en la epoca 0/1/2/3 sobre 10 tareas nuevas (NLL de la GT y si el
DFS la encuentra al umbral de produccion). Es entrenamiento **continuo**: la epoca 2 son los
pesos de la 1 seguidos entrenando, que es exactamente el "cargar los mismos pesos y seguir" que
pide la decision sobre saturacion. Prediccion pre-registrada: **satura en la epoca 1-2**.

**Primer intento: perdido entero.** 4h de T4 (timeout exacto a 14400s), **cero lineas de salida**
y cero resultados. Causa: `colab run` ata la corrida al canal de salida del cliente; cuando ese
canal se rompe, expira el timeout, la VM se libera y se pierde todo — y la persistencia estaba
**solo al final**. Agravante: el ADC activo era de **cuenta4 (de agi-3)**, no de las cuentas 1-3
de este proyecto, asi que ademas se quemo la quota del proyecto vecino.

Tres correcciones, ya aplicadas:
1. **Persistencia incremental**: `persistir()` sube la curva parcial a Kaggle tras CADA medida.
2. **Guardado del adapter** por epoca (`/content/persist/adapter`, sobrescrito) — permite
   continuar en la epoca 4/5 sin repetir las anteriores.
3. **Ejecucion desacoplada**: `colab new` + `upload` + `exec` de un lanzador que hace
   `Popen(start_new_session=True)` a `/content/log.txt`, y sondeo con llamadas de segundos. El
   cliente ya no puede matar la corrida.

**Segundo intento: el techo real de Colab libre son ~65 min de VM.** Con la plomeria nueva la
corrida arranco bien y persistio la linea base:

```
[MEDIDA epoca0_base] resueltas 6/10 | NLL mediana 0.142 | pool 8
[persist] epoca0_base subido a Kaggle          <- ya no se pierde nada
  [epoca 1] 500/2000 loss 0.1103 (927s)
  [epoca 1] 1000/2000 loss 0.1200 (1882s)
  [epoca 1] 1500/2000 loss 0.1676 (2822s)
```

y a los **65 min** la VM fue reclamada a mitad de la epoca 1 (`Session appears to be lost
(404/401)`). Numeros que salen de ahi y que valen para planificar cualquier corrida futura:
**1,85 s/ejemplo** de entrenamiento en T4 (⇒ 62 min por epoca con N=2000, imposible en una
sesion) y **~7 min por ronda de medida** (10 tareas, NLL + DFS con presupuesto 45 s).

Rediseño en consecuencia: **N=500** (epoca ≈ 15 min, epoca+medida ≈ 22 min ⇒ dos epocas por
sesion) y corrida **REANUDABLE** — `EPOCA_INI=k` descarga el adapter de la epoca k del dataset
de Kaggle, lo carga con `set_peft_model_state_dict` y sigue, arrastrando las medidas previas.
Nota metodologica: el estado del optimizador NO se persiste, asi que en cada frontera de sesion
el momento de AdamW se reinicia; como las fronteras coinciden con fronteras de epoca, el efecto
es pequeño, pero esta anotado por si la curva muestra escalones justo ahi.

**Dato sustantivo ya visible, antes de la curva**: la perdida de entrenamiento arranca en
**0,11** y la NLL de la GT en holdout es **0,142** con el modelo base. El 2B ya SFT-eado
**modela ARC-Heavy casi perfectamente desde el primer paso** — no hay apenas perdida que
reducir. Refuerza la prediccion pre-registrada (satura en la epoca 1-2) y explica por que el
barrido de VOLUMEN de D1/D2 saturaba: el cuello no es cuanto se entrena sobre este corpus.

**Aviso de lectura**: la base ya resuelve **6/10**, asi que solo quedan 4 tareas por ganar y
una vale 10 puntos porcentuales. La señal fiable sera la **NLL mediana**, que se mueve de forma
continua; si baja epoca a epoca sin que suban las resueltas, la lectura correcta es "mejora
real que no alcanza a cruzar el umbral", que no es lo mismo que "no mejora".

### 10-bis. RESULTADO: la curva no satura, DESTRUYE (y el porque no es el que parecia)

| etapa | resueltas | NLL mediana | pool | loss (mismos 200 ejemplos) |
|---|---|---|---|---|
| base | **6/10** | **0,142** | 8 | — |
| epoca 1 | 3/10 | 2,996 | 3 | 0,11971 |
| epoca 2 | **2/10** | **9,089** | 2 | 0,12471 |

Monotono y sistematico: en la epoca 1 empeoran **9 de las 10 tareas**, incluidas las que la base
clavaba (6bcdb01e 0,007→3,738; b2bc3ffd 0,022→2,254; 4df5b0ae 0,247→9,246). La prediccion
pre-registrada era "satura en la epoca 1-2"; el resultado la refuta por el lado malo — no
satura, degrada. **Pregunta del usuario respondida: dos pasadas no son mejores que una, y una
ya es peor que ninguna.**

**El dato que reinterpreta todo lo anterior**: `TRAIN` se baraja UNA vez y no se rebaraja entre
epocas, asi que `loss_final` (media de las ultimas 200 perdidas) cubre **los mismos 200
ejemplos** en ambas pasadas. Una segunda pasada sobre datos identicos deberia BAJAR su perdida.
**Subio: 0,11971 → 0,12471.** Eso no es desplazamiento de distribucion, es el optimizador sin
converger: **lr 1e-4 constante, sin warmup, sobre un modelo que ya esta en perdida 0,11 es
demasiado caliente**, y cada paso salta por encima del minimo.

Y esa es **exactamente la receta de D1/D2/D4** (`colab_data_scaling.py:237`: AdamW lr=1e-4,
ACCUM=8, sin scheduler; identica aqui) — toda la familia de escala que fracaso en 27,08 y 28,33.

⇒ La conclusion correcta NO es "las epocas no sirven" sino: **a lr 1e-4 el entrenamiento es
destructivo, y hay prueba interna de que 1e-4 es demasiado alto**. Falsable y barato: barrido de
lr a UNA epoca (`LRS=2e-5,5e-6` en el mismo script, con `SALTAR_BASE=1`), con la base ya medida
dos veces identicas como control. La pregunta del barrido es la correcta, no "¿cual daña menos?"
(a lr→0 se recupera la base trivialmente) sino **¿existe algun lr al que ARC-Heavy MEJORE sobre
6/10 o 0,142?** Si ninguno lo hace, la rama se cierra con una razon medida en vez de una
sospecha.

### 10-ter. CONTROL DE FORMATO: la curva de escala de ARC-Heavy medía reparación, no capacidad

`SIN_ROLES=1` mide el MISMO modelo base sobre las MISMAS 10 tareas quitando del prompt
unicamente los ids 11/12. Es la unica manipulacion, asi que la poblacion queda controlada:

| base | resueltas | NLL total (mediana) | pool |
|---|---|---|---|
| CON marcadores de rol | **6/10** | 0,142 | 8 |
| SIN marcadores | **0/10** | 18,567 | 0 |

**CUIDADO CON LAS UNIDADES** (error que cometi y corrijo): `colab_data_scaling.py` promedia por
token (`.mean()`) y este script suma sobre toda la rejilla (`.sum()`). Convertido a NLL/token
con la longitud real de cada solucion:

| | NLL/token (mediana) |
|---|---|
| base CON roles | **0,0013** |
| base SIN roles | **0,2013** |
| base de D1 (sin roles por construccion) | **0,327** |

La base de D1 cae **junto a** la base sin roles de hoy, y **150x peor** que la base con formato
correcto. Y el remate: el mejor punto que D1 alcanzo entrenando (0,055/token) sigue siendo
**~40x peor que el modelo SIN ENTRENAR con formato correcto** (0,0013).

⇒ **La curva de escala de ARC-Heavy (D1: 1/12→5→7→7) estaba saliendo de un agujero que cavo
nuestro propio harness.** No midio que ARC-Heavy diera capacidad; midio que el LoRA aprende a
operar sin marcadores de rol. Eso explica que D4 —el SFT del 4B construido sobre esa evidencia—
se desplomara a 27,08 en produccion, que si recibe el formato correcto: se entreno para reparar
un problema que produccion no tiene, y ademas a un lr que daña.

**Cabo suelto honesto**: el 2B fusionado SUBIO +2,2 en el LB oculto (19,86 / 21,25 vs 18,33), y
eso paso por produccion con formato correcto. Hipotesis no medida: el TTT de produccion (r=256,
rango completo sobre embed/lm_head) repara parte del daño del SFT — es lo que muestra §9, donde
el TTT ayuda sobre la base fusionada (3→4) y no sobre la original (5→5) — y en el 2B, lejos de
su techo, el neto sale positivo mientras que en el 4B sale negativo.

**Consecuencia sobre el plan**: el barrido de lr pierde casi todo su valor. Sea cual sea el lr,
entrenar ARC-Heavy tendria que superar una base que ya esta en 0,0013/token; el mejor caso
realista es "no dañar", no "mejorar". El modo `LRS` queda en el script por si hace falta, pero
la prioridad vuelve al TTT con pasadas, que es donde vive el 34,8% del presupuesto.
