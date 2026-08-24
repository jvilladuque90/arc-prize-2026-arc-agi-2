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

## 11. TTT con PASADAS (19-ago): tres pasadas para volver al punto de partida

`scripts/colab_2b_epochs.py`, 6 tareas nuevas de holdout, midiendo tras CADA epoca de TTT.
Base `original` (la configuracion del campeon):

| tarea | base | pasada 1 | pasada 2 | pasada 3 | veredicto |
|---|---|---|---|---|---|
| 6bcdb01e | 0,007 ✓ | 0,002 ✓ | 0,001 ✓ | 0,001 ✓ | sigue bien |
| 9caf5b84 | 2,625 | 2,267 | 2,274 | **0,481 ✓** | **GANA** |
| 84551f4c | 2,582 | 7,416 | 14,246 | 13,760 | sigue mal |
| 5ad8a7c0 | 0,004 ✓ | 0,004 ✓ | 0,002 ✓ | 0,002 ✓ | sigue bien |
| b2bc3ffd | 0,022 ✓ | 7,791 | 9,922 | 10,181 | **PIERDE** |
| 25e02866 | 0,001 ✓ | 0,001 ✓ | 0,000 ✓ | 0,000 ✓ | sigue bien |

**4/6 sin TTT = 4/6 con TTT.** Curva por pasadas 3 → 3 → 4: la tercera pasada solo sirve para
**volver al nivel que ya habia sin TTT**, nunca lo supera, y cuesta 224-400 s/tarea frente a los
~103 s de una pasada. ⇒ **Mas pasadas de TTT tampoco son la palanca.** Cuarta muestra
independiente con el mismo veredicto (T2a corregido, TTT fiel 0/10=0/10, 2x2 5/8=5/8, esta
4/6=4/6).

Dos observaciones nuevas:

**(a) El efecto es DETERMINISTA, no ruido.** Gana y pierde exactamente las mismas dos tareas que
el 2x2 de §9 (rescata 9caf5b84, hunde b2bc3ffd) en dos configuraciones distintas (1 pasada con
n_aug del campeon vs 3 pasadas). Y las tres tareas que la base clava con NLL 0,001-0,007 salen
intactas. El TTT no es neutro por no hacer nada: es neutro **porque se compensa**.

**(b) El TTT infla el pool sin acertar mas**: 5 → 8 candidatos en la base original con las mismas
4 correctas; 4 → 9 en la fusionada con 1 correcta mas. Añade candidatos equivocados y diluye la
seleccion, que es el factor 0,90 de la descomposicion del score. Argumento adicional para
`lb33-nottt`: quitar el TTT no solo libera 34,8% del presupuesto, tambien deja de contaminar el
pool.

### 11-bis. Idea derivada (HIPOTESIS, no resultado): TTT SELECTIVO por tarea

Si se salta el TTT en las tareas que la base ya resuelve con confianza y solo se aplica donde
falla: **5/6 en vez de 4/6** aqui, y **6/8 en vez de 5/8** en el 2x2. Con el reparto de tiempo
de produccion (TTT 34,8% / decode 57,2%) el coste sale casi igual — decodificar primero sin TTT
y parar si el candidato es confiable ahorra el TTT en la mayoria de tareas, a cambio de un
segundo decode en la minoria.

**Debilidad, por delante**: ese +1 descansa en **dos tareas concretas** (9caf5b84 y b2bc3ffd),
medidas dos veces cada una. Son n=2 tareas distintas, no 14. Ademas el disparador tiene que ser
OBSERVABLE sin conocer la respuesta — la confianza del mejor candidato sin TTT — y ese numero
**no se registro** (`dfs` devuelve las hojas pero no sus scores). Medirlo es barato y es el
siguiente paso natural; hasta entonces es una hipotesis.

## 12. EL DISPARADOR EXISTE (19-ago): separacion perfecta en el tramo barato

Idea del usuario: un **TTT auditado en cascada** — una pasada barata retira las tareas faciles y
el presupuesto liberado se concentra en las dificiles. Para que el tier 1 sea seguro hace falta
un disparador observable. `scripts/colab_tier1.py` registra el **score acumulado de cada hoja**
del DFS (el DFS de produccion se queda con la hoja y tira el score), sin TTT, sobre las 24
tareas mas baratas del holdout no-AGI1.

| grupo | n | rango de `best_score` |
|---|---|---|
| **correctas** | **16** | 0,0008 – **0,4800** |
| **confiadas y EQUIVOCADAS** | **4** | **1,1088** – 1,4996 |
| sin candidato | 4 | — |

**Separacion perfecta. τ=0,5 retira 16/24 con precision 1,000 Y recall 1,000**: no se equivoca en
ninguna y no deja fuera ninguna resoluble. Hueco entre el peor acierto y el mejor error: **2,3x**.

Barrido completo: τ ∈ {0,001 … 1,0} da precision 1,000 siempre (retirando de 1 a 16 tareas). Solo
al llegar al **umbral que usa produccion, 1,609, la precision cae a 0,80** — deja entrar
exactamente esos 4 candidatos equivocados.

**Mecanismo, no coincidencia**: cuando el modelo acierta, el score de su mejor candidato ES la
NLL de la verdad (0,0072 vs 0,007; 0,4800 vs 0,476 — coinciden en las 16). Cuando se equivoca,
su mejor candidato vive en 1,1-1,5 mientras la verdad esta en 2,0-3,2. **El modelo no esta igual
de seguro cuando se equivoca**, y esa es una propiedad del modelo, no de esta muestra.

**LA ADVERTENCIA, que es lo que decide si esto llega a un submit**: estas 24 son las mas baratas
del holdout y el modelo resuelve **16/24 = 67% sin ningun TTT**, frente al ~30% de produccion en
el test real. Poblacion dramaticamente mas facil. Sobre el test el tier 1 retiraria muchas menos
(el ahorro caeria de ~23% a quiza ~10%) y, peor, **en tareas dificiles los candidatos CORRECTOS
son menos confiados y se mueven hacia la banda de los erroneos: el hueco puede cerrarse**. Es
exactamente el modo de fallo de `lb33-cap` (−2,5 pt: umbral validado en una poblacion, aplicado
en otra). Por eso lo siguiente que se mide es la **franja 24-48 por coste** (`OFFSET=24`), que
es lo unico que puede romper el umbral.

Coste medido del tier 1: **282 s de decode para las 24** (11,8 s/tarea); las 8 supervivientes
suman 116 s, que es el unico decode que se pagaria dos veces.

### 12-bis. CORRECCION y ampliacion: la poblacion, el eval, y la prueba contra PRODUCCION

**Correccion**: dije "las 24 mas baratas". Falso — el holdout no-AGI1 del train tiene **exactamente
24 tareas** (train AGI-2: 1000 total, 112 en la banda de holdout, 24 fuera de AGI-1). `[:24]` las
coge todas. La advertencia sobre "tramo facil" era infundada por ese lado.

Pero al verificarlo aparece algo peor: son tareas del conjunto de ENTRENAMIENTO de AGI-2 y el
modelo base es de terceros. NLL de 0,0008 no es confianza, es **memorizacion**. Si los aciertos
son memorizaciones, su confianza extrema es un artefacto y la separacion podria serlo.

**Eval AGI-2 no-AGI1 (40 tareas)** — el test buscaba aciertos con confianza INTERMEDIA, que en el
train no habia ninguno:

| grupo | n | rango |
|---|---|---|
| correctas | 32 | 0,0002 – **0,6966** |
| confiadas y EQUIVOCADAS | 3 | **0,8643** – 1,2273 |
| sin candidato | 5 | — |

Aparecieron los aciertos intermedios (0,6266 y 0,6966, muy por encima del peor acierto del train)
**pero los errores se movieron con ellos**: el hueco se mantiene, ahora 1,24x en vez de 2,3x.
Juntando las dos poblaciones (64 tareas): peor acierto **0,6966**, mejor error **0,8643** ⇒
**τ=0,75 separa 48 aciertos de 7 errores sin fallar ni una vez**.

Salvedad: el eval resuelve **32/40 = 80%** (el train, 67%) frente al ~30% de produccion. Las dos
poblaciones estan casi con certeza vistas por el modelo.

**LA PRUEBA CONTRA PRODUCCION (gratis, del log del propio campeon)**: `ArcDecoder` guarda el
`beam_score` de cada candidato y `benchmark_selection_algos` imprime el resumen. En
`arc-agi2-lb33-baseline`:

```
avg correct beam score:  0.03724
max correct beam score:  0.48566
```

36 candidatos correctos, **todos por debajo de 0,486**, mediana 0,00082, p90 0,208. Es el stack
REAL (4B + TTT + augmentaciones), no nuestro banco de 2B, y da la misma frontera.

**Que prueba y que NO prueba**: el log solo imprime lineas `ALL_CORRECT`, asi que confirma
**recall** (todo acierto tiene score bajo) pero **no dice nada de precision** — no vemos los
scores de los candidatos equivocados de produccion. La precision la sostienen solo nuestras 64
tareas. La combinacion es: recall validado en produccion con el modelo grande, precision medida
en el 2B sobre dos poblaciones.

**Consecuencia inmediata y barata**: el umbral de aceptacion del DFS del campeon es **1,609**,
mas de 3x por encima del peor acierto que ese mismo kernel produjo (0,486). Esta admitiendo
candidatos que **nunca** son correctos: gasta busqueda en ellos y ademas diluye el pool. Bajarlo
podaria mas fuerte sin perder nada de lo observado.

⚠️ **Pero es la MISMA FORMA de argumento que hundio a `lb33-cap`** (−2,5 pt): "es monotono, no
puede perder". Alli fallo por desplazamiento de poblacion (el cap ataba el 54% de las subtareas
del test contra el 16% del eval). Aqui el riesgo gemelo es que en el test oculto —mas dificil—
un acierto tenga score 0,9 y un umbral de 0,6 lo pode. Mitigacion: elegir τ con margen (≈1,0,
que sigue podando mucho respecto a 1,609 y deja 2x sobre el maximo observado) y tratarlo como
A/B en el LB, no como verdad asumida.

### 12-ter. El intento con el 4B: NULO (medicion rota, no resultado)

`MODELO=4b` sobre 16 tareas de eval dio **0/16 correctas**, con NLL de la verdad entre 5 y 28 en
tareas donde el 2B marca 0,01-0,7. **No es un resultado, es una medicion rota**: el 4B es el
campeon a 30,14 y el 2B ronda 18-21; no puede sacar cero donde el pequeño acierta el 80%.

Firma del fallo, que apunta a la CARGA del modelo y no al umbral:
- **12 de 16 tareas con pool 0 y solo 4-9 s de decode**, lejos del presupuesto de 45 s ⇒ no se
  quedo sin tiempo: **ningun token paso el umbral en la primera posicion**, o sea distribucion
  de salida plana o desplazada.
- `gt_nll` es teacher-forced (un forward, sin busqueda) ⇒ no depende del presupuesto. Que salga
  15-28 donde el 2B saca 0,03 solo se explica por el modelo.
- Excepcion: `bf45cf4b` da gt_nll 0,033 pero pool 0 agotando los 45 s. Es el unico caso
  compatible con "modelo bien, busqueda lenta", asi que el diagnostico no es del todo limpio.

Causa mas probable: se carga con `torch_dtype=torch.float16` un checkpoint distribuido en
**bfloat16**, y **la T4 no soporta bf16 nativo** (capacidad 7.5; hace falta 8.0). Al 2B le sento
bien; el 4B tiene rangos mayores y la conversion puede desbordar. Alternativa: que el 4B espere
alguna variacion de formato que el 2B tolera.

**Antes de volver a correrlo**: (1) cargar en float32 o usar L4/A100 con bf16; (2) validar con un
gate barato — NLL de la verdad en UNA tarea que el 2B clava; si no baja de ~1, abortar sin gastar
la sesion. Es el mismo failsafe que ya nos salvo en el commit de E1-v2.

⇒ **La casilla de precision del 4B sigue VACIA.** El cambio de umbral del DFS no debe ir a un
slot hasta cerrarla.

### 12-quater. El smoke de lb33-nottt (22-ago) CORRIGE la lectura del umbral

Commit de `lb33-nottt` COMPLETO: **smoke reload 3,0/4 vs 2,5 historico del campeon** con ambos
algoritmos de seleccion — pipeline intacto sin TTT, submitido (55680284, PENDING).

Pero su log añade el dato que faltaba y que CORRIGE §12-bis: **sin TTT, los candidatos
correctos del 4B suben de score**. En el smoke (mismas tareas que el campeon usa de reload):

- con TTT (log del campeon): max correct beam score **0,486**
- sin TTT (log de nottt): correctos en **1,55 / 1,48 / 1,44 / 1,28** (aa4ec2a5, 36a08778...)

Mecanismo obvio en retrospectiva: el TTT adapta el modelo a la tarea y AFILA su confianza en lo
que acierta; sin TTT el mismo acierto sale menos confiado. ⇒ **el umbral y el TTT interactuan**:

1. "1,609 esta mal puesto" era cierto **para el campeon CON TTT** (3x sobre su max de 0,486).
2. Para el pipeline SIN TTT — que es el que acabamos de submitir — **1,609 esta casi bien
   calibrado**: hay aciertos en 1,55, y τ=0,75 o 1,0 los podaria.
3. Matiz: esos scores son por-augmentacion y la seleccion agrega sobre augmentaciones (aa4ec2a5
   tambien aparece en 0,71/0,91), asi que una poda a 1,0 debilita pero no necesariamente mata al
   candidato. Aun asi, el margen de seguridad que §12-bis daba por sobrado ya no lo esta.

**Regla que queda**: si nottt se convierte en campeon, el τ de poda se deriva de SU log (los
ALL_CORRECT del kernel que realmente corre), no del log del campeon viejo ni del banco 2B. Cada
pipeline calibra su propio umbral — cuarta vez que el instrumento es parte del experimento.

## 13. tier1-4b (22-ago): la casilla se cierra EN CONTRA, y reinterpreta el banco entero

Kernel `arc-agi2-tier1-4b` (L4, bf16 nativo, gate PASADO con 0,472 < 1,0 — el modelo cargo
bien), las MISMAS 40 tareas de eval del banco 2B:

| | 2B | 4B |
|---|---|---|
| correctas | 32/40 | **3/40** |
| peor acierto | 0,6966 | 0,8742 |
| mejor error | 0,8643 | **0,5800** |
| separacion | limpia (hueco 1,24x) | **SE SOLAPA** (τ=0,75 → precision 0,333) |

**(a) La cascada/tier-1 muere para el 4B.** Los errores confiados del 4B (0,58-0,71) viven por
debajo de aciertos (0,87). No hay τ que separe en el modelo que realmente se envia. El barrido
completo: precision 1,0 solo hasta τ=0,5 reteniendo 1 sola tarea; a 0,75 cae a 0,333.

**(b) El 3/40 vs 32/40 reinterpreta TODO el banco 2B.** El 2B resolvia el 80% del eval con NLL
0,0002 porque LO TIENE MEMORIZADO (checkpoint sft141); el 4B (sft139), con gt_nll de 4-18 en las
mismas tareas, no. Consecuencia grave: **las "4 muestras de TTT neto cero" se midieron sobre
poblaciones donde la base 2B ya sabia las respuestas** — sin margen para que el TTT aportara. En
el 4B, cuya base saca 3/40 en tareas no memorizadas, el TTT + augmentaciones + 1200 s/tarea
pueden ser exactamente lo que lo sube al ~30% de produccion.

**(c) El submit de nottt (55680284, en vuelo) queda como el experimento DEFINITIVO**: A/B limpio
del valor del TTT en el test oculto contra 30,14. Las lecturas prerreg ya cubrian este giro:
<28 = el TTT era esencial y el banco 2B engañaba por contaminacion. El smoke 3,0/4 que valido el
submit era sobre tareas de TRAIN (memorizadas por el 4B) — valida el pipeline, no el techo.

**(d) Los experimentos pendientes de Colab (poda 2B, TTT selectivo 2B) pierden valor**: median
propiedades de un modelo contaminado en poblaciones vistas. No gastar en ellos hasta ver el
score de nottt.

Nota de coste: el 4B en L4 decodifico las 40 en 397 s (menos que el 2B en T4, 581 s) porque 25
pools salieron vacios — la base sin TTT ni augmentaciones apenas encuentra candidatos en eval.

## 14. La NLL revisada (22-ago): el defecto esta en la AGREGACION, y la anatomia lo prueba

Pregunta del usuario: ¿es la NLL la perdida adecuada? La NLL aparece 3 veces (perdida de
entrenamiento, regla de aceptacion del DFS, seleccion). Analisis de los datos ya medidos: la
regla de aceptacion suma NLLs contra un presupuesto FIJO (1,609) ⇒ impuesto al tamaño (L=30
entra con 94,8%/celda; L=300 exige 99,5%). En el 4B/eval: 37/40 verdades fuera del pool, 9 con
confianza/celda ≥98% — extremo 135a2760: 870 celdas al 99,8%, excluida por 0,04 nats.

**Experimento 1 — presupuesto LINEAL por posicion (`dfs-budget`, L4): NULO, y falsifica mi
prediccion.** τ(pos)=max(1.609, b·pos): admisibilidad estructural 3→7 (predije 8-12), barrido
de b PLANO (7 con cualquier b∈[0.015, 0.03]), busqueda real 3→3 resueltas con +56% de decode
(621 s vs 397) y pools inflados de basura (1818057f: pool 10, 0 correctos).

**El hallazgo que deja: la ANATOMIA por celda.** Mediana sobre 40 tareas (L mediana 260):
- celdas que concentran el 50% de la NLL total: **2**
- fraccion de la NLL en las 5 peores celdas: **82%**

La NLL de una verdad es una **llanura en ~0 con 2-5 picos** en las celdas donde se decide la
regla. Consecuencias:
1. **Aceptacion**: un pico de 1,5 nats consume casi todo el presupuesto ⇒ produccion solo
   resuelve tareas donde el modelo no duda en NINGUNA celda de decision. La forma correcta no
   es escalar con la longitud sino **descontar los top-k picos**.
2. **Entrenamiento**: la señal vive en ~5 celdas de 260 ⇒ el CE uniforme la diluye ~50x. La
   perdida PONDERADA hacia celdas dificiles pasa de especulacion a experimento motivado.
3. **TTT selectivo** (idea del usuario): "¿cuantas celdas dudosas tiene el mejor candidato?" es
   la observable fina para decidir por tarea — el trabajo del TTT, cuando funciona, es aplastar
   exactamente esos picos.

**Experimento 2 — descuento de picos (`dfs-spikes`, en vuelo):** aceptar si (suma − top-3
picos) < 1,609 con tope de 2,0 nats/paso. La cota descontada es monotona no-decreciente ⇒ la
poda del DFS sobrevive. Prerreg: admisibilidad ≥12/40; el riesgo es el coste de busqueda (las
exenciones abren ramas en las celdas de decision); si el decode excede ~2x, la regla pide K/CAP
mas estrictos. Guarda los **vectores por celda completos** ⇒ toda regla futura se evalua
offline gratis.

### 14-bis. RESULTADO: la familia de reglas de aceptacion queda CERRADA (tres puntos)

| regla | GT en pool | decode | pool total |
|---|---|---|---|
| fija 1,609 (produccion) | 3/40 | 397 s | ~40 |
| lineal τ(pos) | 3/40 | 621 s | 84 |
| descuento de picos (K=3, cap 2,0) | 3/40 | **1.095 s** | **173** |

Cada iteracion duplica el coste y resuelve LO MISMO. Prerreg de picos falsificada tambien:
admisibilidad 5/40 (peor que la lineal, 7) — el tope de 2,0 nats/paso mata casi todo, y ese
detalle revela el fondo del asunto:

**Los picos son PROFUNDOS: 2-10 nats por celda de decision** (p(token correcto) ≤ 8%). Ninguna
cirugia de agregacion rescata eso: la informacion NO esta en la distribucion del modelo base
en esas celdas, y explorar tokens al 8% abre ~12 ramas por celda dudosa (la explosion es el
decode de 1.095 s con pools de basura). Tres reglas distintas con 3/40 identico ⇒ el cuello es
el PRIOR, no la regla.

**Lo que si mata picos** (reencuadre que deja el nulo):
1. El **TTT** — su trabajo mecanico ahora tiene nombre: aplastar picos de 2-10 nats en celdas
   de decision. El veredicto de si lo consigue en el oculto es el score de nottt.
2. El **ensemble de augmentaciones** de produccion (n_aug=16): una celda dudosa en la vista
   identidad puede ser obvia en la rotada. Nuestro banco decodifica UNA vista — por eso ve
   3/40 donde produccion con augs+TTT ve ~30%. Las augmentaciones SON el mata-picos existente.
3. La **perdida ponderada** en entrenamiento sube de prioridad: la señal vive en celdas al 8%
   y el CE uniforme la promedia con 250 celdas triviales.

Los vectores por celda completos de las 40 verdades quedaron en el dataset del kernel
(`dfs_spikes.json`): cualquier regla futura se evalua offline sin GPU. No gastar mas L4 en
reglas de aceptacion de vista unica.

## 15. VEREDICTO (22-ago): lb33-nottt = 9,58. El TTT es EL MOTOR del pipeline

Submission 55680284 COMPLETE: **9,58** contra el 30,14 del campeon. Lectura prerreg `<28`
en su version extrema. La descomposicion del score quedo medida en tres puntos limpios:

| configuracion | score | fuente |
|---|---|---|
| 4B base, vista unica, sin TTT | ~7,5% | banco (3/40 eval) |
| + ensemble de 16 augmentaciones | **9,58** | LB, test oculto |
| + TTT por tarea | **30,14** | LB, campeon |

⇒ augmentaciones solas ≈ +2 pt; **TTT ≈ +20,6 pt (dos tercios del score) por el 34,8% del
reloj**. Coherente con la anatomia de §14: la base llega a las celdas de decision con picos de
2-10 nats y el TTT es lo unico que los aplasta. El smoke 3,0/4 que valido el submit media
tareas de TRAIN (memorizadas) — valido el pipeline, no el techo, como estaba anotado.

**Todo el "TTT neto cero" del banco 2B queda refutado en el oculto** — era contaminacion, como
anticipo tier1-4b (§13). El experimento costo 1 slot y compro: (a) el A/B mas limpio posible
del componente mas caro del sistema; (b) la refutacion definitiva de un banco viciado antes de
que motivara mas decisiones; (c) la reparticion augs/TTT que ningun banco podia dar.

**Reordenamiento estrategico**:
1. **El campeon sigue siendo 30,14** (nottt era una submission experimental; nada que revertir).
2. **"Liberar el 34,8% para cobertura" esta MUERTO.** El TTT produce 20,6 pt con ese tercio del
   reloj: es el componente MAS rentable, no el menos.
3. La pregunta correcta pasa a ser **como amplificar el TTT**, no como quitarlo:
   - **CE ponderado hacia celdas de pico** durante el TTT (la anatomia lo motiva: su trabajo es
     aplastar picos; una perdida enfocada en ellos podria aplastarlos mas rapido/mejor).
   - **TTT selectivo re-encuadrado**: ya no "saltarlo donde la base acierta" (la base casi nunca
     acierta: 3/40) sino **reasignar presupuesto DENTRO del TTT** — menos pasos donde converge
     rapido, mas donde los picos resisten.
   - **SFT transductivo del test** como etapa previa barata (un adapter sobre las demos de las
     240 tareas) si hiciera falta recuperar reloj.
4. ⚠️ **ttt8 (n_aug 16→8) queda en cuarentena**: su luz verde se midio en el banco 2B
   contaminado. Re-verificar sobre 4B limpio antes de confiar en el.
5. Los bancos futuros de TTT: **4B + eval + augmentaciones** — la vista unica quedo invalidada
   como instrumento para preguntas de TTT (§14-bis) y el 2B para cualquier pregunta (§13).

## 16. TRIAGE del TTT en el 4B (22-ago): el presupuesto fijo de pasos esta mal para el 75%

`ttt-triage` (L4, 35 min, TTT fiel r=256+embed/head+rsLoRA, 48 pasos, 20 tareas eval con las
3 que la base resuelve): base 2/20 → TTT 3/20 en vista identidad; **MATA: 0** (el TTT del 4B
NO destruye tareas resueltas — el fenomeno b2bc3ffd no aparece; la proteccion por salto es
innecesaria); RESCATA: 1 (28a6681f, 7,75→0,78, sus 2 picos >2n aplastados a 0 — el mata-picos
en vivo). Picos netos 31→37: el TTT CREA picos nuevos en el grupo que daña.

**El hallazgo esta en las CURVAS (NLL de la GT en pasos 0/12/24/36/48)** — taxonomia:

| grupo | n | implicacion |
|---|---|---|
| mejora y sigue a 48 | 5 | querrian MAS pasos (2b83f449: 18,0→1,31 y bajando) |
| mejora y SOBREAJUSTA | 8 | optimo en 12-36; seguir destruye (eee78d87: 3,87@12→17,2@48; 4c3d4a41: 2,30@24→8,14@48) |
| dañada desde el paso 12 | 7 | el TTT nunca les ayuda en esta vista |

⇒ **el presupuesto fijo de pasos (48 aqui, ~128 en produccion) esta mal para ~75% de las
tareas**. El "TTT selectivo" correcto no es saltar/aplicar sino **early stopping por tarea**.

**Gate del meta-aprendizaje: NEGATIVO.** Las que mejoran convergen en 12-36 pasos — el
problema no es velocidad de adaptacion sino cuando parar y en quien invertir. Reptile/MAML
archivado con razon medida.

**Verificador sobre pools volcados: caza 11,8%** (2/17 incorrectos, 0 FP). Los candidatos
equivocados de la transduccion conservan paleta y tamaño. Cerrado como filtro de adorno.

**En vuelo: `ttt-loo`** (primer kernel shardeado en 4 GPUs — patron obligatorio nuevo, ver
CLAUDE.md): ¿la NLL de una demo APARTADA (observable en produccion) predice el punto optimo
de parada (oraculo)? Curvas pareadas cada 6 pasos. Si correlacionan, el graft es **early
stopping por LOO en el TTT del campeon** — politica de control, barata, y con doble
dividendo: protege el optimo de las sobreajustadas Y devuelve reloj de las dañadas/rapidas.

### 16-bis. LOO vs politica FIJA (22-ago): la fija corta gana — graft lb33-ttt05

`ttt-loo` (10 min de pared shardeado ×4, ~0,35 h-quota — el patron nuevo rinde): optimos
coinciden solo 10/20, pero el regret dice mas:

| politica | regret mediano | regret total |
|---|---|---|
| parar por LOO (adaptativo) | 0,25 | 34,4 |
| no parar (48 fijo) | 1,45 | 82,6 |

El LOO captura ~60% del beneficio... pero **apartar la demo debilita el TTT mismo**
(2b83f449: 1,31 con todas las demos vs 17,08 con k−1) y le corta el rescate a 28a6681f
(parar@6 → 7,39 vs 3,05@36). Coste de informacion alto con 2-3 demos.

**La politica FIJA corta, medida sobre las curvas del triage (todas las demos):**

| parar en | regret total | tareas mejor que @48 |
|---|---|---|
| 0 (sin TTT) | 81,3 | 10 |
| **12** | **37,9** | 12 |
| 24 | 61,6 | 11 |
| 48 | **94,5** (la PEOR) | — |

⇒ parar en 12 ≈ LOO sin sacrificar demos. **El TTT de produccion sobre-entrena ×2-4 para la
mayoria.** Salvedad: los rescates tardios existen (28a6681f necesita ~36) ⇒ el graft
conservador es LA MITAD, no un cuarto.

**GRAFT `lb33-ttt05` (commiteado):** el campeon con `num_train_epochs=0.5` en el TTT — UN
numero. Media exposicion conservando la diversidad de las 16 vistas. Doble efecto esperado:
menos sobreajuste (8/20 tareas destruidas por el exceso en el banco) + ~17% del reloj total
liberado a cobertura. Converge con lo que ttt8 intentaba (menos vistas), pero con evidencia
del 4B limpio en vez del banco contaminado. Caveats anotados: mapeo banco(48)↔produccion
(1 epoca) es aproximado; proxy = NLL de la GT en vista identidad vs decode agregado.
**Prerreg: ≥30,14 adoptar / 29-30,14 ruido / <29 revertir** (la adaptacion reducida duele a
escala). Gate: smoke reload ≥2,5 o no se submite.

## 17. VEREDICTO ttt05 (23-ago): 26,39 — rama "revertir", y el post-mortem del mapeo

Submission 55701778 = **26,39** (−3,75 vs 30,14). El campeon sigue 30,14. El error fue de
MAPEO, y el banco lo predecia leido correctamente:

**El TTT del campeon entrena 1 EPOCA: cada secuencia augmentada se ve UNA vez.** Eso ES el
"paso 12" del banco (cada vista una pasada) — el punto dulce. El regimen sobreentrenado eran
NUESTRAS 4 epocas, no los 128 pasos de produccion (128 secuencias × 1 pasada, no 4 × 32).
Con 0.5 epocas, la mitad de las vistas NUNCA se adapto → sus decodes llegaron debiles a la
agregacion → la votacion de 16 vistas perdio la mitad de sus votantes entrenados.

**Lo que queda establecido:**
1. **El eje de pasos esta cerrado por los DOS extremos**: <1 epoca = −3,75 en LB (medido);
   >1 epoca = sobreajuste en banco (4 epocas destruian 8/20). **La receta del campeon ya esta
   en su optimo en ese eje.** La amplificacion debe venir de: forma de la perdida (CE
   ponderado a celdas de pico, dentro de la unica epoca), dosis de augmentaciones por tarea,
   o seleccion — NO de la cantidad de pasos.
2. Confirmacion redundante del valor marginal del TTT: −50% de su tiempo = −3,75 pt que la
   cobertura extra no compenso. La calidad por tarea domina sobre el conteo de tareas.
3. **Regla de banco endurecida**: proxy de vista identidad + epocas propias NO mapea al
   decode agregado con epoca unica. Todo banco de TTT futuro replica el calendario EXACTO de
   produccion (1 epoca sobre n_aug secuencias) y decodifica con agregacion.

Coste del aprendizaje: 1 slot (idle de todas formas) + ~1,8 h-quota ya contadas. Ledger de
la semana: ~7/10.

## 18. CE focal (24-ago): NEGATIVO con mecanismo — y la agregacion validada

`ttt-focal` (shardeado x4, 12 min, ~0,4 h-quota): CE focal (1-p)^2 vs estandar, epoca unica
exacta, decode agregado de 8 vistas con des-transformacion (inversion verificada 300/300) y
voto probmul-like, 12 tareas estratificadas.

- **Resueltas 1/12 = 1/12, pero focal produce MENOS candidatos siempre** (estandar 2-4 donde
  llena pool; focal 0-1). Mecanismo: la aceptacion del DFS exige confianza casi perfecta en
  TODAS las celdas del camino; el focal sacrifica el grueso para afilar picos y los caminos
  caen del umbral. **Familia focal-multiplicativa CERRADA.** Si el CE ponderado revive, sera
  ADITIVO (peso extra solo en celdas >2 nats, resto en 1.0).
- **BONUS: la agregacion validada en nuestro banco** — e8686506 ("dañada", jamas resuelta en
  vista identidad) salio resuelta en AMBOS brazos via la votacion de 8 vistas. El mecanismo
  central de produccion, reproducido.
- Limitacion: presupuesto de 15 s/vista (incluye prefill) hambreo los decodes medianos
  (38007db0, resoluble por la base, salio pool 0). El veredicto es "focal <= estandar", no un
  conteo confiable. NO se re-corre: la señal direccional es consistente y el margen de quota
  (~2,3 h) no se gasta en re-litigar.

Linea de amplificacion, estado: (a) pasos CERRADO (§17); (a') focal CERRADO; quedan (b) dosis
de augmentaciones por tarea, (c) seleccion, (a'') CE aditivo en celdas de pico. Bancos futuros
de decode agregado: presupuesto >=45 s/vista.
