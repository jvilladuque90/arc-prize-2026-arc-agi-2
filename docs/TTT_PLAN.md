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

## 19. BANDA DE RUIDO del campeon (24-ago): 30,14 y 29,31

Re-run identico del campeon (seed process-salted resamplea): **29,31** vs 30,14 original.
Diferencia 0,83 por puro ruido de ejecucion, consistente con la banda ±0,7 del 2B
(19,86/21,25). Con dos muestras: centro ~29,7, banda util ~±0,85.

**Regla de decision recalibrada (vale para TODO A/B futuro):**
- adoptar un graft: **>=31** (una banda por encima de 30,14)
- revertirlo: **<=28,5**
- 28,5-31 = RUIDO: no se adopta ni se revierte por si solo; pide segunda muestra.

Veredictos pasados re-examinados: ttt05 (26,39, -3,75) y nottt (9,58) se sostienen con
holgura; la familia de escala (27,08 real; 28,33 a ~2 bandas) se sostiene. El nivel real del
campeon es ~29,5-30,1; el peloton (33,89) esta a ~4 puntos reales.

Prueba candidata para el slot de manana: TERCERA muestra de varianza (dos puntos definen mal
una banda; tres la definen razonablemente). Costo cero quota.

## 20. Analisis offline (25-ago): mueren SELECCION y CE aditivo; nace "consenso entre vistas"

Cero GPU, sobre los JSON ya persistidos (triage + focal).

**Seleccion: MUERTA.** En las 4 configuraciones medidas (triage base/ttt, focal
estandar/focal), las perdidas por seleccion son **0**: cuando la GT entra al pool, siempre se
elige como top-1. El factor 0,90 no tiene margen recuperable; el cuello esta aguas arriba
(que la GT ENTRE al pool).

**CE aditivo hacia celdas de pico: CONTRAINDICADO.** El TTT aplasta 8 picos pero **CREA 14**,
y en **7/20 tareas crea mas de los que quita**. Enfocar la perdida en picos amplificaria un
proceso que ya genera picos nuevos. Cerrada antes de nacer (junto con la focal
multiplicativa de §18, la familia de perdidas ponderadas queda cerrada entera).

**Hallazgo nuevo: la DISPERSION DEL VOTO entre vistas es una observable util.** En el decode
agregado, el perfil de votos separa dos regimenes:
- consenso fuerte: b0039139 [17,51 / 5,74], e8686506 [11,25 / 2,96 / 1,63] — jerarquia clara
- consenso debil: 78332cb0 [1,457 / 1,417 / 1,408] — empate tecnico, las vistas discrepan

⇒ **Palanca viva unica: DOSIS DE AUGMENTACIONES POR TAREA.** Produccion reparte 16 vistas
uniformes; reasignarlas segun el consenso observado (mas vistas donde el voto esta empatado,
menos donde ya hay jerarquia) es politica de asignacion — barata, injertable, y ataca
GT-en-pool. Diseño del banco: medir curva de GT-en-pool vs n_vistas (4/8/16/24) y correlacionar
con la dispersion del voto a 8 vistas, para saber si la dispersion PREDICE que mas vistas
ayuden. Presupuesto >=45 s/vista (leccion del focal).

## 21. LA BANDA REAL (25-ago, n=3): 29,03 ± 1,27 — y casi todos los A/B del mes eran ruido

Tres corridas del MISMO kernel, misma version: **30,14 · 29,31 · 27,64**.

```
media 29,03 | desv 1,27 | rango 2,50
diferencias = 0,83 / 1,67 / 2,50 = exactamente 1, 2 y 3 x 0,8333
=> quantum = 1 tarea de ~120. El ruido son POCAS TAREAS que entran o no.
```

**El campeon nunca fue 30,14**: esa fue la tirada afortunada. Su nivel real es ~29.

**Re-lectura honesta de los veredictos del mes** (z contra 29,03 ± 1,27):

| submission | score | z | veredicto real |
|---|---|---|---|
| lb33-cap | 27,64 | −1,09 | **DENTRO del ruido** (identico a un re-run del campeon) |
| merged+fast | 28,33 | −0,55 | DENTRO del ruido |
| merged | 27,08 | −1,54 | DENTRO del ruido (borde) |
| ttt05 | 26,39 | −2,08 | fuera: real |
| nottt | 9,58 | −15,3 | fuera: real |

⚠️ **CORRECCION IMPORTANTE**: llevo toda la semana citando "la leccion de lb33-cap" (−2,5 pt)
como ancla metodologica. **Ese experimento no fallo**: saco exactamente lo que saca el campeon
en una mala tirada. El PRINCIPIO (calibrar en la poblacion correcta) sigue siendo sano, pero
no lo demostro ese experimento. Igual la familia de escala: se sostiene cerrada por el control
de formato del banco (§10-ter), NO por el leaderboard.

**Consecuencia estrategica dura**: con 1 submission/dia y banda ±1,27, un A/B de una sola
muestra solo detecta efectos >2,5 pt. Nuestras palancas vivas valen ~1 tarea (0,83 pt).
**Sin reducir el ruido, ninguna se puede validar jamas en el LB.**

**LA PALANCA METODOLOGICA: fijar la semilla.** El kernel usa
`aug_dataset.augment(seed=hash(bk) % 1024**2)` en la fase de scoring, y `hash()` de strings en
Python va **salado por proceso** (PYTHONHASHSEED aleatorio). Cada corrida sortea augmentaciones
distintas. Fijando `PYTHONHASHSEED=0` (o un hash estable tipo md5) el kernel pasa a ser
reproducible y los A/B se vuelven **pareados**: la diferencia observada es el efecto del graft,
no el sorteo. No sube el score por si mismo; **hace legibles todos los experimentos futuros**.
(Nota: queda ruido residual de timing — que tareas caben en las 12 h depende de la maquina.)

**Plan**: (1) mie/jue: re-submits gratis del campeon → n=4, n=5, banda mas fina; (2) viernes
tras el reset de quota: commit del kernel determinista y re-submits repetidos para verificar
que la varianza colapsa; (3) solo entonces, A/B de las palancas de ~1 tarea.

## 22. lb33-cap RECUPERADO (26-ago): su "fallo" era ruido, y vuelve al stack

Re-test gratis (kernel ya commiteado, cero cuota): **30,14** — exactamente la mejor tirada del
campeon.

| brazo | muestras | media |
|---|---|---|
| campeon | 30,14 · 29,31 · 27,64 | 29,03 |
| **lb33-cap** | **27,64 · 30,14** | **28,89** |

Diferencia de medias **−0,14 pt (−0,17 tareas), z = −0,12** ⇒ **INDISTINGUIBLES**. Los dos
brazos producen literalmente el mismo conjunto de valores {27,64 · 29,31 · 30,14}.

⇒ **El tope de longitud no daña.** Queda rehabilitado como acelerador **×1,10-1,13** con doble
respaldo: banco (la GT cabe en el 100% de 1248 subtareas) y LB (2 muestras indistinguibles).
Entra al stack de velocidad del viernes.

**Leccion de metodo, ahora con dos casos**: re-testear ramas descartadas cuesta CERO cuota
(los kernels ya estan commiteados; solo el commit gasta) y recupero un activo real. Es la
mejor prueba por defecto para un slot ocioso — mejor que acumular muestras de varianza, que
ya no cambian ninguna decision.

⚠️ Detalle operativo caro de aprender: `kaggle competitions submit -v 1` fallo con 400 porque
**solo la ULTIMA version del kernel conserva el output**. `lb33-cap` tenia 3 versiones y la
buena era la v3. El CLI no dice cual es el problema; el replay con `kagglesdk` lo revelo
("Did not find provided Notebook Output File"). Al re-submitir kernels viejos: probar sin `-v`
o con la version mas alta.

## 23. Preparado sin gastar cuota (26-ago): el par determinista del viernes

Todo el trabajo del viernes construido y verificado en local, coste **CERO**:

**Auditoria de no-determinismo del campeon**: `hash(bk)` es la **unica** fuente salada. Todo
lo demas usa semillas explicitas (42/1/2) y las permutaciones de numpy van tras
`np.random.seed(seed)` dentro de `augment()`. Un solo punto que arreglar.

**Descubrimiento al desempaquetar `lb33_cap`**: no eran uno sino **dos injertos aplicados** —
(a) **cheap-first**: el campeon ordena las tareas ALFABETICAMENTE (`sorted(data.keys())`);
el injerto ordena por coste proxy (celdas de sus grids) barata→cara, drenando mas tareas antes
de `end_time`. Cobertura pura, **sin coste por tarea**. (b) **tope de longitud por tarea**
(`task_max_new`), ×1,10-1,13. El comentado "GRAFT 3" (tiempo 1200→600 s) **nunca se aplico**:
el codigo sigue en `spend_time > 1200`. ⇒ el bundle que salio neutro en el LB (§22) contiene
solo los dos injertos monotonos, ambos ya justificados.

**Par construido** (cambio comun: `seed=hash(bk)` → `md5(bk)[:8]`, mas un import):
- **`lb33-det`** = CONTROL: campeon + determinismo. Nada mas.
- **`lb33-stack`** = TRATAMIENTO: determinismo + cheap-first + tope de longitud.

Verificado en local: 1 reemplazo exacto de seed y 1 import por notebook (asserts en el
builder), `hash(bk)` ausente, las 6 celdas de codigo compilan en ambos (las celdas 2 y 7 son
magias de shell `!pip`/`!UNSLOTH`, identicas al baseline), y las semillas md5 dan enteros
estables.

**Protocolo del viernes**: commit de `lb33-det` (~1,8 h-cuota) → submit 2 dias seguidos.
**Si los dos scores son IDENTICOS, el determinismo funciona** y los A/B pasan a ser pareados
(un efecto de 1 tarea, hoy invisible bajo ±1,27, se vuelve legible). Entonces commit de
`lb33-stack` y su submit da el A/B pareado de los injertos de cobertura. Si los scores
difieren, domina el ruido de timing (que tareas caben en 12 h) y el LB queda confirmado como
instrumento de baja resolucion — el banco carga con todo.

## 24. Smoke del banco de dosis (26-ago, Colab 2B): el observable era el equivocado

Cadena completa validada end-to-end sin gastar cuota de Kaggle: banco → persistencia →
descarga → analisis offline por submuestreo. 4 tareas, 16 vistas, 533 s en T4.

**Vistas que contienen la GT: [16/16 · 10/16 · 3/16 · 15/16]** — hay rango dinamico real
incluso en el 2B contaminado, que es lo que hace la pregunta interesante.

| tarea | n=2 | n=4 | n=8 | n=12 | n=16 | beneficio | prod@4 | ratio@4 |
|---|---|---|---|---|---|---|---|---|
| e8686506 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | +0,00 | 1,00 | 0,000 |
| 20270e3b | 0,93 | 0,97 | 1,00 | 1,00 | 1,00 | +0,07 | 0,75 | 0,322 |
| **7b5033c1** | **0,40** | 0,68 | 0,62 | 0,95 | **1,00** | **+0,60** | **0,50** | 0,000 |
| b0039139 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | +0,00 | 1,00 | 0,134 |

**El smoke hizo justo lo que un smoke debe hacer: refutar mi eleccion de observable.** La
dispersion del voto (§20) NO discrimina — `7b5033c1` (beneficio +0,60) y `e8686506` (beneficio
0,00) tienen ratio identico (0,000) y un solo candidato. Con pocas vistas no hay competencia
que medir, asi que el empate de votos no distingue "ya resuelta" de "apenas encuentra nada".

**El observable bueno es la ESCASEZ**: `productividad` = fraccion de vistas que devuelven algun
candidato. Correlacion con el beneficio de 4→16 vistas: **−0,924** (contra −0,399 de la
dispersion). Mecanismo: vistas vacias = el modelo apenas engancha = mas vistas son mas billetes
de loteria; vistas todas productivas y coincidentes = la tarea ya esta decidida.

⚠️ **n=4 y modelo contaminado**: esto NO es veredicto, es una **hipotesis sobre que medir**,
que es exactamente el producto legitimo de un banco de mecanismo. El veredicto sale del 4B en
L4 (~0,75 h-cuota shardeado) con el harness ya depurado y el observable ya corregido — sin
gastar L4 en descubrir que la metrica no servia.

Coste del hallazgo: **cero cuota de Kaggle** (T4 gratis + analisis local).

## 25. La SEGUNDA fuente de no-determinismo (1-sep): el tope de reloj por tarea

Antes de gastar cuota en `lb33-det` audité si las compuertas de reloj del pipeline llegan a
morder. Los logs del campeón son gratis, así que la auditoría costó **cero cuota**.

El commit de `lb33-cap` corre el smoke público de 4 tareas, una por GPU. Tiempos por tarea:

| rank | tarea | TTT+setup | decode | total | lote más lento |
|---|---|---|---|---|---|
| 0 | 36a08778 | 215,2 s | 859,6 s | 1074,9 s | 203,7 s |
| 1 | aa4ec2a5 | 324,2 s | 308,5 s | 632,8 s | 90,2 s |
| 2 | 0934a4d8 | 400,2 s | 145,3 s | 545,6 s | 43,3 s |
| 3 | 981571dc | 865,5 s | 386,2 s | **1252,1 s** | 107,6 s |

Dos lecturas, una tranquilizadora y otra no.

**El tope de 540 s del DFS nunca muerde.** El lote más lento medido es 203,7 s, a menos de la
mitad del límite. El DFS termina por agotamiento de haces (`num_alive_beams == 0`), que es una
salida *determinista*: la poda por `max_score` es el presupuesto real, no el reloj. Esa
compuerta se deja intacta.

**El tope de 1200 s por tarea sí muerde, y por un pelo.** `981571dc` terminó en 1252,1 s. No
imprimió timeout porque la compuerta se evalúa **al inicio de cada lote**: su último lote
arrancó con `spend_time` = 1159,6 s y corrió hasta 1252,1. Es decir, **40,4 s de margen sobre
1160 — un 3,5% de jitter de reloj decidía si esa tarea decodificaba 16 vistas o 12.**

Y ahí está el mecanismo que faltaba: una tarea vale 0,8333 pt y la sigma observada es 1,27
(≈1,5 tareas). **La granularidad del ruido coincide con "una o dos tareas pierden su último
lote según cómo venga la máquina ese día".** El seed salado (§21) explicaba una parte; esto
explica la otra, y `md5` no la toca.

Corolario incómodo: commitear `lb33-det` con solo el fix de md5 habría dado dos scores
distintos **aunque el fix fuese correcto**, y lo habríamos leído como "el determinismo falló".

**Injerto 2**: se elimina el tope de reloj por tarea; se conserva el deadline global (la
protección contra morir sin escribir submission, que con σ pequeña y cero `stop!` en el log no
parece morder). El cambio es **monótono positivo** por añadidura: las tareas que perdían su
último lote ahora decodifican las 16 vistas. Riesgo aceptado: el run crece unos pocos puntos
porcentuales, acotado estructuralmente en 4 lotes por entrada de test.

Dato de presupuesto corregido de paso: el commit son **1734 s de pared ≈ 0,96 h-cuota**, no las
1,8 que tenía estimadas. El rerun de scoring sobre el oculto va por cuenta de Kaggle.

⚠️ Salvedad: las 4 tareas del smoke son la muestra pública, no el oculto de 120. Que el
mecanismo existe y muerde está **medido**; qué fracción del oculto roza el tope es
extrapolación.

## 26. El banco de dosis v1 salió INVÁLIDO (1-sep): el TTT vio 16 secuencias, no 128

Corrida mecánicamente perfecta y científicamente nula. Los 4 shards terminaron, las 16 tareas
se procesaron, los JSON se fundieron. Y el resultado es basura: **GT en alguna vista en solo
3 de 16 tareas**, 6 aciertos de vista sobre 256 posibles.

La delación está en el propio registro: **`n_seqs_ttt` = 16 en las 16 tareas, y `ttt_s` entre
14 y 27 s** — cuando el TTT de producción tarda 215-865 s por tarea. Un orden de magnitud de
menos no es ruido, es otro experimento.

**Causa**: confundí el `n` de `augment(n)` con el número de vistas. En producción
`augment(n)` = **8 transformaciones D4 × n permutaciones de color**, así que el entrenamiento
`augment(n=16)` son **128 secuencias** y el decode `augment(n=2)` son 16. Mi harness generaba
16 vistas sueltas para AMBOS. El lado del decode estaba bien por coincidencia numérica; el del
TTT iba **8× corto**. Con 16 pasos en vez de 128 el 4B sencillamente no se adapta, y sin
adaptación no hay nada que dosificar: la pregunta del banco no llega ni a plantearse.

**Por qué el smoke del 2B no lo cazó** — y esta es la lección que vale más que el experimento:
el 2B memoriza el eval, así que resuelve esas tareas **sin necesitar TTT**. Un vehículo
contaminado puede validar la fontanería (des-transformación, persistencia, submuestreo) pero es
**estructuralmente incapaz de validar la fidelidad de la adaptación**, porque no depende de
ella. El smoke del 26-ago hizo bien su trabajo declarado y aun así dejó pasar esto: no fue
descuido del smoke, fue un límite del vehículo. Al catálogo de reglas de banco (§8) se añade:
**lo que el banco no necesita para funcionar, el banco no lo verifica.**

Coste: **0,84 h-cuota** (1516 s de pared × 2).

**Arreglo**, con test local que ejecuta la función real del worker antes de gastar GPU:
`build_vistas_train` construye el producto cruzado 8×16 y se comprueba que salen 128
secuencias, 8 transformaciones D4 distintas, 16 permutaciones distintas, la identidad
exactamente una vez, y que la misma semilla da las mismas vistas. Además el worker lleva
`assert len(v_train) == 128` en caliente: si el calendario se vuelve a desviar, el shard muere
en la primera tarea en vez de gastar la corrida entera produciendo números creíbles y falsos.

## 27. Banco de dosis v2 (2-sep): el arreglo funcionó, la rama se cierra por prerreg

Con el calendario corregido el banco pasa a ser fiel: **`n_seqs_ttt` = 128 en las 16 tareas y
`ttt_s` entre 103 y 206 s**, ya en el orden de producción (215-865 s en tareas más caras). El
TTT hace su trabajo: las vistas que contienen la GT suben de **6 a 13** y los candidatos por
vista de ~0,4 a ~0,7. El arreglo era el correcto.

**Lectura prerregistrada: `r(productividad@4, beneficio 4→16) = +0,215`.** El prerreg decía
r ≥ −0,30 ⇒ **cerrar la rama de dosis**. Se cierra. La dispersión del voto tampoco discrimina
(+0,133), consistente con su refutación en el smoke.

Y hay que decir lo que debilita esa lectura, porque es visible sin mirar la r: **13 de 16
tareas no se resuelven a NINGUNA dosis**, así que su "beneficio 0" es una propiedad de la
construcción y no evidencia sobre la hipótesis. Tareas informativas reales: **3**. El prerreg
suponía n=16 informativas y se ejecutó con n=3.

| tarea | n=2 | n=4 | n=8 | n=12 | n=16 | beneficio | prod@4 |
|---|---|---|---|---|---|---|---|
| e8686506 | 0,50 | 0,82 | 0,85 | 1,00 | 1,00 | **+0,50** | 0,75 |
| 4c3d4a41 | 0,70 | 0,85 | 0,95 | 1,00 | 1,00 | **+0,30** | 1,00 |
| 20270e3b | 0,23 | 0,35 | 0,60 | 0,42 | 0,00 | **−0,23** | 0,50 |

Hallazgo lateral con mecanismo: **`20270e3b` empeora con más vistas** (0,60 en n=8 → 0,00 en
n=16). Más vistas no son gratis: un candidato equivocado pero consistente acumula voto y ahoga
al correcto. Eso ya rompe por sí solo la premisa cómoda de "más vistas nunca hacen daño".

**Por qué NO se re-corre con más tareas.** El banco costó 1,27 h-cuota para 3 tareas
informativas; llegar a 16 informativas pide ~5× ≈ 6 h-cuota, el presupuesto de experimentos de
la semana entera. Y el cálculo cambió hoy: **si el determinismo se confirma, un injerto de
dosis se puede medir DIRECTAMENTE en producción como A/B pareado por ~1 h-cuota**, contra el
score real del oculto en vez de contra una correlación proxy sobre una población que el modelo
apenas resuelve. El banco era la herramienta correcta cuando los A/B eran ilegibles; deja de
serlo en cuanto dejen de serlo.

Coste: 1,27 h-cuota (2278 s de pared × 2). Día completo: 0,97 (det) + 0,84 (v1 inválida)
+ 1,27 = **3,08 h-cuota**.

Arreglado de paso un bug de reporte en `augdose_analysis.py`: `informe()` imprimía la
interpretación "r NEGATIVA fuerte ⇒ palanca VIVA" **cualquiera que fuese el signo de r**. Ahora
aplica la regla prerreg por signo y magnitud, y avisa cuántas tareas son degeneradas — que es
justo el aviso que faltaba aquí.

## 28. lb33-det = 31,11 (2-sep): el score más alto del proyecto, y una predicción cumplida

`lb33-det` (campeón + los dos injertos de determinismo) puntúa **31,11**. Contexto: la banda
del campeón es 29,03 ± 1,27 (n=3) y su mejor tirada histórica fue 30,14. Esto es **z = +1,63**
y **+2,08 pt sobre la media**, ≈ 2,5 tareas.

Lo que sostiene la lectura no es el z —una muestra sola nunca lo sostendría— sino que
**la predicción estaba prerregistrada y con mecanismo**. Antes de ver el número quedó escrito
que quitar el tope de reloj por tarea es *monótono positivo* porque las tareas truncadas
recuperan su último lote de 4 vistas, y que por tanto se esperaba ≥ 29,03. La magnitud
observada (≈2,5 tareas) encaja con "una o dos tareas por corrida perdían su último lote",
que es exactamente lo medido en §25 (40,4 s de margen sobre el tope en `981571dc`).

Es decir: la auditoría de los logs no solo evitó un experimento confundido — **encontró
puntos**. El tope de 1200 s no era solo una fuente de ruido, era una fuga de cobertura.

**Esto reordena el uso del slot de mañana.** Con 31,11 en la mano, el segundo submit de
`lb33-det` deja de medir una cosa y pasa a medir dos a la vez:

- Si devuelve **exactamente 31,11** ⇒ el determinismo está cerrado (misma salida bit a bit)
  **y** el +2,08 no es una tirada afortunada, porque un kernel determinista no puede tener
  suerte dos veces. Nuevo campeón confirmado y, desde ese momento, todos los A/B son legibles.
- Si devuelve **otra cosa** ⇒ queda una tercera fuente de no-determinismo y 31,11 es una
  muestra de una distribución todavía ancha.

Frente a eso, submitir `lb33-stack` primero daría un número difícil de leer: sin saber si el
determinismo aguanta, una diferencia de ~1 pt contra 31,11 cabe en cualquier banda plausible.
El orden vuelve a ser **3-sep: `det` #2 · 4-sep: `stack`**, que es el que maximiza información
por slot. `lb33-stack` ya está commiteado, así que no se pierde nada esperando un día.

⚠️ Hasta que llegue esa segunda muestra, 31,11 es **n=1**: prometedor y con mecanismo, no
todavía un veredicto.

## 29. Auditoría de descartes silenciosos (3-sep): `cut_to_len` NO es una fuga — cerrado gratis

Tras descubrir que el tope de reloj por tarea era una fuga de cobertura (§25, §28), apliqué el
mismo método al resto del solver: buscar dónde se **descarta trabajo en silencio**.

El candidato serio era `cut_to_len`. No trunca la secuencia: **elimina demostraciones enteras
del prompt**, una a una desde el principio, hasta que cabe en `max_seq_length = 8192`. Se aplica
al entrenamiento del TTT (`max_len=8192`) y al decode (`max_len=8192−931=7261`). Si mordiera,
las tareas grandes estarían aprendiendo y decodificando con menos ejemplos de los que tienen —
exactamente la misma forma de fuga que el tope de reloj.

Medido **offline y sin gastar cuota**, contando tokens de rejilla (celdas + saltos + marcadores):

| población | tareas | TTT (>8192) | decode (>7261) | demos descartadas |
|---|---|---|---|---|
| eval AGI-2 | 120 | **0 (0%)** | 2 (2%) | **2** de 358 |
| test público | 240 | **0 (0%)** | 0 (0%) | **0** de 767 |

**No muerde.** Subir `max_seq_length` recuperaría 2 demostraciones en 120 tareas, a cambio de
más memoria y más tiempo de prefill en todas. La rama se cierra antes de gastar un solo minuto
de L4.

Vale la pena registrar el contraste con §25: **el mismo método, aplicado con el mismo esfuerzo,
dio +2 pt en un caso y un cierre en el otro.** Esa es la proporción normal, y es barata: ambas
respuestas salieron de leer datos que ya teníamos.

Pendiente sin resolver, anotado para no perderlo: `arc-agi_test_challenges.json` trae **240
tareas / 259 entradas de test**, mientras que el eval son 120 / 172. Si el oculto fuese de 240,
la extrapolación de reloj (~7,3 h con 120) se iría a ~14,6 h contra un deadline de 11,83 h y el
recorte global sería habitual. La evidencia disponible apunta a 120 (el quantum de ruido de
0,8333 = 1/120, y que las corridas terminan sin `stop!`), pero **no está verificado** y conviene
tenerlo presente antes de añadir más trabajo por tarea.

## 30. VEREDICTO (3-sep): el determinismo NO se cerró — pero el canal mejoró y el quantum es 1/240

`lb33-det`, dos corridas del kernel idéntico: **31,11 y 30,69**. El prerreg era explícito: si no
son iguales, **queda una tercera fuente de no-determinismo**. Queda. Los dos injertos (semilla
md5 + eliminación del tope de reloj por tarea) no bastan.

Y a la vez, el canal de medida es otro:

| | corridas | media | rango |
|---|---|---|---|
| campeón (`lb33-cap`) | 30,14 · 29,31 · 27,64 | 29,03 | **2,50** |
| `lb33-det` | 31,11 · 30,69 | **30,90** | **0,42** |

**Nivel +1,87 pt y dispersión ~6× más estrecha.** Ambas cosas son lo que predice el mecanismo:
quitar el tope de reloj añade cobertura (sube el nivel) y elimina una moneda al aire gobernada
por el jitter (estrecha el rango). ⚠️ Con n=2 el rango **no** es una estimación de σ; es una
observación compatible con el mecanismo, no un resultado establecido.

**El quantum queda resuelto, y no es el que suponíamos.** Todas las diferencias observadas son
múltiplos casi exactos de **1/240 = 0,4167**, no de 1/120:

| diferencia | pt | tareas a 1/240 |
|---|---|---|
| 30,14 − 29,31 | 0,83 | **1,99** |
| 29,31 − 27,64 | 1,67 | **4,01** |
| 31,11 − 30,69 | 0,42 | **1,01** |

Es decir: **el oculto son 240 unidades de puntuación**, coherente con las 240 tareas de
`arc-agi_test_challenges.json`, y las dos corridas de `det` difieren en **exactamente una
tarea**. Eso cierra la pregunta abierta en §29 con la respuesta incómoda: la extrapolación de
reloj relevante es **~14,6 h contra un deadline de 11,83 h**, no las ~7,3 h que calculé
suponiendo 120.

**Esto reformula el A/B de mañana.** Si el deadline global muerde, se están descartando tareas
de la cola cada corrida — lo que sería a la vez la tercera fuente de ruido *y* una pérdida
sistemática de cobertura. Y `lb33-stack` es justo la sonda: **cheap-first** hace que el recorte
caiga en la cola cara en vez de en un punto arbitrario, y el **tope de longitud** mete más
tareas antes del corte. Deja de ser "dos injertos monótonos que quizá suman un poco" y pasa a
ser un test de si el deadline muerde.

**Prerreg de `lb33-stack`** (escrito antes de submitir):
- **≥ 31,3** (≥ +1 tarea sobre la media de `det`): el deadline muerde y el orden de la cola es
  palanca de cobertura → adoptar y explorar la familia de presupuesto global.
- **30,5 – 31,3**: indistinguible de `det` con el rango observado → el deadline no muerde de
  forma apreciable; los dos injertos son marginales y la rama se archiva.
- **≤ 30,3**: alguno de los dos injertos hace daño → revertir.

Nota metodológica: aunque el determinismo no esté cerrado, **el canal ya sirve para mucho más
que antes**. Con un rango de ~1 tarea, un injerto que valga 2-3 tareas es legible con una sola
muestra; con la banda vieja hacía falta >2,5 pt (6 tareas).

### 30 bis. Predicción escrita ANTES del resultado de `stack`: el deadline probablemente NO muerde

La magnitud del ruido residual discrimina entre las dos hipótesis, y apunta en contra de la mía.

Si el deadline global mordiera y estuviera cortando la cola, la variación entre corridas vendría
de **dónde cae el corte**. La velocidad de máquina en Kaggle varía fácil un ±3%, y un ±3% sobre
~200 tareas procesadas desplaza el corte unas **±6 tareas**. Lo observado entre las dos corridas
de `det` es **1 tarea**. Un orden de magnitud menos.

Con el deadline no mordiendo, en cambio, todas las tareas se procesan y la única variación es la
no-determinación de coma flotante en GPU (atómicos en los kernels de atención/matmul) volteando
tareas que están justo en la frontera de decisión. **Una tarea de 240 encaja exactamente con eso.**

Entonces mi extrapolación de ~14,6 h es probablemente **demasiado alta**: sale de promediar 4
tareas del smoke público, que no tienen por qué representar al oculto.

**Predicción, escrita antes de ver el número**: si esto es correcto, `lb33-stack` debería caer en
la banda central prerregistrada (**30,5-31,3**, "indistinguible de `det`, archivar"), porque sin
deadline mordiendo tanto cheap-first como el tope de longitud son marginales. Si en cambio sale
**≥ 31,3**, mi razonamiento es erróneo, el deadline sí muerde y se abre la familia de
presupuesto global.

**Siguiente test si sale la banda central** (queda como candidata única la coma flotante):
re-commitear `lb33-det` sin cambiar una línea y comparar los dos logs del smoke **beam score a
beam score**. Si los 4 puzzles dan números bit a bit idénticos, la GPU es determinista y la
tercera fuente está en otro sitio; si difieren en los últimos decimales, queda identificada.
Cuesta ~1 h-cuota y es una respuesta limpia.

## 31. VEREDICTO `lb33-stack` (4-sep): 29,44 — REVERTIR, y una corrección al historial

`lb33-stack` = `det` + cheap-first + tope de longitud por tarea: **29,44**. El prerreg era
≤ 30,3 ⇒ revertir. **Se revierte.** `lb33-det` sigue siendo el campeón.

| | score | vs media de `det` (30,90) |
|---|---|---|
| `det` #1 | 31,11 | +0,21 |
| `det` #2 | 30,69 | −0,21 |
| **`stack`** | **29,44** | **−1,46 pt = −3,5 tareas** |

La caída es de **3 tareas incluso contra la peor corrida de `det`**, siete veces el rango
observado entre las dos corridas del campeón. No es ruido de la banda nueva.

**Mi predicción de §30 bis era errónea.** Escribí que `stack` caería en la banda central por ser
los injertos marginales; cayó por debajo de la banda inferior. La hipótesis de que el deadline
global no muerde sigue viva por el argumento de magnitud, pero **este resultado no la apoya**:
lo que muestra es que al menos uno de los dos injertos hace daño activo.

**Corrección al historial (§22).** `lb33-cap` llevaba **estos mismos dos injertos** sobre la base
vieja y puntuó 27,64 y 30,14. En §22 leí el 30,14 como "recuperado, vuelve al stack" porque
igualaba al campeón. Con el canal estrecho se ve que aquello era **la tirada afortunada de una
configuración peor**: la media de esas dos corridas es 28,89, consistente con el 29,44 de hoy.
**El "rescate" de lb33-cap fue un error de lectura**, y el mes pasado se adoptaron dos injertos
que restan.

Esto es exactamente el pago del canal estrecho: un efecto de **1,46 pt** era **invisible** con la
regla vieja (que exigía >2,5 pt para declarar nada) y hoy se lee sin ambigüedad.

**Defecto de diseño propio, que hay que anotar**: empaqueté DOS injertos en un brazo, así que el
resultado negativo **no dice cuál de los dos es el culpable** (ni si es la interacción). Con un
solo slot al día, agrupar cambios ahorra días pero compra ambigüedad; con el canal ya estrecho,
esa economía dejó de compensar. **Regla nueva: un injerto por brazo.**

**Sospechoso principal, con mecanismo**: el tope de longitud
`task_max_new = min(931, 4·max(salidas de train, entrada de test) + 2)`. Si en el oculto alguna
tarea tiene la salida más larga que ese tope, su respuesta **no se puede generar jamás** — pérdida
garantizada, no probabilística. La verificación de que "la GT cabe en el 100% de 1248 subtareas"
se hizo sobre **train+eval**, no sobre el oculto, que es justo la población que decide. Es el
mismo error de vehículo de §26 en otra forma: validar en la población cómoda.

**Siguiente A/B (un solo injerto)**: `det` + cheap-first, sin tope de longitud. Si iguala a `det`,
el tope de longitud es el culpable y cheap-first se puede conservar; si también cae, el culpable
es cheap-first o la interacción.

## 32. El culpable, con mecanismo probado (5-sep): el tope de longitud asume una expansión ≤4×

Medición offline, coste cero, sobre las poblaciones que tenemos etiquetadas. El tope injertado es
`task_max_new = min(931, 4·max(salidas de train, entrada de test) + 2)`; comprobamos si la GT cabe:

| población | subtareas | violaciones | margen mín. | margen p1 | con margen <50 |
|---|---|---|---|---|---|
| eval | 172 | **0** | **1** | 1 | **21 (12,2%)** |
| train | 1076 | **0** | **1** | 1 | 54 (5,0%) |

Cero violaciones — por eso la validación original dijo "la GT cabe en el 100%". Pero el dato que
esa validación no miró: **el margen mínimo es de UN token**, y el 12% del eval va con menos de 50
de holgura. El tope no es holgado, es **de filo**.

**El mecanismo, en una frase: el tope asume que la salida nunca se expande más de 4× en tokens
sobre la mayor rejilla vista, y ARC no garantiza eso.**

| transformación | GT (tokens) | tope | |
|---|---|---|---|
| teselado 2×2 de una 10×10 | 420 | 438 | cabe |
| **teselado 3×3** de una 10×10 | **930** | 438 | **imposible de generar** |
| teselado 4×4 de una 10×10 | 1640 | 438 | imposible de generar |

El teselado ×3 es una familia clásica de ARC. En cuanto el oculto contiene una, su respuesta
**no se puede emitir jamás** — pérdida garantizada, no probabilística. Con el p1 del eval en un
margen de 1 token, que el oculto (240 tareas, población distinta) tenga varias violaciones es lo
esperable, no lo excepcional. **Y las magnitudes cuadran**: unas pocas subtareas imposibles
explican las −3,5 tareas de `stack`.

**Por qué el baseline es seguro y el injerto no.** El campeón usa `max_new_tokens = 931` fijo, que
no es un número arbitrario: la rejilla ARC más grande legal es 30×30 = 900 celdas + 29 saltos +
1 EOS = **930 tokens**. Es decir, 931 **cubre cualquier rejilla legal con uno de sobra**. El
injerto cambió una cota *estructural* (lo que el formato permite) por una cota *estadística* (lo
que se ha visto en este task), y las cotas estadísticas fallan justo en la cola.

**Consecuencia práctica**: cheap-first queda probablemente **absuelto**, y ya no hace falta gastar
un slot ni ~1 h-cuota en el A/B de aislamiento; la evidencia mecánica es más fuerte que la que
daría esa medida. Si algún día se quiere recuperar velocidad por esta vía, la forma correcta es
un tope **con techo estructural** (p. ej. `min(931, k·celdas_entrada + margen)` con k que cubra el
teselado 3×3), nunca uno derivado solo de lo visto en train.

## 33. `det` n=3 (5-sep): 30,28 — el prerreg falla, pero el resultado de fondo se consolida

Tercera muestra de `lb33-det`: **30,28**. El prerreg era 30,5-31,3 ⇒ **cae fuera**. Mi estimación
de rango con n=2 (0,42) era **optimista**, que es el error clásico de muestra pequeña: dos puntos
no acotan una dispersión.

| | muestras | media | rango | σ |
|---|---|---|---|---|
| campeón viejo | 30,14 · 29,31 · 27,64 | 29,03 | 2,50 (6 tareas) | 1,273 |
| **`lb33-det`** | 31,11 · 30,69 · 30,28 | **30,69** | 0,83 (2 tareas) | **0,415** |

Aun así el canal es **3,1× más estrecho**, que era el objetivo. Y con n=3 en ambos brazos aparece
lo importante: **separación completa**, la peor corrida de `det` (30,28) está por encima de la
mejor del campeón viejo (30,14). Test sin supuestos (Mann-Whitney, n=m=3): **p = 1/20 = 0,05**.
Sumado a que la predicción era prerregistrada y con mecanismo, **`lb33-det` es campeón de verdad:
+1,66 pt = +4 tareas.**

**El veredicto de `stack` sobrevive**, y conviene decir con qué fuerza exactamente. `stack` (29,44)
queda por debajo de las tres muestras de `det`, y por **0,84 pt (2 tareas) bajo la peor**, que es
más que todo el rango de `det`. En z son −3,0, pero con σ estimada de n=3 ese número es frágil;
lo que sostiene el veredicto no es el z sino **el mecanismo probado en §32** (el teselado 3×3 es
literalmente ingenerable bajo ese tope). Estadística moderada + mecanismo demostrado.

**Regla de decisión actualizada** (sustituye a la de §21, que pedía >2,5 pt): con σ ≈ 0,42, una
sola muestra distingue efectos de **~1 pt (2,5 tareas)**. Sigue sin servir para injertos de una
tarea; para esos hacen falta 2-3 muestras por brazo.

⚠️ **Patrón a vigilar, no conclusión**: las tres muestras bajan de forma monótona y **equiespaciada
— exactamente una tarea entre consecutivas** (31,11 → 30,69 → 30,28). Con n=3 una secuencia
monótona sale por azar 1 de 6 veces, así que no prueba nada. Pero si continuara, apuntaría a una
**deriva** (la máquina del rerun volviéndose más lenta, o carga creciente en Kaggle) y eso
reabriría la hipótesis del deadline global, que §30 bis daba por poco probable. La cuarta muestra
lo dirá; es gratis.

## 34. Barrido del rango del TTT (5-sep): r=64 es gratis, r=16 es el filo, y la velocidad SATURA

12 tareas del eval, 4 rangos, pareado dentro de tarea, solo fase de TTT. Rango de las 7
proyecciones; `embed_tokens`/`lm_head` clavadas en 256 (ahí r=256 ya es rango completo sobre
matrices [16, 2560]); escala rsLoRA fijada en 2,0 en todos los brazos con `alpha = 2·√r`.

| r | params | loss media | Δ media | Δ mediana | peores | p (signo) | TTT medio | speedup |
|---|---|---|---|---|---|---|---|---|
| **256** | 529,8 M | 0,0469 | — | — | — | — | 147,4 s | 1,00× |
| **64** | 133,4 M | 0,0460 | −0,0010 | +0,0034 | 8/12 | 0,39 | 109,9 s | **1,34×** |
| **16** | 34,4 M | 0,0436 | −0,0034 | +0,0063 | 10/12 | **0,039** | 101,0 s | **1,46×** |
| 4 | 9,6 M | 0,0980 | +0,0511 | +0,0478 | 11/12 | 0,006 | 100,8 s | 1,46× |

**La media miente en r=16 y el test de signo lo destapa.** Su Δ media es negativa (parece
mejor) pero **10 de 12 tareas empeoran**; la media está dominada por un único outlier,
`20270e3b`, que mejora −0,2037 mientras las otras once suman +0,1631. Con la mediana y el
signo, r=16 **sí degrada de forma sistemática**, aunque poco. Es exactamente el tipo de lectura
que un promedio solo habría dejado pasar.

**Veredicto prerregistrado**: **r=64 es el candidato limpio** — ningún test detecta degradación
(media, mediana y signo de acuerdo) y ahorra **25,5% del TTT = 8,9% del pipeline completo**.
r=16 es el filo: 31,5% de ahorro pero con degradación sistemática detectable. r=4 queda
descartado (11/12 peores, p=0,006).

**Hallazgo estructural: la velocidad SATURA en r=16.** r=4 tarda 100,8 s frente a los 101,0 s
de r=16 — dos décimas de diferencia con 3,6× menos parámetros. Es decir, **por debajo de r=16
el coste ya no está en los parámetros del adaptador sino en el forward/backward del modelo
base congelado**. Corolario para la familia de "reducción de dimensionalidad": seguir
recortando parámetros **no puede** dar más velocidad. La siguiente palanca tiene que atacar el
cómputo del base, y la candidata natural es **adaptar solo las últimas K capas**: si la capa
adaptada más temprana está en la posición K, el backward no necesita propagarse por debajo de
ella, y eso sí recorta cómputo real.

⚠️ **Y el aviso que decide si esto vale algo: un TTT más rápido, por sí solo, NO sube el score.**
El pipeline no reinvierte el tiempo liberado; simplemente termina antes. La ganancia solo se
materializa si (a) el deadline global muerde, en cuyo caso caben más tareas, o (b) el ahorro se
**gasta** explícitamente en más vistas de decode. Por eso el veredicto de la cuarta muestra de
`det` (¿hay deriva? ¿muerde el deadline?) es el que gobierna qué hacer con estos 8,9%: si el
deadline no muerde, r=64 debe injertarse **junto a un gasto**, no solo.

## 35. `lb33-r64` commiteado y submitido (6-sep): el speedup transfiere a la MITAD

Injerto: rango del TTT 256 → 64, `alpha` 32 → 16 para dejar la escala rsLoRA en 2,0 exacta.
No hizo falta `rank_pattern` (que unsloth podría no propagar): `embed_tokens` y `lm_head` son
[16, 2560], su rango efectivo es `min(r,16)=16` tanto con 256 como con 64, así que **siguen en
rango completo**. Commit sano: 1426 s de pared (**0,79 h-cuota**), cero timeouts, 4 GPUs.

**El speedup transfiere a menos de la mitad de lo prometido.** Comparación por tarea contra el
log de `det` (mismas 4 tareas del smoke público):

| tarea | loss det | loss r64 | TTT det | TTT r64 | Δ reloj |
|---|---|---|---|---|---|
| 0934a4d8 | 3,04e−03 | 2,86e−03 | 416 s | 351 s | −15,7% |
| 36a08778 | 5,05e−03 | 7,15e−03 | 199 s | 182 s | −8,6% |
| 981571dc | 2,60e−04 | 3,12e−05 | 868 s | 790 s | −9,0% |
| aa4ec2a5 | 5,78e−03 | 6,63e−03 | 302 s | 267 s | −11,6% |
| **media** | | | **446 s** | **397 s** | **−11,0%** |

**El banco predijo −25,5% y producción da −11,0%: el 43% de lo prometido.** Causa probable: mi
banco usa un bucle de PyTorch con *gradient checkpointing activado*, mientras producción usa el
trainer de unsloth con `use_gradient_checkpointing=False`. Perfiles de cómputo distintos ⇒
sensibilidad distinta al número de parámetros del adaptador. **Regla que se añade: un banco de
VELOCIDAD solo es válido si replica el arnés de entrenamiento de producción, no solo su
calendario.** (El de fidelidad de adaptación ya lo sabíamos, §26; este es el gemelo para el
reloj.)

La adaptación, en cambio, no se rompe: la loss final queda en el mismo orden de magnitud, mejor
en 2 de 4 tareas y sin ningún caso degenerado. Y el smoke **sube de 2,5 a 3,0** (gana
`36a08778_1`), el primer movimiento del smoke en toda la campaña — con la salvedad obvia de que
son 4 tareas y no decide nada.

⚠️ **Aviso de método, que casi me cuesta un informe falso**: al extraer las loss con una regex
`([\d.]+)` capturé `3.12` de `3.122444e-05` y descarté el exponente, concluyendo que `981571dc`
había **divergido** (loss 3,12 ≈ peor que uniforme). Era al revés: **3,12e−05, ocho veces mejor
que `det`**. Parsear números en notación científica con una regex de dígitos y puntos es un
generador de falsos positivos alarmantes; el valor siempre se lee completo antes de interpretarlo.

**Prerreg vigente** contra `det` (30,69 ± 0,415): ≥31,3 adoptar (el deadline muerde) · 30,3-31,3
neutro (el ahorro hay que **gastarlo** en más vistas de decode; r=64 queda validado como
financiación) · ≤30,0 revertir. Con el speedup real en −11% y no −25%, la rama "el deadline
muerde y esto se convierte en tareas" pide un margen que ahora es la mitad, así que el resultado
más probable es la banda neutra.

## 36. VEREDICTO `lb33-r64` (6-sep): 29,72 — REVERTIR, y la loss del TTT queda invalidada como proxy

**29,72.** El prerreg era ≤30,0 ⇒ revertir. Se revierte. **`lb33-det` (30,69) sigue siendo el
campeón.** El injerto queda **−2,34 tareas** bajo la media de `det` y **−1,34 bajo su peor
corrida**; z = −2,35, por debajo de las tres muestras.

**El banco dijo "sin degradación detectable" y producción dice −2,3 tareas. Hay que entender por
qué, porque el fallo es del método, no de la suerte.**

Mirando las loss del propio banco, el diagnóstico estaba a la vista y no lo leí:

| r | loss mín | mediana | máx |
|---|---|---|---|
| 256 | 0,0021 | 0,0213 | 0,3169 |
| 64 | 0,0015 | 0,0355 | 0,1480 |
| 16 | 0,0030 | 0,0417 | 0,1132 |
| 4 | 0,0192 | 0,1012 | 0,2133 |

**Todas las loss están esencialmente en cero.** Con 128 vistas augmentadas de tres o cuatro
demostraciones, **cualquier** adaptador con capacidad suficiente las memoriza; la métrica está
**saturada** y por eso no podía ordenar 256 frente a 64. Solo despertó en r=4, cuando la
capacidad cae tanto que ni memorizar se puede.

Y el fondo del asunto: **la loss del TTT mide el ajuste a las DEMOSTRACIONES, no la solución de
la entrada de test.** Son cosas distintas — es exactamente la distinción entre memorizar y
generalizar. Lo que r=256 aporta sobre r=64 no es capacidad para ajustar las demos (ambos
llegan a ~1e−3) sino, aparentemente, capacidad para codificar una transformación que **transfiere
a la rejilla de test**. El proxy medía el lado equivocado de esa frontera.

**Regla que queda (y que invalida la familia entera de ablaciones "baratas")**: para cualquier
pregunta sobre la CALIDAD del TTT, la loss de entrenamiento **no sirve**; hay que medir aguas
abajo — como mínimo GT-en-pool sobre la entrada de test retenida. Eso es caro (exige decode) y
por tanto la promesa de "ablacionar hiperparámetros barato midiendo la loss" **está muerta**.

**Lectura positiva, que reorienta la dirección**: este resultado dice que **r=256 está haciendo
trabajo real**. La familia "abaratar el TTT" pierde su premisa — el TTT no tenía grasa que
recortar en el eje del rango. Encaja con lo ya sabido (`nottt` 9,58, `ttt05` 26,39, `ttt8` en
cuarentena): **cada intento de darle menos al TTT ha salido negativo, sin excepción.** La
pregunta interesante se invierte: si menos capacidad resta, ¿r=512 suma? Es la primera vez que la
evidencia apunta a *amplificar* y no a recortar.

Coste del experimento: 0,79 h-cuota y un slot, y compró el cierre de un eje entero más la
invalidación de un proxy que iba a gobernar todo el plan de ablaciones.

## 37. `lb33-r512` commiteado (7-sep) y la respuesta MEDIDA a r=768: no cabe

Commit sano: 1763 s (**0,98 h-cuota**), cero timeouts, **cero OOM** (las guardas nuevas no
dispararon), smoke reload 2,5. Submitido, ref 56068258.

**La memoria, con el tercer punto medido:**

| r | params | entrenamiento | inferencia |
|---|---|---|---|
| 64 | 132,5 M | 10.110 MB | 15.597 MB |
| 256 | 529,8 M | 13.150 MB | 17.109 MB |
| **512** | 1.059,6 M | **19.059 MB** | **19.125 MB** |

**La predicción de INFERENCIA acertó exacta: 19.125 MB previstos, 19.125 medidos.** La de
entrenamiento falló: predije 17.204 y salieron 19.059 (**+10,8%**), porque el crecimiento **no es
lineal**: 8,02 bytes/param entre r=64 y 256, pero **11,70 entre 256 y 512**.

**Y eso cierra r=768 con datos, no con opinión.** Extrapolando con la pendiente reciente:

> r=768 → entrenamiento ≈ **24.968 MB = 24,4 GB**, contra los **24,0 GB** que declara la L4
> (y ~22,5 útiles tras el contexto CUDA). **OOM seguro, y en el TTT, no en el decode.**

La diferencia importa: un OOM en decode cuesta un lote de 4 vistas; **un OOM en TTT cuesta la
tarea entera**, porque sin adaptación esa tarea no se resuelve. Con 240 tareas en el oculto, eso
no es un riesgo marginal sino una pérdida sistemática. **r=768 queda descartado por memoria,
independientemente de lo que diga el score de r=512.**

Nota metodológica que vale para el futuro: **el binding constraint se movió**. A r=256 mandaba
la inferencia (17.109 vs 13.150); a r=512 mandan las dos por igual (19.125 vs 19.059) y hacia
arriba manda el **entrenamiento**, que además crece más deprisa. Extrapolar el eje equivocado
habría dado luz verde a r=768.

**Si el score de r=512 sube y quisiéramos más rango**, primero hay que pagar la memoria del
entrenamiento, no la del decode. Candidatos, en orden de menor a mayor intrusión:
`use_gradient_checkpointing=True` (hoy está en `False`; cambia activaciones por cómputo),
optimizador de 8 bits si unsloth no lo usa ya, y solo en último caso fusionar el adaptador antes
del decode. Todos son cambios de comportamiento que exigirían su propio A/B.

## 38. VEREDICTO `lb33-r512` (7-sep): 24,31 — el eje del rango queda CERRADO POR ARRIBA Y POR ABAJO

**24,31.** Contra los 30,69 del campeón son **−6,38 pt = −15,3 tareas**. Revertido. Y la magnitud
es la que enseña algo, porque **no puede ser un efecto de capacidad**:

| brazo | capacidad | Δ reloj por tarea | resultado |
|---|---|---|---|
| `r64` | menos | **−11,0%** | −0,97 pt = **−2,3 tareas** |
| `r512` | más | **+11,9%** | −6,38 pt = **−15,3 tareas** |

**`r512` hace 6,6× más daño que `r64`, y en la dirección donde había MÁS capacidad.** Si esto
fuese el eje de la capacidad, más rango no podría costar seis veces más que quitarlo. Lo que
cambia monótonamente entre los dos brazos no es la capacidad sino **el coste por tarea**: r=512
tarda un **+11,9%** en cada tarea (TTT +11,7%, medido en las 4 del smoke).

**Dos mecanismos lo explicarían y no puedo distinguirlos con lo que Kaggle expone**:
(a) el **deadline global** trunca la cola — un +11,9% sobre 240 tareas se come la holgura y deja
fuera las últimas; (b) **OOM en el TTT** de las tareas grandes — el pico medido fue 19,1 GB en la
mayor de 4 tareas, y las guardas nuevas convierten cada OOM en una tarea perdida. El log de la
corrida oculta no es accesible, así que ambas quedan como hipótesis. **La conclusión práctica no
depende de cuál sea**: r=512 es inutilizable en este pipeline.

**El eje del rango queda cerrado por ambos lados**: 64 resta por capacidad, 512 resta por
recursos, **256 es el punto de operación** y no hay nada que rascar ahí.

**Y el aviso estratégico, que vale más que el experimento**: el pipeline está **cerca de un
acantilado de recursos**. Un +11,9% de coste por tarea costó 15 tareas. Cualquier injerto que
añada trabajo por tarea — más vistas de decode, más pasos de TTT, thinking — carga con un riesgo
muy superior a su coste nominal, y hay que presupuestarlo **antes** de diseñarlo, no después.

⚠️ **Error mío, y van tres con el mismo patrón**: validé la holgura de memoria sobre el smoke
público de **4 tareas** (19,1 GB de pico, "22% de margen") y lo di por representativo de **240**.
Es la misma equivocación de población que ya me costó el banco de dosis (§26) y el tope de
longitud (§32): **el smoke de 4 tareas no puede acotar los extremos del oculto**, ni de tiempo ni
de memoria. Para presupuestar recursos hay que razonar sobre la distribución completa de tamaños
—que sí tenemos offline— y no sobre la muestra pública.

## 39. EL PRESUPUESTO (7-sep): el campeón corre al 95% de su reloj, y eso rige todo lo demás

En vez de seguir adivinando el mecanismo de §38, se **infiere desde el propio leaderboard**, que
es más fiable que cualquier modelo de coste:

> `r512` añadió **+11,9%** de coste por tarea y perdió **15,3 tareas = 6,4%** de las 240. Luego
> completó el **93,6%**. Si el campeón completa el 100%, entonces
> `deadline / coste_base = 1,119 × 0,936 = 1,048`.
>
> **El campeón corre al 95% de su presupuesto de tiempo. Holgura real: 4,8%.**

**Y con eso los tres resultados encajan en un solo modelo, sin piezas sueltas:**

| brazo | efecto | por qué el score |
|---|---|---|
| `det` | completa el 100%, sobra 4,8% | **30,69** |
| `r64` | −11% de reloj, −capacidad | ahorrar **no da nada** (ya terminábamos); solo se ve la pérdida de capacidad → **29,72** |
| `r512` | +11,9% de reloj | **se pasa del deadline** y pierde el 6,4% de la cola → **24,31** |

Esto **resuelve** la ambigüedad de §38 (deadline vs OOM) a favor del **deadline**, y de paso
explica por qué `r64` no ganó pese a ser más rápido: **no hay premio por terminar antes.**

### El número que gobierna el resto de la campaña

> **PRESUPUESTO OPERATIVO: ningún injerto puede añadir más de ~4,8% de coste por tarea.**

| injerto candidato | coste estimado | veredicto |
|---|---|---|
| **thinking** (cadenas de razonamiento) | ×2-5 en decode | **IMPOSIBLE** |
| decode 16 → 24 vistas | +28,6% del total | **IMPOSIBLE** |
| más pasos de TTT | proporcional | **IMPOSIBLE** |
| scoring 8 → 4 augmentaciones | **−4% (ahorra)** | viable, calidad por medir |
| `default_weights` a RAM | 0% tiempo, −1 GB GPU | neutro en score |

Esto **mata el pendiente del SFT con thinking tal y como estaba planteado**: no es que sea mala
idea, es que no cabe en el reloj. Para que entrara habría que financiarlo con un ahorro
equivalente, y el único ahorro identificado sin pérdida de capacidad conocida es el scoring.

### Auditoría de memoria (lo que pidió el usuario)

`arc_solver.py:317` guarda `default_weights` — **una copia completa del adaptador (530 M params
en bf16 ≈ 1,04 GB) residente en GPU toda la corrida**, usada solo para resetear entre tareas.
Pasarla a RAM es gratis en tiempo (una copia CPU→GPU de 1 GB por tarea ≈ 0,1 s contra 400-900 s
de tarea) y libera ~1 GB. Confirmado además que la pendiente de memoria medida (3,81 bytes/param
en inferencia) corresponde exactamente a **dos** copias del adaptador en bf16 (2+2), que es la
firma de este duplicado.

⚠️ Pero con el diagnóstico de arriba, **la memoria NO es el cuello de botella: lo es el tiempo.**
El campeón nunca ha registrado un OOM. Liberar 1 GB compra robustez frente a la cola de tareas
grandes, no puntos. **No justifica un slot por sí solo**, y meterlo junto a otra cosa violaría la
regla de un injerto por brazo.

### Lectura estratégica honesta

Tres injertos consecutivos revertidos (`stack`, `r64`, `r512`) y un presupuesto de reloj del
4,8%. El campeón es un **óptimo local sólido y estrecho**: casi cualquier cosa que se mueva o
cuesta capacidad o cuesta tiempo, y ambas restan. Las ganancias que queden **no están en los
hiperparámetros de este pipeline**; están en hacer el mismo trabajo más barato (para luego
gastarlo) o en un cambio de método.
