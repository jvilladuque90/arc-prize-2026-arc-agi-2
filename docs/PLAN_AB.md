# Plan A / B — la campaña de saltos grandes (desde 2026-09-09)

## Marco (decisión del usuario, 9-sep)

> *"no pienses en tiempo, estamos muy atrás; si eso abre la posibilidad de saltos de score
> grandes vale la pena"*

Estamos en **30,69** contra un líder en **57%**. Un +1 o +2 no cambia nada: solo consume
calendario. **Queda descartado todo lo que viva en la banda de ruido**: el Qwen3.5-2B por
velocidad (vale 0, no hay comprador para el reloj liberado), el ajuste de rango, el de vistas.

Deadline: **2026-11-02**. Presupuesto autorizado para esta campaña: **10 h-cuota**
(el tope semanal del proyecto sigue siendo 10 h y el pool se comparte con agi-3).

## El diagnóstico sobre el que se construye todo

- `score ≈ P(GT en el pool) ≈ 31%` ⇒ **la selección ya es sin pérdidas**. Mejor scoring vale 0,
  mejor verificación *de lo que ya generamos* vale 0. Solo cuenta **cambiar lo que se genera**.
- El modelo acierta la **forma en el 71%** y el contenido en el 7%: aplica *una* transformación
  plausible con seguridad; falla en **cuál**.
- Fase 0 (0 h-cuota): la entrada de test cae fuera de la envolvente de las demos y ese hueco
  **crece con el nivel del benchmark** (AGI-1 0,500 → AGI-2 train 0,750 → AGI-2 eval 1,161,
  p=2e-8). El exceso es **contenido, no vacío**: área ×1,43, tinta ×1,45, **componentes ×1,00**
  (mismos objetos, más grandes). Y la **salida** exigida cae fuera de la envolvente de las salidas
  demo en el **94,7%** de las tareas.

## Plan A — entrenar contra el déficit de escala

Parchear la transducción para que extrapole escala. SFT continuado sobre `sft139` con datos
sintéticos donde la regla se demuestra a escala pequeña y el objetivo va a escala grande,
calibrado al ×1,43 de área y ×1,45 de tinta medidos.

- Techo estimado **+10-15**. Riesgo medio. Coste ~7-10 h-cuota.
- Virtud: apoyado en **nuestra propia** medición.
- Riesgo principal: que los datos sintéticos enseñen el **atajo** ("si la entrada es grande,
  agranda la salida") en vez de la generalización. Sería la tercera lección de data sintética.

## Plan B — inducción aprendida a un DSL compacto + verificación exacta

El modelo emite un **programa**; el programa se ejecuta sobre las entradas de las demostraciones
y solo entran al pool los que **reproducen todas** las salidas demo. Es la mitad de "inducción"
del híbrido que gana arriba.

**Las dos razones por las que lo habíamos cerrado son falsas para esta versión:**

1. *"El modelo no tiene palabras (16 tokens)"* → **no hace falta modelo nuevo**: se extiende la
   matriz de embeddings de 16 a ~64 filas con tokens de DSL y se entrenan solo esas filas en un
   SFT continuado sobre `sft139`. Se conserva su conocimiento de ARC, el formato de roles y el TTT.
2. *"El thinking cuesta ×8-116 de reloj"* → **emitir un programa es más barato que emitir la
   rejilla**: 30-60 tokens frente a 100-931. Si el programa se ejecuta para producir la salida, el
   decode de la rejilla desaparece. **El techo del 4,8% no bloquea esto**; puede ser negativo.

- Techo **+25 o más** si el DSL cubre. Riesgo **alto**. Coste ~20-40 h-cuota.
- Precedente honesto: tenemos dos negativos en la vecindad, pero fueron **composición mecánica**
  (búsqueda sobre un DSL). Esto es **emisión aprendida**: el modelo propone, la búsqueda solo
  verifica. Es una diferencia real, no una garantía.
- Riesgo que manda: **la expresividad del DSL es el techo**. Si no expresa AGI-2, no hay
  entrenamiento que lo abra.

## Relación entre A y B: NO son excluyentes

- **B se traga el mecanismo de A.** El problema de escala es un problema de *transducción*: existe
  porque hay que emitir token a token una rejilla mayor que ninguna vista. **Un programa es
  indiferente a la escala.** Eso sube el valor de B y baja el de A.
- **Son complementarias en cobertura**: B resuelve lo que el DSL expresa, la transducción cubre el
  resto. La unión **es** la arquitectura híbrida que gana. Ninguna sola llega.
- **Pueden compartir inferencia**: una sola adaptación de TTT (34,8% del reloj) y dos modos de
  decode.
- **Restricción real 1**: el LB no atribuye cambios agrupados (lección `lb33-stack`) ⇒ **medir por
  separado aunque se desplieguen juntas**.
- **Restricción real 2**: cada SFT es una tirada del dado sobre el 71% de formas correctas que ya
  tenemos. Dos SFT encadenados son dos tiradas.

## El orden lo decide una puerta que cuesta 0 h-cuota

**Puerta de B** (primero, porque decide cuál de las dos es el plan principal): medir la
**cobertura y la precisión de verificación** de un DSL de juguete por enumeración, en CPU, sobre
las tareas etiquetadas. Prerregistro en `docs/PREREG_dsl.md`.

- Si el DSL cubre bien ⇒ **B es el plan principal**, A pasa a respaldo para lo que el DSL no exprese.
- Si cubre mal ⇒ **B muere en la puerta** y A es el plan principal, sin haber gastado GPU.

**Puerta de A**: validar el proxy `nll/celda de la GT` con datos ya pagados (`augdose` + `rbuyer`)
y generar los datos de escala. También CPU.

## Regla de instrumentación que cambia la campaña

El **leaderboard es nuestro instrumento más barato y potente**: n=240 por ~2 h-cuota del commit.
El banco `rbuyer` costó 1,6 h-cuota para 7 pares de los que solo **2** informaron. El LB es del
orden de **100× más informativo por hora-cuota** para efectos de capacidad.

⇒ **Dejar de gastar cuota en comprar mediciones y gastarla en comprar capacidad.** Lo caro es
entrenar; medirlo no lo es. Límites: un injerto por brazo, 1 slot/día, y su banda de ruido (±0,4)
resuelve ~1 punto ≈ 2,4 tareas — un arreglo que ataque el 69% de las tareas debe ser mucho mayor.

---

# ACTUALIZACIÓN 2026-09-09 (tras las puertas gratis): B muerto, y A REDIRIGIDO

## Plan B: muerto en la puerta (0 h-cuota)

`score_dsl` = **0,0%** en eval AGI-2 y **0 de 353** tareas AGI-2 en total, con el mismo DSL sacando
**5,1%** en AGI-1 y **87,2% de precisión de verificación**. Detalle en `docs/PREREG_dsl.md`.
La premisa de B (la verificación exacta filtra) era **correcta**; lo que no hay es qué verificar.

## Plan A: el objetivo era el equivocado, y ahora está medido

Yo había planteado A como *"la entrada de test es más grande, entrena la extrapolación de escala"*.
**Los datos dicen que la entrada no tiene nada que ver.** Sobre las 23 tareas de eval con resultado
del campeón (`augdose` + `rbuyer`):

| | Spearman con `vistas_con_gt` |
|---|---|
| `d` de la **ENTRADA** de test | **+0,004** — no predice **nada** |
| `d` de la **SALIDA** de test | **−0,234** |

Y la tasa de acierto condicionada es lo que manda:

| régimen | GT en algún pool | población en eval |
|---|---|---|
| salida **DENTRO** de la envolvente de las salidas demo | **3/4 = 75%** | **18%** |
| salida **FUERA** | **2/19 = 11%** | **82%** |

**El campeón es casi competente (75%) cuando la respuesta se parece en forma y estadística a las
salidas que las demos ya mostraron, y casi inútil (11%) cuando no.** Y el 82% de las tareas están
en el régimen difícil. Reconstruye el score global: 0,18×75% + 0,82×11% ≈ 22%, del orden del ~19-31%
que medimos.

⚠️ n=23 (4 frente a 19 en el corte). Mann-Whitney p=0,078 a una cola. **Es una hipótesis con
soporte cuantificado, no un hecho establecido.** Confirmarla es barato y no necesita GPU: ampliar
`d_salida` × resultado sobre más tareas ya pagadas.

## Qué hay que entrenar, entonces

De las 98 tareas de eval con la salida fuera de la envolvente, el rasgo que la saca es:

| rasgo | domina | se sale (alguno) |
|---|---|---|
| **H** | 35% | 64% |
| **W** | 17% | 60% |
| log área | 8% | 72% |
| nº colores | 14% | 38% |
| nº componentes | 10% | 18% |
| densidad | 9% | 37% |

⇒ **60% del hueco es tamaño de la salida, 39% es contenido.**

**El objetivo del entrenamiento cambia**: no es *"aplica la regla a una entrada más grande"* sino
**"produce una salida cuyas dimensiones y estadística caen FUERA del rango que las demostraciones
mostraron, deduciéndolas de la entrada"**. Es extrapolación **del lado de la salida**, que es
justo donde un autorregresivo se rompe y donde todos los gradientes del TTT apuntan al lado
contrario (las salidas demo son más pequeñas en el 94,7% de las tareas).

## Orden de trabajo

1. **(0 h-cuota)** Confirmar el mecanismo con más n, y generar los datos sintéticos calibrados al
   perfil medido (60% tamaño / 39% contenido del lado de la salida).
2. **(GPU)** SFT continuado sobre `sft139` con esos datos. Un solo injerto, atribuible.
3. **(1 slot)** Medir en el leaderboard, que es n=240 por ~2 h-cuota.

---

# RESULTADO `sft-outenv` (2026-09-09, ~2,0 h-cuota): BANCO VACUO — `sft139` memoriza el training set

El detector de vacuidad prerregistrado **saltó**.

| | valor |
|---|---|
| VAL NLL/token **antes** de entrenar | **0,0039** |
| VAL NLL/token después | 0,0087 (**+122,7%**) |
| pasos por rango | 2.073 (de 2.308) |
| re-particiones / secuencias | 727 / 9.232 |

**0,0039 por token = 99,61% de probabilidad por token.** Sobre una salida objetivo de 150 tokens
(la mediana), eso es un **55,6% de probabilidad de emitir la salida ENTERA correcta** — sin TTT,
sin adaptación, sobre tareas que el split apartó como "nunca vistas en este entrenamiento".

Comparación que lo cierra: el **mismo** modelo, **ya adaptado por TTT**, sobre eval, tiene
`gt_nll` de 0,010 a 0,312 por token (banco `rbuyer`). **El modelo sin adaptar es 3-80× más
confiado sobre el training set que el modelo adaptado sobre eval.** Eso no es capacidad, es
**memorización**: `sft139` ya vio las 1.000 tareas de entrenamiento de ARC.

Que el entrenamiento **empeorase** la métrica (+122,7%) es coherente: meter un adaptador nuevo
sobre un ajuste ya perfecto solo puede perturbarlo.

## Lo que esto mata, que es más que esta corrida

**El training set de ARC no sirve como fuente de señal para `sft139`, se re-particione como se
re-particione.** No es un fallo del diseño de la re-partición —que funcionó: 727 re-particiones
reales, cero fuga entre train y val, supervisión correcta verificada en 300/300 secuencias— sino
de la **fuente de datos**. Y explica por qué "más SFT sobre datos de ARC" no podía funcionar.

## Fuentes de datos que sí podrían servir (verificar la vacuidad ANTES de entrenar)

1. **re-ARC** — generadores procedurales de las reglas de ARC-1 que producen **instancias nuevas a
   tamaño arbitrario**. Encaja perfecto con lo que necesitábamos: demos a escala pequeña y objetivo
   a escala grande, con reglas reales y **sin memorización**. Limitación honesta: son reglas de
   ARC-1, y la puerta del DSL sugiere que AGI-2 juega a otra cosa ⇒ transferencia parcial.
2. **`nvidia/Nemotron-SFT-ARC-AGI-v1`** — 67K/207K muestras, CC BY 4.0, **con los IDs de eval en
   lista negra** (limpio respecto a nuestro instrumento de medida).
3. BARC (ya hay entradas `barc-*` en el registro de kernels).

**En los tres casos, correr primero el detector de vacuidad**: si la NLL del objetivo sale ~0 antes
de entrenar, esa fuente tampoco sirve. Cuesta minutos y acaba de ahorrarnos interpretar mal 2 h-cuota.

## La alternativa que NO necesita datos nuevos, y que ahora es la favorita

El mecanismo medido es que **todos los gradientes del TTT apuntan a salidas más pequeñas** que la
exigida (las salidas demo lo son en el 94,7% de las tareas). Eso se puede atacar **sin entrenar
nada y sin datos**: reponderar la pérdida del TTT hacia la demostración de salida más grande, o
dejarla fuera y adaptarse sobre las pequeñas para que el TTT ensaye la extrapolación con los pares
que la tarea ya trae.

- Coste: **~0 de reloj** (mismo presupuesto de 128 secuencias, distinta composición).
- Es **un solo injerto**, medible en el leaderboard, que es n=240 por ~2 h-cuota.
- No depende de resolver el problema de la fuente de datos.

---

# `gate35` (2026-09-10, ~0,5 h-cuota): mi hipótesis del artefacto era FALSA

Re-medición del Qwen3.5-4B con **diagnóstico de carga** y **control entre espejos**.

## El diagnóstico de carga

| modelo | clase elegida | claves ausentes | inesperadas | params |
|---|---|---|---|---|
| `q3_4b` titular | `AutoModelForCausalLM` | **0** | 0 | 4.022M |
| `m_lima` (Qwen3.5-4B) | `AutoModelForImageTextToText` | **0** | 0 | **4.539M** |
| `m_thanh` (Qwen3.5-4B) | `AutoModelForImageTextToText` | **0** | 0 | **4.539M** |

`gate-base` había cargado 4.206M con `AutoModelForCausalLM`; con la clase correcta son 4.539M.
Faltaban **333M**… **y no cambian nada**.

## El resultado, idéntico al anterior hasta la cuarta cifra

| candidato | n | ratio mediano | gana | signo p | fwd × |
|---|---|---|---|---|---|
| `m_lima` | 24 | **1,299** | 8/24 | 0,968 | 1,43 |
| `m_thanh` | 24 | **1,299** | 8/24 | 0,968 | 1,44 |

**Control entre espejos: ratio mediano 1,0000, desviación máxima 0,0000** — son bit a bit el mismo
checkpoint. Y `nll/celda` mediano = **0,4361**, exactamente el mismo número que dio `gate-base` con
la carga "incompleta".

⇒ Los 333M que faltaban eran **solo la torre de visión**. `AutoModelForCausalLM` cargaba el tronco
de texto **perfectamente**. La medición original era correcta y **mi sospecha era infundada**.

## VEREDICTO (caso 2 del prerreg): la rama muere de verdad

**Qwen3.5-4B tiene peor prior de ARC que Qwen3-4B a igual tamaño** (ratio 1,299, gana 8/24,
p=0,968) **y encima no cabe en el reloj** (fwd ×1,43 contra un techo de ×1,048). Ahora está medido
con carga limpia y doble espejo: no hay a qué agarrarse.

## Estado de "modernizar a 3.5", completo

| vía | veredicto |
|---|---|
| **3.5-4B** (misma talla, más capacidad) | ❌ **peor prior**, medido limpio ⇒ muerta |
| **3.5-2B** (misma capacidad, menos reloj) | empata (0,989) pero el reloj liberado **no tiene comprador** ⇒ vale ~0 |
| 3.5-9B | ~2,2× de parámetros ⇒ fuera del techo de reloj por aritmética |
| 3.5-27B / 35B-A3B MoE | ya cerradas |

**El eje del modelo base queda agotado con mediciones sólidas.** La ganancia que buscamos no está ahí.

## Lo que sí deja, reutilizable

- `AutoModelForCausalLM` sobre un VLM de Qwen3.5 **carga el tronco de texto correctamente** y
  descarta la torre de visión sin corromper nada. No hay que temer esa ruta.
- **Cargar siempre con `output_loading_info=True`** y elegir la clase por *claves ausentes*, no por
  "la primera que no lanza excepción". Cuesta nada y convierte una sospecha en un hecho.
- Cuando un resultado parece implausible, el orden correcto es **instrumentar la sospecha**, no
  archivarla ni asumirla. Aquí la instrumentación costó 0,5 h-cuota y la sospecha era mía y falsa.

---

# AUDITORÍA (2026-09-10, a petición del usuario): ¿hay algún error?

## Lo que se verificó y está BIEN

- **`ttford` no tiene bug.** `cut_to_len` recorta por delante (`v[1:]`), así que la demostración
  mayor puesta al final **se conserva**; `max_seq_length=8192` hace el recorte raro; las claves
  `.ex` siguen siendo únicas porque el prefijo ya lleva los descriptores de D4/color. El −5,00 es
  efecto real del injerto, no de la fontanería.
- **El 75%/11% no es solo "barato = resuelto"**: el coste por sí solo no predice el resultado
  (Mann-Whitney p=0,301) mientras `d_salida` sí (Fisher exacto p=0,018 con n=24). La entrada sigue
  sin predecir nada (Spearman −0,035).
- `gate35`, la puerta del DSL, `sft-outenv` y la Fase 0 se sostienen.

## Los TRES errores encontrados — todos de razonamiento, ninguno de código

**1. "No hay comprador para el reloj" nunca se estableció.** `rbuyer` dio 7 pares de los que solo
**2 informaron**; yo mismo lo califiqué de *"medición inasequible, que no es lo mismo que un
negativo"*. Y sin embargo en los mensajes siguientes lo usé **tres veces como negativo** para cerrar
el camino del Qwen3.5-2B. **La pregunta del comprador está ABIERTA.** El 2B (empata el prior a
0,378× de tronco) está bloqueado por una pregunta sin respuesta, no por una respuesta negativa.

**2. Toda la evidencia por tarea viene del decil más barato del eval.** Las 24 tareas con resultado
del campeón tienen coste mediano **781** frente a **1.586** del eval completo — **percentil 11** —
y máximo 2.132 frente a 7.200. El 75%/11% son **3 de 4 tareas** en ese decil. Y no solo esto:
"la selección es sin pérdidas" y "16 vistas = 1 hipótesis" salen de las mismas 16 tareas baratas.
CLAUDE.md ya advertía que *"el smoke público de 4 tareas NO acota los extremos del oculto"*: lo
repetí con 24 en vez de 4. **El oculto son 240 tareas de toda la distribución; nuestro mapa cubre
el 10% más fácil.**

**3. El fallo de `ttford` tiene dos mecanismos y "dosis" solo es segura bajo uno.** (a) *La
diversidad de orden es estructural* — mi lectura. (b) *Atajo posicional*: "lo último es lo grande"
se aprende en el TTT y se filtra al decode, donde la consulta de test también va última. En el
eval, el GT del test es **más pequeño** que la mayor salida demo en el **15%** de las tareas —
ahí el atajo engaña. Bajo (b), dosificar a 16/128 instala la misma sesgo, más débil. **No puedo
distinguir (a) de (b) con el score del LB**, y "dosis, no sustitución" lo di por bueno sin
esa distinción.

## Consecuencia

La condición *"si no hay error, adelante con dosis"* **no se cumple**. El siguiente gasto útil no es
otra submission a ciegas sino **cerrar el punto ciego del error 2**, del que dependen los otros dos.

---

# `strat` (2026-09-11, 2,5 h-cuota): el mapa real, y una palanca que cae y otra que nace

**Cae**: la envolvente de salida. En 20 tareas estratificadas p5-p95, dentro 1/4 = 25% frente a
fuera 3/16 = 19% (Fisher p=0,62). El 75%/11% era un artefacto del decil barato. Con ello mueren
el mecanismo de `ttford` y **"dosis, no sustitución"** — sin gastar su slot. Detalle en
`docs/PREREG_strat.md`.

**Nace**: el **proxy validado**. `NLL/celda de la GT post-TTT` predice GT-en-pool con Spearman
−0,430 y separación ×11,6 (0,0032 vs 0,0370); el pre-TTT no sirve (+0,086). Informa en el 100% de
las tareas frente al 20% de GT-en-pool ⇒ **~10× de potencia por hora-cuota**.

**El mapa**: GT-en-pool 20% global, plano por coste (17/33/12% por tercios). El coste no explica
el fallo. Contaminación medida: 1/20 tareas con `pre` = 0 (`36a08778`) ⇒ excluir `pre` ≈ 0.

## Estado del plan tras strat

| palanca | estado |
|---|---|
| envolvente de salida / dosis | ❌ refutada en dificultad real |
| modelo base (3.5-4B, 27B, MoE) | ❌ cerradas con mediciones limpias |
| Qwen3.5-2B (empata a 0,378× tronco) | ⏸ bloqueada por la **pregunta del comprador**, que ahora **sí es asequible** |
| proxy NLL post-TTT | ✅ **validado** — infraestructura para lo siguiente |

## Siguiente paso que se deriva, no que se inventa

Reabrir la pregunta del comprador **con el proxy**: `r512` vs `r256` sin presión de reloj, pareado,
medido por NLL post-TTT en las ~20 tareas estratificadas (excluyendo contaminadas). Con todas las
tareas informativas, n=20 pareado sí tiene potencia para un efecto moderado. Es la pregunta que
decide si el 2B vale +12 o vale 0, y por primera vez se puede contestar a ~2 h-cuota.

---

# `rb2` (2026-09-12, 2,5 h-cuota): la pregunta del comprador, contestada a medias

Con el proxy validado, r512 vs r256 sin presión de reloj sobre 15 pares (los 4 más caros se
perdieron: r512 costó ×1,32 en TTT, no ×1,15). **Ratio mediano 0,839, r512 mejor en 10/15, signo
p=0,151.** Pasa la magnitud, falla la significancia ⇒ **empate por prerreg ⇒ rama Qwen3.5-2B
bloqueada.** Sensibilidad (no veredicto): sin la tarea memorizada con denominador 0, p=0,090.

**Estado del tablero tras cinco días:**

| palanca | estado |
|---|---|
| modelo base (3.5-4B, 27B, MoE) | ❌ cerradas con mediciones limpias |
| Qwen3.5-2B | ⏸ bloqueada: el rango paga poco (~−16% NLL ≈ +2 pt) y no supera el listón |
| envolvente de salida / dosis / ttford | ❌ refutadas |
| DSL / inducción | ❌ 0 de 353 en AGI-2 |
| SFT sobre el training set | ❌ memorizado por sft139 |
| proxy NLL post-TTT | ✅ validado — la única palanca de *medida* que ganamos |

**Campeón intacto: `lb33-det` 30,69 ± 0,415.** Lo que la campaña ha producido es un mapa de lo
que NO funciona, con mediciones en las que se puede confiar, y un instrumento 10× más barato.

---

# AUDITORÍA de `gate35` (2026-10-31, a petición del usuario): el veredicto «peor de verdad» está SOBREAFIRMADO

Lo que se midió bien: carga con 0 claves ausentes, dos espejos bit a bit iguales, misma población y
mismo prompt para ambos brazos, NLL solo sobre los tokens de la GT. Lo que NO se controló:

1. **Variante.** Los espejos (`limamateus`, `thanhduc1102`) son `Qwen/Qwen3.5-4B`, el modelo
   **post-entrenado** (piensa por defecto, `<think>`). Se midió completado crudo sin plantilla de
   chat. El post-training de 3.5 es más agresivo que el de Qwen3, así que el ratio 1,299 mezcla
   «prior de rejillas» con «castigo por sacar a un modelo de chat de su formato». Existe
   `Qwen/Qwen3.5-4B-Base` (y `qwen-lm/qwen-3/4b-base` como titular justo) y **no se probó**. Para
   un SFT tipo NVARC lo que importa es la Base.
2. **Kernels.** El config oficial tiene 24 de 32 capas `linear_attention` (Gated DeltaNet) y 8 de
   atención completa. Sin `flash-linear-attention` y `causal-conv1d` (no están en la imagen ni en
   `tf52-wheels`), transformers cae al fallback de torch: numéricamente correcto (diferencia media
   de logits ~3e-4 según el test de paridad de OLMo-core) pero **lento**. El `fwd ×1,43` es muy
   probablemente artefacto del fallback: un híbrido 3:1 con kernels debería igualar o batir al
   Qwen3-4B denso. **La puerta de reloj NO está medida limpia.** La de prior (1,299) no se
   explica por esto.
3. **Sin control absoluto.** Ninguna NLL sobre texto conocido para validar la ruta de carga; el
   control entre espejos solo prueba que son los mismos bytes.
4. **Vocabulario 248K** (vs 151K): 0,64B de parámetros en embeddings atados; recortable a 16 filas
   como hizo Sorokin, igual que planeamos para el 2B.
5. Población = 24 tareas más baratas (percentil 11), ya anotado el 10-sep.

**Veredicto corregido:** «Qwen3.5-4B post-entrenado, en completado crudo y con fallback de
kernels, tiene peor NLL de rejilla que Qwen3-4B post-entrenado». NO está establecido que la
generación 3.5 tenga peor prior a igual talla. Re-medición limpia = `Qwen3.5-4B-Base` vs
`Qwen3-4B-Base`, kernels FLA instalados, control de texto plano, ~0,5 h-cuota + subir 9 GB.
**Aun con ratio ≤0,85, la vía exige rehacer el SFT de NVARC (266K secuencias, 10–20 h-cuota,
historial 0/2)** ⇒ no es para el 2-nov; es la primera pregunta de la próxima temporada.

## RESULTADO `gate35b` (2026-10-31, 388 s de pared ≈ 0,22 h-cuota, 36 tareas, 4 modelos, carga limpia 0 claves ausentes)

| candidato vs `Qwen3-4B-Base` | n | ratio mediano | gana | p(signo) | fwd |
|---|---|---|---|---|---|
| **`Qwen3.5-4B-Base`** (baratas) | 24 | **1,012** | 11/24 | 0,73 | ×1,25 |
| `Qwen3.5-4B-Base` (medias) | 11 | 1,084 | 1/11 | 1,00 | ×1,17 |
| `Qwen3.5-4B-Base` (todas) | 35 | **1,044** | 12/35 | 0,98 | — |
| `Qwen3.5-2B-Base` (todas) | 36 | 1,293 | 3/36 | 1,00 | ×0,52 |
| `Qwen3-4B` post-entrenado (todas) | 36 | **1,444** | 1/36 | 1,00 | ×1,08 |

Control absoluto (nats/token, inglés / python): 4B-Base 1,77/0,49 · 3.5-4B-Base 1,77/0,41 ·
3.5-2B-Base 1,83/0,47 · 4B-post 2,23/0,52 ⇒ las cuatro rutas de carga están sanas.

**Lectura.**
1. **La auditoría tenía razón en el mecanismo**: el post-training castiga el completado crudo
   (Qwen3-4B post/base = 1,444). El 1,299 de `gate35` era en gran parte formato, no prior.
2. **Pero corregido el protocolo, la generación 3.5 NO es mejor a igual talla**: 3.5-4B-Base
   empata con Qwen3-4B-Base en baratas (1,012, 11/24) y pierde en medias (1,084, 1/11).
   Ratio global 1,044, gana 12/35. Falla el prerreg (≤0,85 y ≥2/3) por el lado del empate.
3. El 3.5-2B-Base es claramente peor (1,293, 3/36); el 0,989 de septiembre era el post-entrenado
   contra un post-entrenado, es decir, dos castigos que se cancelaban.
4. fwd ×1,17-1,25 con FLA disponible (`is_flash_linear_attention_available()=True`); no se
   pudo confirmar por introspección que las capas usaran el kernel (el detector no encontró el
   atributo), así que el coste real queda entre ×1,0 y ×1,25. El 2B es ×0,52.

**Veredicto corregido y definitivo para esta temporada:** Qwen3.5-4B-Base tiene el MISMO prior de
rejilla que Qwen3-4B-Base. Cambiar de base no da ventaja de partida; lo único que podría darla es
que el 3.5 se entrene mejor en el SFT, y eso solo se sabe haciendo el SFT (10–20 h-cuota, 0/2).
Con un prior empatado y un coste ≥×1,0, no hay caso alcista medible: **vía cerrada con medición
limpia.** Datos: `scripts/_gate35b.json`. La versión «3.5-4B es peor» queda sustituida por «3.5-4B
empata».


## RESULTADO sft35 (1-nov-2026): la generación 3.5 tampoco APRENDE mejor

SFT pareado (9K seq NVARC, vocab 16, LoRA r=64 todas las lineales, 3 h, L4) de Qwen3-4B-Base (2 semillas) vs Qwen3.5-4B-Base (4 semillas): NLL/celda post-SFT mediana 0,176–0,177 vs 0,160–0,179, medias 0,237 vs 0,241–0,249, loss a tokens iguales idéntica, 0 exactas greedy en todos, 3.5 un 8 % más lenta. **Empate ⇒ vía cerrada definitivamente** junto a gate35b (prior). Detalle y limitaciones en `docs/PREREG_sft35.md`.
