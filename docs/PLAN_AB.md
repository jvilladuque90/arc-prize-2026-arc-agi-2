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
