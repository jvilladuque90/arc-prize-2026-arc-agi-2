# Tácticas de entrenamiento e inferencia — explicación técnica extensa

> Documento de referencia técnica del proyecto ARC Prize 2026 (ARC-AGI-2, track offline de Kaggle).
> Explica **el problema**, **cada táctica que usamos o probamos** (qué es, por qué existe, cómo la
> implementamos, qué medimos) y **dónde estamos**. Complementa
> [working_note_es.md](working_note_es.md) (narrativa y resultados) y
> [RESULTS.md](RESULTS.md) (números duros).
>
> Última actualización: 2026-07-29.

---

## Parte I — El problema

### 1. Qué pide ARC-AGI-2

Cada **tarea** es un puzzle independiente que trae:

- unos pocos **pares de demostración** (`train`): grids de entrada → grids de salida,
- uno o más **grids de test**, de los que hay que predecir la salida exacta.

Un **grid** es una matriz 2D de hasta 30×30 celdas, cada una con un color 0–9 (0 = negro/fondo).
La transformación que relaciona entrada y salida es **abstracta** (simetrías, conteo, gravedad,
contención, relleno, escalado, correspondencias de color…) y **se debe inferir solo de los
ejemplos**: no hay descripción textual, no hay dos tareas con la misma regla.

**La métrica es `pass@2` con exact-match:** por cada grid de test se envían **dos intentos**, y
solo cuenta si uno reproduce la salida **celda por celda**. No hay crédito parcial. Un grid de
30×30 con 899 celdas correctas y 1 mal vale exactamente lo mismo que uno completamente en blanco.

Esta métrica es más dura de lo que parece y condiciona todo el diseño: no sirve "acercarse". En
nuestros primeros experimentos medimos tareas con **95–98% de celdas correctas y cero puntos** —
por eso descartamos temprano las métricas de similitud y solo trackeamos exactas.

### 2. Por qué es difícil (y por qué el 2 importa)

ARC está diseñado para medir **generalización composicional**, no conocimiento memorizado:

- **Los humanos rondan el 60%**; las máquinas offline con cómputo capeado, el 20–30%.
- **ARC-AGI-2 vs ARC-AGI-1:** el survey 2026 documenta una degradación **universal de 2.5–3×**
  al pasar de AGI-1 a AGI-2, en *todos* los paradigmas. La causa medida es la **profundidad
  composicional**: la solución media pasa de ~1.3 pasos (AGI-1) a ~2.7 pasos (AGI-2). Ya no basta
  con reconocer "una" transformación; hay que **encadenar varias**, y el espacio de búsqueda
  explota.
- **Es "knowledge-bound".** Un LLM base sin entrenamiento específico en ARC saca ≈0 exactas. Lo
  medimos nosotros: nuestros SFT propios con 8K ejemplos sintéticos (`train v10/v11`) dieron
  **cero soluciones exactas** pese a bajar la loss a 0.043. El conocimiento sobre transformaciones
  de grids no está en el pre-entrenamiento; hay que ponerlo.

### 3. Las restricciones de la competencia (y lo que implican)

| Restricción | Valor | Implicación de diseño |
|---|---|---|
| Internet | **prohibido** | Modelo, pesos y dependencias deben venir de Kaggle Models/Datasets adjuntos. Todo el desarrollo es "a ciegas": no se puede `pip install` ni descargar en el kernel de submission. |
| Hardware | **4× NVIDIA L4** (24 GB cada una, 96 GB total) | Un 8B en bf16 (16 GB) **no cabe para entrenar** en una sola L4. Fuerza paralelismo de modelo o cuantización. |
| Tiempo | **12 h** por submission | Es la restricción dominante. Con 240 tareas, son **180 s/tarea** de media si se quisieran cubrir todas. Cualquier método que gaste más obliga a dejar tareas sin resolver. |
| Métrica | **pass@2**, exact-match | Dos hipótesis por tarea; conviene que sean **diversas**, no dos variantes de lo mismo. |
| Scoring | **~50% semi-privado** (visible) + **~50% privado** (oculto, decide el ranking) | No sobre-optimizar el LB visible: el ranking final se juega en datos que nunca vemos. |
| Cadencia | **1 submission/día** | Cada experimento en el LB real cuesta un día. Obliga a validar offline antes de gastar el slot. |
| Quota GPU | ~30 h/semana | Presupuesto explícito. Regla propia: **dejar 2 h de holgura** para submissions. El *rerun* de scoring NO gasta quota; solo "Save & Run All". |

**La consecuencia central de las restricciones:**

```
score  ≈  Cobertura  ×  Calidad
          (cuántas       (cuántas de
           tareas te      esas aciertas)
           da tiempo
           a atacar)
```

Como el tiempo es fijo, **cobertura y calidad compiten**. Gastar más cómputo por tarea sube la
calidad pero baja la cobertura. Todo nuestro trabajo de optimización se ha movido en esta frontera,
y varios de nuestros resultados negativos se explican exactamente por ella (§Parte III).

---

## Parte II — Las tácticas, explicadas

Aquí está el núcleo: qué hace cada pieza, por qué existe y cómo la implementamos.

### 4. SFT — Supervised Fine-Tuning (el conocimiento base)

**Qué es.** Entrenamiento supervisado clásico: se le dan al modelo pares (entrada, salida deseada)
y se ajustan sus pesos para maximizar la probabilidad de la salida. En ARC, la "entrada" es la
tarea serializada como texto y la "salida" es el grid correcto (transducción) o un programa que lo
produce (inducción).

**Por qué es imprescindible.** Es donde vive la habilidad. Los ganadores lo confirman:

- **NVARC** (1º 2025, 24.03% privado) generó **266.000 puzzles sintéticos** y con ellos hizo SFT
  de un Qwen3-4B. Ese SFT *es* su ventaja competitiva.
- **BARC** hizo lo análogo para inducción: ~400K ejemplos (tarea → programa Python) generados con
  GPT-4, para entrenar un Llama-3.1-8B.

**Nuestra posición.** No re-hacemos el SFT de transducción: **reutilizamos el checkpoint de NVARC**
(`sorokin/qwen3_4b_grids15_sft139`), lo que nos sitúa de entrada al nivel del ganador. Es una
decisión deliberada: pararse sobre el SFT de otro salta la parte más cara y donde somos más débiles
(nuestros intentos con 8K sintéticos dieron cero exactas).

⚠️ **Contaminación importante:** el SFT de ese checkpoint **incluyó el set público de evaluación de
ARC-AGI-2**. Por tanto **nunca validamos ese modelo contra `arc-agi_evaluation_challenges.json`** —
mediría memorización, no capacidad. Solo el leaderboard oculto nos da señal honesta. (Documentado
en [LB33_AUDIT.md](LB33_AUDIT.md); nos costó aprenderlo: un smoke de 4 tareas daba ~perfecto y era
completamente engañoso.)

### 5. LoRA — Low-Rank Adaptation (cómo entrenamos sin romper la memoria)

**Qué es.** En vez de actualizar los pesos `W` de cada capa (miles de millones de parámetros), LoRA
congela `W` y aprende dos matrices pequeñas `A` (d×r) y `B` (r×d) con **r muy pequeño**, de forma
que el peso efectivo es `W + BA`. Solo se entrenan `A` y `B`.

**Por qué la usamos.**
1. **Memoria:** entrenar un 8B completo requiere ~4× los pesos en memoria (pesos + gradientes +
   dos momentos de Adam). Con LoRA se entrenan ~0.1–1% de los parámetros → cabe en las L4.
2. **Velocidad:** menos parámetros que actualizar, backward más barato.
3. **Reversibilidad:** el adapter es un fichero pequeño e independiente; se puede aplicar, quitar o
   fusionar (`merge_and_unload`) sin tocar el modelo base. Esto es **crítico para TTT** (§6).

**Los hiperparámetros y qué significan:**

| Parámetro | Qué controla | Nuestros valores |
|---|---|---|
| `r` (rango) | Capacidad del adapter. Más alto = puede aprender más, cuesta más memoria | **256** en el TTT del campeón (muy alto — el TTT necesita capacidad para "reprogramar" el modelo hacia una tarea); **16–32** en nuestros SFT |
| `lora_alpha` | Escala de la actualización (`BA · alpha/r`) | 32 |
| `target_modules` | Qué capas adapta | El campeón adapta **todo**: `q,k,v,o` (atención) + `gate,up,down` (MLP) + **`embed_tokens` y `lm_head`** (las embeddings — inusual, pero en ARC el vocabulario relevante son 12 tokens y reajustar sus embeddings ayuda) |
| `use_rslora` | Escala estabilizada (`alpha/√r` en vez de `alpha/r`) | ✔ — importante con r=256, si no la actualización es demasiado grande |

### 6. TTT — Test-Time Training (la táctica clave del campeón)

**Qué es.** La idea que separa a los ganadores del resto: **entrenar el modelo sobre la propia
tarea, en el momento de resolverla**. Cada tarea de ARC trae sus pares de demostración; en vez de
solo ponerlos en el prompt, se hace un **fine-tuning corto sobre ellos** antes de generar la
respuesta.

**Por qué funciona.** El survey 2026 lo cuantifica: **+20–30 puntos** sobre el mismo modelo
congelado, y **todo sistema que supera el 70% en AGI-1 lo usa**. La intuición: la tarea define su
propia mini-distribución; unos pocos pasos de gradiente sobre sus ejemplos mueven al modelo mucho
más cerca de esa regla concreta que cualquier prompt.

**Cómo lo implementa nuestro pipeline (por tarea):**

1. **Restaurar** el adapter a su estado inicial (`set_peft_model_state_dict` con los pesos
   guardados). Esto es esencial: sin restaurar, la tarea N heredaría el entrenamiento de la N−1 y
   se contaminarían.
2. **Construir el set de entrenamiento** a partir de los pares de la tarea, **aumentado ×16**
   (§7): 16 variantes geométricas/de color de los mismos ejemplos.
3. **Entrenar 1 época** con LoRA r=256, `lr=5e-5`, `adamw_torch`, scheduler coseno, batch 1.
4. **Cambiar a modo inferencia** (`FastLanguageModel.for_inference`), decodificar (§8), puntuar
   (§9) y seleccionar (§10).
5. Liberar memoria (`gc.collect()`, `torch.cuda.empty_cache()`) y pasar a la siguiente tarea.

**El coste.** Esto es lo que hace que cada tarea cueste ~900 s en vez de ~5 s. Es el origen del
conflicto cobertura↔calidad: el TTT es tan caro que **el modelo no llega a atacar las 240 tareas
en 12 h**. Ese es literalmente nuestro cuello de botella medido.

### 7. Aumentación de datos (por qué ARC la permite y la exige)

**Qué es.** ARC tiene una propiedad que casi ningún dominio tiene: **las transformaciones válidas
del grid preservan la semántica de la tarea**. Si rotas 90° la entrada *y* la salida, la regla
sigue siendo la misma regla. Si permutas los colores de forma consistente, también.

**Las aumentaciones que usamos:**

- **Transposición** (×2)
- **Rotaciones** de 90°, 180°, 270° (×4)
- **Permutación aleatoria de los 10 colores** (consistente entre entrada y salida)
- **Reordenación de los ejemplos de demostración** (`shuffle_ex`) — el orden no debería importar

**Para qué sirven, en tres sitios distintos:**

1. **En el TTT (×16):** convierte 3–5 pares de demostración en ~50–80 ejemplos de entrenamiento.
   Sin esto, el fine-tuning por tarea sobre-ajustaría al instante.
2. **En la inferencia (×16 "subkeys"):** se decodifica la *misma* tarea bajo 16 vistas distintas
   (permutaciones × rotaciones × transposición). Cada vista da candidatos; luego se **des-transforma**
   cada candidato (`invert_mod`) para volver al marco original. Esto genera **diversidad real**.
3. **En el re-scoring (×8):** para evaluar cuán buena es una respuesta candidata (§9).

**Detalle de implementación importante:** invertir la transformación tiene que ser exacto. El
pipeline mantiene la transformación codificada en la propia *clave* del ejemplo
(`tarea.rot90.permute0492…`) y la deshace operación por operación en orden inverso.

### 8. Decodificación DFS (`turbo_dfs`) — buscar en vez de muestrear

**El problema con la generación normal.** Un LLM genera token a token. Con muestreo se obtiene
diversidad pero poca precisión; con greedy, una sola respuesta. Para exact-match hace falta
**explorar sistemáticamente** las secuencias de alta probabilidad.

**Qué hace `turbo_dfs`.** Una búsqueda en profundidad sobre el árbol de tokens, **podada por
probabilidad acumulada**:

- El vocabulario relevante son **12 tokens**: los dígitos `0-9`, el salto de línea (`Ċ`) y el
  fin-de-secuencia. No hace falta considerar los ~150.000 tokens del vocabulario de Qwen.
- Cada rama acumula su **NLL** (log-verosimilitud negativa). Se **poda** cuando supera el umbral
  `max_score = -log(0.2)`, es decir, cuando la secuencia acumulada baja del 20% de probabilidad.
- Se expande en **lotes**: mantiene N ramas vivas y avanza todas a la vez, reutilizando el
  **KV cache** (`past_key_values`) — de ahí "turbo".
- Devuelve **todas** las secuencias completas encontradas bajo el umbral, ordenadas por score.

**Optimización que aportamos.** El baseline transfería los logits del vocabulario completo a CPU
en cada paso. Reescribimos el cálculo para que sea **algebraicamente idéntico** pero transfiriendo
solo los 12 valores relevantes (`index_select` sobre los tokens ARC, `logsumexp` para la
normalización). Mismo resultado, mucho menos tráfico GPU→CPU.

**Por qué DFS y no beam search.** El beam mantiene un ancho fijo; el DFS con umbral de probabilidad
adapta el esfuerzo: en tareas donde el modelo está seguro explora poco y termina rápido; donde
duda, explora más. Encaja con el presupuesto de tiempo variable.

### 9. Re-scoring por consistencia aumentada (`calc_scores`)

**El problema.** El DFS devuelve varios candidatos con su score de generación (`beam_score`). Pero
ese score está sesgado: mide la confianza del modelo *bajo la vista concreta* en la que generó.

**La táctica.** Para cada candidato se hace una **evaluación teacher-forced bajo 8 aumentaciones**:
se le presenta al modelo la tarea transformada y la respuesta candidata (también transformada), y
se mide la log-verosimilitud de esa respuesta. Un candidato **realmente correcto** debería ser
probable bajo *todas* las vistas; uno que fue un accidente de una vista concreta, no.

Esto es una forma barata de **verificación por consistencia**: no ejecuta nada (a diferencia de la
síntesis de programas, §11), pero explota la invarianza del dominio.

**Implementación:** un solo `forward` sin KV cache (no hace falta, es teacher-forced), recogiendo
solo las log-probs de los tokens objetivo. Se guarda `score_aug` (8 valores) por candidato.

### 10. Selección de los 2 intentos

Con `pass@2` hay que elegir **dos** de entre todos los candidatos. Dos criterios implementados:

- **`score_kgmon` (consenso):** `nº de veces que apareció el candidato − media(score_aug)`.
  Domina el **conteo**: premia a los grids que múltiples vistas produjeron *independientemente*.
  Es un voto por mayoría corregido por confianza. **Es el que usa el campeón.**
- **`score_full_probmul_3` (confianza):** suma de `(baseline − beam_score)` más la media de
  `Σ(baseline − score_aug)`. Domina la **probabilidad**: premia al candidato del que el modelo está
  más seguro, aunque lo haya producido una sola vista.

**Nuestro experimento (fallido):** inspirados por el paper de "trace judging" de AGI-2 —donde un
juez LLM fuerte recupera respuestas *minoritarias pero correctas* que el consenso descarta—
probamos un **ensemble de selección**: `attempt_1` = consenso, `attempt_2` = confianza cuando
discrepan. Resultado: **neutro** (29.31% vs 29.72%). La razón: nuestro "juez" son dos fórmulas
**correlacionadas** sobre las mismas señales (beam + augmentaciones), no un razonador independiente.
Cuando discrepan, ninguna es fiablemente mejor.

### 11. Síntesis de programas verificable (el paradigma alternativo)

**La idea.** En vez de que el modelo *dibuje* el grid de salida (transducción), que **escriba un
programa Python** que transforme entrada→salida (inducción). Entonces:

```
generar K programas  →  EJECUTAR cada uno sobre los pares de demostración
                     →  quedarse solo con los que reproducen TODOS exactamente
                     →  aplicarlos al test
```

**Por qué es atractivo.** El paso de verificación es **real**: no es una heurística de confianza,
es una comprobación ejecutable. Un programa que reproduce los 3–4 pares de demostración exactamente
tiene una probabilidad alta de ser la regla correcta. Los aciertos son **fiables**, lo que los hace
ideales para *añadir* cobertura sin arriesgar lo que ya funciona.

**Cómo lo implementamos** (siguiendo la receta de BARC, ver [PROGRAM_SYNTHESIS.md](PROGRAM_SYNTHESIS.md)):

- **Prompt:** plantilla de chat Llama-3.1; los grids se serializan como **nombres de color**
  ("Black Blue Red…"), no como dígitos.
- **Librería común:** los programas generados hacen `from common import *` y usan primitivas de un
  módulo de 1.551 líneas (`Color`, `flood_fill`, `blit`, `crop`, `find_connected_components`,
  detección de simetrías…). La vendorizamos en `vendor/barc/common.py`.
- **Extracción:** regex sobre el bloque ` ```python … ``` `.
- **Ejecución sandboxeada:** `exec` con **timeout** (SIGALRM) para que un programa con bucle
  infinito no cuelgue el kernel; captura de cualquier excepción (muchos programas generados
  crashean, es normal).
- **Verificación:** `np.array_equal` contra cada salida esperada.

**Lo que medimos (FASE 0):** el modelo BARC-8B **congelado** logra un **verifiable-hit rate del
6.25%** en ARC-AGI-2 (1 de 16 tareas produce un programa que pasa todos los pares) — y **ese
acierto generalizó** al test. El mecanismo funciona; el modelo, entrenado en 2024 sobre AGI-1, es
débil en AGI-2 (el "cliff" §2). Además es caro: ~485 s/tarea con K=16.

### 12. Generación de data sintética compositiva (nuestra táctica propia)

**El razonamiento.** Si el cuello es el SFT (§4) y AGI-2 exige profundidad composicional (§2),
entonces necesitamos **datos de entrenamiento más compositivos**. NVARC y BARC generaron los suyos
con GPT-4 — imposible para nosotros (offline, sin presupuesto de API).

**El hallazgo que lo hizo posible.** Los ~163 *seeds* de BARC no son ejemplos estáticos: son
**generadores ejecutables**. Cada uno trae:

```python
# concepts: simetría, relleno
# description: en la entrada verás… para hacer la salida…
def main(input_grid):      # el solver
def generate_input():      # genera entradas aleatorias de esta familia
```

Ejecutar `generate_input()` → `main()` produce pares (entrada, salida) reales e **ilimitados**.

**Nuestra composición (A→B):**

```
entrada  ← generate_input_A()
x        ← main_A(entrada)         # primer paso
salida   ← main_B(x)               # segundo paso
```

y el **programa objetivo** del SFT concatena ambos cuerpos más un `transform` que los encadena.

**Filtros de calidad que impusimos:** descartar identidad (entrada == salida), salidas degeneradas
(un solo color), y —clave— exigir que **ambos pasos hagan algo** en la mayoría de los pares (si no,
es una tarea de 1 paso disfrazada).

**La propiedad valiosa:** cada tarea generada se **verifica ejecutando su propio programa** contra
todos sus pares. **100% de las tareas generadas verifican** → la data es **correcta por
construcción**, sin necesidad de ningún LLM. Generamos **13.271 tareas** (128 conceptos, 6.037
combinaciones distintas de seeds) en ~1,7 h de CPU, **sin gastar nada de quota GPU**.

**El resultado (negativo) y por qué.** El SFT sobre esa data entrenó limpio (loss 4.07 → 0.50) pero
el A/B dio **0/16 frente al 1/16 del congelado**. La causa, ya diagnosticada:

> **Verificable ≠ útil.** Encadenar dos seeds *aleatorias* produce tareas correctas pero **no
> ARC-like**: las tareas reales tienen una regla **coherente e inferible** por un humano; una
> cadena de dos transformaciones sin relación semántica a menudo **no es inferible en absoluto**.
> El modelo aprendió la **forma** (programas de dos partes) en vez del razonamiento.

*Si se retoma esta vía*, el arreglo sería componer solo pares **semánticamente compatibles** y
emitir el programa **inline** (una función coherente) en vez de dos cuerpos + dispatcher.

### 13. `cheap-first` — la única palanca de cobertura que funcionó

**El problema medido.** El pipeline procesa las tareas de una cola compartida entre 4 workers. Con
~900 s/tarea y 12 h, **no da tiempo a las 240**: las que no se alcanzan quedan como `[[0]]`
(placeholder) y valen cero.

**La táctica.** Ordenar la cola de **barata a cara**, usando como proxy de coste la **suma de
celdas** de todos los grids de la tarea (correlaciona con la longitud del prompt y por tanto con el
tiempo de TTT y decodificación). Desempate por clave, para determinismo.

**Por qué funciona.** No cambia *nada* del cómputo por tarea: solo el orden. Pero drena más tareas
antes del muro de las 12 h. **Medido: 28.47% → 30.14% (+1.67).** Es nuestro único injerto que
sobrevivió al test público, y confirmó que **la cobertura era el cuello**.

### 14. Restricciones de ingeniería offline (lo que cuesta que algo *corra*)

Buena parte del esfuerzo real no es algorítmico sino de infraestructura. Lo documentamos porque
cada punto costó iteraciones:

- **No hay `bitsandbytes`** en ninguna de las imágenes disponibles → **no se puede hacer 4-bit /
  QLoRA** ni usar `adamw_8bit`. Esto elimina la solución estándar para entrenar un 8B en 24 GB.
- **Solución adoptada:** `device_map="auto"` reparte el modelo 8B bf16 **entre las 4 L4s** (96 GB
  totales) → entra sin cuantizar y sin necesidad de gradient-checkpointing.
- **Efecto colateral:** con el modelo repartido, las `labels` (GPU 0) y los `logits` del `lm_head`
  (GPU 3) están en dispositivos distintos → `RuntimeError`. Lo resolvimos con un `DevTrainer`
  (subclase de `Trainer`) cuyo `compute_loss` mueve las labels al device de los logits.
- **Imagen Docker pinneada obligatoria.** La imagen "latest" de Kaggle rotó a Python 3.12 con
  `torchao 0.10`, incompatible con `peft` (exige >0.16) → `ImportError` al aplicar LoRA. **Cualquier
  kernel que use peft debe usar la imagen pinneada.**
- **Los modelos montan vía symlink** en `/kaggle/input`: `glob(recursive=True)` **no** los cruza;
  hay que usar `os.walk(followlinks=True)`.
- **`model_sources` es case-sensitive** (`Transformers`, no `transformers`) y Kaggle **descarta en
  silencio** una referencia inválida (la metadata queda `[]`) → siempre verificar la metadata
  committeada.
- **Modelos gated:** la copia de BARC en Kaggle está bloqueada por la licencia de Llama (403). El
  repo original en HuggingFace es abierto → lo resolvimos con un kernel de descarga (CPU+internet)
  que produce nuestra propia copia como artefacto reutilizable.
- **El CLI de Kaggle en Windows** falla al leer notebooks con acentos (cp1252) y **el push falla en
  silencio** (exit≠0, sin mensaje) → forzamos `PYTHONUTF8=1` y comprobamos el returncode. Este bug
  nos hizo enviar una versión equivocada del kernel una vez.
- **Los logs de la imagen BYOD vuelven vacíos** al descargarlos → para cualquier cosa que haya que
  depurar, usar la imagen estándar y **escribir los resultados a un fichero JSON** en
  `/kaggle/working`, no confiar en el log.

---

## Parte III — Dónde estamos

### 15. Resultados medidos (leaderboard semi-privado, `pass@2`)

| # | Configuración | Score | Δ |
|---|---|---|---|
| 0 | Baseline LB33/NVARC (auditado limpio) | 28.47% | — |
| 1 | **+ `cheap-first`** | **30.14%** | **+1.67** ✅ **campeón** |
| — | + presupuesto de decode adaptativo | 28.06% | −2.08 ❌ |
| — | ensemble con Qwen3-2B | ≈30% | break-even ❌ |
| — | ensemble con TRM (split temporal) | 27.22% | −2.92 ❌ |
| — | re-tirada del campeón (test de varianza) | 29.72% | ruido |
| — | ensemble de selección (consenso+confianza) | 29.31% | neutro ❌ |

**Cuatro tiradas del mismo pipeline: 28.47 / 30.14 / 29.72 / 29.31.** El ruido run-to-run es de
~±1%, y **~30% es un techo sistémico** para nosotros con esta configuración.

### 16. Las tres conclusiones estructurales (todas medidas)

**(a) El modelo fuerte quiere *todo* el cómputo.** Probamos dos ensembles con partners más débiles
(Qwen3-2B: 18.33% solo; TRM recursivo de 7M: 8.33% solo). Ambos fallaron por la **misma razón
airtight**: el Qwen 4B es *coverage-limited* (no llega a las 240 tareas), así que cualquier cómputo
cedido a otro modelo **le quita cobertura**, y el partner débil no repone lo que desplaza. En el
caso del TRM lo medimos exactamente: darle al Qwen 9.5 h en vez de 11.4 h costó ~3% de cobertura y
el TRM rellenó ~0 del tail caro. **Corolario: no volver a intentar ensembles con partners más
débiles.**

**(b) El punto de operación del tiempo ya es óptimo.** Tres mediciones: más tiempo por tarea
(presupuesto adaptativo) → −2.08; el punto actual (1.200 s) → 30.14%; menos tiempo (cap 600 s) →
degradación severa de calidad. **Ambas direcciones empeoran.**

**(c) Estamos en el techo del paradigma equivocado.** El survey 2026 sitúa la **transducción pura
en 5–40%** y muestra que los ganadores son **híbridos** (inducción + transducción + TTT, 50–79%).
Nuestro ~30% *es* el nivel del ganador 2025 (cuyo checkpoint usamos). Superarlo requiere cambiar de
paradigma, no ajustar parámetros.

### 17. Estado del Paso 3 (SFT de inducción) — cerrado con negativo

| Fase | Qué era | Estado |
|---|---|---|
| **0** | Probe de viabilidad de la síntesis verificable | ✅ **6.25%** hit rate, y generalizó → el mecanismo funciona |
| **A** | Montar y validar la infra de SFT del 8B offline | ✅ loss 1.65 → 1.08 |
| **B** | Generar data compositiva propia | ✅ **13.271 tareas verificadas**, 0 quota GPU |
| **C** | SFT sobre esa data + A/B | ✅ SFT limpio (4.07 → 0.50) · ❌ **A/B: 0/16 vs 1/16** |

**Las tres lecciones metodológicas** (documentadas en RESULTS.md, valen para todo lo que venga):

1. **Verificable ≠ útil.** La corrección de la data sintética no implica su **relevancia
   distribucional**. Datos 100% correctos pero con la forma equivocada enseñan la forma.
2. **El instrumento de medida es parte del experimento.** El SFT alargó las salidas del modelo y el
   evaluador las truncaba a 1024 tokens sin cerrar el bloque de código → medimos el evaluador, no
   el modelo. Al cambiar la distribución de salida, **revisar siempre los límites del evaluador**.
3. **Poder estadístico.** Un probe de n=16 sobre tasas del ~6% **no puede distinguir 0/16 de 1/16**.
   Para decidir sobre mejoras pequeñas hace falta n mucho mayor (caro: ~1.000 s/tarea) o una
   métrica más densa que el exact-match por tarea.

### 18. Activos reutilizables construidos

Aunque el Paso 3 diera negativo, queda infraestructura que sirve para cualquier intento futuro:

- **Pipeline de SFT del 8B offline** validado (`device_map=auto` + `DevTrainer`, sin bitsandbytes).
- **Generador de data verificable** (`src/arc/seedgen.py` + 163 seeds vendorizados): produce tareas
  correctas por construcción sin LLM y sin quota GPU.
- **Pipeline de síntesis verificable** completo: prompt, extracción, ejecución sandboxeada con
  timeout, verificación.
- **Copia propia del BARC-8B** (evita el gating de Llama) como artefacto de Kaggle.
- **Recetario de infraestructura offline** (§14) — el conocimiento que más iteraciones costó.

### 19. Opciones abiertas (pendiente de decisión)

1. **Rediseñar la data compositiva:** componer solo pares semánticamente coherentes + programa
   inline. Reusa toda la infra. Corrige la causa raíz diagnosticada, pero es otra ronda larga con
   probabilidad incierta.
2. **Medir el híbrido congelado (1 submission):** la incógnita nunca medida es si los ~6%
   verificables del BARC congelado son **disjuntos** del 30% del Qwen. Si lo son, suman hasta +6%
   **sin entrenar nada**. Es barato y cierra una pregunta abierta.
3. **TTT sobre la inducción:** la palanca de +20–30 pt de la receta ganadora, que aún no hemos
   aplicado al lado de inducción.
4. **Consolidar el ~30%** y dejar de invertir en apuestas de alta incertidumbre.
