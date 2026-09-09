# Sobre los hombros del ganador: una baseline honesta de transducción, un espacio de palancas agotado, y la apuesta por programas verificables — Working Note (ARC Prize 2026 / ARC-AGI-2)

*Working Note — ARC Prize 2026, ARC-AGI-2 (Kaggle, track offline).*
**Autor:** Julian Camilo Villa Duque.

> Documento vivo. Objetivo: **ganar**, no solo participar. Se actualiza con cada experimento —
> problema, features, estrategias exitosas/fallidas, resultados, métricas, proceso. Números duros en
> [RESULTS.md](RESULTS.md). **Explicación técnica extensa de cada táctica de entrenamiento e
> inferencia y del problema: [TACTICAS.md](TACTICAS.md)** (en inglés: [TACTICS.md](TACTICS.md)).
> Docs acompañantes: [PROGRAM_SYNTHESIS.md](PROGRAM_SYNTHESIS.md), [SFT_PLAN.md](SFT_PLAN.md),
> [LB33_AUDIT.md](LB33_AUDIT.md), [STRATEGY.md](STRATEGY.md).

---

## El problema en breve (primer de 60 segundos)

**ARC-AGI-2** da, por tarea, unos pocos pares input→output de **grids** (celdas de color 0–9) que
ejemplifican una transformación abstracta; hay que inferir la regla y aplicarla a un test input
oculto. Está diseñado para ser **fácil para humanos (~60%) y duro para máquinas** — premia la
**generalización composicional**, no la memorización.

**Restricciones (Kaggle, track offline):**

| Restricción | Valor |
|---|---|
| Internet | ninguno (offline); modelo + deps desde Kaggle Models/Datasets adjuntos |
| Hardware / tiempo | L4×4 (96 GB), **12 h** por submission |
| Métrica | **pass@2** (2 intentos/test), exact-match |
| Scoring | **~50% semi-privado** (LB público, visible tras cada submit) + **~50% privado** (oculto, decide el ranking) |
| Cadencia | **1 submission/día**; el rerun de scoring **no** gasta quota GPU (solo Save&Run la gasta) |

**Referencias.** SOTA privado 2025 = **NVARC 24.03%**. Los frontier APIs (GPT-5.5 ~85%) **no** son
comparables — corren online y sin cap; este track es offline y capeado en cómputo.

**Qué exige "ganar".** El campeón 2025 llegó arriba con **data sintética a escala + SFT +
test-time training (TTT) + ensemble**. Esa es la receta de referencia que replicamos y superamos.
No hay atajo barato por encima de ~30%.

---

## Abstract

Nuestro sistema actual es el **pipeline de transducción LB33/NVARC** (checkpoint Qwen3-4B SFT'd en
ARC + TTT LoRA + decodificación DFS) con un único injerto — **orden cheap-first** — que lo sube de
28.47% a **30.14%** semi-privado drenando más tareas antes del muro de 12 h. Establecemos, con
cuatro tiradas del LB (28.47 / 30.14 / 29.72 / 29.31), que **~30% es un techo sistémico firme para
nosotros** con este pipeline, y que **coincide con el nivel del ganador privado 2025** cuyo
checkpoint reutilizamos — nuestro score *es* el nivel del ganador. Esta nota está deliberadamente
cargada hacia **lo que falló**, porque los resultados negativos definen la frontera. Cada palanca
barata/media dentro del paradigma de transducción está ya **agotada con medición**: ensemble
arquitectónicamente diverso (un partner Qwen3-2B es break-even; un TRM recursivo de 7M bajo split
temporal *regresa* a 27.22%, porque el Qwen fuerte es coverage-limited y cualquier partner más
débil que le quite cómputo pierde más cobertura de la que añade); ajuste del presupuesto de tiempo
por tarea (adaptativo −2.08, cap 600 s degradó severamente); y calidad de selección (un ensemble de
selección fundamentado en el SOTA — "juez holístico" — es **neutro** a 29.31%, porque nuestros dos
scorers son heurísticas correlacionadas, no un juez fuerte de razonamiento). Una revisión completa
del SOTA explica el muro: el survey 2026 sitúa la **transducción pura en un techo de 5–40%** y
muestra que los ganadores son **híbridos** (inducción + transducción + TTT, 50–79%); los tops del
leaderboard son o frontier-API (prohibidos offline) u opacos. La conclusión disciplinada es que las
mejoras reales requieren un **cambio de paradigma**, no más ajustes — así que nos comprometemos con
la receta completa del ganador en tres pasos escalonados: (1) un híbrido de dos modelos *congelados*
ya-SFT'd — **síntesis de programas verificable de BARC** (generar solvers Python → ejecutar →
quedarse con los que reproducen **todos** los train pairs → aplicar al test) unida a nuestra
transducción NVARC, cuyos aciertos verificados *añaden* cobertura sin sacrificar el 30%; (2) **TTT**
encima; y (3) el trabajo largo y decisivo — **nuestro propio SFT con 100–266K puzzles sintéticos**,
el núcleo de lo que hizo ganar a NVARC, que diferimos hasta que los pasos 1–2 se estanquen y que
señalamos honestamente como nuestra capacidad más débil (nuestros intentos de SFT con 8K sintéticos
dieron cero soluciones exactas). Reportamos la contribución medida de cada componente y somos claros
en que hoy estamos al nivel del fork público, no en la cabeza del board.

*Nota de alcance: compartimos todo el razonamiento, los insights de datos y cada resultado negativo
— ahí está el valor. Las constantes/features exactas quedan en el repo mientras la competencia está
viva.*

---

## 1. La tarea, y a qué se reduce

Cada tarea es un conjunto de pares de demostración más uno (o pocos) test inputs; predecir cada test
output exacto, dos intentos. Dos hechos moldean todo el enfoque:

- **Es knowledge-bound.** El razonamiento ARC está atado a lo que el modelo ya sabe de
  transformaciones de grids. Un LLM base *sin SFT-de-ARC* saca ≈0 exactas (medido: nuestro train
  v10/v11 con 8K ejemplos sintéticos → cero exactas). NVARC necesitó **266K puzzles sintéticos** para
  llegar a 24%. El SFT — no lo nuevo del base — es donde vive la habilidad.
- **Score ≈ Cobertura × calidad.** El pipeline es time-bounded (TTT + DFS por tarea bajo 12 h con 4
  workers). Qué tareas terminan antes del muro es tan decisivo como la exactitud por tarea — por eso
  nuestro único injerto exitoso es de *cobertura*, y por eso cualquier cómputo dado a un modelo más
  débil es un impuesto de cobertura.

---

## 2. Insights (paradigmas y la forma del techo)

- **La jerarquía de paradigmas (survey 2026).** Transducción neural pura: **5–40%** (la más débil,
  nuestra familia). Síntesis de programas: 26–79% pero con un **cliff AGI-1→AGI-2** severo.
  **Híbrido** (inducción + transducción + TTT): **50–79%, dominante.** Todo sistema >70% en AGI-1
  usa TTT, genera ≥10 hipótesis y verifica.
- **Qué separa ~55% de ~30%:** múltiples hipótesis, percepción robusta, **razonamiento híbrido con
  verificación**, refinamiento iterativo, TTT (+20–30 pt). Los sistemas de ~30% son estáticos,
  congelados-tras-entrenar, de una sola representación.
- **Los números de los ganadores son SFT-driven.** El moat de NVARC son sus 266K sintéticos + SFT;
  ARChitects usó 102 pasos de refinamiento recursivo; MindsAI TTFT + augmentation voting. Ninguno es
  un ajuste ligero.
- **El techo offline es real.** Los híbridos frontier-API llegan a 72.9% (p.ej. trace-judging
  holístico sobre GPT-5.2/Gemini-3/Opus, $39/tarea) — **inaplicable offline**. El techo capeado es
  mucho menor; igualar a NVARC (~24% privado / ~30% semi-privado) ya es "nivel ganador", y superarlo
  exige out-R&D-ear al ganador o un método offline genuinamente nuevo.

---

## 3. Método (pipeline actual desplegado)

Un tracker de transducción ARC-SFT congelado con un injerto de cobertura:

```
  tarea test ─▶ cola cheap-first (barata→cara por coste de celdas del grid)
              ─▶ por tarea, 4 workers GPU:
                   TTT: LoRA r=256, 16 augmentaciones, 1 época sobre los pares de la tarea
                   DECODE: turbo-DFS (umbral de prob acumulada) → candidatos
                   SCORE: re-scoring por consistencia aumentada (calc_scores)
              ─▶ SELECT top-2 (score_kgmon) ─▶ submission (attempt_1, attempt_2)
```

> **Cada componente (TTT, LoRA, DFS, re-scoring, selección, cheap-first) está explicado en detalle
> en [TACTICAS.md](TACTICAS.md) — qué es, por qué existe y cómo lo implementamos.**

Adoptamos el fork público LB33 de la receta NVARC (auditado limpio — sin hardcode/leakage; ver
LB33_AUDIT.md), reutilizando el checkpoint `sorokin/qwen3_4b_grids15_sft139`. El **único** cambio
que sobrevivió el test público es **cheap-first** (§4): ordenar la cola barata→cara para drenar más
tareas antes de `end_time`, recortando placeholders `[[0]]`. Todo lo demás (matemática del DFS,
re-scoring, TTT, seeds, selección) se mantiene verbatim del baseline auditado.

---

## 4. Amplitud y profundidad de la exploración (los negativos valen igual)

**Lo que funcionó (transfirió al LB semi-privado):**

- **Orden cheap-first.** 28.47 → **30.14** (+1.67). La cola era alfabética; ordenar por un proxy de
  coste (celdas del grid) drena primero las tareas baratas, dejando menos sin terminar. Puro
  throughput, sin cambiar el cómputo por tarea. **Confirma que la cobertura era el cuello.**

**Lo que NO transfirió (con análisis — esto define el muro):**

- **Ensemble con un LLM más débil (Qwen3-2B).** Break-even. El 4B es coverage-limited (~900 s/tarea,
  no drena 240 en 12 h), así que en diseño compartido el 2B nunca corre en el sobrante, y en split
  temporal desplaza cobertura del fuerte. Net ≈ 30%.
- **Ensemble con un TRM arquitectónicamente diverso (recursivo 7M).** 30.14 → **27.22** (−2.92). Un
  split temporal en un kernel (Qwen 9.5 h + TRM epochs=500 + merge "rellena-huecos" que nunca
  sacrifica un intento del Qwen) igual perdió: darle 9.5 h en vez de ~11.4 h costó ~3% de cobertura,
  y el TRM (8.3% solo) no rellenó casi nada del tail caro no cubierto. **Lección airtight: el modelo
  fuerte coverage-limited quiere *todo* el cómputo; cualquier partner más débil es net-negativo.**
- **Presupuesto de decode adaptativo por tarea.** 30.14 → 28.06 (−2.08). Medir desde el *fin del
  TTT* dio a cada tarea más tiempo total → menos tareas terminadas → menos cobertura.
- **Cap de 600 s por tarea.** Smoke degradado severamente (pérdida de calidad); no enviado.
- **Ensemble de selección (idea del juez holístico, arXiv 2606.31543).** 29.72 → **29.31** (neutro).
  El juez del paper es GPT-5.2 leyendo trazas de razonamiento para recuperar respuestas
  minoritarias-correctas; el nuestro son dos fórmulas de scoring correlacionadas (consenso kgmon vs
  confianza probmul) sobre las mismas señales — cuando discrepan, ninguna es fiablemente mejor. **La
  selección no es la palanca offline.** (El mismo paper contraindica loops de refinamiento — colapsan
  diversidad — así que no lo construimos.)
- **Nuestro SFT con 8K sintéticos (train v10/v11).** Cero soluciones exactas. Data muy por debajo de
  la escala 266K de NVARC; **el SFT es nuestra capacidad más débil.**

**Un gap fantasma, resuelto.** Somos **byte-idénticos** (los 4 ficheros core + config + imagen
Docker + modelo + sources, verificado) al notebook público del pelotón de 33.89%, pero sacamos
~30%. Tres tiradas frescas (28.47/30.14/29.72) quedan por debajo de la banda apretada del pelotón —
un gap **sistémico**, no varianza, cuya causa no pudimos identificar en remoto (probables versiones
exactas de sources distintas, o ajustes no publicados que el pelotón lleva). Dejamos de perseguirlo:
**~30% es el score honesto de este pipeline para nosotros.**

**La meta-lección recurrente:** dentro del paradigma de transducción, cada palanca barata está
gastada; añadir un modelo *más débil* siempre resta (impuesto de cobertura); el score está en un
óptimo local medido en cobertura, presupuesto de tiempo y selección. **Solo un cambio de paradigma
(verificación híbrida) o la receta completa de SFT pueden moverlo.**

---

## 5. Contribución de cada idea

Progresión honesta en el LB semi-privado:

| Paso | Cambio | LB (pass@2) | Δ |
|---|---|---|---|
| 0 | Baseline LB33/NVARC (auditado limpio) | 28.47% | — |
| 1 | + orden cheap-first | **30.14%** | **+1.67** |
| — | + presupuesto decode adaptativo | 28.06% | −2.08 (revertido) |
| — | ensemble + Qwen3-2B | ≈30% | break-even |
| — | ensemble + TRM (split temporal) | 27.22% | −2.92 |
| — | re-tirada del campeón (test de varianza) | 29.72% | ±ruido |
| — | ensemble de selección (kgmon+probmul) | 29.31% | neutro |

Todo lo posterior al paso 1 es un **resultado negativo**. Ese es el hallazgo: el paradigma de
transducción está saturado para nosotros a ~30%, y las cuatro tiradas independientes acotan su ruido
run-to-run en ±~1%.

---

## 6. El plan para ganar — la receta completa en tres pasos (comprometido)

Ganar no es un ajuste; es la receta del campeón: **cargar un SFT → adaptar con TTT → seguir
SFT'eando esos pesos con data sintética propia.** Vamos por los tres en orden, el más barato
primero, midiendo en cada gate.

**Paso 1 — Híbrido de modelos congelados (EN CURSO).** Combinar dos modelos ya-SFT'd sin entrenar:
**síntesis de programas verificable de BARC** + **transducción NVARC**.
- *Fase 0 (probe):* medir el **verifiable-hit rate** de BARC-8B en ARC-AGI-2 — la fracción de tareas
  para las que genera un programa Python que reproduce **todos** los train pairs (alta precisión,
  porque el exact-match en train es una verificación real), y si esos generalizan al test.
- *Fase 1:* si es viable, el sistema de inducción completo + un **merge híbrido** donde los programas
  verificados *añaden* cobertura en tareas que la transducción no cubre, sin desplazar el 30% probado.
- *Umbral:* si el híbrido congelado supera ~30%, ganamos sin tocar pesos.

**Paso 2 — TTT (test-time training).** Fine-tuning ligero por-tarea en inferencia — la palanca que
vale +20–30 pt en la receta ganadora. Automático, barato, por-tarea; aplicado a inducción y/o
transducción según midan los datos.

**Paso 3 — SFT propio con data sintética — EJECUTADO Y CERRADO CON NEGATIVO (2026-07-27).**
Resultado: generamos **13.271 tareas compositivas verificadas** sin LLM ni quota GPU y el SFT
entrenó limpio (loss 4.07→0.50), pero el A/B dio **0/16 vs 1/16 del congelado**. Causa raíz
diagnosticada: **verificable ≠ útil** (ver §4 y [TACTICAS.md](TACTICAS.md) §12). Detalle original
del plan abajo; la infraestructura construida queda reutilizable.

**Paso 3 (plan original) — SFT propio con data sintética.** Replicar el núcleo de
NVARC: **generar 100–266K puzzles sintéticos** (RE-ARC / BARC-Potpourri / concept-mixing, validados
contra specs) y **continuar-SFT** los pesos (partiendo de un checkpoint ya-SFT'd en ARC, no de cero)
para superar el techo de los modelos públicos.
- *Riesgos conocidos:* nuestra área más débil (v10/v11 chocaron con "cero exactas"); intensivo en GPU
  (planificar alrededor del reset semanal y la regla de 2 h de holgura para submissions); difícil
  batir el SFT propio de NVARC.
- *Mitigaciones:* partir de un checkpoint SFT'd + añadir data (no entrenar de cero); medir cobertura
  y conteo de exactas por iteración; no encadenar data-gen sin señal.
- *Estado:* diferido hasta que los pasos 1–2 se estanquen; data-gen, config y métricas se registran
  aquí.

---

## 7. Log vivo (resultados · métricas · proceso)

- **2026-07-11..16.** Adoptado LB33/NVARC → 28.47% → cheap-first **30.14%** (campeón). Ensembles con
  partners débiles (2B, TRM) y ajustes de tiempo: agotados / regresivos.
- **2026-07-20.** Confirmado que ~30% es sistémico (re-tirada 29.72%); somos byte-idénticos al
  notebook público 33.89% — el gap no es método.
- **2026-07-21.** Revisión SOTA → la transducción capa ~40%, ganan los híbridos. Ensemble de
  selección 29.31% (neutro) → palancas de transducción agotadas. Decisión: **síntesis de programas
  verificable**.
- **2026-07-22.** Localizado BARC-8B (único modelo de inducción ARC; Llama-3.1). Receta + `common.py`
  vendorizadas; probe construido (varios gates de integración offline). La copia de ironbar en Kaggle
  está gated por Llama (403); el repo HF es abierto → descargando copia propia. **Comprometidos con
  la receta completa de 3 pasos, incluyendo el Paso 3 (SFT con data sintética), para ganar.**
- **2026-07-23.** Probe FASE 0 completado tras ~8 gates de integración offline + un bug clave
  (`common.py` de master usa TEAL/MAROON; el modelo usa PURPLE/BROWN → 0% falso, parcheado).
  **Verifiable-hit rate = 6.25%** (1/16, K=16) y ese acierto **generalizó** (1/1). Validado: el
  pipeline generar→ejecutar→verificar→aplicar funciona y generaliza. Pero **BARC congelado ≈6% en
  AGI-2** (el cliff) y caro (485s/tarea) → como partner *congelado* enfrenta el mismo impuesto de
  cobertura que el TRM. **Confirma que el Paso 3 (SFT en AGI-2) es necesario** para que la inducción
  sea competitiva — como se anticipó.
- **2026-07-24.** Usuario elige el **Paso 3 (SFT de la inducción)**. Localizado el pipeline completo
  de data-gen+SFT de BARC (seeds ejecutables + config). **Fase A (de-risk de infra) COMPLETA:** SFT
  LoRA del BARC-8B validado offline (loss 1.65→1.08). Config clave: imagen estándar, transformers+peft,
  `device_map=auto` (reparte el 8B en 4 L4s, sin bitsandbytes), `DevTrainer` device-aware para el loss.
  Siguiente: **Fase B** — data-gen compositiva (componer seeds → tareas AGI-2 de 2 pasos).
- **2026-07-25/26. Fases B y C ejecutadas.** **Fase B:** compositor propio (`src/arc/seedgen.py`)
  que encadena seeds ejecutables de BARC (A→B) → **13.271 tareas de 2 pasos VERIFICADAS** (el
  programa compuesto reproduce todos los pares → data correcta por construcción), 128 conceptos,
  6.037 pares de seeds, generadas **sin LLM y sin gastar quota GPU** (kernel CPU). **Fase C:** SFT
  LoRA r=32, 900 steps sobre esa data mezclada con ~25% de la original (anti-olvido) →
  **loss 4.07 → 0.50**; la loss inicial alta confirma que la data compositiva es nueva para el
  modelo. Pendiente: el A/B del verifiable-hit rate (probe con adapter) vs el 6.25% congelado.
- **2026-07-27→08-11. A/B del SFT compositivo — DOS negativos, vía CERRADA.** v1 (composición
  aleatoria): 0/16 vs 1/16 — el modelo aprendió la *plantilla* (`_step_a/_step_b`); dos artefactos
  de medición corregidos por el camino (truncación del evaluador; poder estadístico). v2 (rediseño
  completo: composición semánticamente coherente + filtro de visibilidad + forma natural +
  vocabulario alineado + mezcla 1:1, evaluado con **probe pareado** — mismas 16 tareas, métrica
  densa, base replica el ancla 1/16+generaliza): **SFT'd = 0 train-correct (Δ=−1), exec rate
  80%→50%, generaciones 2.5× más largas, y el modelo imita el estilo de la data ("First…, Then…")
  en tareas reales.** Lección de fondo: **la distribución sintética ≠ la real — el SFT sobre tareas
  compuestas mecánicamente desplaza el prior e interfiere con la capacidad base.** BARC necesitó
  100K+ tareas GPT-4; ese listón no se alcanza offline con composición mecánica.
- **2026-08-11. Nueva familia de palancas: VELOCIDAD (inexplorada).** Recon fresco del LB: la
  cabeza se escapa (nvbanana 67.5, rabbithole 47.8), pelotón en 33-34, corte top-8 ~33.5. Todas
  nuestras perillas medidas cambiaban el *cap* de tiempo, nunca la *eficiencia* por tarea.
  Construidos para el reset del viernes: (1) **smoke instrumentado** (12 tareas por el espectro de
  coste, split TTT/decode/scoring — dato nunca capturado); (2) **injerto ttt8** (TTT n_aug 16→8,
  un solo número: tarea más barata → más cobertura).
- **2026-08-11 (tarde). Carril Colab operativo — tres entregas gratis en un día.** CLI oficial de
  Colab headless desde Windows (T4 gratis, 0 quota Kaggle; setup en docs/COLAB.md). (1) Ablación
  pareada `ttt8`: **luz roja** — ΔNLL(n8) = 48% de ΔNLL(n16): halvar augs halva la adaptación;
  degradado a "solo si TTT ≥ 35% del tiempo". (2) El bench de `dfsbatch` **cazó un bug de
  producción** (la fusión ingenua [0,4,2,6] mezcla formas (h,w)/(w,h) en tareas no-cuadradas →
  tensor irregular → habría crasheado; los 4 lotes del campeón están agrupados por forma a
  propósito) y luego lo **eliminó** con datos: speedup 1.05× (0.77 en la tarea con DFS real) y KV
  proyectado en L4 = 39.3 GB > 22 GB. Queda `scoring4` (augs de re-scoring 8→4, pre-testeable con
  rank-correlation en Colab).
- **2026-08-11 (noche). Nueva familia paralela: ESCALA (docs/SCALE_PLAN.md).** Hipótesis
  (propuesta del usuario): el gap con la cabeza es en buena parte **datos sintéticos + escala**,
  no un método distinto. Los dos negativos de v1/v2 condenan la composición mecánica casera, NO
  los corpora diversos de 100K+ generados por LLM frontera — el campeón mismo es prueba de que ese
  SFT funciona. Fuentes verificadas y descargables: ARC-Heavy 139K (HF, streaming), NVARC
  103K+3.2M (Kaggle), RE-ARC. **D1 lanzado en Colab (gratis)**: 2B campeón + LoRA r=64, curva
  acumulativa N=0→500→2000 sobre ARC-Heavy, eval densa en las mismas 12 tareas reales de la
  ablación + 40 sintéticas held-out (separa "aprende la fuente" de "transfiere a AGI-2" — la
  lección de v1/v2). Cadena pre-registrada: D2 N=8000 → D3 comparación de fuentes (¿qué
  distribución sintética generaliza a AGI-2?) → D4 = el paso 3 real (SFT del 4B en Kaggle) solo
  con dos verdes.
- **2026-08-11/12 (madrugada). D1 + D1-bis: la señal positiva MÁS FUERTE de la campaña.**
  Curva D1 (2B campeón + LoRA sobre ARC-Heavy, N=0→500→1250→2000): exactas 1/12→7/12, cell-acc
  0.47→0.85. Auditoría post-hoc: 9/12 tareas de eval eran de AGI-1 (la población fuente de los
  seeds de BARC) → posible vecindario memorizado → **D1-bis** el mismo día: eval pareada base vs
  adapter sobre las **24 tareas del holdout que NO están en AGI-1** (posteriores a ARC-Heavy →
  contaminación imposible). Veredicto: **exactas 0/24 → 7/24, cell-acc 0.50 → 0.85, NLL mejora
  en 24/24, 0 perdidas — transferencia GENUINA.** Los negativos de v1/v2 condenaban nuestra
  composición mecánica, no los corpora diversos: con la fuente correcta, el paso 3 de la receta
  ganadora VIVE. Gates técnicos del día: OOM en T4 (→ grad checkpointing, pico 5.4GB), adapter
  n=2000 persistido como dataset privado de Kaggle (`arcscale-adapter-2b`) para reanudar sin
  reentrenar. Incógnita restante antes de gastar Kaggle: ¿la ganancia sobrevive ENCIMA del TTT
  por tarea? → **D2-bis lanzado** (brazos pareados TTT vs adapter+TTT en 12 nuevas).
- **2026-08-12/13. Cadena D2/D3 + ENSAYO E1 en el test oculto (idea del usuario: usar el slot
  diario ocioso).** D2: ARC-Heavy SATURA hacia 2-4K ejemplos en el 2B (exactas planas 7-8/24).
  D3: las tres fuentes públicas (ARC-Heavy, NVARC-propia, mezcla) son INDISTINGUIBLES — plateau
  compartido → el grueso del efecto es re-alineación de dominio, no conocimiento de fuente; la
  única palanca de datos restante es una distribución NUEVA (piloto de generación con autor
  semántico). E1: 2B+adapter n4000 fusionado, submitido con la config exacta del baseline 18.33.
  **Primer intento: 0.00** — post-mortem: el config de transformers 5.x pierde `rope_theta: 5M`
  → el stack 4.x del kernel usa el default → atención rota. Fix: aux files ORIGINALES en todo
  merge + el commit ahora VALIDA (carga unsloth + NLL<1.5; costó ~7 de los 22 min de quota).
  Auditoría pre-score a pedido del usuario: diff de pesos (adapter fusionado de verdad,
  controles bit-idénticos), procedencia n4000, diff mecánico vs baseline (1 solo cambio en el
  camino de rerun). **E1-v2 (13-ago): 19.86 = +1.53 sobre 18.33 — transferencia REAL pero
  modesta** (~3× la banda de varianza; entre los umbrales pre-registrados). Coherente con
  D2-bis: el TTT ya recupera parte del SFT. Gates operativos del día: límite diario de T4 gratis
  (~6 sesiones) — cambio de cuenta Google autorizado por el usuario (ADC respaldado); la VM
  fantasma de D2 siguió corriendo tras morir el cliente y salvó el punto n=4000.
- **2026-08-13/14. La palanca de datos se cierra + tirada 2 + piloto de inducción.** claudegen
  v1: 33 reglas compositivas autoradas a mano en el día (2178 tareas verificadas por
  construcción, 0 contaminación con AGI-2) → probe con protocolo D3: el MISMO plateau (7-8/24)
  que las otras cuatro fuentes/mezclas → evidencia QUÍNTUPLE de que el efecto del SFT es
  re-alineación de dominio + competencia genérica de grids, saturada — la palanca de datos se
  cierra en el proxy 2B con un resultado nítido para el paper. Tirada 2 del ensayo E1 (slot
  ocioso del jueves): **21.25** — dos tiradas 19.86/21.25, delta medio **+2.2** vs 18.33 (3-6×
  la dispersión entre tiradas) → transferencia al test oculto CONFIRMADA robusta. Piloto de
  inducción propia (grids→programa `main`, ejecutar y verificar): Qwen2.5-1.5B SFT en 2000
  pares de ARC-Heavy → **0/24 verificado** (regla prerreg: escalar antes de descartar);
  escalación a Coder-3B PERDIDA por infra (la VM fantasma murió antes de persistir) — la rama
  queda sin veredicto limpio.
- **2026-08-15/16. D4: la capacidad rompe el plateau… y el pipeline se come la ganancia. Rama
  fusionada CERRADA.** SFT del 4B en Kaggle (3.2h, N=8000 ARC-Heavy): 0/24 → **11/24 greedy**
  en la eval interna — la capacidad rompe el plateau del 2B ✓. Pero el A/B en el LB: **27.08
  (−3.06 vs 30.14)** → revert prerreg. Diagnóstico convergente y gratis: los pools del fusionado
  MEJORAN (1.6 vs 0.8 cands/tarea; GT-en-pool 8/12 vs 5/12), y el smoke con el stack real da
  3.0 vs 2.5 del campeón (la calidad por tarea NO regresó) → mecanismo: **pools 2× más ricos
  encarecen decode+scoring → cobertura colapsó**. Compensación `lb33-scale-fast` = ttt8 (verde
  re-medido en la base nueva, −17%) + scoring4 (top-2 100% fiel) ≈ −25% t/tarea; la v1 la cazó
  el failsafe del commit (IndentationError — el build ahora compila los módulos embebidos), v2
  smoke reload 3.0. **A/B (55538928): 28.33** — la dirección del mecanismo se confirma (+1.25
  por recortar coste) pero <29 → **lectura prerreg: rama fusionada CERRADA; el campeón sigue
  siendo 30.14**. Para el paper, el par limpio de la semana: el MISMO SFT da +2.2 en el 2B y
  −1.8/−3.1 en el 4B de producción — la interacción calibración×presupuesto-de-pipeline domina
  sobre la capacidad.
- **2026-08-16. Mapa de arquitectura: dónde entra nuestra data y dónde está el techo.**
  A pedido del usuario, [ARCHITECTURE.md](ARCHITECTURE.md) con los números sacados de los
  tensores y del código, no de memoria: Qwen3-4B con vocabulario recortado a **16 tokens**
  (1 token = 1 celda), 36 capas, d_model 2560, GQA 32/8, rope 5M, `lm_head` **no atado**,
  3.634 M parámetros. **Solo hay DOS puntos de inyección de datos**: el SFT (LoRA r=64, 132 M,
  7 proyecciones) y el TTT por tarea (LoRA r=256 + rsLoRA, 530 M, **incluye embed_tokens y
  lm_head**). Hallazgo: esas dos matrices son [16, 2560], **rango máximo 16**, así que r=256
  sobre ellas es *rango completo* — el TTT reescribe entera la tabla de símbolos en cada tarea,
  y nuestro SFT nunca la tocó (verificado tensor a tensor). Candidato a explicar por qué la
  ganancia del SFT se lavaba sobre el TTT. **Descomposición del score: 0,80 cobertura × 0,42
  GT-en-pool × 0,90 selección ≈ 30,1** → techo de TODA la familia velocidad = **+7,5 pt**;
  el ×2 está solo en GT-en-pool. Medido gratis el mismo día: la truncación de contexto NO es
  palanca (2/120 tareas), el `attempt_2` **sí se usa** (4/5 subtareas del smoke), y el scoring
  siembra sus augmentaciones con `hash()` **salteado por proceso** → el mismo kernel no es
  reproducible (consistente con las tiradas 19,86 y 21,25) ⇒ **diferencias <1,4 pt no son
  concluyentes con una sola tirada**.
- **2026-08-16/17. Especulación: exacta pero muerta; sobrevive el tope de longitud.**
  A partir de la pregunta del usuario sobre tokenización por nibbles (que en la literatura es
  cuantización, no tokenización; y para ARC los ARChitects eligieron 1 token/celda a propósito),
  se persiguió la versión *lossless* de la misma idea: decodificación especulativa con borrador
  gratis = la propia grid de entrada (72% de las salidas comparten dimensiones con su entrada y,
  de esas, mediana 85% de celdas idénticas). Cadena: gate de coste ✓ (un forward de 16 tokens
  cuesta igual que uno de 1) → v1 del test de exactitud **VACÍA** (sin TTT el 4B no completa
  ninguna grid al umbral de producción — corolario propio: *el TTT no mejora el pool, es lo que
  hace que el pool exista*) → v2 **cazó un bug real** (explorar una rama sobrescribía la caché
  y corrompía el prefijo: Jaccard 0,0) → v3 corregida → **v4 gate SUPERADO: 15/15 pools
  idénticos, 6 al umbral de producción**. Y entonces el bench decisivo: sobre las **4 vistas
  reales de un lote** la ganancia cae a **×1,25 (lockstep) / ×1,39 (ragged)** contra gates de
  1,4/1,6 → **rama CERRADA con el algoritmo ya verificado correcto**. Motivo: la especulación
  acelera las tareas cortas (×3,4) y no las largas (×1,06-1,3), que son las que consumen el
  presupuesto. Del mismo bench salió lo que sí se adoptó: lotes con longitudes `[72, 650, 72,
  72]` → el campeón usa `max_new_tokens=931` para todas las tareas y una vista que divaga
  arrastra al lote. **`lb33-cap`**: tope = `4 × max(mayor salida de train, entrada de test)`;
  la GT cabe en el **100,0% de 1248 subtareas**, el pool capado fue **subconjunto** del normal
  en 12/12 con 0 GT perdidas, y el smoke en L4×4 dio **reload 2,5 = idéntico al campeón**.
  Submitido (55582027). Gates de infraestructura pagados: layout de montaje (ahora se **busca**
  el archivo), `--accelerator NvidiaL4` obligatorio (el default T4×2 deja sin GPU a 2 de los 4
  procesos), y el CLI de Kaggle **no puede escribir logs en Windows** (charmap vs el emoji de
  unsloth → `.log` de 0 bytes que ocultó el error real durante tres intentos; `PYTHONUTF8=1`).
  Descubierto además que **agi-3 comparte la misma cuenta de Kaggle**: mismas 2 sesiones GPU y
  misma cuota semanal ⇒ el ledger de horas de este repo no es fiable.
- **2026-08-22/23. El TTT es EL MOTOR, y su calendario ya está en el óptimo.** Dos submissions
  que fijan los extremos. **Sin TTT: 9,58** (contra ~30 del campeón) ⇒ **el TTT aporta ~+20 pt,
  dos tercios del score, por el 34,8% del reloj.** No es un refinamiento: es lo que hace que el
  pool exista. Y **TTT a media época: 26,39** ⇒ regresión real. Post-mortem del mapeo: los 128
  pasos de producción **son** 1 época (cada secuencia augmentada vista UNA vez), que es justo el
  punto dulce que el banco había medido; media época deja la mitad de las vistas sin adaptar y
  el voto pierde votantes. **Eje "cantidad de TTT" cerrado por ambos lados.**
- **2026-08-24/26. La banda de ruido: casi todos los A/B del mes eran ilegibles.** Tres corridas
  del **mismo kernel, misma versión**: 30,14 / 29,31 / 27,64 → media **29,03**, σ **1,27**.
  El campeón nunca fue 30,14; esa fue la tirada afortunada. Con 1 submission/día y esa banda,
  **una sola muestra solo detecta efectos >2,5 pt**, y las palancas vivas valen ~1 tarea. Causa
  localizada en el código: el scoring siembra augmentaciones con `hash(bk)`, y `hash()` de
  strings va **salteado por proceso**.
- **2026-09-01. Auditoría de los logs del campeón (coste CERO) → segunda fuente, y una fuga.**
  Antes de gastar cuota se auditó si las compuertas de reloj muerden. El tope de **540 s del
  DFS nunca muerde** (lote más lento 203,7 s; sale por agotamiento de haces, que es
  determinista). Pero el tope de **1200 s por tarea sí**, y por un pelo: `981571dc` arrancó su
  último lote a **1159,6 s — 40,4 s de margen, un 3,5% de jitter** — de modo que la velocidad de
  la máquina decidía si esa tarea decodificaba 16 vistas o 12. **No era solo ruido: era una fuga
  de cobertura.**
- **2026-09-02/05. `lb33-det` = NUEVO CAMPEÓN, con predicción prerregistrada.** Dos injertos:
  semilla `md5` estable y **eliminación del tope de reloj por tarea** (deadline global
  conservado). Tres corridas: **31,11 / 30,69 / 30,28** → media **30,69**, σ **0,415**.
  **Separación completa** frente al campeón viejo (la peor de `det`, 30,28, supera a la mejor
  del viejo, 30,14) → Mann-Whitney n=m=3, **p = 0,05**. **+1,66 pt = +4 tareas**, y el canal de
  medida queda **3,1× más estrecho**. Corolario operativo: con σ≈0,42 **una sola muestra
  distingue ~1 pt (2,5 tareas)**; antes hacían falta >2,5 pt.
- **2026-09-03. El quantum es 1/240, no 1/120.** Todas las diferencias históricas son múltiplos
  casi exactos de 0,4167: 0,83→1,99 tareas · 1,67→4,01 · 0,42→1,01. Confirmado en el código:
  el rerun puntúa contra `arc-agi_test_challenges.json`, que son **240 tareas / 259 entradas**.
  ⚠️ El determinismo **NO** se cerró (31,11 ≠ 30,69): queda una tercera fuente, probablemente
  coma flotante en GPU. Se decidió **abandonar su persecución**: vale ~1 tarea y quitarla
  exigiría algoritmos deterministas más lentos.
- **2026-09-04/05. El canal estrecho paga inmediatamente: `stack` 29,44 y un falso rescate
  corregido.** `stack` (= `det` + cheap-first + tope de longitud) queda **−3,5 tareas** bajo la
  media de `det` y −2 bajo su peor corrida ⇒ revertir. Y obliga a corregir el historial:
  `lb33-cap` llevaba **esos mismos dos injertos** y su 30,14 se había leído como "recuperado";
  era **la tirada afortunada de una configuración peor**. Un efecto de 1,46 pt era **invisible**
  con la regla vieja. Lección: **una banda ancha no solo esconde efectos reales, también fabrica
  falsos rescates.** El culpable se identificó **offline y gratis**: el tope
  `4·max(salidas de train, entrada de test)+2` no produce violaciones en eval/train, pero su
  **margen mínimo es de UN token** y el 12,2% del eval va con menos de 50 de holgura; asume una
  expansión ≤4× que ARC no garantiza — **un teselado 3×3 de una 10×10 pide 930 tokens contra un
  tope de 438: ingenerable**. El `max_new_tokens=931` del baseline, en cambio, es una cota
  **estructural** (30×30 = 900 celdas + 29 saltos + 1 EOS = 930) y por eso es seguro. Error de
  diseño propio anotado: se empaquetaron **dos** injertos en un brazo ⇒ **regla nueva: un
  injerto por brazo**.
- **2026-09-01/02. Banco de dosis de augmentaciones: cerrado, con un post-mortem que vale más
  que el experimento.** La v1 corrió mecánicamente perfecta y salió **científicamente nula**: el
  TTT entrenaba con **16 secuencias en vez de 128** (confundí el `n` de `augment(n)` con el
  número de vistas; en producción `augment(n)` = 8 D4 × n permutaciones). Lo delató el registro
  del *calendario* — `n_seqs_ttt`=16 y `ttt_s` de 14-27 s contra los 215-865 s de producción —
  no el resultado. **Por qué el smoke del 2B no lo cazó: ese modelo memoriza el eval y resuelve
  sin necesitar TTT.** Regla general: **lo que el banco no necesita para funcionar, el banco no
  lo verifica.** La v2, ya fiel, dio r(escasez, beneficio de más vistas) = **+0,215** contra un
  prerreg de ≤−0,60 ⇒ **rama cerrada**.

### Tabla consolidada: las 24 submissions al leaderboard

⚠️ Todo lo anterior al 2026-09-02 se midió con σ=1,27, donde una muestra sola solo resuelve
efectos >2,5 pt. Los "veredictos" de esa época se marcan como no concluyentes cuando lo son.

| fecha | qué probaba | score | veredicto |
|---|---|---|---|
| 07-07 | pipeline 3B v10 inicial | 0,83 | superado |
| 07-11 | **LB33 baseline** (NVARC 4B + TTT r=256 + DFS) | 28,47 | punto de partida |
| 07-13 | + cheap-first | 30,14 | no concluyente |
| 07-14 | + presupuesto de decode adaptativo | 28,06 | negativo |
| 07-15 | Qwen3-2B solo | 18,33 | no compite |
| 07-17 | TRM solo (7M recursivo) | 8,33 | no compite |
| 07-19 | ensemble 4B + TRM | 27,22 | negativo |
| 07-20 | campeón cheap-first cap1200 | 29,72 | muestra de la banda |
| 07-22 | ensemble de selección (kgmon + probmul) | 29,31 | neutro |
| 08-12 | escala 2B + ARC-Heavy n4000 | 0,00 | bug (rope_theta) |
| 08-13 | escala 2B v2 | 19,86 | — |
| 08-14 | varianza de la anterior | 21,25 | σ del 2B |
| 08-15 | **4B + ARC-Heavy N=8000** | 27,08 | negativo → familia escala cerrada |
| 08-16 | fusionado + ttt8 + scoring4 | 28,33 | no concluyente |
| 08-17 | **tope de longitud por tarea** | 27,64 | **negativo (confirmado 09-04 con mecanismo)** |
| 08-22 | **sin TTT** | **9,58** | **el TTT vale ~+20 pt** |
| 08-23 | TTT a media época | 26,39 | negativo → 1 época es el óptimo |
| 08-24 / 08-25 / 08-26 | varianza del campeón viejo | 29,31 / 27,64 / 30,14 | **σ = 1,27** |
| **09-02 / 09-03 / 09-05** | **`lb33-det`** (md5 + sin tope de reloj) | **31,11 / 30,69 / 30,28** | **CAMPEÓN · +4 tareas · p=0,05** |
| 09-04 | `stack` (det + cheap-first + tope longitud) | 29,44 | revertido |

Bancos que nunca llegaron al LB (offline, Colab o Kaggle): triaje de qué tareas necesitan TTT,
TTT con leave-one-out, CE focal y ponderada, verificador de objetos, dosis de augmentaciones
(dos corridas). Todos cerrados en negativo o invalidados.

### Qué es exactamente el campeón (`lb33-det`, 30,69 ± 0,415)

| componente | configuración |
|---|---|
| modelo base | Qwen3-**4B**, checkpoint `qwen3_4b_grids15_sft139` (**el SFT viene hecho**, derivado de NVARC) |
| paralelismo | 4×L4, un worker por GPU, cola de tareas compartida |
| **TTT** (por tarea) | LoRA **r=256, α=32 con rsLoRA**, sobre 7 proyecciones **+ `embed_tokens` + `lm_head`** (~530 M params); AdamW **lr 5e-5 coseno**, batch 1, **1 época sobre 128 secuencias** (`augment(n=16)` = 8 D4 × 16 permutaciones de color) |
| decode | 16 vistas (`augment(n=2)`) en 4 lotes de 4; DFS con poda `max_score = −log(0,2)`; **`max_new_tokens = 931` fijo** |
| re-scoring | cada candidato re-puntuado con 8 vistas augmentadas por NLL (semilla md5 estable) |
| selección | 2 intentos: `score_full_probmul_3` y `score_kgmon` |
| presupuesto | deadline global 11 h 50 m · tope de 1200 s por tarea **eliminado** · tope de 540 s del DFS conservado |
| **thinking** | **no se usa** — el pipeline no pasa por el chat template; alimenta tokens de dígito crudos con marcadores de rol (ids 11/12) |

### Pendientes abiertos (prerregistrados)

1. **SFT con *thinking* — palanca virgen, nunca probada.** Hoy el pipeline **no usa thinking en
   absoluto**: no hay chat template, el modelo consume dígitos crudos y el SFT del checkpoint
   está hecho sobre ese formato, así que no produce cadenas de razonamiento. Activarlo **no es
   un flag: exige otro SFT**. Plan: **primero banco en Colab (T4, gratis)** — ¿un 2B con SFT
   corto sobre trazas de razonamiento mejora la NLL de la respuesta frente al mismo 2B sin
   ellas? ⚠️ El vehículo debe evitar el error de §26: el 2B memoriza el eval, así que la
   comparación tiene que ser **pareada y sobre tareas que el 2B NO resuelva ya**, o no medirá
   nada. Solo con verde se plantea gastar L4.
2. **TTT eficiente por ablación de hiperparámetros** (propuesta del usuario). Idea: el TTT es el
   motor **y** el 34,8% del reloj; si solo una parte de lo que entrena mueve la loss, entrenar
   *solo esa parte* libera presupuesto para decode **sin recortar adaptación** — que es el modo
   de fallo ya conocido (`ttt05`, `ttt8`). Primer paso, ya hecho: **inventariar qué toca el TTT
   hoy** (tabla de arriba). Los ejes ablacionables son: (a) los **9 módulos objetivo**, con la
   sospecha fuerte de que `embed_tokens`/`lm_head` son especiales — son matrices [16, 2560], de
   **rango máximo 16**, así que r=256 sobre ellas es *rango completo* y el TTT **reescribe entera
   la tabla de símbolos** en cada tarea; (b) el **rango r** (256 es enorme para las proyecciones);
   (c) **lr y scheduler**; (d) el **número de secuencias** (cerrado por arriba y por abajo, no
   tocar). Diseño del banco: medir **la curva de loss del TTT por configuración**, que no
   necesita decode y por tanto es barato (~150-200 s por tarea en 4B). ⚠️ Debe correr en **4B
   sobre eval** — el 2B no sirve porque memoriza y su loss arranca en cero. Y la loss es un
   *proxy*: `ttt05` ya enseñó que un calendario que parece sano puede regresar en producción, así
   que un verde en loss exige confirmación con el pool antes de injertar.

   **EJECUTADO 2026-09-05 (eje del rango).** 12 tareas, 4 rangos, pareado, solo TTT: **r=64 →
   1,34× más rápido sin degradación detectable** (media, mediana y test de signo de acuerdo;
   25,5% del TTT = **8,9% del pipeline**). r=16 → 1,46× pero **10/12 tareas peores** (signo
   p=0,039): la media lo ocultaba porque un outlier mejora −0,204 mientras las otras once suman
   +0,163 — sin el test de signo habríamos adoptado una degradación creyendo que no la había.
   r=4 descartado (11/12 peores). **La velocidad SATURA en r=16**: r=4 tarda lo mismo con 3,6×
   menos parámetros ⇒ por debajo de ahí el coste está en el modelo base congelado y recortar más
   parámetros **no puede** ayudar; la siguiente palanca es **adaptar solo las últimas K capas**,
   que sí acorta el backward. ⚠️ Un TTT más rápido **no sube el score por sí solo**: el pipeline
   no reinvierte el tiempo, termina antes. Solo paga si el deadline global muerde o si el ahorro
   se **gasta** en más vistas de decode.

- *(estado: la familia de cobertura dio su mayor pieza (+4 tareas) por auditoría offline gratuita,
  no por el LB. Ese es ahora el motor: el análisis offline encontró tanto el +2 pt como el
  diagnóstico del tope de longitud sin gastar un slot. El leaderboard queda para confirmar
  efectos ≥2-3 tareas. Sigue pendiente repartir la cuota semanal con agi-3, que comparte cuenta,
  sesiones GPU y cuota)*


### Cierre de la semana (2026-09-06/08): el presupuesto de reloj, y por qué el razonamiento no cabe

**El campeón corre al 95% de su deadline.** Inferido del propio LB, que es más fiable que
cualquier modelo de coste: `r512` añadió **+11,9%** de coste por tarea y perdió **15,3 tareas**
(6,4% de 240) ⇒ `deadline / coste_base = 1,048`. **Holgura real: 4,8%.** Con eso los tres
injertos de la semana encajan en un solo modelo: `det` completa el 100% y le sobra un 4,8%
(**30,69**); `r64` es 11% más rápido pero **ahorrar no da puntos** porque ya terminábamos, así
que solo se ve su pérdida de capacidad (**29,72**); `r512` es 11,9% más lento, **se pasa del
deadline** y pierde la cola (**24,31**). El eje del rango queda cerrado **por ambos lados**:
256 es el punto de operación.

**El tipo de cambio cómputo↔capacidad: ~0,6 tareas por cada 1% de reloj.** Dos canjes
independientes coinciden — 4B→2B cuesta 29,7 tareas por un 50% de reloj (0,59) y r256→r64 cuesta
2,3 tareas por un 3,8% (0,61). De ahí el criterio: **un uso nuevo del cómputo solo compensa si
rinde más de 0,6 tareas por 1%.**

**Y por eso el razonamiento no cabe, medido y no argumentado.** Las trazas del dataset público
`nvidia/Nemotron-SFT-ARC-AGI-v1` son de **6.900-11.600 tokens** contra los 100-931 de una rejilla
nuestra: **×8 a ×116 en decode**, o sea ×4,8 a ×67 en el pipeline completo. Ni el thinking
selectivo sobrevive (el presupuesto da para razonar en **1,7 de 240 tareas**). Confirmación
independiente: el equipo que ganó ARC Prize 2025 —cuyo checkpoint **es el nuestro**— publicó que
su diseño fue *"move all complex reasoning offline into a synthetic data pipeline"* porque *"los
métodos pesados de razonamiento no caben en el runtime de Kaggle"*. Derivamos la misma
restricción por caminos distintos.

**El bloqueo más profundo no es el reloj, es el vocabulario.** Sorokin recortó el vocabulario de
Qwen3 de 151.936 a **16 tokens**. El modelo **no tiene ni una palabra**: razonar en lenguaje no
es caro, es imposible. Y la variante latente que sí cabría —bucles sobre estados ocultos— **ya la
tenemos**: el TTT reprocesa el problema 128 veces en espacio de pesos sin emitir un token, y vale
+20 pt. Su versión arquitectónica (el TRM de 7M, citado por su 44,6% en ARC-AGI-1) está en
nuestra tabla con **8,33** en ARC-AGI-2.

**El diagnóstico de fondo, desde dos bancos independientes.** (1) `ttt-regla`: el mismo modelo
adaptado acierta **5/12** entradas sacadas de la distribución de las demostraciones y **0/12**
entradas de test (McNemar p=0,031) — aprende la regla y la aplica, pero la entrada de test de
ARC-AGI-2 no es una muestra más de esa distribución. (2) De los 179 candidatos guardados por el
banco de dosis, el **71% tiene la forma correcta** y solo el **7% es exacto**: el modelo produce
rejillas bien formadas, del tamaño correcto y con el **contenido equivocado**. Aplica *una*
transformación plausible con seguridad; lo que no acierta es **cuál**.

**Consecuencia estratégica.** Esto explica el techo de ~40% de la transducción pura y por qué
tres injertos consecutivos se revirtieron: no falta cómputo ni mejores hiperparámetros. Falta
**generalizar fuera de la distribución de las demostraciones**, y eso es un cambio de método.
Todo paso intermedio que pudimos diseñar o (a) es "aplicar la regla a algo" —lo que el modelo ya
hace— o (b) no cabe en el 4,8%.

### Cierre del cambio de modelo base (2026-09-09): las tres vías, muertas por reloj

Tras agotar los injertos sobre el pipeline, la pregunta natural fue **cambiar la base**. Se
cerraron las tres vías, todas con aritmética antes de gastar:

1. **Qwen3.5 denso (0,8B/2B/4B/9B)** — el 4B es del mismo tamaño que el nuestro, y adoptarlo
   obliga a rehacer el SFT que hoy heredamos gratis de NVARC. Prior histórico de rehacer SFT:
   0-de-2. Descartada sin gasto.
2. **Qwen3.8-27B denso** — coste de reloj ×3,45 sobre el campeón. Con el 100% de cobertura
   actual, ese factor deja un techo *teórico* de 29,0% < 30,69 medido. Imposible por aritmética,
   descartada sin gasto.
3. **Qwen3.5-35B-A3B (MoE, 3B activos)** — la única que sobrevivía a la aritmética a priori
   (×0,59 conservador en 4-bit), así que se midió.

**La puerta `gate-moe`.** Prerregistro: *el MoE de stock debe tener mejor prior sobre ARC que el
Qwen3-4B de stock* — `ratio mediano nll/celda MoE/4B <= 0,80` **Y** `gana >= 2/3` sobre 24
tareas de eval, comparación pareada. Correr esto exigió resolver primero que la arquitectura
`qwen3_5_moe` no existe en ninguna imagen de Kaggle: se resolvió con un **dataset privado de
wheels** (`transformers 5.16.1` + dependencias) instalado con `pip --no-index --no-deps` desde el
lanzador del kernel. Cuatro versiones, ~1,1 h-cuota.

**Resultado**: ratio mediano **0,807**, gana **11/16**, test de signo **p = 0,105**. El
prerreg era un `Y` y falla por 0,007. Además `fwd` ×4,07 (bf16 en pipeline sobre las 4 L4;
estimación honesta en 4-bit: ×1,5-3) contra un techo de **×1,048** del presupuesto de reloj — el
mismo techo que `r512` rebasó a ×1,119 para perder 15,3 tareas. **El reloj lo mata aunque el
prior fuera excelente.**

Dos avisos metodológicos que deja la corrida. Las 8 tareas que el MoE no logró medir fueron
**exactamente las 8 de contexto más largo** (2.978-4.742 tokens; OOM por el vocabulario de 151K
del stock, que nuestro modelo de producción no tiene: usa 16 tokens), así que el 0,807 está
medido sobre la mitad corta del banco. Y la **media** del ratio decía "sin cambio" (1,008)
mientras la **mediana** decía −19%: por tercera vez en la campaña, media sola = veredicto
equivocado.

**Estado**: el campeón sigue siendo `lb33-det` (30,69 ± 0,415). El cambio de base queda cerrado
por ~1,1 h-cuota, que es barato para enterrar una rama presupuestada en 10-48 h-cuota.
