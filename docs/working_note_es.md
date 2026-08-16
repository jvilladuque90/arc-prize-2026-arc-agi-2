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
- *(siguiente: palancas sobre el CAMPEÓN — reasignación adaptativa del presupuesto por tarea
  (dispersión 27-790s ya medida) como graft candidato; inducción Coder-3B solo si el banco
  Colab lo permite; ledger ~14.5h-quota con piso de 4h, reset viernes ~20:00; slot diario de
  submission libre desde ~19:00 locales)*
