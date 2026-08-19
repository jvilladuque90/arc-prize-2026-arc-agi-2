# Familia ESCALA — ¿el gap es solo datos sintéticos + escala?

> Documento de estrategia de la apuesta paralela (2026-08-11). Complementa
> [SPEED_PLAN.md](SPEED_PLAN.md) (familia velocidad, en cierre), [TACTICAS.md](TACTICAS.md)
> (fundamentos) y [COLAB.md](COLAB.md) (el carril proxy gratuito). Registro vivo.

---

## 1. La hipótesis (H-escala)

**Enunciado:** el gap entre nosotros (30.14%) y la cabeza del leaderboard (~57%) se explica en
buena parte por **escala y calidad de los datos sintéticos de entrenamiento**, no por un método
cualitativamente distinto. Si es cierta, el camino es: más/mejores datos → SFT del 4B → TTT
encima (el paso 3 de la receta ganadora, que cerramos con datos propios pero **no** con los
corpora públicos grandes).

**Evidencia a favor:**
- El propio campeón es data-driven: NVARC ganó 2025 generando 103K puzzles sintéticos
  (+3.2M aumentados) y entrenando sobre ellos. Nuestro techo actual ES un modelo entrenado así.
- BARC (2024) mostró que escalar datos generados por GPT-4o (ARC-Heavy, 200K tareas) mejora
  monótonamente el score de modelos 8B en AGI-1.
- Nadie ha publicado un método nuevo que explique el 57% — escala de datos es la explicación
  más parsimoniosa disponible.

**Evidencia en contra (nuestros propios negativos):**
- SFT compositivo v1/v2: dos negativos medidos. PERO eran **nuestra** composición mecánica
  (~300 tareas, plantillas): el modelo aprendió estilo, no competencia. Ese resultado condena
  la composición mecánica casera a escala pequeña — **no** condena corpora diversos de 100K+
  generados por LLM frontera (que es exactamente lo que el campeón sí usó con éxito).

**Lo que este experimento NO puede confirmar:** que la hipótesis explique el 57% completo.
Sí puede: (a) matarla barata si la pendiente es plana, (b) darle luz verde medible al paso 3
con datos públicos existentes.

## 2. Fuentes de datos sintéticos disponibles (verificado 2026-08-11)

| Fuente | Tamaño | Acceso | Relación con el campeón |
|---|---|---|---|
| **ARC-Heavy (BARC)** | 139K filas (200K generadas) | HF `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` (6.2 GB, streaming OK) | Incierta — no está documentado si NVARC la usó |
| **NVARC Synthetic** | 103K puzzles | Kaggle `sorokin/nvarc-synthetic-puzzles` | Es SU training set → más de esto = medir saturación |
| **NVARC Augmented** | 3.2M | Kaggle `sorokin/nvarc-augmented-puzzles` | Ídem (aumentaciones) |
| RE-ARC (Hodel) | ∞ (generadores de las 400 de AGI-1) | GitHub | Distribución AGI-1, misma tarea |
| seedgen propio | ~300 | repo | CERRADO (dos negativos) |

Elección para la curva: **ARC-Heavy** — fuente distinta a la del campeón, diversa, ya filtrada
por ejecución. Formato por fila: `examples` = lista de pares `[input, output]` (grids de
floats 0–9). NVARC-Synthetic queda para la etapa de comparación de fuentes (D3).

## 3. Cadena de experimentos (de gratis a caro)

### D1 — Curva de escala en Colab (GRATIS, sesión 1)
`scripts/colab_data_scaling.py` — NVARC 2B (proxy del campeón) + LoRA r=64, SFT **acumulativo**
sobre ARC-Heavy: puntos **N = 0 (control) → 500 → 2000** en una sesión T4 (~2 h). Métricas
densas tras cada punto (regla del carril: lo absoluto no transfiere, lo relativo sí):

- **Real (12 tareas holdout AGI-2**, las mismas de la ablación ttt8): NLL teacher-forced de la
  respuesta verdadera, cell-acc greedy, exactas.
- **Sintético held-out (40 tareas ARC-Heavy no vistas):** NLL — para separar "aprende la fuente"
  de "transfiere a AGI-2".

**Reglas de lectura (pre-registradas):**

| Resultado | Lectura | Decisión |
|---|---|---|
| NLL real mejora 0→500→2000 sin saturar | H-escala VIVA | D2: sesión 2 con N=8000 |
| Mejora syn pero real plano | La fuente no transfiere a AGI-2 (mismo patrón que v1/v2, ahora con corpus diverso) | Cambiar de fuente (D3) antes de descartar |
| Real EMPEORA (interferencia) | Mismo patrón que v2: el SFT desplaza el prior | ❌ rama ARC-Heavy muerta |
| Todo plano (ni syn mejora) | Bug o saturación total del 2B | Diagnosticar antes de leer |

*Caveat de interpretación: si NVARC ya entrenó con ARC-Heavy, "plano" significa saturación de
la fuente; si no, significa no-transferencia. Ambas cierran la rama ARC-Heavy — la ambigüedad
no cambia la decisión.*

### D2-bis — Interacción con TTT (GRATIS, decide el EV de D4)
`scripts/colab_ttt_adapter.py` — brazos pareados en 12 tareas nuevas: (A) base+TTT vs
(B) base+adapter fusionado+TTT, con el TTT del campeón (n_aug=16). El pipeline de producción
hace TTT por tarea: si el TTT ya captura lo que enseña ARC-Heavy, el SFT no mueve el LB.
**Lectura:** B > A claramente en exactas/cell-acc post-TTT → el SFT compone con TTT → luz
verde D4. B ≈ A → el SFT solo ayuda al modelo "frío" → no paga en producción.

### D2 — Extensión de la curva (GRATIS, condicional a D1 verde)
N=8000 (sesión 2, reanuda desde el adapter guardado). Si la pendiente sigue positiva en log-N
→ informa cuántos ejemplos necesita el SFT del 4B.

### D3 — Generalización / comparación de fuentes (GRATIS, condicional)
A N fijo (p. ej. 2000): ARC-Heavy vs NVARC-Synthetic vs mezcla. Responde la segunda pregunta:
**qué tan generalizable es la generación** — qué distribución sintética transfiere mejor a
AGI-2. Es también material directo de paper (nadie ha publicado esta comparación en AGI-2).

### D4 — El paso 3 real en Kaggle (CARO, solo con D1+D2 verdes)
SFT del **4B campeón** en L4×4 con la mezcla ganadora de D3 (LoRA, ~horas de quota) → TTT
encima → A/B en LB vs 30.14. Es exactamente la receta del ganador 2025, ahora con datos
diversos existentes en vez de nuestra composición mecánica. Solo se paga si el proxy dio
pendiente positiva dos veces.

## 4. Expectativa honesta

- **Mejor caso realista:** pendiente positiva sostenida → D4 sube el techo de calidad del 4B →
  compone con la cobertura (~79%) → algunos puntos de LB. No es un camino directo al 57% —
  es el camino de "cuestión de escala" comprado al precio mínimo.
- **Caso probable:** el 2B de NVARC ya está cerca de saturación en fuentes públicas (ellos ya
  exprimieron esta palanca); la curva lo dirá con números, no con opinión.
- **Valor de paper en cualquier caso:** la curva de escala + la matriz de transferencia de
  fuentes en AGI-2 es un resultado publicable por sí mismo, positivo o negativo.
- Coste hasta la decisión D4: **0 horas de Kaggle** (todo en Colab T4 gratis).

## 4-bis. Ejecución de D4 — runbook del viernes 2026-08-14 (pre-registrado)

Artefactos YA construidos (todo lo previo al gasto de quota):
- **Datos:** dataset privado `arcscale-archeavy-19k` (19K tareas, filtros idénticos al proxy).
- **Kernel de SFT:** `notebooks/scale_sft_4b.ipynb` (`push_kernels.py scale-sft-4b
  --accelerator NvidiaL4`) — LoRA r=64, lr 1e-4, accum 8, `N_TRAIN` a dimensionar con D2;
  eval integrada base vs SFT en las 24 nuevas; salida `adapter_4b` + `merged_4b`.
- **Kernel de inferencia:** `notebooks/lb33_scale.ipynb` (`lb33-scale`) — campeón exacto con
  UN cambio: modelo = `merged_4b` (sin fallback silencioso).
- ⚠️ **No hacer push antes del reset**: el push ejecuta el kernel (gasta quota).

Orden del viernes (reset 19:00; ~30h de quota, regla de 2h de holgura):
1. **19:05 — `lb33-timing` smoke** (~40 min): split TTT/decode/scoring (SPEED_PLAN §4-bis).
2. **~19:50 — push `scale-sft-4b`** con `N_TRAIN` fijado por D2 (~3-5h de L4).
3. **Con el split en mano:** si scoring ≥ ~15% → construir `lb33-scoring4` (un número:
   augs de re-scoring 8→4; proxy ya avalado), smoke y **submission del viernes = A/B scoring4**.
   Si no → guardar el slot.
4. **Sábado:** smoke de `lb33-scale` (~40 min) → **submission del sábado = A/B escala**
   (lb33-scale vs 30.14). Lecturas pre-registradas: `>31` → nuevo campeón, iterar (más datos /
   fuentes D3); `≈30` → neutro, diagnosticar (¿cobertura vs calidad?); `<29.5` → revertir.
5. Gasto total estimado: ~6h → quedan >20h para iterar el fin de semana.

## 5. Registro de resultados de la cadena

| Fecha | Etapa | Resultado | Decisión |
|---|---|---|---|
| 2026-08-11 | D1 (curva Colab) | **Pendiente positiva FUERTE** en N=0→500→1250→2000: real NLL 0.327→0.156→0.055→0.061, cell-acc 0.47→0.73→0.78→**0.85**, exactas **1/12→5→7→7**; syn NLL 0.349→0.081 (satura ~1250 en NLL; cell-acc sigue subiendo). Gate técnico: OOM en T4 → grad checkpointing (pico 5.4GB). Adapter n=2000 guardado en Kaggle (`arcscale-adapter-2b`, privado). **⚠️ Confound detectado post-hoc:** 9/12 tareas de eval son de AGI-1 — la población fuente de los seeds de BARC → posible vecindario memorizado. Las 3 NUEVAS de AGI-2 también mejoran (0/3→2/3 exactas, cell-acc 0.31→0.99) pero n=3 no decide. | ✅ señal suficiente para **D1-bis** (mismo día): eval pareada base vs adapter sobre las **~24 tareas NUEVAS de AGI-2** del holdout — la única población parecida al test oculto. La lectura de D1 queda EN SUSPENSO hasta D1-bis. |
| 2026-08-11 | D1-bis (nuevas AGI-2) | **TRANSFERENCIA GENUINA.** Eval pareada base vs adapter en las **24 tareas del holdout que NO están en AGI-1** (todas posteriores a ARC-Heavy → contaminación imposible): exactas **0/24 → 7/24**, cell-acc 0.50 → 0.85, NLL mejora en **24/24** tareas, 0 exactas perdidas. La señal positiva más fuerte de la campaña. | ✅ **H-escala VIVA con evidencia limpia.** Incógnita restante antes de gastar Kaggle: ¿la ganancia sobrevive ENCIMA del TTT? (el pipeline campeón hace TTT por tarea — si el TTT ya captura lo mismo, el SFT no mueve el LB) → **D2-bis** inmediato. |
| 2026-08-12 | D2-bis (adapter × TTT) | **La ganancia SOBREVIVE encima del TTT.** Pareado en 12 nuevas, TTT del campeón (n_aug=16) en ambos brazos: TTT-solo exactas 1/12, cell-acc 0.70, NLL-post 0.124 vs adapter+TTT exactas **4/12**, cell-acc **0.92**, NLL-post 0.022; adapter mejora NLL-post en **12/12**, +3 exactas, 0 perdidas. (Caveat: TTT proxy simplificado r=64 vs r=256 de producción; greedy, sin DFS/selección — señal relativa.) | ✅ **LUZ VERDE a D4** (SFT del 4B en Kaggle). Antes, gratis: D2 (extender N desde el adapter → dimensionar el dataset de D4) y D3 (¿ARC-Heavy vs NVARC-103K vs mezcla?). D4 se ejecuta tras el reset del viernes en paralelo al runbook de velocidad. |
| 2026-08-12 | D2 (parcial: N=4000) | Dos intentos con gates (flush de GradScaler en chunks múltiplos de 8; luego el límite diario de T4 gratis mató al cliente — pero la VM siguió y subió el punto): **N=4000 → 8/24 exactas (vs 7/24 en N=2000), NLL 0.047 (mejor), cell-acc 0.817 (leve baja)**. La curva se aplana pero no degrada. | `N_TRAIN=8000` del kernel D4 se mantiene (el 4B tiene más capacidad). Punto N=8000 pendiente del reset diario de Colab. |
| 2026-08-12 | **E1 — ENSAYO en test oculto** (idea del usuario: usar el slot diario ocioso) | 2B + adapter ARC-Heavy **n=4000 fusionado** (dataset `arcscale-merged-2b`), config campeona IDÉNTICA a la submission 18.33 del 2B-solo (15-jul) → A/B directo del efecto ARC-Heavy en la distribución real con el stack completo (TTT+DFS+selección). Con solo 22 min de quota: **commit barato** (0 tareas, sin cargar modelo, ~3 min; el rerun de scoring hace el trabajo gratis). Dos gates cazados en el camino: umbral del pool del proxy scoring4; slug con guion + zip aplanado rompían la detección del modelo (fix: detección por contenido). Submission 55450984: **0.00 — FALLA SISTÉMICA, no señal del modelo.** Post-mortem: el merge se guardó con transformers **5.13.1** (Colab), cuyo config nuevo pierde `rope_theta: 5000000` del nivel superior (queda anidado en `rope_parameters`); el stack 4.53 del kernel usó el default (10000) → atención rota → todos los workers produjeron basura → submission placeholder válida → 0.00. También faltaban vocab.json/added_tokens/special_tokens_map. | **E1-v2 (13-ago):** dataset reempacado con los **aux files originales** del sorokin (verificado: rope_theta 5M presente); el commit ahora **VALIDA** (carga unsloth + NLL sanity < 1.5 en una tarea real — el fallo se habría cazado antes de submitir); mismo fix propagado a `scale_sft_4b` (merged_4b lleva aux originales) y a `colab_merge_2b`. Re-submit con el slot de la noche (55469716). **Auditoría pre-score (12-ago noche, a pedido del usuario):** (1) pesos del dataset submitido vs base original — los tensores objetivo de LoRA DIFIEREN (~0.3-0.6% rel) y los controles (embeddings/layernorm) son bit-idénticos → el adapter está fusionado de verdad (descarta el falso ≈18.33 por merge vacío); (2) procedencia = adapter n4000 (json embebido); (3) NLL 0.0002 del commit prueba que el kernel montó el dataset v2 (el v1 roto habría reventado el assert); (4) diff mecánico vs el kernel del baseline: 3 celdas, único cambio en la ruta de rerun = el directorio del modelo. **El score de mañana es interpretable en ambas direcciones.** Lecturas pre-registradas sin cambio: **≥21** verde / **≈18** replantear / **<17** revisar. **SCORE (13-ago): 19.86 = +1.53 sobre 18.33.** Cae entre los umbrales → lectura honesta: **transferencia REAL pero modesta** (~3× la banda de varianza observada ±0.4-0.5; no llega al verde pleno). Coherente con D2-bis: el TTT de producción ya recupera parte de lo que el SFT aporta (la ventaja post-TTT era menor que la pre-TTT), y el test oculto es más duro que las 24 de train. | **D4 sigue GO con expectativas calibradas:** si el 4B gana un delta similar (+1-1.5) → **31.1-31.6** → cruza el umbral pre-registrado ">31 = nuevo campeón". Ya no se espera +3-5. Refuerza: (a) el piloto de generación (distribución nueva) como palanca de mayor techo, (b) las perillas de velocidad re-medidas sobre la base nueva (ttt8) para convertir calidad en cobertura. **TIRADA 2 (14-ago, slot ocioso del jueves): 21.25** — dos tiradas 19.86/21.25, media ~20.6, delta medio **+2.2** vs 18.33 (3-6× la dispersión entre tiradas) → efecto CONFIRMADO robusto; la tirada 2 cruza el verde pleno (≥21). Expectativa del sábado re-recalibrada: 4B con delta similar → **~31.5-32.5**. |
| 2026-08-12 | D2 (completa, N→8000) | Curva final en las 24 nuevas (nota: el resume desde n4000 repitió los ítems 2000-4000 una época extra): **exactas 8 → 7 → 8, cell-acc 0.817 → 0.816 → 0.875, NLL real ~0.041-0.047 estable; syn NLL sigue bajando lento (0.074→0.066)**. | **SATURACIÓN de ARC-Heavy en el 2B hacia 2-4K ejemplos**: más data de la misma fuente ya casi no paga en exactas (cell-acc sube leve). `N_TRAIN=8000` confirmado para D4 (sin degradación, punto dulce). La palanca de calidad pasa a la MEZCLA de fuentes (D3, corriendo) y al piloto de generación con autor semántico (condicional). Adapters n4000/n8000 persistidos en Kaggle. |
| 2026-08-12 | D3 (fuentes, N=2000 fresco por brazo) | **Las tres fuentes son INDISTINGUIBLES en el 2B** (n=24): ARC-Heavy 7/24 · 0.851 · NLL 0.049; NVARC-synthetic 7/24 · 0.873 · NLL 0.029; MIX 8/24 · 0.855 · NLL 0.033. Sorpresa: la data del PROPIO campeón re-entrenada también da el salto completo (0→7) — el grueso del efecto parece ser re-alineación de formato/dominio + competencia genérica de grids, no conocimiento específico de la fuente; y TODAS saturan en el mismo plateau (~7-8/24, ~0.86). | **Receta D4: ARC-Heavy solo, N=8000** (sin mezcla): las fuentes empatan, la mezcla añade superficie de fallo sin ganancia medida, y mantiene una sola variable con E1 (que prueba ARC-Heavy en el test oculto). Implicación estratégica: el plateau compartido de las fuentes públicas convierte al **corpus de distribución NUEVA (piloto Claude-autor compositivo)** en la única palanca de datos restante tras D4. |
| 2026-08-13 | **Piloto claudegen v1** (generación con autor semántico) | Corpus creado el mismo día: **33 reglas compositivas/condicionales autoradas una a una** (máscaras AND/XOR, reparación de simetría, gravedad+recolor, selección por marcador, conteo→estructura, sombras diagonales…), 2178 tareas · 8712 pares verificados por construcción, 36% con cambio de forma, **0 contaminación** con AGI-2. Probe con protocolo D3: **claudegen-2000 = 7/24 · 0.780; MIX cg+heavy = 8/24 · 0.812** — el MISMO plateau que las otras cuatro fuentes/mezclas. | **La palanca de datos se CIERRA en el proxy 2B con evidencia quíntuple**: cinco distribuciones radicalmente distintas (scrapeada GPT-4o, dirigida del campeón, autorada nueva, 2 mezclas) convergen al mismo salto → el efecto es re-alineación de dominio + competencia genérica, saturada. El techo lo decide la CAPACIDAD (A/B del 4B, sábado) o el MÉTODO (TTT/decode/selección). Resultado nítido para el paper. Corpus reutilizable para D4-v2 si el 4B muestra hambre de datos. |
| 2026-08-14 | **Piloto de INDUCCIÓN propia** (apuesta estructural) | Qwen2.5-1.5B-Instruct + LoRA, SFT en 2000 pares (grids→programa `main`) de ARC-Heavy (gates: convención `def main` no `transform` — corrida 1 inválida con n=27; timeout de silencio del cliente ~10min — VM fantasma matada con `colab stop`). Corrida limpia: loss converge ~0.4, pero **0/24 tareas con programa verificado** (144 generaciones, K=6). | Regla pre-registrada: 0-1/24 → escalar modelo antes de descartar → **escalación en curso: Qwen2.5-Coder-3B (máx entrenable en T4) + N=3000**. Si también da ~0 → la rama de inducción-con-modelos-chicos se cierra honesta (referencia: BARC-8B con 100K+ curados daba solo 6.25% congelado) y queda para el paper. |
| 2026-08-15 | **D4 EJECUTADO + A/B en LB** | SFT del 4B en Kaggle (3.2h pared, N=8000): eval interna **0/24 → 11/24 greedy, cell-acc 0.90** — la capacidad rompe el plateau del 2B ✓. Smoke de integración 3/4. **A/B (55520466): 27.08 = −3.06 vs 30.14** — lectura pre-registrada "<29.5 → REVERTIR". | ❌ **El campeón sigue siendo el 4B original (30.14).** La paradoja (greedy interno ↑↑, 2B oculto ↑, 4B producción ↓↓) apunta al mecanismo: **el SFT afila la calibración → el DFS con umbral -log(0.2) tuneado para la calibración original produce pools aún más chicos → peor pass@2**. Cuarta instancia de "el instrumento es parte del experimento", ahora a escala de pipeline. **Diagnóstico gratis en curso**: barrido de umbral sobre el 4B FUSIONADO (vs el original: 0.8 cands medios, GT 5/12 a -log0.2) — si los pools del fusionado son menores, el fix es recalibrar el umbral del DFS para el modelo nuevo (graft candidato del domingo). |
| 2026-08-15/16 | **Diagnóstico del 27.08 + compensación (`lb33-scale-fast`) + A/B final** | Diagnóstico convergente, gratis (Colab + logs históricos): (1) el fusionado GENERA MEJOR sin TTT (1.6 vs 0.8 cands/tarea; GT-en-pool **8/12 vs 5/12** al mismo umbral) — descarta la hipótesis "pools más chicos" de la fila anterior; (2) smoke con el stack REAL (TTT r=256 + DFS + selección): fusionado **3.0 vs 2.5** histórico del campeón — la calidad por tarea NO regresó; (3) ttt8 re-medido sobre la base nueva: **VERDE** (exactas idénticas 5/12, −17% tiempo; era rojo en la base vieja). Mecanismo final: **pools 2× más ricos encarecen decode+scoring → menos tareas procesadas → la cobertura colapsó** (score ≈ cobertura × calidad, atacado por el lado del coste). Compensación con dos grafts pre-medidos: ttt8 + scoring4 (~−25% t/tarea) = `lb33-scale-fast`; v1 cazada por el failsafe del commit (IndentationError), v2 smoke reload 3.0 (grafts sin costo de calidad). **A/B (55538928, 16-ago): 28.33** = +1.25 vs 27.08, −1.81 vs 30.14. | ❌ Lectura prerreg "<29": **rama fusionada CERRADA; el campeón sigue siendo el 4B original (30.14).** La dirección del mecanismo se confirmó (recortar coste recuperó +1.25) pero el déficit residual excede lo que ~25% de tiempo compensa. Mapa nítido para el paper: el MISMO SFT da **+2.2 en el 2B** (oculto) y **−1.8/−3.1 en el 4B de producción** — la interacción calibración×presupuesto-de-pipeline domina sobre la capacidad. Palancas restantes sobre el CAMPEÓN: reasignación adaptativa del presupuesto por tarea (dispersión 27-790s medida), inducción Coder-3B (sin veredicto por infra); scoring4-solo cerrado (7.9% share). |
| 2026-08-19 | **POST-MORTEM: la evidencia de esta rama estaba viciada** | Control de formato (`colab_lora_epochs.py` con `SIN_ROLES=1`): mismo 2B base, mismas 10 tareas de holdout no-AGI1, única manipulación = quitar los ids 11/12 del prompt. **Con marcadores 6/10 y 0,0013 NLL/token; sin marcadores 0/10 y 0,2013.** La base de D1 marcó **0,327 NLL/token** — es decir, D1 midió íntegramente en el régimen ROTO (el `tokenizers` de Colab se comía 'user'/'assistant'; ver TTT_PLAN §8). Su "pendiente positiva fuerte" (exactas 1/12→5→7→7) era el LoRA aprendiendo a operar SIN marcadores, no ganando capacidad en ARC: **el mejor punto que D1 alcanzó entrenando (0,055/token) sigue siendo ~40× peor que el modelo SIN ENTRENAR bien formateado (0,0013)**. ⚠️ Al comparar con corridas viejas hay que convertir unidades: `colab_data_scaling.py` promedia por token (`.mean()`), `colab_lora_epochs.py` suma sobre la rejilla (`.sum()`). | ❌ **D1/D2 quedan invalidados como evidencia de que ARC-Heavy aporta capacidad.** Explica retrospectivamente el desplome de D4 (27,08) y del fusionado (28,33): se entrenó para reparar un problema que producción NO tiene. **El campeón NO está afectado** — producción sí emite los ids 11/12, verificado contra su propio log. Cabo suelto honesto sin explicar: el 2B fusionado subió +2,2 en el LB oculto pasando por producción con formato correcto; hipótesis no medida = el TTT r=256 repara parte del daño y en el 2B, lejos de su techo, el neto sale positivo. |
| 2026-08-19 | **Curva de EPOCAS del LoRA (la pregunta del usuario: ¿más pasadas llegan al objetivo?)** | 2B + LoRA r=64, 500 ejemplos fijos de ARC-Heavy, formato correcto, 10 tareas nuevas: **base 6/10 → época 1 3/10 → época 2 2/10**, monótono, empeoran 9 de 10 tareas. El corpus no se rebaraja, así que `loss_final` cubre los MISMOS 200 ejemplos en ambas pasadas: **subió 0,11971 → 0,12471**. Una segunda pasada sobre datos idénticos debería BAJAR su pérdida. | ❌ **Palanca de épocas muerta**, y por una razón diagnóstica: no es desplazamiento de distribución sino el optimizador sin converger — **lr 1e-4 constante sin warmup sobre un modelo ya en pérdida 0,11 es demasiado caliente**, y es la receta exacta de D1/D2/D4 (`colab_data_scaling.py:237`). El barrido de lr queda implementado (`LRS`) pero de baja prioridad: tendría que superar una base de 0,0013/token. |
