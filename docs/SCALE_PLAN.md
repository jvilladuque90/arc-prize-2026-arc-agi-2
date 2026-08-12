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
| 2026-08-12 | **E1 — ENSAYO en test oculto** (idea del usuario: usar el slot diario ocioso) | 2B + adapter ARC-Heavy **n=4000 fusionado** (dataset `arcscale-merged-2b`), config campeona IDÉNTICA a la submission 18.33 del 2B-solo (15-jul) → A/B directo del efecto ARC-Heavy en la distribución real con el stack completo (TTT+DFS+selección). Con solo 22 min de quota: **commit barato** (0 tareas, sin cargar modelo, ~3 min; el rerun de scoring hace el trabajo gratis). Dos gates cazados en el camino: umbral del pool del proxy scoring4; slug con guion + zip aplanado rompían la detección del modelo (fix: detección por contenido). Submission 55450984: **0.00 — FALLA SISTÉMICA, no señal del modelo.** Post-mortem: el merge se guardó con transformers **5.13.1** (Colab), cuyo config nuevo pierde `rope_theta: 5000000` del nivel superior (queda anidado en `rope_parameters`); el stack 4.53 del kernel usó el default (10000) → atención rota → todos los workers produjeron basura → submission placeholder válida → 0.00. También faltaban vocab.json/added_tokens/special_tokens_map. | **E1-v2 (13-ago):** dataset reempacado con los **aux files originales** del sorokin (verificado: rope_theta 5M presente); el commit ahora **VALIDA** (carga unsloth + NLL sanity < 1.5 en una tarea real — el fallo se habría cazado antes de submitir); mismo fix propagado a `scale_sft_4b` (merged_4b lleva aux originales) y a `colab_merge_2b`. Re-submit con el slot de la noche. Lecturas pre-registradas sin cambio: **≥21** verde / **≈18** replantear / **<17** revisar. |
| — | D2 (N=8000) | *(pendiente del reset diario de Colab)* | — |
| — | D3 (fuentes) | *(condicional)* | — |
| — | D4 (SFT 4B Kaggle) | *(VERDE — diseñar con presupuesto del viernes)* | — |
