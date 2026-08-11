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

### D2 — Extensión de la curva (GRATIS, condicional a D1 verde)
N=8000 (sesión 2, reanuda con más datos). Si la pendiente sigue positiva en log-N → luz verde
al paso 3 real.

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

## 5. Registro de resultados de la cadena

| Fecha | Etapa | Resultado | Decisión |
|---|---|---|---|
| — | D1 (curva Colab) | *(en curso — sesión 1 lanzada 2026-08-11)* | — |
| — | D2 (N=8000) | *(condicional)* | — |
| — | D3 (fuentes) | *(condicional)* | — |
| — | D4 (SFT 4B Kaggle) | *(condicional)* | — |
