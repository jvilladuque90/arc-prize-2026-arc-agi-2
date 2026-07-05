# ARC Prize 2026 — ARC-AGI-2

Solución para la competencia de Kaggle [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).

> **Objetivo:** resolver tareas de razonamiento abstracto (grids de colores) prediciendo
> exactamente 2 outputs por cada input de test. Se acierta si **cualquiera** de los 2
> coincide exactamente con la verdad. El score final es el promedio sobre todos los test outputs.

## Reglas clave de la competencia (resumen)

| Aspecto | Detalle |
|---|---|
| Tipo | **Code competition** — se entrega un **Kaggle Notebook** que corre el modelo |
| Internet en evaluación | **NO** hay acceso a internet durante el scoring → todo debe ir empaquetado (modelo + pesos + código) |
| Runtime | Límite de tiempo de ejecución del notebook (≈12 h; el HW/compute se confirma en la página de reglas) |
| Métrica | `pass@2` por test output, promediado |
| Open source | El código y método deben publicarse con licencia permisiva para optar a premios |
| Premios | ~$700k total: Progreso (~$275k), Grand Prize por writeup (~$275k), Bonus $150k al alcanzar 85% |

> ⚠️ Las reglas exactas (HW, runtime, licencia aceptada) deben confirmarse logueado en
> la [página de reglas](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules).
> La distinta edición **ARC-AGI-3** sí permite APIs/internet; **ARC-AGI-2 NO**.

## Datos

Los datos de la competencia **NO** se versionan en este repo (ver `.gitignore`). Para descargarlos:

```bash
python scripts/download_data.py
```

Estructura (en `data/`):

- `arc-agi_training_challenges.json` — 1000 tareas de entrenamiento (con soluciones aparte).
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json` — 120 tareas públicas de evaluación.
- `arc-agi_evaluation_solutions.json`
- `arc-agi_test_challenges.json` — copia pública del set de test (el privado real está oculto).
- `sample_submission.json` — formato de entrega: `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`.

## Estrategia

La receta dominante en ARC-AGI (ganadores 2024 **y 2025**) es un **LLM pequeño fine-tuneado +
Test-Time Training (TTT)**. Calibración: el 1º puesto 2025 (NVARC, NVIDIA) logró **24.03%** en
ARC-AGI-2 con un modelo ~4B + TTT + corpus sintético masivo + ensemble con TRM. Plan en fases:

1. **Baseline sin ML** — heurísticas/identidad + selección por votación, para validar el pipeline
   de submission de punta a punta.
2. **LLM + LoRA (PEFT)** — fine-tune de un modelo pequeño (recomendado **Qwen2.5-0.5B…7B**)
   sobre tareas ARC serializadas a texto, con **augmentación masiva**:
   - 8 simetrías del grupo diédrico D4 (rotaciones + reflexiones).
   - Permutaciones de colores (preservando estructura).
   - Reordenamiento de los pares de ejemplo; transposición.
   Esto expande las 1000 tareas en cientos de miles de muestras.
3. **Test-Time Training (TTT)** — en inferencia, adaptar un LoRA por-tarea sobre los pocos pares
   `train` de cada tarea de test (y sus augmentaciones), luego predecir. Fue **la** clave para ganar.
4. **Inferencia multi-candidato + scoring por verosimilitud** — generar varios candidatos por
   vista augmentada (sampling/DFS), des-augmentar, y elegir los 2 attempts **puntuando cada
   candidato por su log-verosimilitud bajo vistas augmentadas** (voto simple solo como desempate).
5. **Ingeniería de runtime** — 4 workers (1 por GPU L4), presupuesto de tiempo por tarea y
   escritura incremental del `submission.json`.

> **Nota sobre el modelo:** Qwen2.5 (0.5B–7B) es preferible a **Llama 4** para esta competencia:
> Llama 4 es un MoE enorme, inviable para TTT por-tarea dentro del límite de ~12 h offline en GPU
> de Kaggle. Modelos pequeños permiten muchas iteraciones de TTT y caben en el notebook.

Detalle ampliado en [`docs/STRATEGY.md`](docs/STRATEGY.md).

## Estructura del repo

```
src/arc/            Paquete Python: carga de datos, serialización, augmentación, métrica
scripts/            Utilidades CLI (descarga de datos, preparación de dataset de fine-tune)
docs/               Notas de estrategia y diseño
notebooks/          Notebook(s) de entrega para Kaggle
data/               Datos de la competencia (NO versionado)
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
# credenciales Kaggle en .env (KAGGLE_API_TOKEN, kaggle_username)
python scripts/download_data.py
```

## Licencia

MIT (ver [`LICENSE`](LICENSE)) — licencia permisiva acorde con el requisito de open source de la
competencia. Confirmar en las reglas si exigen CC0/MIT-0.
