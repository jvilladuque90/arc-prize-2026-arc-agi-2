# Auditoría del notebook LB33 (2026-07-11)

Auditoría multi-agente (hardcode / leakage / método / procedencia, con verificación
adversarial de cada hallazgo) sobre el fork `agi-arc-agi2-lb33-minimal-perfpatch`
(original de boristown/【暗黑AGI】). Condición del usuario para adoptarlo como baseline:
sin hardcode ni overfit tramposo.

## Veredicto: LIMPIO — apto como baseline

- **Sin hardcode**: cero lookups por task_id, cero grids literales (solo el placeholder
  `[[0]]`), cero base64/pickle/exec/hash-tables, cero datasets adjuntos, internet OFF.
  La submission se construye solo con salidas del modelo computadas en el propio run.
- **Sin leakage hacia la submission**: `evaluation_solutions.json` solo se usa para
  benchmark impreso fuera del rerun. Los 4 task-IDs literales son un smoke-test que
  actúa únicamente cuando `KAGGLE_IS_COMPETITION_RERUN` no está seteado; el rerun oculto
  procesa las 240 tareas.
- **Procedencia**: el modelo `sorokin/qwen3_4b_grids15_sft139` es el checkpoint del
  release open-source oficial de **NVARC** (Sorokin + Puget, NVIDIA; 1º ARC Prize 2025),
  Apache 2.0, publicado 2025-11-02 — antes de la competencia 2026: memorizar el set
  oculto es materialmente imposible.

## Hallazgo crítico (no es trampa, pero nos afecta)

**El SFT del checkpoint NVARC incluye el evaluation set público de ARC-AGI-2**
(train+test; verificado en el `build_datasets.py` del repo NVARC y admitido en su paper).
Legal y sin efecto en el LB oculto, pero **invalida el eval público como validación
local para este modelo**: cualquier score medido ahí es memorización. Validar solo con
(a) split reservado del training set, (b) tareas sintéticas propias, (c) el LB (1/día).

## El método (por qué da ~33%)

Receta ARChitects/NVARC: por tarea, TTT con LoRA **r=256** (1 época, 128 augmentaciones
D4×color) sobre Qwen3-4B con vocabulario recortado a ~16 tokens de grid; decodificación
**DFS determinista** (enumera todas las salidas con probabilidad acumulada >0.2) bajo 16
augmentaciones; re-scoring teacher-forced bajo 8 augmentaciones frescas; top-2 por
votos+verosimilitud. 4 procesos × 1 L4, presupuesto 12h−10min. Los autores reportaron
29.72% con este checkpoint en el LB 2025 (cayó a 24.03% en el privado — esperar caída
similar público→privado).

## Riesgos residuales

- Datos de entrenamiento del modelo no auditables al 100% (mitigado: release previo a
  la competencia, Apache 2.0, del equipo ganador verificado por ARC Prize).
- Caída histórica público→privado de varios puntos.
- Operativos: depende de su **imagen docker BYOD propia** + kernel de wheels
  `sorokin/pip-install-unsloth-flash-patch` (preservar ambos en la metadata); la
  metadata del fork tenía `enable_gpu: false` (corregir a L4×4 antes del push); tareas
  no alcanzadas quedan `[[0]]`.

## Plan de integración (resumen; detalle en STRATEGY.md)

P0: reproducir el control (~33%) pusheando el fork con metadata corregida — el
Save & Run solo corre el smoke de 4 tareas (~25 min de quota); el rerun de 12h es gratis.
P1: held-out limpio (split de training + sintético propio); jamás validar en eval público.
P2: injertos nuestros para superar 33%: presupuesto por tiers/cobertura (`runner.py` —
LB33 deja `[[0]]` en lo que no alcanza), portar `turbo_dfs`/`calc_scores` a `src/arc/`
con tests, benchmark de selección attempt_1/2, y a medio plazo ensemble con un segundo
modelo entrenado sin el eval público (validador limpio + pool extra de candidatos).
