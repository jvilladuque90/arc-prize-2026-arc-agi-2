# PREREG `nemvac` — ¿sft139 ya conoce los puzzles de Nemotron? (2026-09-30)

## Contexto

- `nvplus` (NVARC+ v2 sin cambios) = **29,44** < 29,86 ⇒ **descartado** por prerreg. El campeón sigue
  siendo `lb33-det` (30,69). Los injertos A1-A3 (sobre NVARC+) quedan sin objeto.
- La apuesta C (Qwen3-8B-Nemotron77K como segundo generador) no cabe en el reloj: el campeón usa el
  95% y el 8B costaría ≥15% (≈ −9 tareas) incluso con vLLM+FP8, que no tenemos.
- Variante que esquiva el reloj: usar los **puzzles** de `nvidia/Nemotron-SFT-ARC-AGI-v1` como datos
  de SFT para `sft139` en su formato nativo (Plan A). Coste de inferencia cero.
- Puerta gratis (`scripts/nemotron_puerta.py`, 3.907 puzzles únicos de 5 subconjuntos): **89% nuevos**
  (ninguna rejilla coincide con ARC train/eval bajo D4), **0 contaminación del eval**, 49% con
  respuesta parseable; escala de AGI-1 (mediana 88 celdas) con un **15,8% de salidas ≥400 celdas**.

## La pregunta que queda

NVIDIA es también el equipo NVARC. Si estos puzzles salen de los sintéticos con los que se entrenó
`sft139`, el modelo los tiene memorizados y no hay señal que aprender (lección `sft-outenv`).
"Nuevos respecto a ARC" no implica "nuevos para sft139".

## Medición

Una pasada hacia delante por puzzle, `sft139` **sin TTT**, vista canónica, tokens a mano con
marcadores de rol (11/12). NLL de la respuesta por celda. Shard ×4.

- **Nemotron grandes**: hasta 150 puzzles nuevos con salida ≥400 celdas.
- **Nemotron resto**: 150 puzzles nuevos al azar con salida <400.
- **Control M (memorizado)**: 40 tareas AGI-1 de training.
- **Control T2**: 40 tareas AGI-2 de training.
- **Control E (no visto)**: 40 tareas del eval (sin las 6 sucias ni `36a08778`).

Posición en escala log entre memorizado y no visto: `r = log(nem/M) / log(E/M)` con medianas.

## Lecturas (fijadas antes)

| resultado | lectura | siguiente paso |
|---|---|---|
| **r ≥ 0,5** en grandes | sft139 no los conoce: hay señal | banco de SFT (Plan A) sobre los grandes |
| 0,2 < r < 0,5 | conocimiento parcial | SFT solo con los puzzles de NLL alta (filtro por puzzle) |
| **r ≤ 0,2** | memorizados | la fuente muere; Plan A sin datos |

Sanidad: M debe salir ≲ 0,01 y E ≈ 0,05 (niveles ya medidos). Si no, el banco está mal construido
y no se lee nada.

Coste previsto: ~15 min de pared ≈ **0,5 h-cuota**.
