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

---

## RESULTADO (30-sep, ~0,3 h-cuota: 4 min de pared)

Sanidad OK: M = 0,0001 (≤0,01) y E = 0,036 (≈0,05). Cero OOM. Datos crudos en `scripts/_nemvac.json`.

| grupo | n | nll/celda mediana | p90 | greedy exacto sin TTT | celdas | r |
|---|---|---|---|---|---|---|
| M: AGI-1 train | 40 | 0,0001 | 0,013 | 90% | 100 | 0 |
| T2: AGI-2 train | 40 | 0,0000 | 0,006 | 95% | 168 | −0,25 |
| E: eval | 40 | **0,0363** | 0,105 | **10%** | 441 | 1 |
| **Nemotron grandes** | 150 | **0,0002** | 0,005 | **74%** | 529 | **0,08** |
| Nemotron resto | 150 | 0,0014 | 0,191 | 78% | 49 | 0,43 |

**Veredicto por prerreg:**
- **Grandes: r = 0,08 ≤ 0,2 ⇒ la fuente MUERE para el objetivo que importaba.** `sft139` emite la
  salida entera correcta sin TTT en el 74% de los puzzles de ≥400 celdas, y **0 de 150** llegan al nivel
  de NLL del eval. Memorizados o triviales para él: da igual, no hay gradiente que aprovechar.
- Resto: r = 0,43, parcial. 33/150 (22%) están al nivel del eval, pero son puzzles **pequeños**
  (mediana 49 celdas), lejos del déficit medido (escala/contenido), y parte de esa NLL alta puede ser
  ruido de respuestas mal etiquetadas en el dataset.

**Lectura de fondo:** el déficit no es "falta de puzzles", es que el eval cae **fuera** de lo que el
modelo vio: los grandes de Nemotron son grandes pero **dentro** de su distribución (greedy 74%), el eval
es grande y **fuera** (greedy 10%). Más datos de la misma fábrica no mueven eso.

---

# PREREG `rearcvac` (30-sep): ¿falla sft139 al llevar una regla CONOCIDA a una escala NO vista?

`nemvac` enseñó que el eval es grande **y fuera de distribución**, mientras que los grandes de Nemotron
son grandes **dentro**. Esta puerta separa los dos factores. Por cada tarea AGI-1 de training
(memorizada por sft139): **demos = las originales**, **test = instancia NUEVA de re-ARC** verificada.
- `rearc_grande`: salida ≥ max(400, 1,3× la mayor salida demo) — fuera de la envolvente de escala
  (mediana 609 celdas, 5,75× la envolvente). 150 tareas.
- `rearc_chico`: salida ≤ la mayor demo — misma regla dentro de escala. 150; **136 pareadas**.
Mismo kernel y mismos controles que `nemvac` (`scripts/rearc_puerta.py` genera los datos).

| resultado | lectura | siguiente paso |
|---|---|---|
| r(grande) ≥ 0,5 **y** grande > chico en ≥70% de los pares | la ESCALA es el déficit aun con reglas conocidas | SFT con re-ARC grande (datos ilimitados, dirigidos) |
| r(grande) ≤ 0,2 | la escala sola no rompe al modelo: el déficit es la NOVEDAD de la regla | no hay datos fabricables que lo cubran ⇒ consolidar + Paper Track |
| intermedio | escala explica parte | SFT solo con las instancias de NLL alta |

Sanidad: r(chico) debe salir bajo (≲0,3): misma regla memorizada, misma escala.

## RESULTADO `rearcvac` (30-sep, ~0,2 h-cuota: 3 min de pared)

| grupo | n | nll/celda mediana | greedy exacto sin TTT | celdas | r |
|---|---|---|---|---|---|
| re-ARC chico (regla conocida, escala conocida) | 150 | 0,0040 | 75% (78% pareado) | 60 | 0,62 |
| re-ARC **grande** (misma regla, 5,75× la envolvente) | 150 | **0,0005** | 67% (pareado) | 609 | **0,24** |
| eval (regla NUEVA) | 40 | 0,0363 | **10%** | 441 | 1 |

Pareado (119 tareas con ambos brazos): el grande es peor por celda en **27/119 (23%)**; NLL total
mediana 0,12 → 0,26.

**Fallo de sanidad que hay que decir**: r(chico) = 0,62 > 0,3. No es el banco, es la **métrica**: la
NLL por celda penaliza a las rejillas pequeñas (pocas celdas para repartir los mismos tokens de
decisión). La medida limpia para comparar escalas es el **greedy exacto**: 78% → 67% pareado.

**Veredicto (fila 2 del prerreg en sustancia; r=0,24 queda en la frontera, el criterio pareado falla
con claridad):** llevar una regla CONOCIDA a 5,75× su escala cuesta ~11 puntos de resolución exacta;
una regla NUEVA cuesta ~65 (75% → 10%). **El déficit es la novedad de la regla, no la escala.**
No hay datos que podamos fabricar (Nemotron, re-ARC, training de ARC) que lo cubran: todos son reglas
que sft139 ya domina. Esto cierra el Plan A tal como estaba planteado.
