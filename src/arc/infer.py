"""Inferencia AIRV: Augment → predict → Invert → seleccionar attempts.

Para cada test input se generan candidatos sobre varias vistas augmentadas (D4 + color),
se des-augmenta cada predicción al espacio original y se eligen los 2 attempts. Dos modos
de selección:

- **"vote"** (legado): 1 predicción greedy por vista, consenso por conteo.
- **"score"** (recomendado; técnica de los top 2024/2025): además de greedy se muestrean
  candidatos extra por vista, y cada candidato único se **puntúa por su log-verosimilitud
  bajo K vistas augmentadas** (forward pass, sin generación). pass@2 es un problema de
  ranking: puntuar > contar.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from .augment import augment_grid, augment_task, color_permutation, invert_grid
from .data import Task
from .grid import format_task_prompt, grid_to_text, text_to_grid

Grid = List[List[int]]


def _aug_specs(n_aug: int, task_id: str) -> List[Tuple[int, List[int]]]:
    specs = [(0, list(range(10)))]  # identidad siempre incluida
    for a in range(1, n_aug):
        cmap = color_permutation(seed=(hash((task_id, a)) & 0xFFFFFFFF), keep_zero=True)
        specs.append((a % 8, cmap))
    return specs


def predict_test_input(
    task: Task,
    test_idx: int,
    predict_fn: Callable[[str], Optional[Grid]],
    n_aug: int = 8,
    sep: str = "",
) -> List[Grid]:
    """Devuelve los candidatos ordenados por nº de votos para un test input dado.

    `predict_fn(prompt) -> Grid|None` encapsula al modelo (ver arc.model.generate_grid).
    """
    votes: Counter = Counter()
    for d4_idx, cmap in _aug_specs(n_aug, task.task_id):
        atask = augment_task(task, d4_idx, cmap)
        prompt = format_task_prompt(atask.train, atask.test[test_idx].input, sep)
        pred = predict_fn(prompt)
        if pred is None:
            continue
        # des-augmentar al espacio original para poder votar entre vistas
        orig = invert_grid(pred, d4_idx, cmap)
        votes[grid_to_text(orig)] += 1
    return [text_to_grid(t) for t, _ in votes.most_common()]


def two_attempts(candidates: List[Grid], fallback: Grid) -> dict:
    """Construye {attempt_1, attempt_2} a partir de candidatos ordenados.

    pass@2 regala un segundo intento: duplicar attempt_1 lo desperdicia. Si solo hay
    un candidato, attempt_2 usa el fallback (identidad) cuando es distinto."""
    a1 = candidates[0] if candidates else fallback
    if len(candidates) > 1:
        a2 = candidates[1]
    else:
        a2 = fallback if fallback != a1 else a1
    return {"attempt_1": a1, "attempt_2": a2}


def predict_task(
    task: Task,
    predict_fn: Callable[[str], Optional[Grid]],
    n_aug: int = 8,
    sep: str = "",
) -> List[dict]:
    """Predice todos los test inputs de una tarea -> lista de {attempt_1, attempt_2}."""
    results = []
    for i, tp in enumerate(task.test):
        cands = predict_test_input(task, i, predict_fn, n_aug=n_aug, sep=sep)
        results.append(two_attempts(cands, fallback=tp.input))
    return results


# --- Selección por scoring de verosimilitud (recomendada) ---------------------

def rank_keys(scores: Dict[str, float], votes: Dict[str, int]) -> List[str]:
    """Ordena candidatos (por clave de texto) por score total desc; votos desempatan."""
    return sorted(scores, key=lambda k: (scores[k], votes.get(k, 0), k), reverse=True)


def predict_test_input_scored(
    task: Task,
    test_idx: int,
    model,
    tokenizer,
    *,
    n_aug: int = 8,
    n_cands: int = 2,
    temperature: float = 0.7,
    n_score_views: int = 4,
    sep: str = "",
    max_new_tokens: int = 2048,
    constrained: bool = True,
) -> List[Grid]:
    """Candidatos ordenados por log-verosimilitud agregada bajo vistas augmentadas.

    1) Generación: por cada una de las `n_aug` vistas, 1 candidato greedy +
       (`n_cands`-1) muestreados; todo des-augmentado al espacio original.
    2) Scoring: cada candidato único se re-serializa en las primeras `n_score_views`
       vistas y se acumula su log P(candidato | prompt de la vista).
    """
    from .model import generate_grid_candidates, score_completions

    specs = _aug_specs(n_aug, task.task_id)
    votes: Counter = Counter()
    for d4_idx, cmap in specs:
        atask = augment_task(task, d4_idx, cmap)
        prompt = format_task_prompt(atask.train, atask.test[test_idx].input, sep)
        preds = generate_grid_candidates(
            model, tokenizer, prompt, sep=sep, n_candidates=n_cands,
            temperature=temperature, max_new_tokens=max_new_tokens,
            constrained=constrained,
        )
        for pred in preds:
            votes[grid_to_text(invert_grid(pred, d4_idx, cmap))] += 1
    if not votes:
        return []

    cand_keys = list(votes)
    cand_grids = {k: text_to_grid(k) for k in cand_keys}
    totals = {k: 0.0 for k in cand_keys}
    for d4_idx, cmap in specs[: max(1, n_score_views)]:
        atask = augment_task(task, d4_idx, cmap)
        prompt = format_task_prompt(atask.train, atask.test[test_idx].input, sep)
        # misma convención que el SFT/TTT: completion = "\n" + grid + "\n" (+ EOS)
        comps = ["\n" + grid_to_text(augment_grid(cand_grids[k], d4_idx, cmap), sep) + "\n"
                 for k in cand_keys]
        for k, s in zip(cand_keys, score_completions(model, tokenizer, prompt, comps)):
            totals[k] += s
    return [cand_grids[k] for k in rank_keys(totals, votes)]


def predict_task_scored(
    task: Task,
    model,
    tokenizer,
    *,
    n_aug: int = 8,
    n_cands: int = 2,
    temperature: float = 0.7,
    n_score_views: int = 4,
    sep: str = "",
    max_new_tokens: int = 2048,
    constrained: bool = True,
) -> List[dict]:
    """Como `predict_task` pero con selección por scoring de verosimilitud."""
    results = []
    for i, tp in enumerate(task.test):
        cands = predict_test_input_scored(
            task, i, model, tokenizer, n_aug=n_aug, n_cands=n_cands,
            temperature=temperature, n_score_views=n_score_views, sep=sep,
            max_new_tokens=max_new_tokens, constrained=constrained,
        )
        results.append(two_attempts(cands, fallback=tp.input))
    return results
