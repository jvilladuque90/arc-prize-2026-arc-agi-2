"""Inferencia AIRV: Augment → predict → Invert → Vote.

Para cada test input se predice sobre varias vistas augmentadas (D4 + color), se
des-augmenta cada predicción al espacio original y se eligen los 2 candidatos por consenso.
Esto estabiliza al modelo frente a orientación/colores y suele subir el pass@2.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, List, Optional, Tuple

from .augment import augment_task, color_permutation, invert_grid
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
    """Construye {attempt_1, attempt_2} a partir de candidatos ordenados por voto."""
    a1 = candidates[0] if candidates else fallback
    a2 = candidates[1] if len(candidates) > 1 else a1
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
