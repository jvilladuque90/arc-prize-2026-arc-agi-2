"""Métrica oficial: pass@2 por test output, promediado.

Cada test output da 1 punto si attempt_1 O attempt_2 coincide exactamente.
"""
from __future__ import annotations

from typing import Dict, List

Grid = List[List[int]]


def exact_match(a: Grid, b: Grid) -> bool:
    return a == b


def score_submission(
    submission: Dict[str, List[dict]],
    solutions: Dict[str, List[Grid]],
) -> float:
    """`submission`: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}.
    `solutions`:   {task_id: [grid, ...]}.
    Devuelve el score promedio en [0, 1]."""
    total = 0
    correct = 0
    for tid, sols in solutions.items():
        attempts = submission.get(tid, [])
        for i, truth in enumerate(sols):
            total += 1
            if i >= len(attempts):
                continue
            a = attempts[i]
            if exact_match(a.get("attempt_1", []), truth) or exact_match(
                a.get("attempt_2", []), truth
            ):
                correct += 1
    return correct / total if total else 0.0
