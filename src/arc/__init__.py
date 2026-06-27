"""Utilidades para ARC Prize 2026 (ARC-AGI-2)."""

from .data import (
    Task,
    TaskPair,
    load_challenges,
    load_solutions,
    load_dataset,
)
from .grid import grid_to_text, text_to_grid
from .metrics import exact_match, score_submission

__all__ = [
    "Task",
    "TaskPair",
    "load_challenges",
    "load_solutions",
    "load_dataset",
    "grid_to_text",
    "text_to_grid",
    "exact_match",
    "score_submission",
]
