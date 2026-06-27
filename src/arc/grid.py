"""Serialización de grids ARC a/desde texto para LLMs.

Representación compacta: cada fila es una cadena de dígitos; filas separadas por '\n'.
Ej:  [[1,2],[3,4]] -> "12\n34". Es reversible con `text_to_grid`.
"""
from __future__ import annotations

from typing import List

Grid = List[List[int]]


def grid_to_text(grid: Grid, sep: str = "") -> str:
    return "\n".join(sep.join(str(c) for c in row) for row in grid)


def text_to_grid(text: str, sep: str = "") -> Grid:
    rows = [r for r in text.strip().splitlines() if r.strip() != ""]
    grid: Grid = []
    for r in rows:
        if sep:
            cells = [c for c in r.split(sep) if c != ""]
        else:
            cells = list(r.strip())
        grid.append([int(c) for c in cells])
    return grid


def format_task_prompt(train_pairs, test_input, sep: str = "") -> str:
    """Construye el prompt textual de una tarea para el LLM (few-shot por ejemplos)."""
    parts = []
    for i, p in enumerate(train_pairs, 1):
        parts.append(f"Example {i}")
        parts.append("Input:\n" + grid_to_text(p.input, sep))
        parts.append("Output:\n" + grid_to_text(p.output, sep))
    parts.append("Test")
    parts.append("Input:\n" + grid_to_text(test_input, sep))
    parts.append("Output:")
    return "\n".join(parts)
