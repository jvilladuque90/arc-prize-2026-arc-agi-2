"""Augmentación de tareas ARC (clave para fine-tune y test-time training).

Transformaciones que preservan la estructura del problema:
  - Grupo diédrico D4: 8 simetrías (rotaciones 0/90/180/270 + reflexiones).
  - Permutaciones de color (biyección sobre 0-9).
Todas son invertibles, de modo que una predicción sobre la vista augmentada puede
des-augmentarse al espacio original (necesario para AIRV / voting en inferencia).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

Grid = List[List[int]]


# --- D4: ocho simetrías como (func, inversa) sobre np.ndarray ---------------

def _d4_ops() -> List[Tuple[str, Callable, Callable]]:
    rot = np.rot90
    return [
        ("identity", lambda a: a, lambda a: a),
        ("rot90", lambda a: rot(a, 1), lambda a: rot(a, -1)),
        ("rot180", lambda a: rot(a, 2), lambda a: rot(a, -2)),
        ("rot270", lambda a: rot(a, 3), lambda a: rot(a, -3)),
        ("fliplr", np.fliplr, np.fliplr),
        ("flipud", np.flipud, np.flipud),
        ("transpose", lambda a: a.T, lambda a: a.T),
        ("anti_transpose", lambda a: np.rot90(a.T, 2), lambda a: np.rot90(a, 2).T),
    ]


D4 = _d4_ops()


def apply_d4(grid: Grid, idx: int) -> Grid:
    a = np.array(grid, dtype=int)
    return D4[idx][1](a).tolist()


def invert_d4(grid: Grid, idx: int) -> Grid:
    a = np.array(grid, dtype=int)
    return D4[idx][2](a).tolist()


# --- Permutación de color ----------------------------------------------------

def color_permutation(seed: int, keep_zero: bool = True) -> List[int]:
    """Devuelve un mapeo de los 10 colores. Si keep_zero, 0 (fondo) se mantiene fijo."""
    rng = np.random.default_rng(seed)
    if keep_zero:
        rest = np.arange(1, 10)
        rng.shuffle(rest)
        return [0] + rest.tolist()
    perm = np.arange(10)
    rng.shuffle(perm)
    return perm.tolist()


def apply_color_map(grid: Grid, mapping: List[int]) -> Grid:
    return [[mapping[c] for c in row] for row in grid]


def invert_color_map(mapping: List[int]) -> List[int]:
    inv = [0] * len(mapping)
    for src, dst in enumerate(mapping):
        inv[dst] = src
    return inv


# --- Augmentación a nivel de grid y de tarea --------------------------------

def augment_grid(grid: Grid, d4_idx: int, color_map: List[int]) -> Grid:
    """Aplica una simetría D4 y luego un mapeo de color (en ese orden)."""
    return apply_color_map(apply_d4(grid, d4_idx), color_map)


def invert_grid(grid: Grid, d4_idx: int, color_map: List[int]) -> Grid:
    """Inversa de `augment_grid`: des-mapea color y luego des-aplica D4."""
    return invert_d4(apply_color_map(grid, invert_color_map(color_map)), d4_idx)


def augment_task(task, d4_idx: int, color_map: List[int]):
    """Devuelve una copia de `task` con TODOS sus grids (train+test) augmentados
    de forma consistente. Mantiene la validez de la tarea (es una simetría)."""
    from .data import Task, TaskPair  # import local para evitar ciclo

    def aug_pair(p):
        out = augment_grid(p.output, d4_idx, color_map) if p.output is not None else None
        return TaskPair(augment_grid(p.input, d4_idx, color_map), out)

    return Task(
        task_id=task.task_id,
        train=[aug_pair(p) for p in task.train],
        test=[aug_pair(p) for p in task.test],
    )
