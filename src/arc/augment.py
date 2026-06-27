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
