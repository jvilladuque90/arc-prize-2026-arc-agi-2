"""Generador de tareas ARC sintéticas.

El diferenciador del enfoque ganador (ARC Prize 2025) fue mover el "razonamiento" a un
pipeline de datos sintéticos: generar muchas tareas ARC-style con una regla consistente y
entrenar un modelo pequeño para inducir esa regla a partir de los ejemplos.

Cada `Rule` define:
  - `params(rng)`  -> dict de parámetros FIJOS para toda la tarea (p.ej. el mapeo de color);
  - `inp(rng, p)`  -> un grid de entrada aleatorio;
  - `out(grid, p)` -> el grid de salida determinado por la regla.

`generate_task` fija los params una vez y genera N pares (mismos params, inputs distintos),
de modo que train y test comparten exactamente la misma regla (como en ARC real).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .data import Task, TaskPair

Grid = List[List[int]]
NDArr = "np.ndarray"


# --------------------------------------------------------------------------- #
# Generadores de grids de entrada
# --------------------------------------------------------------------------- #

def _palette(rng, k: int) -> np.ndarray:
    """k colores distintos de 1..9 (0 reservado a fondo)."""
    return rng.choice(np.arange(1, 10), size=min(k, 9), replace=False)


def gi_uniform(rng, hmin=2, hmax=8, k=4) -> np.ndarray:
    h, w = int(rng.integers(hmin, hmax + 1)), int(rng.integers(hmin, hmax + 1))
    pal = _palette(rng, k)
    return rng.choice(pal, size=(h, w))


def gi_sparse(rng, hmin=4, hmax=12, k=3, density=0.3) -> np.ndarray:
    h, w = int(rng.integers(hmin, hmax + 1)), int(rng.integers(hmin, hmax + 1))
    pal = _palette(rng, k)
    g = np.zeros((h, w), dtype=int)
    mask = rng.random((h, w)) < density
    if not mask.any():
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    g[mask] = rng.choice(pal, size=int(mask.sum()))
    return g


def gi_objects(rng, hmin=8, hmax=14, n_obj=3) -> np.ndarray:
    h, w = int(rng.integers(hmin, hmax + 1)), int(rng.integers(hmin, hmax + 1))
    g = np.zeros((h, w), dtype=int)
    pal = _palette(rng, n_obj)
    for c in pal:
        rh, rw = int(rng.integers(1, max(2, h // 2))), int(rng.integers(1, max(2, w // 2)))
        r0, c0 = int(rng.integers(0, h - rh + 1)), int(rng.integers(0, w - rw + 1))
        g[r0:r0 + rh, c0:c0 + rw] = c
    return g


# --------------------------------------------------------------------------- #
# Reglas (transformaciones)
# --------------------------------------------------------------------------- #

@dataclass
class Rule:
    name: str
    params: Callable
    inp: Callable
    out: Callable


def _color_perm(rng) -> Dict[int, int]:
    src = np.arange(1, 10)
    dst = rng.permutation(src)
    return {0: 0, **{int(s): int(d) for s, d in zip(src, dst)}}


def _apply_cmap(g, cmap) -> np.ndarray:
    out = g.copy()
    for s, d in cmap.items():
        out[g == s] = d
    return out


def _gravity_down(g) -> np.ndarray:
    out = np.zeros_like(g)
    for c in range(g.shape[1]):
        col = g[:, c]
        vals = col[col != 0]
        if len(vals):
            out[g.shape[0] - len(vals):, c] = vals
    return out


def _bbox_crop(g) -> np.ndarray:
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return g
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0)
    return g[r0:r1 + 1, c0:c1 + 1]


def _most_frequent(g) -> np.ndarray:
    vals, cnts = np.unique(g[g != 0], return_counts=True)
    if len(vals) == 0:
        return np.array([[0]])
    return np.array([[int(vals[cnts.argmax()])]])


def _symmetrize_lr(g) -> np.ndarray:
    out = g.copy()
    w = g.shape[1]
    for c in range(w // 2):
        merged = np.where(g[:, c] != 0, g[:, c], g[:, w - 1 - c])
        out[:, c] = merged
        out[:, w - 1 - c] = merged
    return out


def build_rules() -> List[Rule]:
    return [
        Rule("recolor", lambda rng: {"cmap": _color_perm(rng)},
             lambda rng, p: gi_uniform(rng), lambda g, p: _apply_cmap(g, p["cmap"])),
        Rule("rotate", lambda rng: {"k": int(rng.integers(1, 4))},
             lambda rng, p: gi_uniform(rng), lambda g, p: np.rot90(g, p["k"])),
        Rule("flip", lambda rng: {"axis": int(rng.integers(0, 2))},
             lambda rng, p: gi_uniform(rng),
             lambda g, p: np.flip(g, p["axis"])),
        Rule("transpose", lambda rng: {},
             lambda rng, p: gi_uniform(rng), lambda g, p: g.T),
        Rule("tile", lambda rng: {"rh": int(rng.integers(1, 4)), "rw": int(rng.integers(1, 4))},
             lambda rng, p: gi_uniform(rng, hmax=5),
             lambda g, p: np.tile(g, (p["rh"], p["rw"]))),
        Rule("mirror_h", lambda rng: {},
             lambda rng, p: gi_uniform(rng, hmax=6),
             lambda g, p: np.concatenate([g, np.fliplr(g)], axis=1)),
        Rule("mirror_v", lambda rng: {},
             lambda rng, p: gi_uniform(rng, hmax=6),
             lambda g, p: np.concatenate([g, np.flipud(g)], axis=0)),
        Rule("upscale", lambda rng: {"fh": int(rng.integers(2, 4)), "fw": int(rng.integers(2, 4))},
             lambda rng, p: gi_uniform(rng, hmax=5),
             lambda g, p: np.kron(g, np.ones((p["fh"], p["fw"]), dtype=int))),
        Rule("border", lambda rng: {"c": int(rng.integers(1, 10))},
             lambda rng, p: gi_uniform(rng, hmax=8),
             lambda g, p: np.pad(g, 1, constant_values=p["c"])),
        Rule("gravity", lambda rng: {},
             lambda rng, p: gi_sparse(rng), lambda g, p: _gravity_down(g)),
        Rule("crop_bbox", lambda rng: {},
             lambda rng, p: gi_objects(rng), lambda g, p: _bbox_crop(g)),
        Rule("most_frequent", lambda rng: {},
             lambda rng, p: gi_sparse(rng, density=0.5), lambda g, p: _most_frequent(g)),
        Rule("symmetrize", lambda rng: {},
             lambda rng, p: gi_sparse(rng, hmin=4, hmax=10, density=0.25),
             lambda g, p: _symmetrize_lr(g)),
    ]


RULES = build_rules()
RULES_BY_NAME = {r.name: r for r in RULES}


# --------------------------------------------------------------------------- #
# Generación de tareas
# --------------------------------------------------------------------------- #

def _valid(g: np.ndarray) -> bool:
    return g.ndim == 2 and g.size > 0 and g.shape[0] <= 30 and g.shape[1] <= 30 \
        and g.min() >= 0 and g.max() <= 9


def generate_task(rng, rule: Optional[Rule] = None, n_train=4, n_test=1,
                  max_tries=40) -> Optional[Task]:
    """Genera una tarea consistente con una regla. Devuelve None si no logra
    pares válidos y no triviales (input != output)."""
    rule = rule or RULES[int(rng.integers(len(RULES)))]
    p = rule.params(rng)
    pairs: List[TaskPair] = []
    tries = 0
    need = n_train + n_test
    while len(pairs) < need and tries < max_tries:
        tries += 1
        gi = np.asarray(rule.inp(rng, p), dtype=int)
        try:
            go = np.asarray(rule.out(gi, p), dtype=int)
        except Exception:
            continue
        if not (_valid(gi) and _valid(go)):
            continue
        if rule.name != "recolor" and np.array_equal(gi, go):
            continue  # evita pares triviales (salvo recolor, que puede coincidir)
        pairs.append(TaskPair(gi.tolist(), go.tolist()))
    if len(pairs) < need:
        return None
    return Task(task_id=f"synth_{rule.name}",
                train=pairs[:n_train], test=pairs[n_train:need])


def generate_dataset(n_tasks: int, seed: int = 0):
    """Itera `n_tasks` tareas sintéticas (con task_id único)."""
    rng = np.random.default_rng(seed)
    made = 0
    i = 0
    while made < n_tasks:
        i += 1
        task = generate_task(rng)
        if task is None:
            continue
        task.task_id = f"synth_{made:06d}_{task.task_id.split('_', 1)[1]}"
        made += 1
        yield task
