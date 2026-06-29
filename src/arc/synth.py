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


def _components(g: np.ndarray, diag: bool = False) -> List[dict]:
    """Componentes conexos de celdas no-cero del MISMO color (4- u 8-conn)."""
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                color = int(g[i, j]); stack = [(i, j)]; seen[i, j] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in nb:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == color:
                            seen[ny, nx] = True; stack.append((ny, nx))
                comps.append({"color": color, "cells": cells, "size": len(cells)})
    return comps


def _fill_enclosed(g: np.ndarray) -> np.ndarray:
    """Rellena el fondo (0) NO conectado al borde con el color no-cero dominante."""
    h, w = g.shape
    outside = np.zeros((h, w), bool)
    stack = [(i, j) for i in range(h) for j in range(w)
             if (i in (0, h - 1) or j in (0, w - 1)) and g[i, j] == 0]
    for s in stack:
        outside[s] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not outside[ny, nx] and g[ny, nx] == 0:
                outside[ny, nx] = True; stack.append((ny, nx))
    enclosed = (g == 0) & ~outside
    if not enclosed.any():
        raise ValueError("sin huecos encerrados")
    vals, cnts = np.unique(g[g != 0], return_counts=True)
    if len(vals) == 0:
        raise ValueError("grid vacio")
    out = g.copy(); out[enclosed] = int(vals[cnts.argmax()])
    return out


# --- generadores de input especializados (object-centric) -------------------

def gi_blobs(rng, hmin=7, hmax=12) -> np.ndarray:
    """Fondo con n celdas sueltas (no adyacentes) de un mismo color -> n objetos."""
    h, w = int(rng.integers(hmin, hmax + 1)), int(rng.integers(hmin, hmax + 1))
    g = np.zeros((h, w), dtype=int)
    color = int(rng.integers(1, 10))
    n = int(rng.integers(1, 7))
    placed, tries = 0, 0
    nb8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while placed < n and tries < 300:
        tries += 1
        y, x = int(rng.integers(h)), int(rng.integers(w))
        if g[y, x] != 0:
            continue
        if any(0 <= y + dy < h and 0 <= x + dx < w and g[y + dy, x + dx] != 0 for dy, dx in nb8):
            continue
        g[y, x] = color; placed += 1
    return g


def gi_rects(rng, hmin=9, hmax=14, n=3) -> np.ndarray:
    """Varios rectangulos de colores y TAMAÑOS distintos (para 'objeto mayor')."""
    h, w = int(rng.integers(hmin, hmax + 1)), int(rng.integers(hmin, hmax + 1))
    g = np.zeros((h, w), dtype=int)
    pal = _palette(rng, n)
    for c in pal:
        rh, rw = int(rng.integers(1, h // 2 + 1)), int(rng.integers(1, w // 2 + 1))
        r0, c0 = int(rng.integers(0, h - rh + 1)), int(rng.integers(0, w - rw + 1))
        g[r0:r0 + rh, c0:c0 + rw] = c
    return g


def gi_signal_noise(rng, hmin=8, hmax=12) -> np.ndarray:
    """Un rectangulo solido (señal) + ruido de celdas sueltas de otros colores."""
    g = gi_rects(rng, hmin, hmax, n=1)
    h, w = g.shape
    noise_c = int(rng.integers(1, 10))
    for _ in range(int(rng.integers(2, 6))):
        y, x = int(rng.integers(h)), int(rng.integers(w))
        if g[y, x] == 0:
            g[y, x] = noise_c
    return g


def _force_color(g: np.ndarray, c: int, rng) -> np.ndarray:
    """Garantiza que el color `c` aparezca en el grid (para reglas tipo replace)."""
    g = g.copy()
    h, w = g.shape
    for _ in range(max(1, (h * w) // 6)):
        g[int(rng.integers(h)), int(rng.integers(w))] = c
    return g


def gi_hollow(rng, hmin=7, hmax=12) -> np.ndarray:
    """Un rectangulo HUECO (solo borde) de un color, sobre fondo."""
    h, w = int(rng.integers(hmin, hmax + 1)), int(rng.integers(hmin, hmax + 1))
    g = np.zeros((h, w), dtype=int)
    c = int(rng.integers(1, 10))
    rh, rw = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
    r0, c0 = int(rng.integers(0, h - rh + 1)), int(rng.integers(0, w - rw + 1))
    g[r0, c0:c0 + rw] = c; g[r0 + rh - 1, c0:c0 + rw] = c
    g[r0:r0 + rh, c0] = c; g[r0:r0 + rh, c0 + rw - 1] = c
    return g


# --- operaciones simples reutilizables (para componer) ----------------------

def _simple_ops() -> List[dict]:
    return [
        {"params": lambda rng: {"k": int(rng.integers(1, 4))}, "apply": lambda g, p: np.rot90(g, p["k"])},
        {"params": lambda rng: {"axis": int(rng.integers(0, 2))}, "apply": lambda g, p: np.flip(g, p["axis"])},
        {"params": lambda rng: {}, "apply": lambda g, p: g.T},
        {"params": lambda rng: {"cmap": _color_perm(rng)}, "apply": lambda g, p: _apply_cmap(g, p["cmap"])},
        {"params": lambda rng: {}, "apply": lambda g, p: np.concatenate([g, np.fliplr(g)], axis=1)},
        {"params": lambda rng: {}, "apply": lambda g, p: np.concatenate([g, np.flipud(g)], axis=0)},
        {"params": lambda rng: {"d": int(rng.integers(1, 3)) * (1 if rng.random() < .5 else -1)},
         "apply": lambda g, p: np.roll(g, p["d"], axis=int(p.get("ax", 0)))},
    ]


SIMPLE_OPS = _simple_ops()


def _compose_params(rng) -> dict:
    i, j = rng.choice(len(SIMPLE_OPS), size=2, replace=False)
    return {"i": int(i), "j": int(j),
            "pi": SIMPLE_OPS[int(i)]["params"](rng), "pj": SIMPLE_OPS[int(j)]["params"](rng)}


def _compose_apply(g, p) -> np.ndarray:
    g = SIMPLE_OPS[p["i"]]["apply"](g, p["pi"])
    g = SIMPLE_OPS[p["j"]]["apply"](g, p["pj"])
    return g


def _count_objects(g) -> np.ndarray:
    comps = _components(g)
    if not comps:
        raise ValueError("sin objetos")
    return np.full((1, len(comps)), comps[0]["color"], dtype=int)


def _keep_largest(g) -> np.ndarray:
    comps = _components(g)
    if len(comps) < 2:
        raise ValueError("se necesitan >=2 objetos")
    comps.sort(key=lambda c: c["size"])
    if comps[-1]["size"] == comps[-2]["size"]:
        raise ValueError("empate en tamaño")
    out = np.zeros_like(g)
    for (y, x) in comps[-1]["cells"]:
        out[y, x] = comps[-1]["color"]
    return out


def _denoise(g) -> np.ndarray:
    comps = _components(g)
    sizes = [c["size"] for c in comps]
    if 1 not in sizes or all(s == 1 for s in sizes):
        raise ValueError("nada que limpiar")
    out = g.copy()
    for c in comps:
        if c["size"] == 1:
            for (y, x) in c["cells"]:
                out[y, x] = 0
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
        # --- compositivas (clave para ARC-AGI-2) ---
        Rule("compose", _compose_params,
             lambda rng, p: gi_uniform(rng, hmax=5), _compose_apply),
        # --- object-centric ---
        Rule("replace_color",
             lambda rng: {"src": int(rng.integers(1, 10)), "dst": int(rng.integers(1, 10))},
             lambda rng, p: _force_color(gi_uniform(rng), p["src"], rng),
             lambda g, p: np.where(g == p["src"], p["dst"], g)),
        Rule("shift",
             lambda rng: {"dy": int(rng.integers(-3, 4)), "dx": int(rng.integers(-3, 4))},
             lambda rng, p: gi_sparse(rng),
             lambda g, p: np.roll(np.roll(g, p["dy"], 0), p["dx"], 1)),
        Rule("downscale",
             lambda rng: {"fh": int(rng.integers(2, 4)), "fw": int(rng.integers(2, 4))},
             lambda rng, p: np.kron(gi_uniform(rng, hmax=5),
                                    np.ones((p["fh"], p["fw"]), dtype=int)),
             lambda g, p: g[::p["fh"], ::p["fw"]]),
        Rule("mask_color", lambda rng: {"c": int(rng.integers(1, 10))},
             lambda rng, p: gi_rects(rng, n=3),
             lambda g, p: (g != 0).astype(int) * p["c"]),
        Rule("count_objects", lambda rng: {},
             lambda rng, p: gi_blobs(rng), lambda g, p: _count_objects(g)),
        Rule("keep_largest", lambda rng: {},
             lambda rng, p: gi_rects(rng), lambda g, p: _keep_largest(g)),
        Rule("denoise", lambda rng: {},
             lambda rng, p: gi_signal_noise(rng), lambda g, p: _denoise(g)),
        Rule("fill_holes", lambda rng: {},
             lambda rng, p: gi_hollow(rng), lambda g, p: _fill_enclosed(g)),
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
