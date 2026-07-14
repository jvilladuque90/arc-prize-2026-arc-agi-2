"""Held-out de validación limpio (no contaminado) — política de la Vía A/B.

Contexto (ver docs/LB33_AUDIT.md): el checkpoint NVARC de la Vía A **incluyó el set
`evaluation` público de ARC-AGI-2 en su SFT** (y muy probablemente también el `training`,
por ser datos de entrenamiento estándar). Consecuencia:

  - El set `evaluation` público NO sirve para medir NINGÚN modelo que herede ese SFT
    (mide memorización).
  - Un split reservado del `training` es un held-out **limpio para la Vía B** (nuestro
    modelo propio) siempre que la Vía B EXCLUYA esos IDs de su SFT. Para la Vía A
    (checkpoint NVARC) probablemente también está contaminado → para la Vía A el único
    held-out limpio son tareas **sintéticas propias jamás publicadas** (`arc.synth`), como
    sanity-check, además del leaderboard (1 submission/día).

Este módulo define el split de forma **determinista y estable entre procesos/máquinas**
(hash md5 del task_id, no el `hash()` de Python que está salteado por PYTHONHASHSEED),
de modo que no hace falta versionar ninguna lista en `data/` (que está gitignored).
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple

# 1 de cada N tareas de `training` se reserva como held-out (~10%).
HOLDOUT_FRACTION = 10


def _stable_bucket(task_id: str) -> int:
    """Bucket determinista y estable (independiente de PYTHONHASHSEED)."""
    digest = hashlib.md5(task_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % HOLDOUT_FRACTION


def is_holdout(task_id: str) -> bool:
    """True si el task_id cae en el split de validación reservado."""
    return _stable_bucket(task_id) == 0


def split_ids(task_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Devuelve (ids_de_entrenamiento, ids_de_holdout) preservando el orden de entrada."""
    train, hold = [], []
    for t in task_ids:
        (hold if is_holdout(t) else train).append(t)
    return train, hold
