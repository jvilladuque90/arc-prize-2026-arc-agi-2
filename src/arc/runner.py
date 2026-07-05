"""Ejecución sharded con presupuesto de tiempo: 1 proceso por GPU, sin DDP.

TTT e inferencia son independientes por tarea → el paralelismo correcto en L4×4 es
lanzar 4 workers (uno por GPU, vía CUDA_VISIBLE_DEVICES) con un shard de tareas cada
uno, y fusionar los `submission_shard_*.json` al final. Sin comunicación entre GPUs.

Este módulo concentra la lógica compartida entre `scripts/infer_worker.py` y los
notebooks de submit/eval:
  - estimación de costo y sharding balanceado (serpentina sobre costo desc),
  - presupuesto de tiempo por tarea con degradación por tiers (full → mid → low → baseline),
  - loop por tarea con fallback granular y escritura incremental del shard,
  - merge de shards rellenando faltantes con la baseline.

Solo `run_shard` toca torch (lazy); el resto es Python puro y testeable en CPU.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .data import Task
from .grid import grid_to_text, text_to_grid

Grid = List[List[int]]


# --- Baseline (compartida por evaluate / worker / notebooks) ------------------

def baseline_attempts(task: Task) -> List[dict]:
    """attempt_1 = identidad; attempt_2 = output de train más frecuente."""
    outs = [grid_to_text(p.output) for p in task.train if p.output]
    common = Counter(outs).most_common(1)
    mc = text_to_grid(common[0][0]) if common else [[0]]
    return [{"attempt_1": tp.input, "attempt_2": mc} for tp in task.test]


# --- Costo y sharding ---------------------------------------------------------

def _cells(grid: Optional[Grid]) -> int:
    return len(grid) * len(grid[0]) if grid else 0


def estimate_task_cost(task: Task) -> int:
    """Proxy de costo en tokens: el prompt (pares train + test input) se procesa por
    cada test input y por cada vista augmentada; el output se genera token a token."""
    train_cells = sum(_cells(p.input) + _cells(p.output) for p in task.train)
    return sum(train_cells + 2 * _cells(tp.input) for tp in task.test)


def shard_tasks(tasks: Dict[str, Task], n_shards: int) -> List[List[str]]:
    """Reparte task_ids en `n_shards` listas balanceadas por costo (serpentina)."""
    order = sorted(tasks, key=lambda t: estimate_task_cost(tasks[t]), reverse=True)
    shards: List[List[str]] = [[] for _ in range(n_shards)]
    for i, tid in enumerate(order):
        r, pos = divmod(i, n_shards)
        shards[pos if r % 2 == 0 else n_shards - 1 - pos].append(tid)
    return shards


# --- Configuración de inferencia con tiers de degradación ---------------------

@dataclass
class InferConfig:
    selection: str = "score"       # "score" | "vote"
    n_aug: int = 8
    n_cands: int = 2
    temperature: float = 0.7
    n_score_views: int = 4
    max_new_tokens: int = 2048
    sep: str = ""
    ttt: bool = True
    ttt_steps: int = 24
    ttt_lr: float = 1e-4
    ttt_n_aug: int = 4
    ttt_max_seq_len: int = 2048


# tier 0 = config completa; 1 = reducida; 2 = mínima (greedy, 1 vista, sin TTT)
TIER_SPEED = [1.0, 0.4, 0.12]  # tiempo relativo estimado de cada tier vs full


def cfg_for_tier(cfg: InferConfig, tier: int) -> InferConfig:
    if tier <= 0:
        return cfg
    if tier == 1:
        return replace(cfg, n_aug=max(2, cfg.n_aug // 2), n_cands=1,
                       n_score_views=2, ttt_steps=max(8, cfg.ttt_steps // 2))
    return replace(cfg, n_aug=1, n_cands=1, n_score_views=1,
                   selection="vote", ttt=False)


class TimeBudget:
    """Presupuesto global de pared; asigna un allowance uniforme por tarea restante."""

    def __init__(self, total_s: Optional[float], n_tasks: int,
                 clock: Callable[[], float] = time.monotonic):
        self.total_s = total_s
        self.n_tasks = n_tasks
        self.done = 0
        self._clock = clock
        self._t0 = clock()

    def remaining_s(self) -> float:
        if self.total_s is None:
            return float("inf")
        return self.total_s - (self._clock() - self._t0)

    def per_task_s(self) -> float:
        left = max(1, self.n_tasks - self.done)
        return self.remaining_s() / left

    def task_done(self) -> None:
        self.done += 1


def pick_tier(ema_full_s: Optional[float], allowance_s: float) -> int:
    """Elige el tier más rico cuyo tiempo estimado quepa en el allowance.
    Sin historial (ema None) arranca optimista en full."""
    if ema_full_s is None:
        return 0
    for tier, speed in enumerate(TIER_SPEED):
        if ema_full_s * speed <= allowance_s:
            return tier
    return len(TIER_SPEED) - 1


# --- Loop principal por shard ---------------------------------------------------

def run_shard(
    tasks: Dict[str, Task],
    task_ids: List[str],
    *,
    model=None,
    tokenizer=None,
    cfg: InferConfig = InferConfig(),
    time_budget_s: Optional[float] = None,
    out_path: Optional[str | Path] = None,
    min_task_s: float = 30.0,
    log: Callable[[str], None] = print,
) -> Dict[str, List[dict]]:
    """Predice `task_ids` (orden barato→caro) con fallback granular a baseline.

    - Prefill: el shard arranca con baseline para TODAS sus tareas y se escribe a
      disco, de modo que un crash a mitad de camino deja un shard completo y válido.
    - Tras cada tarea se reescribe `out_path` (escritura incremental).
    - `time_budget_s`: presupuesto de pared del shard; degrada n_aug/candidatos/TTT
      por tiers cuando el allowance por tarea se queda corto.
    """
    order = sorted(task_ids, key=lambda t: estimate_task_cost(tasks[t]))
    submission: Dict[str, List[dict]] = {t: baseline_attempts(tasks[t]) for t in order}

    def _flush() -> None:
        if out_path is not None:
            Path(out_path).write_text(json.dumps(submission), encoding="utf-8")

    _flush()
    if model is None:
        return submission

    ttt = None
    if cfg.ttt:
        if any(p.requires_grad for p in model.parameters()):
            from .ttt import TTT

            ttt = TTT(model, tokenizer, steps=cfg.ttt_steps, lr=cfg.ttt_lr,
                      n_aug=cfg.ttt_n_aug, max_seq_len=cfg.ttt_max_seq_len)
        else:
            log("TTT pedido pero el modelo no tiene parámetros entrenables (¿falta LoRA "
                "con is_trainable=True?); continúo sin TTT.")

    budget = TimeBudget(time_budget_s, len(order))
    ema_full: Optional[float] = None
    for n, tid in enumerate(order):
        task = tasks[tid]
        allowance = budget.per_task_s()
        if budget.remaining_s() < min_task_s:
            log(f"[{tid}] sin tiempo restante -> baseline ({n + 1}/{len(order)})")
            budget.task_done()
            continue
        tier = pick_tier(ema_full, allowance)
        c = cfg_for_tier(cfg, tier)
        t0 = time.monotonic()
        try:
            if ttt is not None and c.ttt:
                from .ttt import predict_with_ttt

                ttt.steps = c.ttt_steps
                submission[tid] = predict_with_ttt(
                    task, model, tokenizer, ttt, n_aug_infer=c.n_aug, sep=c.sep,
                    selection=c.selection, max_new_tokens=c.max_new_tokens,
                    **({"n_cands": c.n_cands, "temperature": c.temperature,
                        "n_score_views": c.n_score_views}
                       if c.selection == "score" else {}),
                )
            elif c.selection == "score":
                from .infer import predict_task_scored

                submission[tid] = predict_task_scored(
                    task, model, tokenizer, n_aug=c.n_aug, n_cands=c.n_cands,
                    temperature=c.temperature, n_score_views=c.n_score_views,
                    sep=c.sep, max_new_tokens=c.max_new_tokens,
                )
            else:
                from .infer import predict_task
                from .model import generate_grid

                predict_fn = lambda p: generate_grid(
                    model, tokenizer, p, sep=c.sep, max_new_tokens=c.max_new_tokens)
                submission[tid] = predict_task(task, predict_fn, n_aug=c.n_aug, sep=c.sep)
        except Exception as e:  # fallback granular: la baseline del prefill queda
            log(f"[{tid}] error -> baseline: {e}")
        dt = time.monotonic() - t0
        # normaliza la medición a "tiempo equivalente full" para calibrar tiers
        full_eq = dt / TIER_SPEED[tier]
        ema_full = full_eq if ema_full is None else 0.7 * ema_full + 0.3 * full_eq
        budget.task_done()
        _flush()
        if (n + 1) % 5 == 0 or n + 1 == len(order):
            log(f"{n + 1}/{len(order)} tareas | tier={tier} | {dt:.0f}s/tarea | "
                f"restante={budget.remaining_s() / 60 if time_budget_s else -1:.0f} min")
    return submission


# --- Merge de shards ------------------------------------------------------------

def merge_submissions(shard_paths: List[str | Path],
                      tasks: Dict[str, Task]) -> Dict[str, List[dict]]:
    """Une shards y rellena cualquier tarea faltante con la baseline."""
    merged: Dict[str, List[dict]] = {}
    for p in shard_paths:
        p = Path(p)
        if p.exists():
            merged.update(json.loads(p.read_text(encoding="utf-8")))
    for tid, task in tasks.items():
        if tid not in merged or not merged[tid]:
            merged[tid] = baseline_attempts(task)
    return merged
