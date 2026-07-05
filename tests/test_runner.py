"""Tests de la lógica pura de arc.runner (sin torch/GPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.data import Task, TaskPair
from arc.runner import (InferConfig, TimeBudget, baseline_attempts, cfg_for_tier,
                        estimate_task_cost, merge_submissions, pick_tier, run_shard,
                        shard_tasks)


def make_task(tid: str, size: int = 2, n_train: int = 2, n_test: int = 1) -> Task:
    g = [[1] * size for _ in range(size)]
    return Task(task_id=tid,
                train=[TaskPair(g, g) for _ in range(n_train)],
                test=[TaskPair(g, None) for _ in range(n_test)])


def test_estimate_task_cost_scales_with_size():
    assert estimate_task_cost(make_task("b", size=10)) > estimate_task_cost(make_task("a", size=2))


def test_shard_tasks_partitions_all_tasks():
    tasks = {f"t{i}": make_task(f"t{i}", size=i + 1) for i in range(10)}
    shards = shard_tasks(tasks, 4)
    flat = [t for s in shards for t in s]
    assert sorted(flat) == sorted(tasks)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_shard_tasks_balances_cost():
    tasks = {f"t{i}": make_task(f"t{i}", size=(i % 5) * 6 + 1) for i in range(20)}
    shards = shard_tasks(tasks, 4)
    costs = [sum(estimate_task_cost(tasks[t]) for t in s) for s in shards]
    assert max(costs) <= 2 * min(costs)  # serpentina: shards comparables


def test_time_budget_allowance_shrinks():
    now = [0.0]
    b = TimeBudget(100.0, 4, clock=lambda: now[0])
    assert b.per_task_s() == 25.0
    now[0] = 40.0
    b.task_done()
    assert b.per_task_s() == 20.0  # 60s restantes / 3 tareas
    assert b.remaining_s() == 60.0


def test_time_budget_unbounded():
    b = TimeBudget(None, 3)
    assert b.remaining_s() == float("inf")
    assert pick_tier(1e9, b.per_task_s()) == 0


def test_pick_tier_degrades():
    assert pick_tier(None, 10.0) == 0          # sin historial: optimista
    assert pick_tier(50.0, 100.0) == 0         # cabe full
    assert pick_tier(50.0, 30.0) == 1          # cabe mid (50*0.4=20)
    assert pick_tier(50.0, 7.0) == 2           # solo low (50*0.12=6)
    assert pick_tier(50.0, 1.0) == 2           # nunca por debajo del último tier


def test_cfg_for_tier():
    cfg = InferConfig(n_aug=8, n_cands=2, ttt_steps=24)
    assert cfg_for_tier(cfg, 0) is cfg
    mid = cfg_for_tier(cfg, 1)
    assert mid.n_aug == 4 and mid.n_cands == 1 and mid.ttt_steps == 12
    low = cfg_for_tier(cfg, 2)
    assert low.n_aug == 1 and low.ttt is False and low.selection == "vote"


def test_baseline_attempts_identity_and_most_common():
    g1, g2 = [[1]], [[2]]
    task = Task("t", train=[TaskPair(g1, g2), TaskPair(g1, g2), TaskPair(g2, g1)],
                test=[TaskPair([[5]], None)])
    (att,) = baseline_attempts(task)
    assert att["attempt_1"] == [[5]]
    assert att["attempt_2"] == g2  # output más frecuente en train


def test_run_shard_without_model_writes_baseline(tmp_path):
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(3)}
    out = tmp_path / "shard.json"
    sub = run_shard(tasks, list(tasks), model=None, out_path=out)
    assert sorted(sub) == sorted(tasks)
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == sub
    for tid in tasks:
        assert all("attempt_1" in a and "attempt_2" in a for a in sub[tid])


def test_merge_submissions_fills_missing(tmp_path):
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(4)}
    p1 = tmp_path / "s0.json"
    p1.write_text(json.dumps({"t0": [{"attempt_1": [[9]], "attempt_2": [[9]]}]}),
                  encoding="utf-8")
    merged = merge_submissions([p1, tmp_path / "missing.json"], tasks)
    assert sorted(merged) == sorted(tasks)
    assert merged["t0"][0]["attempt_1"] == [[9]]           # shard respetado
    assert merged["t1"] == baseline_attempts(tasks["t1"])  # faltante -> baseline
