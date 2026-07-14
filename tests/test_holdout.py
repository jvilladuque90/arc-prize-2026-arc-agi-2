"""Tests del split de held-out (determinista, estable, disjunto)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.holdout import HOLDOUT_FRACTION, is_holdout, split_ids


def test_split_is_disjoint_and_complete():
    ids = [f"task{i:04d}" for i in range(1000)]
    train, hold = split_ids(ids)
    assert set(train) | set(hold) == set(ids)
    assert set(train) & set(hold) == set()
    assert len(train) + len(hold) == len(ids)


def test_holdout_fraction_roughly_matches():
    ids = [f"task{i:04d}" for i in range(2000)]
    _, hold = split_ids(ids)
    frac = len(hold) / len(ids)
    # ~1/HOLDOUT_FRACTION, con holgura por variación del hash
    assert 0.5 / HOLDOUT_FRACTION < frac < 1.8 / HOLDOUT_FRACTION


def test_is_holdout_deterministic_within_process():
    assert is_holdout("abc123") == is_holdout("abc123")


def test_is_holdout_stable_across_processes():
    """No debe depender de PYTHONHASHSEED: un subproceso da lo mismo."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from arc.holdout import is_holdout;"
        "print(int(is_holdout('sample_task_id_42')))" % str(ROOT / "src")
    )
    import os

    env = dict(os.environ, PYTHONHASHSEED="12345")
    r1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    env["PYTHONHASHSEED"] = "99999"
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r1.stdout.strip() == r2.stdout.strip()
    # y coincide con el resultado in-process
    assert r1.stdout.strip() == str(int(is_holdout("sample_task_id_42")))
