"""Genera submission.json en el formato de la competencia.

Baseline Fase 0 (sin ML): para cada test input, propone dos candidatos simples
(identidad y el output más frecuente del train). Sirve para validar el pipeline
de entrega de punta a punta antes de meter el LLM.

Uso:  python scripts/make_submission.py --split test --out submission.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.data import load_split  # noqa: E402
from arc.grid import grid_to_text  # noqa: E402


def most_common_output(task) -> list:
    outs = [grid_to_text(p.output) for p in task.train if p.output]
    if not outs:
        return [[0]]
    txt = Counter(outs).most_common(1)[0][0]
    from arc.grid import text_to_grid

    return text_to_grid(txt)


def predict(task, test_input) -> dict:
    # attempt_1: identidad (muchas tareas conservan tamaño/relación directa)
    # attempt_2: output más frecuente visto en train
    return {"attempt_1": test_input, "attempt_2": most_common_output(task)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["training", "evaluation", "test"])
    ap.add_argument("--out", default="submission.json")
    args = ap.parse_args()

    tasks = load_split(args.split)
    submission = {}
    for tid, task in tasks.items():
        submission[tid] = [predict(task, p.input) for p in task.test]

    Path(args.out).write_text(json.dumps(submission), encoding="utf-8")
    print(f"Escrito {args.out} con {len(submission)} tareas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
