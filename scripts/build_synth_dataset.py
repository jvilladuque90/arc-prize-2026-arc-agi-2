"""Materializa un dataset de tareas ARC SINTÉTICAS a JSONL (prompt/completion).

Reutiliza la serialización/augmentación de `build_finetune_dataset.samples_for_task`,
así que el formato es idéntico al del dataset real y ambos se pueden concatenar.

Uso:
  python scripts/build_synth_dataset.py --n-tasks 50000 --n-aug 1 \
      --out data/synth_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from arc.synth import generate_dataset  # noqa: E402
from build_finetune_dataset import samples_for_task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=50000)
    ap.add_argument("--n-aug", type=int, default=1, help="augmentaciones por tarea (1=identidad)")
    ap.add_argument("--max-cells", type=int, default=900)
    ap.add_argument("--sep", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth_train.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for task in generate_dataset(args.n_tasks, seed=args.seed):
            for s in samples_for_task(task, args.n_aug, args.max_cells, args.sep):
                f.write(json.dumps(s) + "\n")
                n_samples += 1

    print(f"Escritas {n_samples} muestras de {args.n_tasks} tareas sintéticas -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
