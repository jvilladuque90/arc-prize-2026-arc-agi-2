"""Materializa el dataset de fine-tune (SFT) augmentado a JSONL.

Por cada tarea genera muestras (prompt -> completion) donde el modelo aprende a predecir
el grid de salida dado un conjunto de ejemplos + un input de consulta. Se usa:
  - leave-one-out sobre los pares `train` (cada par se convierte, por turnos, en la consulta);
  - augmentacion D4 (8 simetrias) x permutaciones de color (consistentes por tarea).

Salida JSONL: {"prompt": ..., "completion": ..., "task_id": ..., "aug": ...}
El campo "completion" termina en '\n' (el grid); el SFT debe añadir EOS.

Uso:
  python scripts/build_finetune_dataset.py --split training --n-aug 8 \
      --out data/finetune_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.augment import augment_task, color_permutation  # noqa: E402
from arc.data import Task, TaskPair, load_split  # noqa: E402
from arc.grid import format_task_prompt, grid_to_text  # noqa: E402


def loo_views(task: Task):
    """Genera vistas leave-one-out: (contexto_train, par_consulta).

    Incluye los pares de `train` (cada uno como consulta, el resto como contexto)
    y, si tienen solucion, los pares de `test`.
    """
    n = len(task.train)
    for i in range(n):
        ctx = task.train[:i] + task.train[i + 1 :]
        if ctx:  # necesita al menos un ejemplo de contexto
            yield ctx, task.train[i]
    for tp in task.test:
        if tp.output is not None:
            yield task.train, tp


def samples_for_task(task: Task, n_aug: int, max_cells: int, sep: str):
    """Emite dicts de muestra para una tarea, con `n_aug` augmentaciones (la 1ra es identidad)."""
    for a in range(n_aug):
        if a == 0:
            d4_idx, cmap = 0, list(range(10))  # identidad
        else:
            # variar D4 y color por indice; semilla determinista por (task, a)
            d4_idx = a % 8
            cmap = color_permutation(seed=hash((task.task_id, a)) & 0xFFFFFFFF, keep_zero=True)
        atask = augment_task(task, d4_idx, cmap)
        for ctx, query in loo_views(atask):
            if query.output is None:
                continue
            in_cells = len(query.input) * len(query.input[0])
            out_cells = len(query.output) * len(query.output[0])
            if max_cells and max(in_cells, out_cells) > max_cells:
                continue
            prompt = format_task_prompt(ctx, query.input, sep)
            completion = "\n" + grid_to_text(query.output, sep) + "\n"
            yield {
                "task_id": task.task_id,
                "aug": a,
                "prompt": prompt,
                "completion": completion,
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training", choices=["training", "evaluation"])
    ap.add_argument("--n-aug", type=int, default=8, help="augmentaciones por tarea (1=solo identidad)")
    ap.add_argument("--max-cells", type=int, default=900, help="descarta grids > N celdas (0=sin limite)")
    ap.add_argument("--sep", default="", help="separador de celdas en la serializacion")
    ap.add_argument("--out", default=str(ROOT / "data" / "finetune_train.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="limita nº de tareas (debug)")
    ap.add_argument("--exclude-holdout", action="store_true",
                    help="excluye del SFT el split de held-out (arc.holdout) para que la "
                         "Vía B tenga validación local limpia (ver docs/STRATEGY.md)")
    args = ap.parse_args()

    tasks = load_split(args.split)
    if args.exclude_holdout and args.split == "training":
        from arc.holdout import is_holdout

        tasks = {t: task for t, task in tasks.items() if not is_holdout(t)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks.values()):
            if args.limit and i >= args.limit:
                break
            for s in samples_for_task(task, args.n_aug, args.max_cells, args.sep):
                f.write(json.dumps(s) + "\n")
                n_samples += 1

    print(f"Escritas {n_samples} muestras de {len(tasks)} tareas -> {out_path}"
          + (" (holdout excluido)" if args.exclude_holdout else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
