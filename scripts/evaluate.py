"""Valida un modelo en el split `evaluation` (120 tareas) con inferencia AIRV.

Pensado para correr en GPU de Kaggle (importa torch/peft solo si se usa el modelo).
Sin `--model-dir` corre la baseline, util para sanity-check del pipeline de scoring.

Uso (en kernel de Kaggle con GPU):
  python scripts/evaluate.py --model-dir /kaggle/input/.../base \
      --lora-dir /kaggle/input/arc-agi2-model/lora-adapter --n-aug 8
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.data import load_split  # noqa: E402
from arc.grid import grid_to_text, text_to_grid  # noqa: E402
from arc.infer import predict_task  # noqa: E402
from arc.metrics import score_submission  # noqa: E402


def baseline_fn_factory(task):
    outs = [grid_to_text(p.output) for p in task.train if p.output]
    common = Counter(outs).most_common(1)
    return text_to_grid(common[0][0]) if common else [[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="modelo base; si falta, usa baseline")
    ap.add_argument("--lora-dir", help="adaptador LoRA opcional")
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tasks = load_split("evaluation", data_dir=args.data_dir)
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    solutions = {t: [p.output for p in task.test] for t, task in tasks.items()}

    predict_fn = None
    if args.model_dir:
        import torch

        from arc.model import generate_grid, load_model

        model, tok = load_model(
            args.model_dir, device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        if args.lora_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora_dir)
        model.eval()
        predict_fn = lambda prompt: generate_grid(model, tok, prompt, constrained=True)

    submission = {}
    for tid, task in tasks.items():
        if predict_fn is not None:
            submission[tid] = predict_task(task, predict_fn, n_aug=args.n_aug)
        else:
            mc = baseline_fn_factory(task)
            submission[tid] = [{"attempt_1": p.input, "attempt_2": mc} for p in task.test]

    score = score_submission(submission, solutions)
    print(f"Score en evaluation ({len(tasks)} tareas): {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
