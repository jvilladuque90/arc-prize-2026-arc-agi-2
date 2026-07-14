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
    ap.add_argument("--selection", choices=["score", "vote"], default="score",
                    help="score = multi-candidato + ranking por log-verosimilitud")
    ap.add_argument("--n-cands", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--score-views", type=int, default=4)
    ap.add_argument("--ttt", action="store_true", help="activa Test-Time Training por tarea")
    ap.add_argument("--ttt-steps", type=int, default=24)
    ap.add_argument("--ttt-lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--eval-set", default="evaluation",
                    choices=["evaluation", "holdout", "synth"],
                    help="'evaluation'=set público (CONTAMINADO para el modelo NVARC); "
                         "'holdout'=split reservado del training (limpio para la Vía B si "
                         "se excluyó del SFT); 'synth'=tareas sintéticas propias (limpio "
                         "para ambas vías, bajo señal). Ver docs/STRATEGY.md.")
    ap.add_argument("--synth-n", type=int, default=100, help="nº de tareas si --eval-set synth")
    ap.add_argument("--synth-seed", type=int, default=12345, help="semilla del held-out sintético")
    args = ap.parse_args()

    if args.eval_set == "synth":
        from arc.synth import generate_dataset

        tasks = {t.task_id: t for t in generate_dataset(args.synth_n, seed=args.synth_seed)}
    else:
        tasks = load_split("training" if args.eval_set == "holdout" else "evaluation",
                           data_dir=args.data_dir)
        if args.eval_set == "holdout":
            from arc.holdout import is_holdout

            tasks = {t: task for t, task in tasks.items() if is_holdout(t)}
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    solutions = {t: [p.output for p in task.test] for t, task in tasks.items()}

    predict_fn = None
    model = tok = ttt = None
    if args.model_dir:
        import torch

        from arc.model import generate_grid, load_model

        if not torch.cuda.is_available():
            # Regla del proyecto: el modelo SOLO corre en GPU; en CPU es inviable.
            raise SystemExit("ERROR: se pidio --model-dir pero no hay GPU. Ejecuta en un "
                             "kernel de Kaggle con GPU (ver scripts/push_kernels.py).")
        model, tok = load_model(args.model_dir, device_map="cuda")
        if args.lora_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora_dir, is_trainable=args.ttt)
        elif args.ttt:
            from arc.model import build_lora

            model = build_lora(model)  # LoRA fresco para adaptar en test-time
        model.eval()
        predict_fn = lambda prompt: generate_grid(model, tok, prompt, constrained=True)
        if args.ttt:
            from arc.ttt import TTT

            ttt = TTT(model, tok, steps=args.ttt_steps, lr=args.ttt_lr, n_aug=4)

    score_kwargs = dict(n_cands=args.n_cands, temperature=args.temperature,
                        n_score_views=args.score_views)
    submission = {}
    for tid, task in tasks.items():
        if predict_fn is not None and ttt is not None:
            from arc.ttt import predict_with_ttt

            extra = score_kwargs if args.selection == "score" else {"constrained": True}
            submission[tid] = predict_with_ttt(task, model, tok, ttt,
                                               n_aug_infer=args.n_aug,
                                               selection=args.selection, **extra)
        elif predict_fn is not None and args.selection == "score":
            from arc.infer import predict_task_scored

            submission[tid] = predict_task_scored(task, model, tok,
                                                  n_aug=args.n_aug, **score_kwargs)
        elif predict_fn is not None:
            submission[tid] = predict_task(task, predict_fn, n_aug=args.n_aug)
        else:
            mc = baseline_fn_factory(task)
            submission[tid] = [{"attempt_1": p.input, "attempt_2": mc} for p in task.test]

    score = score_submission(submission, solutions)
    print(f"Score en {args.eval_set} ({len(tasks)} tareas): {score:.4f}")
    if args.eval_set == "evaluation" and args.model_dir:
        print("AVISO: el set 'evaluation' está CONTAMINADO para el checkpoint NVARC "
              "(estuvo en su SFT). Para la Vía A usa --eval-set synth o el leaderboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
