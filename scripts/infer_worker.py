"""Worker de inferencia para 1 GPU: procesa un shard de tareas y escribe su JSON.

El proceso padre (submit.ipynb / eval multi-GPU) lanza N workers, cada uno con
CUDA_VISIBLE_DEVICES=<i>, y fusiona los shards con arc.runner.merge_submissions.

Uso típico (lanzado por el notebook, no a mano):
  CUDA_VISIBLE_DEVICES=2 python scripts/infer_worker.py \
      --challenges /kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json \
      --out /kaggle/working/submission_shard_2.json --n-shards 4 --shard-idx 2 \
      --model-dir <base> --lora-dir <adapter> --ttt --selection score \
      --time-budget-s 36000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.data import load_dataset  # noqa: E402
from arc.runner import InferConfig, run_shard, shard_tasks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True, help="JSON de challenges (test o eval)")
    ap.add_argument("--out", required=True, help="ruta del submission del shard")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--model-dir", help="modelo base; sin él, todo es baseline")
    ap.add_argument("--lora-dir", help="adaptador LoRA opcional")
    ap.add_argument("--selection", choices=["score", "vote"], default="score")
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--n-cands", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--score-views", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--ttt", action="store_true")
    ap.add_argument("--ttt-steps", type=int, default=24)
    ap.add_argument("--ttt-lr", type=float, default=1e-4)
    ap.add_argument("--ttt-max-seq-len", type=int, default=1024,
                    help="límite de secuencia en TTT (memoria: CE aloca logits fp32)")
    ap.add_argument("--time-budget-s", type=float, default=0,
                    help="presupuesto de pared del shard; 0 = sin límite")
    ap.add_argument("--limit", type=int, default=0,
                    help="usar solo las primeras N tareas (aplicado antes del sharding)")
    args = ap.parse_args()

    tasks = load_dataset(args.challenges)
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    shard_ids = shard_tasks(tasks, args.n_shards)[args.shard_idx]
    tag = f"[shard {args.shard_idx}/{args.n_shards}]"
    log = lambda m: print(f"{tag} {m}", flush=True)
    log(f"{len(shard_ids)} tareas")

    model = tok = None
    if args.model_dir:
        import torch

        from arc.model import load_model

        if not torch.cuda.is_available():
            raise SystemExit(f"{tag} ERROR: worker sin GPU visible.")
        # bf16 solo con soporte nativo (sm>=80); en T4 (sm75) usar fp16
        cap = torch.cuda.get_device_capability(0)[0]
        dtype = "bfloat16" if cap >= 8 else "float16"
        model, tok = load_model(args.model_dir, device_map="cuda", dtype=dtype)
        log(f"modelo cargado en {dtype} "
            f"({torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
        if args.lora_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora_dir,
                                              is_trainable=args.ttt)
        elif args.ttt:
            from arc.model import build_lora

            model = build_lora(model)
        model.eval()

    cfg = InferConfig(
        selection=args.selection, n_aug=args.n_aug, n_cands=args.n_cands,
        temperature=args.temperature, n_score_views=args.score_views,
        max_new_tokens=args.max_new_tokens, ttt=args.ttt,
        ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
        ttt_max_seq_len=args.ttt_max_seq_len,
    )
    run_shard(tasks, shard_ids, model=model, tokenizer=tok, cfg=cfg,
              time_budget_s=args.time_budget_s or None, out_path=args.out, log=log)
    log(f"listo -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
