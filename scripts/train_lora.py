"""Fine-tune SFT con LoRA (PEFT) sobre el dataset JSONL de ARC.

Enmascarado completion-only EXACTO: como cada muestra trae `prompt` y `completion`
por separado, se tokenizan aparte y se ponen labels=-100 en los tokens del prompt, de
modo que la pérdida solo cuenta sobre el grid de salida. (Más robusto que el template
matching de trl, porque "Output:" se repite por cada ejemplo del prompt.)

Uso (en GPU):
  python scripts/train_lora.py \
      --data data/finetune_train.jsonl \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --out models/lora-qwen05b \
      --epochs 1 --batch-size 4 --grad-accum 8 --lr 1e-4 --max-seq-len 2048
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class ArcSFTDataset:
    """Dataset perezoso: tokeniza prompt+completion y enmascara el prompt."""

    def __init__(self, rows, tokenizer, max_seq_len=2048):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.rows = rows  # lista de dicts con 'prompt' y 'completion'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        eos = self.tok.eos_token or ""
        p_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = self.tok(r["completion"] + eos, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + c_ids)[: self.max_seq_len]
        labels = ([-100] * len(p_ids) + c_ids)[: self.max_seq_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}


class PadCollator:
    """Pad dinámico a la longitud máxima del batch (labels con -100)."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        maxlen = max(len(b["input_ids"]) for b in batch)

        def pad(seq, val):
            return seq + [val] * (maxlen - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "finetune_train.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "models" / "lora-qwen05b"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0, help="limita muestras (debug)")
    args = ap.parse_args()

    import torch
    from transformers import Trainer, TrainingArguments

    from arc.model import build_lora, load_model

    rows = list(read_jsonl(args.data))
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"Muestras de entrenamiento: {len(rows)}")

    model, tok = load_model(
        args.model,
        load_4bit=args.load_4bit,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = build_lora(model, r=args.lora_r)
    model.print_trainable_parameters()

    ds = ArcSFTDataset(rows, tok, max_seq_len=args.max_seq_len)
    collator = PadCollator(pad_id=tok.pad_token_id)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Adaptador LoRA guardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
