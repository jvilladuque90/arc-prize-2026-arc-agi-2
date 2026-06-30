"""Publica un adaptador LoRA entrenado como Dataset de Kaggle (para el kernel de inferencia).

Toma una carpeta con `adapter_config.json` + `adapter_model.safetensors` (p.ej. un
checkpoint del Trainer) y crea/actualiza el dataset. Solo sube los ficheros del adaptador
(no el optimizer/scheduler), que es lo único que necesita PeftModel.from_pretrained.

Uso:
  python scripts/push_model_dataset.py --src <dir_checkpoint> --new
  python scripts/push_model_dataset.py --src <dir_checkpoint> -m "v7 step1200"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "arc-agi2-model"
TITLE = "ARC-AGI2 Model"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json"]


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()  # .env tiene prioridad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="carpeta del adaptador (checkpoint)")
    ap.add_argument("--new", action="store_true", help="crea el dataset (primera vez)")
    ap.add_argument("-m", "--message", default="update")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")

    src = Path(args.src)
    assert (src / "adapter_model.safetensors").exists(), f"no hay adapter en {src}"

    stage = Path(tempfile.mkdtemp(prefix="arc_model_ds_")) / "lora-adapter"
    stage.mkdir(parents=True)
    for f in ADAPTER_FILES:
        if (src / f).exists():
            shutil.copy(src / f, stage / f)
    stage = stage.parent
    meta = {"title": TITLE, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    files = [p.name for p in (stage / "lora-adapter").iterdir()]
    print("metadata:", json.dumps(meta)); print("adapter files:", files)
    if args.dry_run:
        print(f"[dry-run] carpeta {stage}"); return 0

    if args.new:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(stage), "-m", args.message,
               "--dir-mode", "zip"]
    print("$", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
