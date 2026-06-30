"""Empaqueta y publica los notebooks como kernels de Kaggle (GPU) via Kaggle CLI.

Crea una carpeta temporal con el notebook + kernel-metadata.json y ejecuta
`kaggle kernels push`. Ambos kernels van SIN internet por defecto: las maquinas L4x4
(96GB) de esta competencia EXIGEN internet deshabilitado, asi que el modelo base y las
deps deben venir de Kaggle Models/Datasets adjuntos. Usa --internet para forzar red
(solo valido en T4x2/P100, que descargarian el modelo de HF).

Ejemplos:
  # Entrenamiento en GPU L4x4 (offline): adjunta codigo + modelo base de Kaggle Models
  python scripts/push_kernels.py train \
      --code-dataset juliancamilovilla/arc-agi2-code \
      --model qwen-lm/qwen2.5/transformers/0.5b-instruct

  # Inferencia/submission offline: adjunta codigo + base + LoRA entrenado
  python scripts/push_kernels.py submit \
      --code-dataset juliancamilovilla/arc-agi2-code \
      --model qwen-lm/qwen2.5/transformers/0.5b-instruct \
      --dataset juliancamilovilla/arc-agi2-model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-2"


def load_env(env_path: Path) -> None:
    # El .env es la fuente de verdad de credenciales: SOBREESCRIBE el entorno del sistema
    # (que puede tener un KAGGLE_API_TOKEN obsoleto que rompería la autenticación).
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def kaggle_username() -> str:
    return (
        os.environ.get("KAGGLE_USERNAME")
        or os.environ.get("kaggle_username")
        or "juliancamilovilla"
    )


KERNELS = {
    "train": {"notebook": "notebooks/train.ipynb", "slug": "arc-agi2-train",
              "title": "arc agi2 train", "gpu": True},
    "submit": {"notebook": "notebooks/submit.ipynb", "slug": "arc-agi2-submit",
               "title": "arc agi2 submit", "gpu": True},
    "eval": {"notebook": "notebooks/evaluate.ipynb", "slug": "arc-agi2-eval",
             "title": "arc agi2 eval", "gpu": True},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=list(KERNELS))
    ap.add_argument("--code-dataset", help="slug user/dataset con src/ y scripts/")
    ap.add_argument("--dataset", action="append", default=[],
                    help="dataset adicional (p.ej. modelo entrenado); repetible")
    ap.add_argument("--model", action="append", default=[],
                    help="Kaggle Models source user/model/...; repetible")
    ap.add_argument("--internet", action="store_true",
                    help="habilita internet (SOLO T4x2/P100; L4x4 lo prohibe)")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
                    help="GPU/TPU. T4 (default) la soporta torch; P100 (sm_60) NO. "
                         "L4x4 no está en el enum de esta CLI: selecciónalo en la UI de Kaggle.")
    ap.add_argument("--dry-run", action="store_true", help="solo genera metadata, no publica")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]

    cfg = KERNELS[args.kernel]
    user = kaggle_username()
    datasets = list(args.dataset)
    if args.code_dataset:
        datasets.insert(0, args.code_dataset)

    meta = {
        "id": f"{user}/{cfg['slug']}",
        "title": cfg["title"],
        "code_file": Path(cfg["notebook"]).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": cfg["gpu"],
        "machine_shape": args.accelerator,
        "enable_internet": args.internet,
        "dataset_sources": datasets,
        "competition_sources": [COMP],
        "model_sources": args.model,
        "kernel_sources": [],
    }

    tmp = Path(tempfile.mkdtemp(prefix="arc_kernel_"))
    shutil.copy(ROOT / cfg["notebook"], tmp / meta["code_file"])
    (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("metadata:\n" + json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"\n[dry-run] carpeta lista en {tmp}")
        return 0

    print(f"\nPublicando kernel '{meta['id']}' ...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp)])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
