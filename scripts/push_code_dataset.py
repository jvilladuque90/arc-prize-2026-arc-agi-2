"""Publica el código del repo (src/ + scripts/) como Dataset de Kaggle via CLI.

Los kernels de Kaggle (train/submit) importan el paquete `arc` desde un dataset adjunto que
contenga `src/arc`. Este script empaqueta `src/` y `scripts/` con su `dataset-metadata.json`
y ejecuta `kaggle datasets create` (primera vez) o `kaggle datasets version` (actualizar).

Uso:
  python scripts/push_code_dataset.py --new                 # crear por primera vez
  python scripts/push_code_dataset.py -m "actualiza ttt"    # nueva version
  python scripts/push_code_dataset.py --dry-run             # solo prepara, no sube
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
SLUG = "arc-agi2-code"
TITLE = "ARC-AGI2 Code"
INCLUDE = ["src", "scripts"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


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
    return (os.environ.get("KAGGLE_USERNAME") or os.environ.get("kaggle_username")
            or "juliancamilovilla")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="store_true", help="crea el dataset (primera vez)")
    ap.add_argument("-m", "--message", default="update", help="mensaje de la nueva version")
    ap.add_argument("--dry-run", action="store_true", help="prepara la carpeta, no sube")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    user = kaggle_username()

    stage = Path(tempfile.mkdtemp(prefix="arc_code_ds_"))
    for name in INCLUDE:
        src = ROOT / name
        if src.exists():
            shutil.copytree(src, stage / name, ignore=IGNORE)
    meta = {
        "title": TITLE,
        "id": f"{user}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    files = [str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()]
    print(f"Carpeta: {stage}")
    print("metadata:", json.dumps(meta))
    print(f"{len(files)} archivos, p.ej.:", files[:6])

    if args.dry_run:
        print("\n[dry-run] no se sube nada.")
        return 0

    if args.new:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(stage),
               "-m", args.message, "--dir-mode", "zip"]
    print("\n$", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
