"""Descarga los datos de la competencia con la Kaggle CLI.

Lee credenciales desde .env (KAGGLE_API_TOKEN, kaggle_username) y descomprime en ./data.
Uso:  python scripts/download_data.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COMP = "arc-prize-2026-arc-agi-2"


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def main() -> int:
    load_env(ROOT / ".env")
    # La Kaggle CLI acepta KAGGLE_USERNAME/KAGGLE_KEY o KAGGLE_API_TOKEN.
    if "kaggle_username" in os.environ:
        os.environ.setdefault("KAGGLE_USERNAME", os.environ["kaggle_username"])
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ:
        print("ERROR: faltan credenciales Kaggle en .env", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Descargando '{COMP}' en {DATA_DIR} ...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMP, "-p", str(DATA_DIR)],
        check=True,
    )
    for zpath in DATA_DIR.glob("*.zip"):
        print(f"Descomprimiendo {zpath.name} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA_DIR)
        zpath.unlink()
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
