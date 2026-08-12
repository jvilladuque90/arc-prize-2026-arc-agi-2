#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Fusiona el 2B campeon + adapter ARC-Heavy y lo sube como dataset privado.

Prepara la PRUEBA DE ENSAYO en el test oculto (docs/SCALE_PLAN.md): el kernel lb33-2b-scale
(config campeona identica a la submission 18.33 del 2B-solo) carga este modelo fusionado ->
A/B directo del efecto ARC-Heavy en la distribucion real, gastando solo el slot diario.

Descarga el adapter en su ULTIMA version del dataset arcscale-adapter-2b (tras D2 = el mejor
punto de la curva; verificar antes la lectura de D2). Merge en bf16 + tokenizer -> dataset
`arcscale-merged-2b` (carpeta merged_2b/).

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_merge_2b.py <KAGGLE_API_TOKEN> <KAGGLE_USERNAME>
"""
import json
import os
import subprocess
import sys
import time

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
KUSER = sys.argv[2] if len(sys.argv) > 2 else ""
assert KUSER, "pasa el KAGGLE_USERNAME como argv[2]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "kaggle",
                "torchao>=0.16"], check=True)

os.makedirs("/content", exist_ok=True)
os.chdir("/content")
subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1",
                "-p", "/content/m2b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m2b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR

subprocess.run(["kaggle", "datasets", "download", "-d", f"{KUSER}/arcscale-adapter-2b",
                "-p", "/content/adapter", "--unzip", "-q"], check=True)
ADIR = "/content/adapter"
assert os.path.exists(os.path.join(ADIR, "adapter_model.safetensors"))
print(f"[setup] modelo + adapter listos ({time.time()-T00:.0f}s)", flush=True)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# merge en CPU si no hay GPU (limite diario de T4 gratis agotado): bf16 en RAM (~5GB), lento
# pero suficiente — merge_and_unload es una pasada de sumas de matrices
DEV = "cuda:0" if torch.cuda.is_available() else None
BASE = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.bfloat16, device_map=DEV)
MODEL = PeftModel.from_pretrained(BASE, ADIR)
MERGED = MODEL.merge_and_unload()
OUT = "/content/out/merged_2b"
os.makedirs(OUT, exist_ok=True)
MERGED.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(MDIR).save_pretrained(OUT)
# metadatos de procedencia (que adapter exactamente se fusiono)
if os.path.exists(os.path.join(ADIR, "data_scaling_ext_result.json")):
    import shutil
    shutil.copy(os.path.join(ADIR, "data_scaling_ext_result.json"), OUT)
print(f"[merge] guardado en {OUT} ({time.time()-T00:.0f}s)", flush=True)

json.dump({"title": "arcscale-merged-2b", "id": f"{KUSER}/arcscale-merged-2b",
           "licenses": [{"name": "CC0-1.0"}]},
          open("/content/out/dataset-metadata.json", "w"))
r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/out", "--dir-mode", "zip"],
                   capture_output=True, text=True)
out = (r.stdout or "") + (r.stderr or "")
if r.returncode != 0 or "already exists" in out:
    r = subprocess.run(["kaggle", "datasets", "version", "-p", "/content/out", "-m", "merge",
                        "--dir-mode", "zip"], capture_output=True, text=True)
print(f"[upload] rc={r.returncode} {(r.stdout or '')[-300:]}", flush=True)
print("RESULT_JSON:" + json.dumps({"ok": r.returncode == 0}), flush=True)
