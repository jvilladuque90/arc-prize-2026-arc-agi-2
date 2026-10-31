"""Genera notebooks/q35base_dl.ipynb: kernel CPU con INTERNET (0 cuota GPU) que baja a
/kaggle/working los checkpoints BASE de Qwen3.5 (4B y 2B) y las wheels puras de
flash-linear-attention, para usarlos offline como kernel_source en gate35b.
Lanzar: PYTHONUTF8=1 python scripts/push_kernels.py q35base-dl --internet
"""
import json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

MD = """# q35base-dl: Qwen3.5-4B-Base + Qwen3.5-2B-Base + wheels FLA (CPU, internet, 0 cuota GPU)

Auditoría de `gate35` (31-oct): se midió el Qwen3.5-4B **post-entrenado** con el fallback de torch
para las 24 capas Gated DeltaNet. Este kernel trae lo que hace falta para medir limpio:
los checkpoints **Base** oficiales y las wheels puras (py3-none-any) de `flash-linear-attention`.
La salida se consume como `kernel_sources` desde `gate35b` (offline)."""

CODE = r'''
import os, sys, json, time, hashlib, subprocess
T0 = time.time()
OUT = "/kaggle/working"
WH = os.path.join(OUT, "wheels"); os.makedirs(WH, exist_ok=True)

# 1) wheels puras de FLA (sin deps: triton/torch ya estan en la imagen)
r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", WH,
                    "flash-linear-attention==0.6.0", "fla-core==0.6.0", "einops"],
                   capture_output=True, text=True)
print("pip download rc=%d" % r.returncode, (r.stdout + r.stderr)[-600:], flush=True)
print("wheels:", sorted(os.listdir(WH)), flush=True)

# 2) checkpoints BASE oficiales (solo ficheros de texto/pesos; sin imagenes de ejemplo)
from huggingface_hub import snapshot_download
PATRONES = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken", "merges.txt", "vocab.json"]
for repo, dest in (("Qwen/Qwen3.5-4B-Base", "q35_4b_base"), ("Qwen/Qwen3.5-2B-Base", "q35_2b_base")):
    d = os.path.join(OUT, dest)
    t = time.time()
    try:
        p = snapshot_download(repo_id=repo, local_dir=d, allow_patterns=PATRONES, max_workers=8)
    except Exception as e:
        print("FALLO %s: %s" % (repo, str(e)[:300]), flush=True)
        continue
    tot = 0; fichs = []
    for root, _, files in os.walk(d):
        for f in files:
            if ".cache" in root: continue
            fp = os.path.join(root, f); n = os.path.getsize(fp); tot += n
            fichs.append((os.path.relpath(fp, d), n))
    print("%s -> %s: %d ficheros, %.2f GB, %.0fs" % (repo, p, len(fichs), tot / 1e9, time.time() - t), flush=True)
    for f, n in sorted(fichs): print("   %-50s %12d" % (f, n), flush=True)
    cfg = json.load(open(os.path.join(d, "config.json")))
    tc = cfg.get("text_config", cfg)
    print("   arch=%s layers=%s hidden=%s vocab=%s tie=%s" % (cfg.get("architectures"), tc.get("num_hidden_layers"),
          tc.get("hidden_size"), tc.get("vocab_size"), cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings"))), flush=True)
    lt = tc.get("layer_types") or []
    print("   layer_types: %d linear_attention / %d full_attention" % (lt.count("linear_attention"), lt.count("full_attention")), flush=True)
    # limpiar la cache de HF dentro de local_dir para no duplicar bytes en la salida
    import shutil
    for root, dirs, _ in os.walk(d):
        for c in dirs:
            if c == ".cache": shutil.rmtree(os.path.join(root, c), ignore_errors=True)

print("RESULT_JSON:" + json.dumps({"wall_s": round(time.time() - T0)}), flush=True)
'''

def celda(src, tipo="code"):
    c = {"cell_type": tipo, "metadata": {}, "source": src.splitlines(keepends=True)}
    if tipo == "code":
        c["execution_count"] = None; c["outputs"] = []
    return c

nb = {"cells": [celda(MD, "markdown"), celda(CODE.lstrip("\n"))],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/q35base_dl.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("OK notebooks/q35base_dl.ipynb")
