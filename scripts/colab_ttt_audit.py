#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] T2a — auditoria del criterio del DFS: ¿que le pide de verdad al modelo?

Linea abierta el 17-ago (TTT_PLAN.md) tras cerrar la familia de cobertura. El DFS solo acepta
un candidato si la grid COMPLETA cumple p(secuencia) > 0.2. Para una salida de L tokens eso
exige una probabilidad media por token de 0.2^(1/L):  L=100 -> 98.4%,  L=372 -> 99.57%,
L=929 -> 99.83%. El TTT optimiza NLL MEDIA; el criterio es masa de secuencia. Desalineados, y
peor cuanto mas grande la salida.

Este primer paso NO entrena nada (solo teacher forcing) y resuelve la FORMA del problema:

  H1 (tamano): NLL_total(GT) crece con L y la fraccion de tareas bajo el umbral colapsa.
  H2 (pre-TTT): ¿puede el modelo reproducir las salidas de sus PROPIOS pares de train
     (leave-one-out) bajo el umbral? Si ni eso, el DFS no tenia ninguna posibilidad.

Metricas por tarea: L, NLL_total(GT), NLL/token(GT), p(GT)>0.2, y lo mismo para cada par de
train en leave-one-out. Sirve de LINEA BASE para medir despues cuanto mueve el TTT (T2b).

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_ttt_audit.py <KAGGLE_API_TOKEN>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json",
          "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)
subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
                "-p", "/content/m4b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m4b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] descargas listas ({time.time()-T00:.0f}s)", flush=True)

import hashlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.nn.attention import SDPBackend, sdpa_kernel

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                             device_map="cuda:0",
                                             attn_implementation="sdpa")
MODEL.eval()
# v2: la v1 murio con OOM (3.73 GiB) en scaled_dot_product_attention — SDPA cayo al camino
# matematico, que materializa la matriz LxL. Forzar el backend eficiente + acotar contexto.
_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
MAX_CTX = 5000
print(f"[setup] 4B cargado | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

THR = -float(np.log(0.2))     # umbral del campeon sobre la NLL total de la secuencia


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def ctx(pairs):
    return "".join(f"<|im_start|>user\n{g2s(p['input'])}<|im_end|>"
                   f"<|im_start|>assistant\n{g2s(p['output'])}<|im_end|>" for p in pairs)


@torch.no_grad()
def nll_total(prompt, answer):
    """NLL acumulada de `answer` dado `prompt` (teacher forcing), igual que el DFS."""
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(answer, return_tensors="pt").input_ids[0]
    ids = torch.cat([pi, ai]).unsqueeze(0).to("cuda:0")
    if ids.shape[1] > MAX_CTX:
        return None, None
    with sdpa_kernel(_SDPA):
        logits = MODEL(ids).logits.float()
    lp = torch.log_softmax(logits[0, pi.numel() - 1:-1], -1)
    tot = -lp.gather(1, ai.unsqueeze(1).to("cuda:0")).sum().item()
    del logits, lp
    torch.cuda.empty_cache()
    return tot, int(ai.numel())


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CH_T = json.load(open("/content/arc-agi_training_challenges.json"))
SO_T = json.load(open("/content/arc-agi_training_solutions.json"))
CH_E = json.load(open("/content/arc-agi_evaluation_challenges.json"))
SO_E = json.load(open("/content/arc-agi_evaluation_solutions.json"))

# poblacion: 24 nuevas del holdout (no-AGI1) + 16 del eval publico, para cubrir tamanos
NEW = [k for k in CH_T if is_holdout(k) and k not in AGI1]
TASKS = ([(k, CH_T[k], SO_T[k]) for k in sorted(NEW)]
         + [(k, CH_E[k], SO_E[k]) for k in sorted(CH_E)[:16]])
print(f"[setup] {len(TASKS)} tareas | umbral NLL total < {THR:.3f} (p > 0.2)", flush=True)

RES = {"tasks": []}
for ti, (tk, task, sol) in enumerate(TASKS):
    prompt = ctx(task["train"]) + (f"<|im_start|>user\n{g2s(task['test'][0]['input'])}"
                                   f"<|im_end|><|im_start|>assistant\n")
    gt_txt = g2s(sol[0]) + "<|im_end|>"
    tot, L = nll_total(prompt, gt_txt)
    if tot is None:
        print(f"[{ti+1}/{len(TASKS)}] {tk} saltada (contexto largo)", flush=True)
        continue
    # H2 pre-TTT: reproducir sus PROPIOS pares de train (leave-one-out)
    tr_rows = []
    for i, p in enumerate(task["train"]):
        others = [q for j, q in enumerate(task["train"]) if j != i]
        if not others:
            continue
        pr = ctx(others) + f"<|im_start|>user\n{g2s(p['input'])}<|im_end|><|im_start|>assistant\n"
        t2, l2 = nll_total(pr, g2s(p["output"]) + "<|im_end|>")
        if t2 is not None:
            tr_rows.append({"nll": round(t2, 3), "L": l2, "pasa": bool(t2 < THR)})
    row = {"task": tk, "L": L, "nll_total": round(tot, 3),
           "nll_por_token": round(tot / max(L, 1), 5),
           "gt_pasa_umbral": bool(tot < THR),
           "prob_media_requerida": round(float(0.2 ** (1.0 / max(L, 1))), 5),
           "train_pasa": sum(r["pasa"] for r in tr_rows), "train_n": len(tr_rows),
           "train_nll_medio": round(float(np.mean([r["nll"] for r in tr_rows])), 3) if tr_rows else None}
    RES["tasks"].append(row)
    print(f"[{ti+1}/{len(TASKS)}] {tk}: L={L:4d} NLL_total={tot:9.2f} "
          f"(/tok {row['nll_por_token']:.4f}) GT_pasa={row['gt_pasa_umbral']} | "
          f"train {row['train_pasa']}/{row['train_n']} bajo umbral", flush=True)

rows = RES["tasks"]
if rows:
    def frac(sel):
        s = [r for r in rows if sel(r)]
        return (len(s), round(100 * sum(r["gt_pasa_umbral"] for r in s) / max(len(s), 1), 1))
    summ = {"n": len(rows),
            "gt_pasa_umbral": sum(r["gt_pasa_umbral"] for r in rows),
            "nll_total_mediana": round(float(np.median([r["nll_total"] for r in rows])), 2),
            "nll_por_token_mediana": round(float(np.median([r["nll_por_token"] for r in rows])), 5),
            "train_pasa_total": sum(r["train_pasa"] for r in rows),
            "train_n_total": sum(r["train_n"] for r in rows),
            "por_tamano": {"L<=150": frac(lambda r: r["L"] <= 150),
                           "150<L<=400": frac(lambda r: 150 < r["L"] <= 400),
                           "L>400": frac(lambda r: r["L"] > 400)}}
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)

json.dump(RES, open("/content/ttt_audit_result.json", "w"), indent=1)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/ttt_audit_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale ttt audit", "id": f"{user}/arcscale-ttt-audit",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "audit", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
