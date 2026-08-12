#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] D1-bis: ¿la ganancia de ARC-Heavy transfiere a tareas NUEVAS de AGI-2?

Contexto (docs/SCALE_PLAN.md): D1 midio pendiente positiva fuerte, pero 9 de las 12 tareas de
eval eran de AGI-1 — la poblacion de la que BARC escribio los seeds de ARC-Heavy → posible
"memorizacion de vecindario". Las 3 nuevas tambien mejoraron (0/3 → 2/3 exactas) pero n=3 no
decide (leccion de poder estadistico). Este script evalua PAREADO base vs adapter (el LoRA
n=2000 de D1, dataset kaggle arcscale-adapter-2b) sobre TODAS las tareas del holdout que NO
estan en AGI-1 (~24) — la unica poblacion que se parece al test set oculto.

Eval-only: sin entrenamiento, ~40 min de T4. Metricas por tarea: NLL teacher-forced,
cell-acc greedy, exacta.

Lectura pre-registrada: si las exactas/cell-acc del adapter superan claramente a la base en
las nuevas → transferencia GENUINA → D2 (extender N) + planear D4 (SFT del 4B en Kaggle).
Si la ganancia se evapora fuera de AGI-1 → la curva de D1 era vecindario → cambiar fuente (D3).

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_data_scaling_newtasks.py <KAGGLE_API_TOKEN>
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

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "kaggle",
                "torchao>=0.16"], check=True)

os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)

subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1",
                "-p", "/content/m2b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m2b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR

subprocess.run(["kaggle", "datasets", "download", "-d", "juliancamilovilla/arcscale-adapter-2b",
                "-p", "/content/adapter", "--unzip", "-q"], check=True)
ADIR = "/content/adapter"
assert os.path.exists(os.path.join(ADIR, "adapter_model.safetensors"))
print(f"[setup] modelo + adapter listos ({time.time()-T00:.0f}s)", flush=True)

# ---- ids de AGI-1 (para excluirlas: son la fuente de los seeds de ARC-Heavy) ----
AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] AGI-1 ids: {len(AGI1)}", flush=True)

import hashlib

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def fmt_pair(inp, out):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n{g2s(out)}<|im_end|>"


def fmt_query(inp):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n"


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
KEYS = [k for k in HOLD if k not in AGI1]
print(f"[setup] {len(KEYS)} tareas NUEVAS de AGI-2 en el holdout", flush=True)

MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL.eval()

ITEMS = {}
for k in KEYS:
    t = CHAL[k]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    if pi.numel() > 6000:
        print(f"[skip] {k}: prompt {pi.numel()} tokens", flush=True)
        continue
    ai = TOK(g2s(SOLS[k][0]) + "<|im_end|>", return_tensors="pt").input_ids[0]
    ITEMS[k] = (pi, ai)
print(f"[setup] {len(ITEMS)} tareas evaluables", flush=True)


@torch.no_grad()
def nll_ids(pi, ai):
    ids = torch.cat([pi, ai]).unsqueeze(0).to("cuda:0")
    logits = MODEL(ids).logits.float()
    lp = torch.log_softmax(logits[0, pi.numel() - 1:-1], -1)
    return -lp.gather(1, ai.unsqueeze(1).to("cuda:0")).mean().item()


def greedy(key, max_new=960):
    pi, _ = ITEMS[key]
    ids = pi.unsqueeze(0).to("cuda:0")
    MODEL.config.use_cache = True
    with torch.no_grad():
        out = MODEL.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=TOK.eos_token_id)
    txt = TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    try:
        pred = [[int(c) for c in line if c.isdigit()] for line in txt.strip().split("\n") if line.strip()]
        gt = SOLS[key][0]
        if not pred or len(pred) != len(gt) or any(len(r) != len(gt[0]) for r in pred):
            return 0.0, False
        p, g = np.array(pred), np.array(gt)
        return float((p == g).mean()), bool((p == g).all())
    except Exception:
        return 0.0, False


def eval_arm(name):
    rows = []
    for i, k in enumerate(ITEMS):
        pi, ai = ITEMS[k]
        nll = nll_ids(pi, ai)
        acc, exact = greedy(k)
        rows.append({"task": k, "nll": round(nll, 4), "cell_acc": round(acc, 4), "exact": exact})
        print(f"[{name} {i+1}/{len(ITEMS)}] {k}: nll={nll:.3f} acc={acc:.2f} exact={exact}",
              flush=True)
    return rows


RES = {"arms": {}}
RES["arms"]["base"] = eval_arm("base")

MODEL = PeftModel.from_pretrained(MODEL, ADIR)
MODEL.eval()
print(f"[setup] adapter n=2000 cargado ({time.time()-T00:.0f}s)", flush=True)
RES["arms"]["adapter"] = eval_arm("adapter")

summ = {}
for name, rows in RES["arms"].items():
    summ[name] = {"nll_media": round(float(np.mean([r["nll"] for r in rows])), 4),
                  "cell_acc_media": round(float(np.mean([r["cell_acc"] for r in rows])), 4),
                  "exactas": sum(r["exact"] for r in rows), "n": len(rows)}
b = {r["task"]: r for r in RES["arms"]["base"]}
a = {r["task"]: r for r in RES["arms"]["adapter"]}
summ["pareado"] = {
    "tareas_donde_adapter_mejora_nll": sum(1 for k in b if a[k]["nll"] < b[k]["nll"]),
    "exactas_ganadas": sum(1 for k in b if a[k]["exact"] and not b[k]["exact"]),
    "exactas_perdidas": sum(1 for k in b if b[k]["exact"] and not a[k]["exact"]),
}
RES["resumen"] = summ
json.dump(RES, open("/content/newtasks_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(summ), flush=True)
