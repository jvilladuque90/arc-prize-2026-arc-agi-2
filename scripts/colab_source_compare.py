#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] D3: comparacion de FUENTES sinteticas a N=2000 (misma receta que D1).

¿Que distribucion transfiere mejor a AGI-2? Brazos frescos (base 2B + LoRA nueva por brazo):
  A) NVARC-synthetic 2000  — la data del PROPIO campeon (esperado: saturacion; ya la vio)
  B) MIX 1000 NVARC + 1000 ARC-Heavy — ¿la diversidad compone?
Referencia medida (D1, mismo protocolo): ARC-Heavy-2000 = 7/24 exactas, cell-acc 0.851,
NLL 0.0489 en las 24 tareas NUEVAS de AGI-2.

Responde la pregunta de generalizacion de la generacion (SCALE_PLAN D3) y fija la mezcla de
datos de D4 (SFT del 4B del viernes). Formato NVARC: cada json = lista de ~30 pares de una
regla -> contexto 3 pares + objetivo 1 (mismo builder que ARC-Heavy).

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_source_compare.py <KAGGLE_API_TOKEN>
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

subprocess.run(["kaggle", "datasets", "download", "-d", "sorokin/nvarc-synthetic-puzzles",
                "-p", "/content/nvarc", "--unzip", "-q"], check=True)
subprocess.run(["kaggle", "datasets", "download", "-d", "juliancamilovilla/arcscale-archeavy-19k",
                "-p", "/content/heavy", "--unzip", "-q"], check=True)

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] descargas listas ({time.time()-T00:.0f}s)", flush=True)

import hashlib
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MAX_LEN = 2048


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def fmt_pair(inp, out):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n{g2s(out)}<|im_end|>"


def fmt_query(inp):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n"


def valid_grid(g):
    return (g and g[0] and len(g) <= 30 and len(g[0]) <= 30
            and all(len(r) == len(g[0]) for r in g)
            and all(0 <= int(c) <= 9 for r in g for c in r))


def build_item(pairs):
    prompt = "".join(fmt_pair(i, o) for i, o in pairs[:-1]) + fmt_query(pairs[-1][0])
    ans = g2s(pairs[-1][1]) + "<|im_end|>"
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(ans, return_tensors="pt").input_ids[0]
    if pi.numel() + ai.numel() > MAX_LEN:
        return None
    return torch.cat([pi, ai])


# ---- pool NVARC: archivos ordenados (determinista), 4 primeros pares por archivo ----
NV_FILES = []
for root, dirs, files in os.walk("/content/nvarc"):
    for f in sorted(files):
        if f.endswith(".json"):
            NV_FILES.append(os.path.join(root, f))
NV_FILES.sort()
NVARC_ITEMS = []
for fp in NV_FILES:
    try:
        d = json.load(open(fp))
        pairs = [(p["input"], p["output"]) for p in d[:4]
                 if valid_grid(p["input"]) and valid_grid(p["output"])]
        if len(pairs) < 4:
            continue
        it = build_item(pairs)
        if it is not None:
            NVARC_ITEMS.append(it)
        if len(NVARC_ITEMS) >= 2000:
            break
    except Exception:
        continue
print(f"[pool] NVARC: {len(NVARC_ITEMS)} items de {len(NV_FILES)} archivos "
      f"({time.time()-T00:.0f}s)", flush=True)

# ---- pool ARC-Heavy (snapshot, mismo builder) para la mezcla ----
HEAVY_ITEMS = []
JSONL = None
for root, dirs, files in os.walk("/content/heavy"):
    if "arc_heavy_19k.jsonl" in files:
        JSONL = os.path.join(root, "arc_heavy_19k.jsonl")
assert JSONL
with open(JSONL) as fh:
    for line in fh:
        pairs = [tuple(p) for p in json.loads(line)["pairs"]]
        it = build_item(pairs)
        if it is not None:
            HEAVY_ITEMS.append(it)
        if len(HEAVY_ITEMS) >= 1000:
            break
print(f"[pool] ARC-Heavy: {len(HEAVY_ITEMS)} items", flush=True)

ARMS = {
    "nvarc2000": NVARC_ITEMS[:2000],
    "mix1000_1000": NVARC_ITEMS[:1000] + HEAVY_ITEMS[:1000],
}
for name, items in ARMS.items():
    random.Random(7).shuffle(items)

# ---- eval: las 24 tareas NUEVAS (poblacion limpia) ----
def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
KEYS = [k for k in HOLD if k not in AGI1]
ITEMS = {}
for k in KEYS:
    t = CHAL[k]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    if pi.numel() > 6000:
        continue
    ai = TOK(g2s(SOLS[k][0]) + "<|im_end|>", return_tensors="pt").input_ids[0]
    ITEMS[k] = (pi, ai)
print(f"[setup] {len(ITEMS)} tareas de eval", flush=True)

LCFG = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])


def run_arm(name, items):
    global MODEL
    base = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                                device_map="cuda:0")
    MODEL = get_peft_model(base, LCFG)
    MODEL.enable_input_require_grads()
    MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    MODEL.config.use_cache = False

    opt = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    ACCUM = 8
    MODEL.train()
    t0 = time.time()
    opt.zero_grad()
    for n, full in enumerate(items):
        ids = full.unsqueeze(0).to("cuda:0")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = MODEL(ids, labels=ids)
        scaler.scale(out.loss / ACCUM).backward()
        if (n + 1) % ACCUM == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        if (n + 1) % 500 == 0:
            print(f"  [{name}] {n+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    if len(items) % ACCUM != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
    MODEL.eval()
    torch.cuda.empty_cache()

    rows = []
    for k in ITEMS:
        pi, ai = ITEMS[k]
        with torch.no_grad():
            ids = torch.cat([pi, ai]).unsqueeze(0).to("cuda:0")
            logits = MODEL(ids).logits.float()
            lp = torch.log_softmax(logits[0, pi.numel() - 1:-1], -1)
            nll = -lp.gather(1, ai.unsqueeze(1).to("cuda:0")).mean().item()
        gids = pi.unsqueeze(0).to("cuda:0")
        MODEL.config.use_cache = True
        with torch.no_grad():
            gout = MODEL.generate(gids, max_new_tokens=960, do_sample=False,
                                  pad_token_id=TOK.eos_token_id)
        MODEL.config.use_cache = False
        txt = TOK.decode(gout[0, gids.shape[1]:], skip_special_tokens=True)
        try:
            pred = [[int(c) for c in line if c.isdigit()]
                    for line in txt.strip().split("\n") if line.strip()]
            gt = SOLS[k][0]
            ok = pred and len(pred) == len(gt) and all(len(r) == len(gt[0]) for r in pred)
            acc = float((np.array(pred) == np.array(gt)).mean()) if ok else 0.0
            exact = bool(ok and (np.array(pred) == np.array(gt)).all())
        except Exception:
            acc, exact = 0.0, False
        rows.append({"task": k, "nll": round(nll, 4), "cell_acc": round(acc, 4), "exact": exact})
    summ = {"nll_media": round(float(np.mean([r["nll"] for r in rows])), 4),
            "cell_acc_media": round(float(np.mean([r["cell_acc"] for r in rows])), 4),
            "exactas": sum(r["exact"] for r in rows), "n": len(rows),
            "train_secs": round(time.time() - t0, 1)}
    print(f"RESULT_ARM:{name}:" + json.dumps(summ), flush=True)
    del MODEL, base
    torch.cuda.empty_cache()
    return {"resumen": summ, "detalle": rows}


RES = {"referencia_archeavy2000_D1": {"exactas": 7, "cell_acc_media": 0.8506, "nll_media": 0.0489},
       "arms": {}}
for name, items in ARMS.items():
    RES["arms"][name] = run_arm(name, items)

json.dump(RES, open("/content/source_compare_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps({k: v["resumen"] for k, v in RES["arms"].items()}), flush=True)
