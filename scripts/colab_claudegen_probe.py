#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Piloto claudegen: ¿una distribucion NUEVA (autor semantico) rompe el plateau?

D3 midio que las fuentes publicas empatan en el 2B (~7-8/24, cell-acc ~0.85-0.87). Este probe
mide el corpus claudegen v1 (33 reglas compositivas/condicionales autoradas, 2178 tareas
verificadas, 0 contaminacion) con el MISMO protocolo (N=2000 fresco, lr 1e-4, accum 8):
  A) claudegen-2000 solo
  B) MIX 1000 claudegen + 1000 ARC-Heavy
Referencias medidas: ARC-Heavy 7/24 - 0.851 - NLL 0.049 | NVARC 7/24 - 0.873 - 0.029 |
MIX(nvarc+heavy) 8/24 - 0.855 - 0.033.

Lectura pre-registrada:
- >=10/24 o cell-acc >=0.90 en A o B -> el plateau ERA de datos -> escalar claudegen (mas
  reglas/tareas) y añadirlo a la mezcla de D4-v2.
- ~7-8/24 -> el plateau es del MODELO/protocolo a 2B -> la palanca de datos se cierra en el
  proxy; la capacidad (4B, sabado) decide.
- <<7 -> la distribucion interfiere (revisar mezcla de reglas).

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_claudegen_probe.py <KAGGLE_API_TOKEN>
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

subprocess.run(["kaggle", "datasets", "download", "-d", "juliancamilovilla/arcscale-claudegen-v1",
                "-p", "/content/cg", "--unzip", "-q"], check=True)
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


def build_item(pairs):
    prompt = "".join(fmt_pair(i, o) for i, o in pairs[:-1]) + fmt_query(pairs[-1][0])
    ans = g2s(pairs[-1][1]) + "<|im_end|>"
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(ans, return_tensors="pt").input_ids[0]
    if pi.numel() + ai.numel() > MAX_LEN:
        return None
    return torch.cat([pi, ai])


def load_jsonl(path, n):
    items = []
    with open(path) as fh:
        for line in fh:
            pairs = [tuple(p) for p in json.loads(line)["pairs"]]
            it = build_item(pairs)
            if it is not None:
                items.append(it)
            if len(items) >= n:
                break
    return items


CG = None
for root, dirs, files in os.walk("/content/cg"):
    if "claudegen_v1.jsonl" in files:
        CG = os.path.join(root, "claudegen_v1.jsonl")
HV = None
for root, dirs, files in os.walk("/content/heavy"):
    if "arc_heavy_19k.jsonl" in files:
        HV = os.path.join(root, "arc_heavy_19k.jsonl")
assert CG and HV
CG_ITEMS = load_jsonl(CG, 2000)
HV_ITEMS = load_jsonl(HV, 1000)
print(f"[pool] claudegen {len(CG_ITEMS)} | heavy {len(HV_ITEMS)}", flush=True)

ARMS = {
    "claudegen2000": CG_ITEMS[:2000],
    "mix_cg1000_heavy1000": CG_ITEMS[:1000] + HV_ITEMS[:1000],
}
for name, items in ARMS.items():
    random.Random(7).shuffle(items)


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


RES = {"referencias": {"archeavy2000": {"exactas": 7, "cell_acc": 0.851, "nll": 0.049},
                       "nvarc2000": {"exactas": 7, "cell_acc": 0.873, "nll": 0.029},
                       "mix_nvarc_heavy": {"exactas": 8, "cell_acc": 0.855, "nll": 0.033}},
       "arms": {}}
for name, items in ARMS.items():
    RES["arms"][name] = run_arm(name, items)

json.dump(RES, open("/content/claudegen_probe_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps({k: v["resumen"] for k, v in RES["arms"].items()}), flush=True)
