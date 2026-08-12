#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] D2: extension de la curva de escala N=2000 -> 4000 -> 8000 (reanuda del adapter).

D1/D1-bis/D2-bis dieron verde (transferencia genuina + sobrevive al TTT). Esta etapa dimensiona
el dataset de D4 (SFT del 4B en Kaggle): ¿la curva sigue subiendo en log-N o satura?

- Reanuda el LoRA n=2000 (dataset kaggle arcscale-adapter-2b) con is_trainable=True.
- Datos: ARC-Heavy streaming en el MISMO orden determinista que D1 (40 holdout + salta los
  2000 ya vistos) -> 6000 frescos.
- Eval en cada punto sobre las 24 tareas NUEVAS de AGI-2 (poblacion limpia, la de D1-bis)
  + 40 sinteticas held-out. Optimizador fresco (el estado de AdamW no se persistio — se anota).
- Tras cada punto: guarda el adapter y sube version al dataset (n4000/n8000) — si la sesion
  muere, el progreso queda persistido.

Lectura: pendiente positiva en las nuevas 2000->8000 -> D4 con dataset grande (>=16K);
saturacion -> D4 con ~4-8K basta (mas barato en quota).

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_data_scaling_ext.py <KAGGLE_API_TOKEN> [<KAGGLE_USERNAME>]
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
KUSER = sys.argv[2] if len(sys.argv) > 2 else ""
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "kaggle", "datasets",
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def fmt_pair(inp, out):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n{g2s(out)}<|im_end|>"


def fmt_query(inp):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n"


# ---------------------------------------------------------------- pool (mismo orden que D1)
from datasets import load_dataset

MAX_LEN = 2048


def parse_row(row):
    try:
        pairs = []
        for ex in row["examples"]:
            gi = [[int(c) for c in r] for r in ex[0]]
            go = [[int(c) for c in r] for r in ex[1]]
            for g in (gi, go):
                if not g or not g[0] or len(g) > 30 or len(g[0]) > 30:
                    return None
                if any(len(r) != len(g[0]) for r in g):
                    return None
                if any(c < 0 or c > 9 for r in g for c in r):
                    return None
            pairs.append((gi, go))
            if len(pairs) == 4:
                break
        return pairs if len(pairs) >= 3 else None
    except Exception:
        return None


def build_item(pairs):
    prompt = "".join(fmt_pair(i, o) for i, o in pairs[:-1]) + fmt_query(pairs[-1][0])
    ans = g2s(pairs[-1][1]) + "<|im_end|>"
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(ans, return_tensors="pt").input_ids[0]
    if pi.numel() + ai.numel() > MAX_LEN:
        return None
    return {"prompt_ids": pi, "ans_ids": ai, "full_ids": torch.cat([pi, ai])}


t0 = time.time()
STREAM = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                      split="train", streaming=True)
SYN_HOLD, TRAIN = [], []
N_SKIP, N_FRESH, N_HOLD = 2000, 6000, 40   # los primeros 2000 validos ya se entrenaron en D1
n_valid = 0
for row in STREAM:
    pairs = parse_row(row)
    if pairs is None:
        continue
    item = build_item(pairs)
    if item is None:
        continue
    if len(SYN_HOLD) < N_HOLD:
        SYN_HOLD.append(item)
        continue
    n_valid += 1
    if n_valid <= N_SKIP:
        continue
    TRAIN.append(item)
    if len(TRAIN) >= N_FRESH:
        break
random.Random(11).shuffle(TRAIN)
print(f"[pool] {len(SYN_HOLD)} holdout + {N_SKIP} saltados + {len(TRAIN)} frescos "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- tareas reales NUEVAS
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
print(f"[setup] {len(ITEMS)} tareas nuevas evaluables", flush=True)

# ---------------------------------------------------------------- modelo (reanuda adapter)
BASE = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL = PeftModel.from_pretrained(BASE, ADIR, is_trainable=True)
MODEL.enable_input_require_grads()
MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
MODEL.config.use_cache = False
print(f"[setup] 2B + adapter n=2000 reanudado ({time.time()-T00:.0f}s)", flush=True)


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
    MODEL.config.use_cache = False
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


def eval_point(n_seen):
    MODEL.eval()
    torch.cuda.empty_cache()
    t0 = time.time()
    real = []
    for k in ITEMS:
        pi, ai = ITEMS[k]
        nll = nll_ids(pi, ai)
        acc, exact = greedy(k)
        real.append({"task": k, "nll": round(nll, 4), "cell_acc": round(acc, 4), "exact": exact})
    syn_nll = [nll_ids(it["prompt_ids"], it["ans_ids"]) for it in SYN_HOLD]
    point = {"n": n_seen,
             "real": {"nll_media": round(float(np.mean([r["nll"] for r in real])), 4),
                      "cell_acc_media": round(float(np.mean([r["cell_acc"] for r in real])), 4),
                      "exactas": sum(r["exact"] for r in real), "n_tareas": len(real),
                      "detalle": real},
             "syn": {"nll_media": round(float(np.mean(syn_nll)), 4)},
             "eval_secs": round(time.time() - t0, 1)}
    print("RESULT_POINT:" + json.dumps({k: v for k, v in point.items() if k != "real"}
                                       | {"real": {k2: v2 for k2, v2 in point["real"].items()
                                                   if k2 != "detalle"}}), flush=True)
    return point


ACCUM = 8
OPT = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=1e-4)
SCALER = torch.cuda.amp.GradScaler()


def train_chunk(items):
    MODEL.config.use_cache = False
    MODEL.train()
    torch.cuda.empty_cache()
    t0 = time.time()
    OPT.zero_grad()
    for n, it in enumerate(items):
        ids = it["full_ids"].unsqueeze(0).to("cuda:0")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = MODEL(ids, labels=ids)
        SCALER.scale(out.loss / ACCUM).backward()
        if (n + 1) % ACCUM == 0:
            SCALER.step(OPT)
            SCALER.update()
            OPT.zero_grad()
        if (n + 1) % 500 == 0:
            print(f"  [train] {n+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    if len(items) % ACCUM != 0:   # flush SOLO si quedan gradientes (si no, GradScaler asserta)
        SCALER.step(OPT)
        SCALER.update()
        OPT.zero_grad()
    MODEL.eval()
    return time.time() - t0


def save_and_push(tag, res):
    try:
        out = f"/content/adapter_{tag}"
        MODEL.save_pretrained(out)
        json.dump(res, open(os.path.join(out, "data_scaling_ext_result.json"), "w"), indent=1)
        if KUSER:
            json.dump({"title": "arcscale-adapter-2b", "id": f"{KUSER}/arcscale-adapter-2b",
                       "licenses": [{"name": "CC0-1.0"}]},
                      open(os.path.join(out, "dataset-metadata.json"), "w"))
            r = subprocess.run(["kaggle", "datasets", "version", "-p", out, "-m", tag],
                               capture_output=True, text=True)
            print(f"[adapter] version {tag}: rc={r.returncode}", flush=True)
    except Exception as e:
        print(f"[adapter] no subido ({tag}): {e}", flush=True)


RES = {"points": []}
RES["points"].append(eval_point(2000))   # punto de partida (adapter reanudado)
for lo, hi in ((0, 2000), (2000, 6000)):
    secs = train_chunk(TRAIN[lo:hi])
    n_total = 2000 + hi
    print(f"[curva] +{hi-lo} en {secs:.0f}s (total {time.time()-T00:.0f}s)", flush=True)
    RES["points"].append(eval_point(n_total))
    save_and_push(f"n{n_total}", RES)

p0 = RES["points"][0]
RES["resumen"] = {
    "curva_real_nll": [p["real"]["nll_media"] for p in RES["points"]],
    "curva_real_cellacc": [p["real"]["cell_acc_media"] for p in RES["points"]],
    "curva_real_exactas": [p["real"]["exactas"] for p in RES["points"]],
    "curva_syn_nll": [p["syn"]["nll_media"] for p in RES["points"]],
}
json.dump(RES, open("/content/data_scaling_ext_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(RES["resumen"]), flush=True)
