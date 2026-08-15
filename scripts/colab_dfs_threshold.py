#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Barrido del umbral del DFS (4B real): ¿pools más anchos = más GT en el pool?

El proxy de scoring4 midió que con el umbral de producción (-log 0.2) 7/12 tareas producen
1-2 candidatos → el pass@2 casi no tiene de dónde elegir. Con scoring4 abaratando el
re-scoring, un umbral más ancho podría pagar. Este barrido mide, por tarea y umbral:
n_candidatos, ¿la GT está en el pool?, y segundos de DFS.

Umbrales: -log(0.2) (producción) | -log(0.05) | -log(0.01). 12 tareas medianas/grandes.
Lectura: si -log(0.05) sube materialmente la tasa GT-en-pool con coste de tiempo moderado →
candidata `dfswide` para A/B (combinada con scoring4 para pagar el re-scoring extra).

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_dfs_threshold.py <KAGGLE_API_TOKEN>
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)

t0 = time.time()
# argv[2] == "merged": usa el 4B FUSIONADO con ARC-Heavy (output del kernel scale-sft-4b)
# en vez del campeon original — diagnostico del A/B 27.08 (¿pools mas chicos por calibracion?)
USE_MERGED = len(sys.argv) > 2 and sys.argv[2] == "merged"
if USE_MERGED:
    # sin --file-pattern: el kaggle de pip en la VM no lo soporta; bajar todo (~8GB, VM aguanta)
    subprocess.run(["kaggle", "kernels", "output", "juliancamilovilla/arc-agi2-scale-sft-4b",
                    "-p", "/content/m4b"], check=True)
else:
    subprocess.run(["kaggle", "models", "instances", "versions", "download",
                    "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
                    "-p", "/content/m4b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m4b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR
print(f"[setup] 4B {'FUSIONADO' if USE_MERGED else 'original'} en {MDIR} "
      f"({time.time()-t0:.0f}s)", flush=True)

import hashlib
import random as _rnd

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL.eval()
print(f"[setup] modelo cargado | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
PAD_ID, EOS_ID, NL_ID = 13, 15, 10
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
MAX_SCORE = 0.0   # se fija por umbral en el barrido


def turbo_dfs(logits, max_new, scores, pos, cache, t_start, budget):
    n = logits.size(0)
    lf = logits.float()
    nll = (torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
           + torch.logsumexp(lf, dim=-1, keepdim=True) - lf.index_select(-1, _TOKID)).cpu()
    suffixes = defaultdict(list)
    cand = {}
    for i in range(n):
        cand[i] = []
        for j, t in enumerate(ARC_TOKENS):
            sc = nll[i, j].item()
            if sc < MAX_SCORE:
                if t == EOS_ID:
                    suffixes[i].append((sc, [t]))
                elif max_new > 1:
                    cand[i].append((sc, t))
        cand[i].sort(key=lambda x: x[0])
    while time.time() - t_start < budget:
        toks, scs, alive = [], [], 0
        for i in range(n):
            if cand[i]:
                sc, t = cand[i].pop(0)
                toks.append(t)
                scs.append(sc)
                alive += 1
            else:
                toks.append(PAD_ID)
                scs.append(1000.0)
        if alive == 0:
            break
        out = MODEL(input_ids=torch.tensor(toks, device="cuda:0", dtype=torch.long).view(-1, 1),
                    position_ids=torch.full((n, 1), pos, device="cuda:0"),
                    past_key_values=cache, use_cache=True, return_dict=True)
        nxt = turbo_dfs(out.logits[:, -1], max_new - 1, scs, pos + 1, out.past_key_values,
                        t_start, budget)
        for bi, beams in nxt.items():
            for sc, st in beams:
                st.insert(0, toks[bi])
                suffixes[bi].append((sc, st))
    return suffixes


@torch.no_grad()
def dfs_call(prefixes, max_new=400, budget=150):
    ids = torch.tensor(prefixes, device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    return turbo_dfs(out.logits[:, -1], max_new, [0.0] * ids.size(0), ids.size(1),
                     out.past_key_values, time.time(), budget)


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


GEOS = [lambda a: a, lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a.T, 1), lambda a: np.rot90(a.T, 3)]
GEOS_INV = [lambda b: b, lambda b: np.rot90(b, -2),
            lambda b: np.rot90(b, -1).T, lambda b: np.rot90(b, -3).T]


def make_views(task, seed=3):
    rnd = _rnd.Random(seed)
    perms = []
    for _ in range(2):
        p = list(range(10))
        tail = p[1:]
        rnd.shuffle(tail)
        p[1:] = tail
        perms.append(np.array(p))
    views = []
    for gi, g in enumerate(GEOS):
        for pj, pm in enumerate(perms):
            def tr(x, g=g, pm=pm):
                return pm[g(np.array(x))]
            txt = "".join(f"<|im_start|>user\n{g2s(tr(p['input']))}<|im_end|>"
                          f"<|im_start|>assistant\n{g2s(tr(p['output']))}<|im_end|>"
                          for p in task["train"])
            txt += (f"<|im_start|>user\n{g2s(tr(task['test'][0]['input']))}<|im_end|>"
                    f"<|im_start|>assistant\n")
            views.append({"gi": gi, "pm": pm, "ids": TOK.encode(txt)})
    return views


def toks_to_grid(toks):
    rows, row = [], []
    for t in toks:
        if t == EOS_ID:
            break
        if t == NL_ID:
            rows.append(row)
            row = []
        elif 0 <= t <= 9:
            row.append(t)
        else:
            return None
    if row:
        rows.append(row)
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        return None
    return np.array(rows)


def canonical(grid, view):
    inv = np.argsort(view["pm"])
    return GEOS_INV[view["gi"]](inv[grid])


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
TASKS = HOLD[40:64:2]

THRESHOLDS = {"t020": -float(np.log(0.2)), "t005": -float(np.log(0.05)),
              "t001": -float(np.log(0.01))}

RES = {"tasks": []}
for tk in TASKS:
    task = CHAL[tk]
    views = make_views(task)
    L = len(views[0]["ids"])
    if any(len(v["ids"]) != L for v in views) or L > 6000:
        print(f"[{tk}] saltada (ctx {L})", flush=True)
        continue
    gt = np.array(SOLS[tk][0])
    row = {"task": tk, "ctx": L}
    for name, thr in THRESHOLDS.items():
        globals()["MAX_SCORE"] = thr
        t0 = time.time()
        res = dfs_call([v["ids"] for v in views])
        secs = time.time() - t0
        pool = {}
        for vi, beams in res.items():
            for sc, toks in beams:
                g = toks_to_grid(toks)
                if g is None:
                    continue
                c = canonical(g, views[vi])
                key = c.tobytes() + bytes(str(c.shape), "ascii")
                if key not in pool or sc < pool[key][0]:
                    pool[key] = (sc, c)
        cands = [c for _, c in pool.values()]
        gt_in = any(c.shape == gt.shape and (c == gt).all() for c in cands)
        row[name] = {"n_cands": len(cands), "gt_in_pool": bool(gt_in),
                     "secs": round(secs, 1)}
    RES["tasks"].append(row)
    print(f"[{tk}] " + " | ".join(f"{n}: {row[n]['n_cands']}c gt={row[n]['gt_in_pool']} "
                                  f"{row[n]['secs']}s" for n in THRESHOLDS), flush=True)

summ = {}
for name in THRESHOLDS:
    rows = [r[name] for r in RES["tasks"]]
    summ[name] = {"cands_media": round(float(np.mean([r["n_cands"] for r in rows])), 1),
                  "gt_en_pool": sum(r["gt_in_pool"] for r in rows),
                  "secs_media": round(float(np.mean([r["secs"] for r in rows])), 1),
                  "n": len(rows)}
RES["resumen"] = summ
json.dump(RES, open("/content/dfs_threshold_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(summ), flush=True)
