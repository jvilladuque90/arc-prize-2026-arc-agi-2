#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Proxy de rank-correlation para el injerto `scoring4` (augs de re-scoring 8->4).

El campeon re-puntua cada candidato del DFS bajo 8 augmentaciones (consistencia aumentada) y
selecciona top-2 (pass@2). El injerto `scoring4` usaria 4 augs -> ~mitad del tiempo de scoring.
Riesgo unico: fidelidad de la SELECCION. Este proxy lo mide sin gastar Kaggle:

1. 4B campeon real (grids15) en T4 fp16; ~10 tareas holdout con trabajo DFS real.
2. Pool de candidatos por tarea: turbo_dfs (port fiel) sobre 8 vistas de forma igual.
3. Cada candidato (canonico) se puntua S8 = NLL media bajo las 8 vistas y S4 = bajo 4
   (subconjunto anidado — exactamente lo que haria el injerto).
4. Metricas: Spearman(S8,S4), acuerdo top-1/top-2, y con la GT INYECTADA al pool: ¿la GT queda
   en top-2 bajo S4 siempre que quedaba bajo S8? (fidelidad pass@2 directa).

Lectura pre-registrada (SPEED_PLAN §5): proxy OK si acuerdo top-2 >= ~90% Y la GT nunca sale
del top-2 al pasar de S8 a S4. Si falla -> `scoring4` muere y (con decode dominando el split
del viernes) la familia velocidad queda cerrada con el mapa completo.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_scoring4_proxy.py <KAGGLE_API_TOKEN>
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
subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
                "-p", "/content/m4b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m4b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR
print(f"[setup] 4B en {MDIR} ({time.time()-t0:.0f}s)", flush=True)

import hashlib
import random as _rnd

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL.eval()
print(f"[setup] modelo cargado | vocab={len(TOK)} | mem={torch.cuda.memory_allocated()/1e9:.1f}GB",
      flush=True)

# ---- vocabulario minimo del campeon: ids 0-9 = digitos, 10 = \n, 15 = im_end, 13 = pad ----
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
PAD_ID, EOS_ID, NL_ID = 13, 15, 10
MAX_SCORE = -float(np.log(0.2))
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")


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
def dfs_call(prefixes, max_new=400, budget=120):
    ids = torch.tensor(prefixes, device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    return turbo_dfs(out.logits[:, -1], max_new, [0.0] * ids.size(0), ids.size(1),
                     out.past_key_values, time.time(), budget)


# ---- 8 vistas de forma igual (geometrias que preservan (h,w) x 2 permutes de color) ----
def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


GEOS = [lambda a: a,
        lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a.T, 1),
        lambda a: np.rot90(a.T, 3)]
GEOS_INV = [lambda b: b,
            lambda b: np.rot90(b, -2),
            lambda b: np.rot90(b, -1).T,
            lambda b: np.rot90(b, -3).T]


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
            views.append({"gi": gi, "pj": pj, "pm": pm, "ids": TOK.encode(txt)})
    return views


def toks_to_grid(toks):
    """suffix del DFS -> grid canonicalizable (digitos + NL, termina en EOS)."""
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


def grid_to_toks(grid):
    out = []
    for i, row in enumerate(grid):
        if i:
            out.append(NL_ID)
        out.extend(int(c) for c in row)
    out.append(EOS_ID)
    return out


def canonical(grid, view):
    inv = np.argsort(view["pm"])
    return GEOS_INV[view["gi"]](inv[grid])


@torch.no_grad()
def nll_under_view(view, cand_canon):
    tv = view["pm"][GEOS[view["gi"]](cand_canon)]
    ans = grid_to_toks(tv)
    ids = torch.tensor(view["ids"] + ans, device="cuda:0", dtype=torch.long).unsqueeze(0)
    logits = MODEL(ids).logits.float()
    lp = torch.log_softmax(logits[0, len(view["ids"]) - 1:-1], -1)
    at = torch.tensor(ans, device="cuda:0")
    return -lp.gather(1, at.unsqueeze(1)).mean().item()


# ---- tareas ----
def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
TASKS = HOLD[40:60:2]   # 10 tareas medianas/grandes (donde el DFS trabaja de verdad)

RES = {"tasks": []}
MAX_CANDS = 8
for tk in TASKS:
    task = CHAL[tk]
    views = make_views(task)
    L = len(views[0]["ids"])
    if any(len(v["ids"]) != L for v in views) or L > 6000:
        print(f"[{tk}] saltada (vistas desiguales o ctx {L} muy largo)", flush=True)
        continue
    t0 = time.time()
    res = dfs_call([v["ids"] for v in views])
    # pool: candidatos canonicos deduplicados, mejores por score DFS
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
    cands = [c for _, c in sorted(pool.values(), key=lambda x: x[0])[:MAX_CANDS]]
    gt = np.array(SOLS[tk][0])
    gt_in_pool = any(c.shape == gt.shape and (c == gt).all() for c in cands)
    if not gt_in_pool:
        cands.append(gt)   # GT inyectada: mide fidelidad pass@2 aunque el DFS no la halle
    if len(cands) < 3:
        print(f"[{tk}] pool insuficiente ({len(cands)} cands, dfs {time.time()-t0:.0f}s)", flush=True)
        continue
    s8 = [float(np.mean([nll_under_view(v, c) for v in views])) for c in cands]
    s4 = [float(np.mean([nll_under_view(v, c) for v in views[:4]])) for c in cands]
    r8, r4 = np.argsort(s8), np.argsort(s4)
    gt_idx = next(i for i, c in enumerate(cands) if c.shape == gt.shape and (c == gt).all())
    row = {"task": tk, "n_cands": len(cands), "gt_de_dfs": bool(gt_in_pool),
           "spearman": round(float(spearmanr(s8, s4).statistic), 4),
           "top1_igual": bool(r8[0] == r4[0]),
           "top2_igual": bool(set(r8[:2].tolist()) == set(r4[:2].tolist())),
           "gt_top2_s8": bool(gt_idx in r8[:2]), "gt_top2_s4": bool(gt_idx in r4[:2]),
           "secs": round(time.time() - t0, 1)}
    RES["tasks"].append(row)
    print(f"[{tk}] cands={row['n_cands']} rho={row['spearman']} top2_igual={row['top2_igual']} "
          f"gt@2: s8={row['gt_top2_s8']} s4={row['gt_top2_s4']} ({row['secs']}s)", flush=True)

ts = RES["tasks"]
assert ts, "ninguna tarea produjo pool suficiente"
RES["resumen"] = {
    "n_tareas": len(ts),
    "spearman_media": round(float(np.mean([t["spearman"] for t in ts])), 4),
    "acuerdo_top1": round(sum(t["top1_igual"] for t in ts) / len(ts), 3),
    "acuerdo_top2": round(sum(t["top2_igual"] for t in ts) / len(ts), 3),
    "gt_top2_s8": sum(t["gt_top2_s8"] for t in ts),
    "gt_top2_s4": sum(t["gt_top2_s4"] for t in ts),
    "gt_perdidas_por_s4": sum(1 for t in ts if t["gt_top2_s8"] and not t["gt_top2_s4"]),
}
json.dump(RES, open("/content/scoring4_proxy_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(RES["resumen"]), flush=True)
