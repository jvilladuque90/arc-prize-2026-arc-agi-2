"""[Colab T4] De-risk del injerto dfsbatch con el 4B REAL (antes de gastar Kaggle).

Responde las DOS incognitas del injerto preferido (fusion de lotes DFS 4x4->2x8):
1. GANANCIA: tiempo de DFS con 2 llamadas de batch 4 vs 1 llamada de batch 8 (mismas 8 vistas,
   mismo umbral) -> ratio de utilizacion. Es relativo -> transfiere de T4 a L4.
2. MEMORIA: bytes de KV por token/secuencia medidos empiricamente -> extrapolacion lineal a
   contexto 8192 con batch 8 -> veredicto de si cabe en la L4 (24GB) sin OOM.

Usa el checkpoint campeon (sorokin qwen3_4b_grids15, vocab minimo de ~16 tokens) y un port fiel
de turbo_dfs (poda por -log(0.2), expansion lockstep con KV cache, solo 12 tokens ARC).

Uso:  colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_dfsbatch_bench.py <KAGGLE_API_TOKEN>
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                "-f", "arc-agi_training_challenges.json", "-p", "/content", "-q"], check=True)
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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL.eval()
print(f"[setup] modelo cargado | vocab={len(TOK)} | mem={torch.cuda.memory_allocated()/1e9:.1f}GB",
      flush=True)

# ---- vocabulario ARC del campeon (vocab minimo del checkpoint grids15) ----
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]   # digitos, \n, im_end
PAD_ID, EOS_ID = 13, 15
MAX_SCORE = -float(np.log(0.2))

_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")


def turbo_dfs(logits, max_new, scores, pos, cache, t_start, budget):
    """Port fiel del DFS del campeon (poda por prob acumulada, expansion lockstep)."""
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
def dfs_call(prefixes, max_new=400, budget=240):
    ids = torch.tensor(prefixes, device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    res = turbo_dfs(out.logits[:, -1], max_new, [0.0] * ids.size(0), ids.size(1),
                    out.past_key_values, time.time(), budget)
    return sum(len(v) for v in res.values())


# ---- prompts realistas: 8 vistas de la MISMA FORMA (como el injerto real) ----
# En tareas no-cuadradas rot90/transpose cambian la longitud tokenizada -> el injerto fusiona
# solo grupos de forma igual. Aqui: 4 geometrias que PRESERVAN (h,w) x 2 permutes de color.
import random as _rnd


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def views_same_shape(task, seed=3):
    geos = [lambda a: a,
            lambda a: np.rot90(a, 2),
            lambda a: np.rot90(a.T, 1),    # transpose.rot90 -> (h,w)
            lambda a: np.rot90(a.T, 3)]    # transpose.rot270 -> (h,w)
    rnd = _rnd.Random(seed)
    perms = []
    for _ in range(2):
        p = list(range(10))
        tail = p[1:]
        rnd.shuffle(tail)
        p[1:] = tail
        perms.append(np.array(p))
    out = []
    for g in geos:
        for pm in perms:
            def tr(x, g=g, pm=pm):
                return pm[g(np.array(x))]
            txt = "".join(f"<|im_start|>user\n{g2s(tr(p['input']))}<|im_end|>"
                          f"<|im_start|>assistant\n{g2s(tr(p['output']))}<|im_end|>"
                          for p in task["train"])
            txt += (f"<|im_start|>user\n{g2s(tr(task['test'][0]['input']))}<|im_end|>"
                    f"<|im_start|>assistant\n")
            out.append(TOK.encode(txt))
    return out


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda kk: cost(CHAL[kk]))
TASKS = [HOLD[20], HOLD[40], HOLD[60]]   # tamanos chico/medio/grande

RES = {"tasks": [], "kv": {}}
for tk in TASKS:
    views = views_same_shape(CHAL[tk])
    L = len(views[0])
    assert all(len(v) == L for v in views), "vistas de longitud desigual"
    r = {"task": tk, "ctx_tokens": L}

    # --- config campeon: 2 llamadas de batch 4 ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n1 = dfs_call(views[:4]) + dfs_call(views[4:])
    r["t_2x4"] = round(time.time() - t0, 1)
    r["beams_2x4"] = n1
    r["peak_2x4_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # --- config injerto: 1 llamada de batch 8 (doble presupuesto por llamada) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n2 = dfs_call(views, budget=480)
    r["t_1x8"] = round(time.time() - t0, 1)
    r["beams_1x8"] = n2
    r["peak_1x8_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    r["speedup"] = round(r["t_2x4"] / max(r["t_1x8"], 0.1), 2)
    RES["tasks"].append(r)
    print(f"[{tk}] ctx={L}tok | 2x4: {r['t_2x4']}s/{n1}beams/{r['peak_2x4_GB']}GB | "
          f"1x8: {r['t_1x8']}s/{n2}beams/{r['peak_1x8_GB']}GB | speedup={r['speedup']}", flush=True)

# ---- extrapolacion de memoria KV a la L4 (batch 8, ctx 8192) ----
base = torch.cuda.memory_allocated() / 1e9   # pesos del modelo
big = max(RES["tasks"], key=lambda x: x["ctx_tokens"])
kv_gb = big["peak_1x8_GB"] - base
kv_per_tok_seq = kv_gb / (big["ctx_tokens"] * 8)
proj_l4 = base * (2 / 2) + kv_per_tok_seq * 8192 * 8   # bf16 = mismos bytes que fp16
RES["kv"] = {"model_GB": round(base, 2), "kv_por_token_seq_MB": round(kv_per_tok_seq * 1024, 4),
             "proyeccion_L4_batch8_ctx8192_GB": round(base + kv_per_tok_seq * 8192 * 8, 2),
             "L4_capacity_GB": 22.0,
             "cabe_en_L4": bool(base + kv_per_tok_seq * 8192 * 8 < 20.5)}
speeds = [t["speedup"] for t in RES["tasks"]]
RES["resumen"] = {"speedup_medio": round(float(np.mean(speeds)), 2),
                  "speedup_rango": [min(speeds), max(speeds)]}
json.dump(RES, open("/content/dfsbatch_bench.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps({**RES["resumen"], **RES["kv"]}), flush=True)
