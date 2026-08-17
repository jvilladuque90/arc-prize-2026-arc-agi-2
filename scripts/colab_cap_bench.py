#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Validacion del injerto `lb33-cap`: tope de longitud por tarea en el DFS.

El campeon acota el decode con max_new_tokens=931 (el maximo de una grid 30x30) para TODAS las
tareas. El bench lockstep (17-ago) mostro lotes con longitudes [72, 650, 72, 72]: el lote corre
hasta la vista mas lenta, asi que UNA vista que divaga arrastra a las otras tres.

Tope propuesto = 4 x max(mayor salida de train, entrada de test) en tokens, conservador sobre
ambas orientaciones (las vistas son transformaciones D4). Ya verificado FUERA de la GPU:
  - la GT cabe en el 100.0% de 1248 subtareas de train+eval (asi que el recorte solo puede
    caer sobre candidatos que NO son la respuesta -> monotono);
  - en el test real el tope MUERDE en el 54% de las subtareas (mediana 786, p10 226).

Lo que falta y necesita GPU: que en el DFS REAL por lotes el pool capado sea un SUBCONJUNTO del
no capado, que nunca se pierda la GT, y cuanto tiempo ahorra de verdad.

Caveat declarado: sin TTT y al umbral de produccion el 4B no completa ninguna grid (medido), asi
que el DFS se corre a -log(0.01) para que decodifique de verdad. El regimen no es identico al de
produccion (donde el TTT vuelve confiado al modelo), pero es el mas cercano medible gratis.

Lecturas pre-registradas:
  - pool_capado SUBCONJUNTO de pool_normal y GT nunca perdida -> VERDE: el tope es seguro, se
    sube el kernel y se submitea.
  - alguna GT perdida -> ROJO: la regla del tope esta mal calibrada; recalcular el multiplicador.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_cap_bench.py <KAGGLE_API_TOKEN>
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"):
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
print(f"[setup] descargas listas ({time.time()-T00:.0f}s)", flush=True)

import hashlib
import random as _rnd

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                             device_map="cuda:0")
MODEL.eval()
print(f"[setup] 4B cargado | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
PAD_ID, EOS_ID, NL_ID = 13, 15, 10
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
MAX_SCORE = -float(np.log(0.01))
FWD = {"n": 0}


def turbo_dfs(logits, max_new, scores, pos, cache, t0, budget):
    """Port fiel del turbo_dfs del campeon (lote sincronizado, cache compartida)."""
    n = logits.size(0)
    lf = logits.float()
    nll = (torch.as_tensor(scores, dtype=torch.float32, device=lf.device).view(n, 1)
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
    while time.time() - t0 < budget:
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
        FWD["n"] += 1
        nxt = turbo_dfs(out.logits[:, -1], max_new - 1, scs, pos + 1, out.past_key_values,
                        t0, budget)
        for bi, beams in nxt.items():
            for sc, st in beams:
                st.insert(0, toks[bi])
                suffixes[bi].append((sc, st))
    return suffixes


@torch.no_grad()
def dfs_batch(view_ids, max_new, budget=240):
    ids = torch.tensor(view_ids, device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    FWD["n"] += 1
    return turbo_dfs(out.logits[:, -1], max_new, [0.0] * ids.size(0), ids.size(1),
                     out.past_key_values, time.time(), budget)


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


GEOS = [lambda a: np.asarray(a), lambda a: np.rot90(np.asarray(a), 2)]
GEOS_INV = [lambda b: b, lambda b: np.rot90(b, -2)]


def make_views(task, seed=2):
    rnd = _rnd.Random(seed)
    perms = []
    for _ in range(2):
        p = list(range(10))
        tail = p[1:]
        rnd.shuffle(tail)
        p[1:] = tail
        perms.append(np.array(p))
    views = []
    for gi, geo in enumerate(GEOS):
        for pm in perms:
            def tr(x, geo=geo, pm=pm):
                return pm[geo(x)]
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


def pool_of(res, views):
    pool = set()
    for vi, beams in res.items():
        for sc, toks in beams:
            g = toks_to_grid(toks)
            if g is None:
                continue
            inv = np.argsort(views[vi]["pm"])
            c = GEOS_INV[views[vi]["gi"]](inv[g])
            pool.add((c.shape, c.tobytes()))
    return pool


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


def tlen(g):
    h, w = len(g), len(g[0])
    return h * w + max(h, w)


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
TASKS = HOLD[:12]

RES = {"tasks": []}
for ti, tk in enumerate(TASKS):
    task = CHAL[tk]
    views = make_views(task)
    L = len(views[0]["ids"])
    if any(len(v["ids"]) != L for v in views) or L > 6000:
        print(f"[{ti+1}/{len(TASKS)}] {tk} saltada (ctx {L})", flush=True)
        continue
    cap = int(min(931, 4 * max([tlen(p["output"]) for p in task["train"]]
                               + [tlen(t["input"]) for t in task["test"]]) + 2))
    gt = np.array(SOLS[tk][0])
    gt_key = (gt.shape, gt.tobytes())

    FWD["n"] = 0
    t0 = time.time()
    r_base = dfs_batch([v["ids"] for v in views], 931)
    s_base, f_base = time.time() - t0, FWD["n"]
    p_base = pool_of(r_base, views)

    FWD["n"] = 0
    t0 = time.time()
    r_cap = dfs_batch([v["ids"] for v in views], cap)
    s_cap, f_cap = time.time() - t0, FWD["n"]
    p_cap = pool_of(r_cap, views)

    row = {"task": tk, "cap": cap, "muerde": cap < 931,
           "pool_base": len(p_base), "pool_cap": len(p_cap),
           "subconjunto": p_cap <= p_base,
           "gt_base": gt_key in p_base, "gt_cap": gt_key in p_cap,
           "gt_perdida": (gt_key in p_base) and (gt_key not in p_cap),
           "fwd_base": f_base, "fwd_cap": f_cap,
           "secs_base": round(s_base, 1), "secs_cap": round(s_cap, 1)}
    RES["tasks"].append(row)
    print(f"[{ti+1}/{len(TASKS)}] {tk}: cap {cap} ({'muerde' if row['muerde'] else 'inactivo'}) "
          f"| pool {len(p_base)}->{len(p_cap)} subconj={row['subconjunto']} "
          f"gt {row['gt_base']}->{row['gt_cap']} | fwd {f_base}->{f_cap} "
          f"(x{f_base/max(f_cap,1):.2f}) | {s_base:.0f}s->{s_cap:.0f}s", flush=True)

rows = RES["tasks"]
if rows:
    fb, fc = sum(r["fwd_base"] for r in rows), sum(r["fwd_cap"] for r in rows)
    sb, sc = sum(r["secs_base"] for r in rows), sum(r["secs_cap"] for r in rows)
    summ = {"n": len(rows), "n_muerde": sum(r["muerde"] for r in rows),
            "todas_subconjunto": all(r["subconjunto"] for r in rows),
            "gt_perdidas": sum(r["gt_perdida"] for r in rows),
            "gt_en_pool_base": sum(r["gt_base"] for r in rows),
            "x_forwards": round(fb / max(fc, 1), 2),
            "x_tiempo": round(sb / max(sc, 0.1), 2)}
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)

json.dump(RES, open("/content/cap_result.json", "w"), indent=1)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/cap_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale cap bench", "id": f"{user}/arcscale-cap",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "cap", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
