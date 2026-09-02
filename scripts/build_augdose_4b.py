#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genera notebooks/augdose_4b.ipynb: banco de DOSIS de augmentaciones en el 4B, L4x4.

Veredicto pendiente tras el smoke del 26-ago (TTT_PLAN §24): el observable NO es la
dispersion del voto sino la ESCASEZ (`productividad` = fraccion de vistas que devuelven
algun candidato; r = -0,924 con el beneficio de 4->16 vistas en n=4 del 2B).

Este kernel lo mide en el vehiculo valido: 4B + eval + augmentaciones + TTT con el
calendario EXACTO de produccion (1 epoca sobre 16 vistas). Shardeado en las 4 GPUs
(obligatorio: la L4x4 se paga entera). Decodifica 16 vistas UNA vez y persiste QUE
produjo cada una -> la curva 2/4/8/12/16 sale OFFLINE por submuestreo con
scripts/augdose_analysis.py, que ya esta testeado sin GPU.

Poblacion: las 16 tareas mas baratas del eval de ARC-AGI-2 que NO estan en ARC-AGI-1
(6 del eval si lo estan; se excluyen porque el SFT pudo verlas). Las 4 primeras son
exactamente las del smoke del 2B, asi que el smoke queda como subconjunto comparable.
"""
import json
import os

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tareas del eval de AGI-2 que TAMBIEN estan en ARC-AGI-1 (calculado en local contra el
# repo fchollet/ARC-AGI). Se empotran para conservar el filtro sin internet: la L4x4
# prohibe internet.
SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

WORKER = r'''
import glob, hashlib, json, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()

SHARD = int(os.environ.get("SHARD", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "4"))
N_TAREAS = int(os.environ.get("N_TAREAS", "16"))
N_VISTAS = int(os.environ.get("N_VISTAS", "16"))
BUDGET = float(os.environ.get("BUDGET_VISTA", "45"))
WALL_MAX = float(os.environ.get("WALL_MAX", "4800"))
SUCIAS = set(json.loads(os.environ["SUCIAS"]))
OUT = "/kaggle/working/shard%d.json" % SHARD

import numpy as np
import torch
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

# --- gate de carga: aborta ESTE shard, no la maquina ---
DATA = "/kaggle/input/arc-prize-2026-arc-agi-2"
CHAL = json.load(open(DATA + "/arc-agi_evaluation_challenges.json"))
SOLS = json.load(open(DATA + "/arc-agi_evaluation_solutions.json"))

MDIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR, "no se encontro el modelo en /kaggle/input"
print("[s%d] modelo en %s" % (SHARD, MDIR), flush=True)

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))
MAX_SEQ = 3500
LR = 5e-5


def coste(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


# poblacion GLOBAL ordenada, y de ahi el shard: el indice global no cambia con N_SHARDS
KEYS_ALL = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))[:N_TAREAS]
MIS_KEYS = KEYS_ALL[SHARD::N_SHARDS]
print("[s%d] %d tareas de %d" % (SHARD, len(MIS_KEYS), len(KEYS_ALL)), flush=True)


def grid_tokens(g):
    out = []
    for i, row in enumerate(g):
        out.extend(int(c) for c in row)
        if i < len(g) - 1:
            out.append(NL)
    return out


def pair_tokens_mask(inp, out):
    t = [IM_START, USER, NL] + grid_tokens(inp) + [EOS]
    m = [0] * len(t)
    head = [IM_START, ASSIST, NL]
    body = grid_tokens(out) + [EOS]
    return t + head + body, m + [0] * len(head) + [1] * len(body)


def query_tokens(inp):
    return [IM_START, USER, NL] + grid_tokens(inp) + [EOS] + [IM_START, ASSIST, NL]


def d4(g, k):
    a = np.array(g)
    if k >= 4:
        a = a.T
    for _ in range(k % 4):
        a = np.rot90(a)
    return a.tolist()


def d4_inv(g, k):
    a = np.array(g)
    for _ in range(k % 4):
        a = np.rot90(a, -1)
    if k >= 4:
        a = a.T
    return a.tolist()


def aplicar(g, k, perm):
    return [[perm[c] for c in fila] for fila in d4(g, k)]


def invertir(g, k, perm):
    inv = [0] * 10
    for i, p in enumerate(perm):
        inv[p] = i
    return d4_inv([[inv[c] for c in fila] for fila in g], k)


def build_vistas(pares, seed, n):
    # vistas de DECODE: produccion usa augment(n=2) = 8 D4 x 2 perms = 16 vistas.
    rng = random.Random(seed)
    vistas = []
    for v in range(n):
        if v == 0:
            k, perm, orden = 0, list(range(10)), list(range(len(pares)))
        else:
            k = rng.randrange(8)
            perm = rng.sample(range(10), 10)
            orden = rng.sample(range(len(pares)), len(pares))
        vistas.append({"k": k, "perm": perm, "orden": orden})
    return vistas


def build_vistas_train(pares, seed, n_perm=16):
    # vistas de TTT: produccion usa augment(n=16) = 8 transformaciones D4 x 16
    # permutaciones de color = 128 SECUENCIAS, cada una vista UNA vez (1 epoca).
    # El PRODUCTO CRUZADO importa: 128 pasos, no 16. La primera version de este banco
    # generaba 16 vistas sueltas -> 8x menos adaptacion -> el 4B no aprendia nada
    # (3 de 16 tareas con GT en alguna vista). El smoke del 2B no lo detecto porque
    # ese modelo memoriza el eval y resuelve sin necesitar TTT.
    rng = random.Random(seed)
    perms = [list(range(10))] + [rng.sample(range(10), 10) for _ in range(n_perm - 1)]
    vistas = []
    for k in range(8):
        for perm in perms:
            vistas.append({"k": k, "perm": perm,
                           "orden": rng.sample(range(len(pares)), len(pares))})
    return vistas


base = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
LCFG = LoraConfig(r=256, lora_alpha=32, use_rslora=True, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj",
                                  "embed_tokens", "lm_head"])
MODEL = get_peft_model(base, LCFG)
for _, p_ in MODEL.named_parameters():
    if p_.requires_grad:
        p_.data = p_.data.float()
MODEL.enable_input_require_grads()
MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
INIT = {k: v.detach().clone() for k, v in get_peft_model_state_dict(MODEL).items()}
try:
    import bitsandbytes as bnb
    OPT_CLS = bnb.optim.AdamW8bit
except Exception:
    OPT_CLS = torch.optim.AdamW
print("[s%d] modelo listo mem=%.1fGB (%.0fs)"
      % (SHARD, torch.cuda.memory_allocated() / 1e9, time.time() - T00), flush=True)


@torch.no_grad()
def prefill(toks, chunk=512):
    cache, out = None, None
    for i in range(0, len(toks), chunk):
        with sdpa_kernel(_SDPA):
            out = MODEL(input_ids=torch.tensor([toks[i:i + chunk]], device="cuda:0",
                                               dtype=torch.long),
                        past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
    return out.logits[:, -1], cache


def toks_to_grid(toks):
    rows, row = [], []
    for t in toks:
        if t == NL:
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
    return rows


@torch.no_grad()
def dfs(prompt, depth, budget):
    t0 = time.time()
    last, cache0 = prefill(prompt)
    leaves, path = [], []

    def rec(logits, score, cache, left):
        if time.time() - t0 > budget or left <= 1:
            return
        lf = logits.float()
        nll = (score + torch.logsumexp(lf, dim=-1, keepdim=True)
               - lf.index_select(-1, _TOKID)).cpu()
        cands = []
        for j, t in enumerate(ARC_TOKENS):
            sc = nll[0, j].item()
            if sc < THR:
                if t == EOS:
                    leaves.append((sc, list(path)))
                else:
                    cands.append((sc, t))
        for sc, t in sorted(cands):
            if time.time() - t0 > budget:
                return
            with sdpa_kernel(_SDPA):
                o = MODEL(input_ids=torch.tensor([[t]], device="cuda:0", dtype=torch.long),
                          past_key_values=cache, use_cache=True, return_dict=True)
            path.append(t)
            rec(o.logits[:, -1], sc, o.past_key_values, left - 1)
            path.pop()

    rec(last, 0.0, cache0, depth)
    return leaves


def entrenar_ttt(task, vistas_train):
    # UNA epoca sobre las vistas: calendario EXACTO de produccion (128 secuencias vistas
    # una vez). Recortarlo es el modo de fallo de ttt05 (26,39) -- no se toca.
    MODEL.config.use_cache = False
    MODEL.train()
    opt = OPT_CLS([p for p in MODEL.parameters() if p.requires_grad], lr=LR)
    seqs = []
    for v in vistas_train:
        toks, mask = [], []
        for idx in v["orden"]:
            p = task["train"][idx]
            t, m = pair_tokens_mask(aplicar(p["input"], v["k"], v["perm"]),
                                    aplicar(p["output"], v["k"], v["perm"]))
            toks += t
            mask += m
        if len(toks) <= MAX_SEQ:
            seqs.append((toks, mask))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(len(seqs), 1))
    for toks, mask in seqs:
        ids = torch.tensor([toks], device="cuda:0", dtype=torch.long)
        lab = torch.tensor([[t if m else -100 for t, m in zip(toks, mask)]],
                           device="cuda:0", dtype=torch.long)
        with sdpa_kernel(_SDPA):
            out = MODEL(input_ids=ids, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in MODEL.parameters() if p.requires_grad], 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
    del opt, sch
    MODEL.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    MODEL.eval()
    MODEL.config.use_cache = True
    return len(seqs)


RES = {"tareas": [], "modelo": "4b", "n_vistas": N_VISTAS, "budget_vista": BUDGET,
       "shard": SHARD}
for i, k in enumerate(MIS_KEYS):
    if time.time() - T00 > WALL_MAX:
        print("[s%d] WALL_MAX alcanzado, corto en %d/%d" % (SHARD, i, len(MIS_KEYS)),
              flush=True)
        break
    task = CHAL[k]
    gt = SOLS[k][0]
    pares = [(p["input"], p["output"]) for p in task["train"]]
    seed = int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 1024**2

    set_peft_model_state_dict(MODEL, {kk: v.clone() for kk, v in INIT.items()})
    v_train = build_vistas_train(pares, seed)
    assert len(v_train) == 128, "el TTT debe ver 128 secuencias como produccion"
    t0 = time.time()
    n_seqs = entrenar_ttt(task, v_train)
    t_ttt = time.time() - t0

    v_dec = build_vistas(pares, seed + 1, N_VISTAS)
    vistas_out, t0 = [], time.time()
    for v in v_dec:
        prompt = []
        for idx in v["orden"]:
            p = task["train"][idx]
            t, _ = pair_tokens_mask(aplicar(p["input"], v["k"], v["perm"]),
                                    aplicar(p["output"], v["k"], v["perm"]))
            prompt += t
        prompt += query_tokens(aplicar(task["test"][0]["input"], v["k"], v["perm"]))
        gt_v = grid_tokens(aplicar(gt, v["k"], v["perm"])) + [EOS]
        try:
            hojas = dfs(prompt, min(931, len(gt_v) + 40), BUDGET)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hojas = []
        cands = []
        for sc, camino in sorted(hojas, key=lambda x: x[0])[:8]:
            g = toks_to_grid(camino)
            if g is None:
                continue
            try:
                cands.append({"grid": invertir(g, v["k"], v["perm"]), "score": round(sc, 4)})
            except Exception:
                continue
        vistas_out.append(cands)
        torch.cuda.empty_cache()
    t_dec = time.time() - t0

    n_con_gt = sum(1 for cs in vistas_out if any(c["grid"] == gt for c in cs))
    RES["tareas"].append({"task": k, "gt": gt, "vistas": vistas_out, "n_seqs_ttt": n_seqs,
                          "ttt_s": round(t_ttt, 1), "decode_s": round(t_dec, 1),
                          "vistas_con_gt": n_con_gt})
    print("[s%d %d/%d] %s: GT en %d/%d vistas | cands/vista %.1f | ttt %.0fs dec %.0fs"
          % (SHARD, i + 1, len(MIS_KEYS), k, n_con_gt, N_VISTAS,
             sum(len(c) for c in vistas_out) / max(N_VISTAS, 1), t_ttt, t_dec), flush=True)
    json.dump(RES, open(OUT, "w"))          # persistencia INCREMENTAL por tarea

json.dump(RES, open(OUT, "w"))
print("[s%d] fin %.0fs" % (SHARD, time.time() - T00), flush=True)
'''

LANZADOR = '''import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = %s
WORKER_SRC = open("/kaggle/working/_worker_src.py").read()
open("/kaggle/working/worker.py", "w").write(WORKER_SRC)

procs = []
for i in range(4):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), SHARD=str(i), N_SHARDS="4",
               N_TAREAS="16", N_VISTAS="16", BUDGET_VISTA="45", WALL_MAX="4800",
               SUCIAS=json.dumps(SUCIAS), PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "/kaggle/working/worker.py"], env=env,
                         stdout=open("/kaggle/working/log%%d.txt" %% i, "w"),
                         stderr=subprocess.STDOUT)
    procs.append(p)
    print("lanzado worker %%d (GPU %%d)" %% (i, i), flush=True)

for i, p in enumerate(procs):
    p.wait()
    print("worker %%d termino rc=%%s (%%.0fs)" %% (i, p.returncode, time.time() - T0), flush=True)

tareas = []
for i in range(4):
    f = "/kaggle/working/shard%%d.json" %% i
    if os.path.exists(f):
        try:
            tareas += json.load(open(f))["tareas"]
        except Exception as e:
            print("shard %%d ilegible: %%s" %% (i, e), flush=True)
json.dump({"tareas": tareas, "modelo": "4b", "n_vistas": 16},
          open("/kaggle/working/augdose.json", "w"))

for i in range(4):
    print("\\n" + "=" * 20 + " LOG WORKER %%d " %% i + "=" * 20, flush=True)
    try:
        print(open("/kaggle/working/log%%d.txt" %% i).read()[-4000:], flush=True)
    except Exception as e:
        print("sin log: %%s" %% e, flush=True)

print("RESULT_JSON:" + json.dumps({
    "n": len(tareas),
    "vistas_con_gt": [t["vistas_con_gt"] for t in tareas],
    "wall_s": round(time.time() - T0)}), flush=True)
''' % json.dumps(SUCIAS)


def celda(src, tipo="code"):
    d = {"cell_type": tipo, "metadata": {},
         "source": src.splitlines(keepends=True)}
    if tipo == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d


MD = """# Banco de DOSIS de augmentaciones — 4B, L4x4 shardeado

Pregunta (prerregistrada): **¿la ESCASEZ de candidatos con pocas vistas predice que más
vistas ayuden?** Observable `productividad` = fracción de vistas que devuelven algún
candidato. En el smoke del 2B (n=4) su correlación con el beneficio de 4→16 vistas fue
**−0,924**, frente a −0,399 de la dispersión del voto, que quedó refutada.

**Lecturas prerreg:** r ≤ −0,6 con n=16 ⇒ el disparador es real y la reasignación de vistas
por tarea pasa a ser injerto candidato. −0,6 < r < −0,3 ⇒ débil, no injertar. r ≥ −0,3 ⇒
la palanca de dosis muere y se cierra la rama.

Decodifica 16 vistas UNA vez y persiste qué produjo cada una; la curva 2/4/8/12/16 sale
offline por submuestreo (`scripts/augdose_analysis.py`). TTT con el calendario exacto de
producción (1 época). 4 workers, uno por GPU, shard `KEYS[i::4]`, persistencia por tarea.
"""


def main():
    nb = {"cells": [celda(MD, "markdown"),
                    celda("%%writefile /kaggle/working/_worker_src.py\n" + WORKER.lstrip("\n")),
                    celda(LANZADOR)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    ruta = os.path.join(AQUI, "notebooks", "augdose_4b.ipynb")
    json.dump(nb, open(ruta, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # --- verificaciones locales (sin GPU) ---
    import ast
    ast.parse(WORKER)
    ast.parse(LANZADOR)
    assert "KEYS_ALL[SHARD::N_SHARDS]" in WORKER, "falta el shard"
    assert "CUDA_VISIBLE_DEVICES" in LANZADOR, "falta el aislamiento de GPU"
    assert WORKER.count("json.dump(RES, open(OUT") == 2, "falta persistencia incremental"
    assert "num_train_epochs" not in WORKER
    assert "hashlib.md5" in WORKER, "la semilla debe ser estable entre procesos"

    # FIDELIDAD DEL TTT: el fallo de la v1 fue entrenar con 16 secuencias en vez de 128.
    # Se ejecuta la funcion real del worker y se cuenta.
    ns = {"random": __import__("random")}
    ini = WORKER.index("def build_vistas_train")
    exec(WORKER[ini:WORKER.index("\n\n\n", ini)], ns)          # noqa: S102
    pares = [(0, 0)] * 3
    vt = ns["build_vistas_train"](pares, 123)
    assert len(vt) == 128, "el TTT debe ver 128 secuencias, vio %d" % len(vt)
    assert len({v["k"] for v in vt}) == 8, "faltan transformaciones D4"
    assert len({tuple(v["perm"]) for v in vt}) == 16, "faltan permutaciones de color"
    assert sum(1 for v in vt if v["k"] == 0 and v["perm"] == list(range(10))) == 1, \
        "la vista identidad debe aparecer exactamente una vez"
    # determinismo: misma semilla -> mismas vistas
    assert ns["build_vistas_train"](pares, 123) == vt, "build_vistas_train no es determinista"

    print("OK notebooks/augdose_4b.ipynb")
    print("  worker y lanzador compilan; shard, aislamiento de GPU y persistencia presentes")
    print("  TTT: %d secuencias (8 D4 x 16 perms) = calendario de produccion" % len(vt))


if __name__ == "__main__":
    main()
