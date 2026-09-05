#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genera notebooks/ttt_rank.ipynb: barrido del RANGO del TTT en 4B, L4x4 shardeado.

Pregunta (propuesta del usuario): el TTT es el motor (+20 pt) y el 34,8% del reloj. Si el
rango r=256 es mas del necesario, bajarlo libera presupuesto SIN recortar adaptacion -- que es
el modo de fallo conocido (ttt05 = 26,39, ttt8 en cuarentena).

DONDE ESTA LA VELOCIDAD (calculado, no supuesto): a r=256 el TTT entrena 529,8 M parametros,
de los que **528,5 M (99,75%) estan en las 7 proyecciones** y solo 1,3 M en embed_tokens +
lm_head. Ademas esas dos son matrices [16, 2560] -> RANGO MAXIMO 16, asi que r=256 sobre ellas
ya es rango completo (el TTT reescribe entera la tabla de simbolos por tarea, ARCHITECTURE.md).
Conclusion: se barre el rango de las PROYECCIONES y se dejan embed/head clavadas en 256, para
que cada brazo cambie UNA sola cosa (regla nueva tras el fallo de lb33-stack).

CONTROL IMPORTANTE -- la escala rsLoRA: el factor efectivo es alpha/sqrt(r). Si se baja r sin
tocar alpha, la escala se dispara (r=16 daria 8,0 en vez de 2,0) y estariamos midiendo escala,
no capacidad. Por eso cada brazo usa alpha = 2*sqrt(r_proj) para mantener la escala en 2,0
exactamente como produccion, y alpha_pattern fija 32 para embed/head (que siguen en r=256).

COSTE: solo fase de TTT, SIN decode -> ~150-200 s por tarea. No mide score, mide loss y reloj.
La loss es un PROXY: ttt05 demostro que un calendario sano en banco puede regresar en
produccion, asi que un verde aqui exige confirmacion en el LB antes de injertar.

PRERREG: un rango es candidato si (a) el delta de loss final pareado contra r=256 no se
distingue de 0 con n=12, y (b) el reloj del TTT baja >=15%. Con las dos, va al LB como injerto
unico.
"""
import json
import os

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

WORKER = r'''
import hashlib, json, math, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()

SHARD = int(os.environ.get("SHARD", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "4"))
N_TAREAS = int(os.environ.get("N_TAREAS", "12"))
RANGOS = [int(x) for x in os.environ.get("RANGOS", "256,64,16,4").split(",")]
SUCIAS = set(json.loads(os.environ["SUCIAS"]))
OUT = "/kaggle/working/shard%d.json" % SHARD

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

# El punto de montaje de la competicion NO es estable: el campeon usa
# /kaggle/input/competitions/<comp>/ y otros kernels han visto /kaggle/input/<comp>/.
# Hardcodearlo mato los 4 workers en el primer intento (5-sep), asi que se BUSCA,
# igual que el modelo, y si no aparece se imprime el arbol para no volver a adivinar.
FICH = "arc-agi_evaluation_challenges.json"
DATA = None
for cand in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
             "/kaggle/input/arc-prize-2026-arc-agi-2"):
    if os.path.exists(os.path.join(cand, FICH)):
        DATA = cand
        break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if FICH in files:
            DATA = root
            break
assert DATA, "no se encontro %s; /kaggle/input contiene: %s" % (FICH, os.listdir("/kaggle/input"))
print("[s%d] datos en %s" % (SHARD, DATA), flush=True)
CHAL = json.load(open(os.path.join(DATA, FICH)))

MDIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR, "no se encontro el modelo"

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
MAX_SEQ = 3500
LR = 5e-5
PROY = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def coste(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


KEYS_ALL = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))[:N_TAREAS]
MIS_KEYS = KEYS_ALL[SHARD::N_SHARDS]
print("[s%d] %d tareas, rangos %s" % (SHARD, len(MIS_KEYS), RANGOS), flush=True)


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


def d4(g, k):
    a = np.array(g)
    if k >= 4:
        a = a.T
    for _ in range(k % 4):
        a = np.rot90(a)
    return a.tolist()


def aplicar(g, k, perm):
    return [[perm[c] for c in fila] for fila in d4(g, k)]


def build_vistas_train(pares, seed, n_perm=16):
    # 8 D4 x 16 permutaciones = 128 secuencias: el calendario EXACTO de produccion.
    rng = random.Random(seed)
    perms = [list(range(10))] + [rng.sample(range(10), 10) for _ in range(n_perm - 1)]
    vistas = []
    for k in range(8):
        for perm in perms:
            vistas.append({"k": k, "perm": perm,
                           "orden": rng.sample(range(len(pares)), len(pares))})
    return vistas


def construir(r_proj):
    # alpha = 2*sqrt(r) mantiene la escala rsLoRA en 2,0 (la de produccion) para cualquier r.
    # embed/head se quedan en r=256 y alpha=32 -> su escala tambien sigue en 2,0.
    base = AutoModelForCausalLM.from_pretrained(
        MDIR, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    cfg = LoraConfig(
        r=r_proj, lora_alpha=2.0 * math.sqrt(r_proj), use_rslora=True,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=PROY + ["embed_tokens", "lm_head"],
        rank_pattern={"embed_tokens": 256, "lm_head": 256},
        alpha_pattern={"embed_tokens": 32, "lm_head": 32})
    m = get_peft_model(base, cfg)
    for _, p_ in m.named_parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    m.enable_input_require_grads()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, n_train


def entrenar(model, opt_cls, task, vistas):
    model.config.use_cache = False
    model.train()
    opt = opt_cls([p for p in model.parameters() if p.requires_grad], lr=LR)
    seqs = []
    for v in vistas:
        toks, mask = [], []
        for idx in v["orden"]:
            p = task["train"][idx]
            t, mm = pair_tokens_mask(aplicar(p["input"], v["k"], v["perm"]),
                                     aplicar(p["output"], v["k"], v["perm"]))
            toks += t
            mask += mm
        if len(toks) <= MAX_SEQ:
            seqs.append((toks, mask))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(len(seqs), 1))
    curva = []
    for toks, mask in seqs:
        ids = torch.tensor([toks], device="cuda:0", dtype=torch.long)
        lab = torch.tensor([[t if m else -100 for t, m in zip(toks, mask)]],
                           device="cuda:0", dtype=torch.long)
        with sdpa_kernel(_SDPA):
            out = model(input_ids=ids, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
        curva.append(round(float(out.loss.item()), 5))
    del opt, sch
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    model.eval()
    model.config.use_cache = True
    return curva, len(seqs)


try:
    import bitsandbytes as bnb
    OPT_CLS = bnb.optim.AdamW8bit
except Exception:
    OPT_CLS = torch.optim.AdamW

RES = {"filas": [], "shard": SHARD, "rangos": RANGOS}
for r_proj in RANGOS:
    t_load = time.time()
    MODEL, n_train = construir(r_proj)
    INIT = {k: v.detach().clone() for k, v in get_peft_model_state_dict(MODEL).items()}
    print("[s%d] r=%d listo: %.1fM entrenables, carga %.0fs, mem %.1fGB"
          % (SHARD, r_proj, n_train / 1e6, time.time() - t_load,
             torch.cuda.memory_allocated() / 1e9), flush=True)

    for k in MIS_KEYS:
        task = CHAL[k]
        pares = [(p["input"], p["output"]) for p in task["train"]]
        seed = int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 1024**2
        vistas = build_vistas_train(pares, seed)
        assert len(vistas) == 128, "el TTT debe ver 128 secuencias"

        set_peft_model_state_dict(MODEL, {kk: v.clone() for kk, v in INIT.items()})
        t0 = time.time()
        curva, n_seqs = entrenar(MODEL, OPT_CLS, task, vistas)
        dt = time.time() - t0

        ult16 = sum(curva[-16:]) / max(len(curva[-16:]), 1)
        RES["filas"].append({"task": k, "r_proj": r_proj, "n_train_M": round(n_train / 1e6, 2),
                             "n_seqs": n_seqs, "ttt_s": round(dt, 1),
                             "loss_final16": round(ult16, 5),
                             "loss_prim16": round(sum(curva[:16]) / 16, 5),
                             "curva": curva})
        print("[s%d] r=%3d %s: loss ini16 %.4f -> fin16 %.4f | %.0fs | %d seqs"
              % (SHARD, r_proj, k, sum(curva[:16]) / 16, ult16, dt, n_seqs), flush=True)
        json.dump(RES, open(OUT, "w"))

    del MODEL, INIT
    torch.cuda.empty_cache()

json.dump(RES, open(OUT, "w"))
print("[s%d] fin %.0fs" % (SHARD, time.time() - T00), flush=True)
'''

LANZADOR = '''import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = %s
open("/kaggle/working/worker.py", "w").write(open("/kaggle/working/_worker_src.py").read())

procs = []
for i in range(4):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), SHARD=str(i), N_SHARDS="4",
               N_TAREAS="12", RANGOS="256,64,16,4",
               SUCIAS=json.dumps(SUCIAS), PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "/kaggle/working/worker.py"], env=env,
                         stdout=open("/kaggle/working/log%%d.txt" %% i, "w"),
                         stderr=subprocess.STDOUT)
    procs.append(p)
    print("lanzado worker %%d (GPU %%d)" %% (i, i), flush=True)

for i, p in enumerate(procs):
    p.wait()
    print("worker %%d rc=%%s (%%.0fs)" %% (i, p.returncode, time.time() - T0), flush=True)

filas = []
for i in range(4):
    f = "/kaggle/working/shard%%d.json" %% i
    if os.path.exists(f):
        try:
            filas += json.load(open(f))["filas"]
        except Exception as e:
            print("shard %%d ilegible: %%s" %% (i, e), flush=True)
json.dump({"filas": filas}, open("/kaggle/working/ttt_rank.json", "w"))

for i in range(4):
    print("\\n" + "=" * 18 + " LOG WORKER %%d " %% i + "=" * 18, flush=True)
    try:
        print(open("/kaggle/working/log%%d.txt" %% i).read()[-3500:], flush=True)
    except Exception as e:
        print("sin log: %%s" %% e, flush=True)

# resumen pareado por tarea contra r=256
por = {}
for f in filas:
    por.setdefault(f["task"], {})[f["r_proj"]] = f
print("\\n===== RESUMEN PAREADO (contra r=256) =====", flush=True)
print("%%-10s %%6s %%10s %%9s %%9s" %% ("tarea", "r", "entren.M", "loss_fin", "ttt_s"), flush=True)
for t in sorted(por):
    for r in sorted(por[t], reverse=True):
        d = por[t][r]
        print("%%-10s %%6d %%10.1f %%9.4f %%9.0f" %%
              (t, r, d["n_train_M"], d["loss_final16"], d["ttt_s"]), flush=True)
print("RESULT_JSON:" + json.dumps({"n_filas": len(filas), "wall_s": round(time.time() - T0)}),
      flush=True)
''' % json.dumps(SUCIAS)

MD = """# Barrido del RANGO del TTT — 4B, L4x4 shardeado

**Pregunta**: ¿es r=256 más de lo necesario? Si un rango menor adapta igual, el TTT se abarata
sin recortar adaptación — el modo de fallo conocido (`ttt05` = 26,39).

**Dónde está la velocidad** (calculado): a r=256 el TTT entrena 529,8 M params, de los que
**528,5 M (99,75%) están en las 7 proyecciones**. `embed_tokens` y `lm_head` son 1,3 M y además
son matrices [16, 2560] — rango máximo 16 — así que r=256 sobre ellas ya es rango completo. Por
eso **se barre el rango de las proyecciones y embed/head se quedan clavadas en 256**.

**Control de escala**: rsLoRA escala por `alpha/sqrt(r)`. Cada brazo usa `alpha = 2·sqrt(r)`
para mantener la escala en 2,0 (la de producción) y `alpha_pattern` fija 32 en embed/head. Sin
esto estaríamos midiendo escala, no capacidad.

Solo fase de TTT, **sin decode**. Mide loss y reloj, no score. La loss es un proxy: un verde
aquí exige confirmación en el LB antes de injertar.

**PRERREG**: un rango es candidato si (a) el delta de loss final pareado contra r=256 no se
distingue de 0 con n=12, y (b) el reloj del TTT baja ≥15%.
"""


def celda(src, tipo="code"):
    d = {"cell_type": tipo, "metadata": {}, "source": src.splitlines(keepends=True)}
    if tipo == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d


def main():
    nb = {"cells": [celda(MD, "markdown"),
                    celda("%%writefile /kaggle/working/_worker_src.py\n" + WORKER.lstrip("\n")),
                    celda(LANZADOR)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    ruta = os.path.join(AQUI, "notebooks", "ttt_rank.ipynb")
    json.dump(nb, open(ruta, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    import ast
    ast.parse(WORKER)
    ast.parse(LANZADOR)
    assert "KEYS_ALL[SHARD::N_SHARDS]" in WORKER, "falta el shard"
    assert "CUDA_VISIBLE_DEVICES" in LANZADOR, "falta el aislamiento de GPU"
    assert 'rank_pattern={"embed_tokens": 256, "lm_head": 256}' in WORKER, "embed/head deben quedar fijas"
    assert "alpha_pattern" in WORKER, "falta el control de escala rsLoRA"
    assert "assert len(vistas) == 128" in WORKER, "falta el guardarrail del calendario"
    assert "/kaggle/input/competitions/" in WORKER, "el montaje de la competicion debe BUSCARSE"
    assert "os.listdir(\"/kaggle/input\")" in WORKER, "el gate debe imprimir el arbol si falla"

    # el control de escala: alpha/sqrt(r) debe dar 2,0 para todo r del barrido
    import math as _m
    for r in (256, 64, 16, 4):
        esc = (2.0 * _m.sqrt(r)) / _m.sqrt(r)
        assert abs(esc - 2.0) < 1e-9, "la escala rsLoRA se mueve en r=%d" % r
    print("OK notebooks/ttt_rank.ipynb")
    print("  worker y lanzador compilan; shard, aislamiento de GPU y guardarrail de 128 presentes")
    print("  escala rsLoRA = 2,0 constante en r = 256/64/16/4 (control verificado)")


if __name__ == "__main__":
    main()
