#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genera notebooks/ttt_regla.ipynb — ¿el 4B con TTT sabe APLICAR la regla a una entrada nueva?

Origen: propuesta del usuario de un paso intermedio "entender→condensar→especificar" en espacio
de REJILLA, la única variante expresable con un vocabulario de 16 tokens (TTT_PLAN §43) y la
única de toda la familia del razonamiento que entra en el presupuesto (+3-5% contra +4,8%).

Antes de invertir en un SFT de formato nuevo, la pregunta barata: **¿el modelo YA sabe hacerlo?**

DISEÑO — dos decodes por tarea sobre el MISMO modelo adaptado, que es lo que lo hace diagnóstico:
  A) se retira el ÚLTIMO par de train; el TTT entrena solo con los N-1 restantes; luego se
     decodifica la ENTRADA retirada y se compara con su salida conocida.
     -> mide aplicar la regla a una entrada NUEVA pero de la MISMA distribución que las demos.
  B) se decodifica la ENTRADA DE TEST y se compara con la solución del eval.
     -> mide la tarea real.

LECTURA PRERREGISTRADA:
  * A >> B  (A−B ≥ 3 tareas de 12): la regla SÍ se aprende y se aplica; lo que falla es la
    novedad de la entrada de test. Un paso intermedio en rejilla —que es justo "aplicar la regla
    a algo"— NO desbloquearía el test. Rama del razonamiento condensado: CERRAR.
  * A ≈ B: el cuello es aprender la regla, no aplicarla. Un paso intermedio que fuerce a
    explicitar la transformación podría ayudar. Rama: SEGUIR, y el SFT se justifica.
  * A y B ambos ~0: el TTT con N-1 demos no basta; el banco no discrimina y habría que repetir
    sin retirar par (perdiendo el control de "entrada nueva").

Coste: 12 tareas, shard ×4, TTT fiel de 128 secuencias + 8 decodes por tarea ≈ 0,9 h-cuota.
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
N_VISTAS = int(os.environ.get("N_VISTAS", "4"))
BUDGET = float(os.environ.get("BUDGET_VISTA", "45"))
SUCIAS = set(json.loads(os.environ["SUCIAS"]))
OUT = "/kaggle/working/shard%d.json" % SHARD

import numpy as np
import torch
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

# el montaje de la competicion NO es estable: se busca (leccion del 5-sep)
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
assert DATA, "no se encontro %s; /kaggle/input: %s" % (FICH, os.listdir("/kaggle/input"))
CHAL = json.load(open(os.path.join(DATA, FICH)))
SOLS = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))
print("[s%d] datos en %s" % (SHARD, DATA), flush=True)

MDIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR, "no se encontro el modelo"

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))          # identico a max_score de produccion
MAX_SEQ, LR = 3500, 5e-5


def coste(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


# >=3 pares de train para que al retirar uno queden al menos 2 demostraciones
KEYS_ALL = sorted([k for k in CHAL if k not in SUCIAS and len(CHAL[k]["train"]) >= 3],
                  key=lambda k: coste(CHAL[k]))[:N_TAREAS]
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


def build_vistas_train(pares, seed, n_perm=16):
    # 8 D4 x 16 permutaciones = 128 secuencias: calendario EXACTO de produccion
    rng = random.Random(seed)
    perms = [list(range(10))] + [rng.sample(range(10), 10) for _ in range(n_perm - 1)]
    vistas = []
    for k in range(8):
        for perm in perms:
            vistas.append({"k": k, "perm": perm,
                           "orden": rng.sample(range(len(pares)), len(pares))})
    return vistas


def build_vistas_dec(pares, seed, n):
    rng = random.Random(seed)
    v = []
    for i in range(n):
        if i == 0:
            v.append({"k": 0, "perm": list(range(10)), "orden": list(range(len(pares)))})
        else:
            v.append({"k": rng.randrange(8), "perm": rng.sample(range(10), 10),
                      "orden": rng.sample(range(len(pares)), len(pares))})
    return v


base = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
LCFG = LoraConfig(r=256, lora_alpha=32, use_rslora=True, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                  "up_proj", "down_proj", "embed_tokens", "lm_head"])
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


def decodificar(demos, consulta, gt, vistas):
    """Devuelve (resuelta, vistas_con_gt, cands_por_vista). Des-transforma al marco original."""
    con_gt, n_c = 0, 0
    for v in vistas:
        prompt = []
        for idx in v["orden"]:
            inp, out = demos[idx]
            t, _ = pair_tokens_mask(aplicar(inp, v["k"], v["perm"]),
                                    aplicar(out, v["k"], v["perm"]))
            prompt += t
        prompt += query_tokens(aplicar(consulta, v["k"], v["perm"]))
        gt_v = grid_tokens(aplicar(gt, v["k"], v["perm"])) + [EOS]
        try:
            hojas = dfs(prompt, min(931, len(gt_v) + 40), BUDGET)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hojas = []
        hit = False
        for sc, camino in sorted(hojas, key=lambda x: x[0])[:8]:
            g = toks_to_grid(camino)
            if g is None:
                continue
            n_c += 1
            try:
                if invertir(g, v["k"], v["perm"]) == gt:
                    hit = True
            except Exception:
                pass
        con_gt += 1 if hit else 0
        torch.cuda.empty_cache()
    return (con_gt > 0), con_gt, n_c / max(len(vistas), 1)


def entrenar(task_pares, vistas):
    MODEL.config.use_cache = False
    MODEL.train()
    opt = OPT_CLS([p for p in MODEL.parameters() if p.requires_grad], lr=LR)
    seqs = []
    for v in vistas:
        toks, mask = [], []
        for idx in v["orden"]:
            inp, out = task_pares[idx]
            t, m = pair_tokens_mask(aplicar(inp, v["k"], v["perm"]),
                                    aplicar(out, v["k"], v["perm"]))
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
        torch.nn.utils.clip_grad_norm_([p for p in MODEL.parameters() if p.requires_grad], 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
    del opt, sch
    MODEL.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    MODEL.eval()
    MODEL.config.use_cache = True
    return len(seqs)


RES = {"filas": [], "shard": SHARD}
for i, k in enumerate(MIS_KEYS):
    task = CHAL[k]
    pares = [(p["input"], p["output"]) for p in task["train"]]
    # se RETIRA el ultimo par: el TTT no lo ve y su entrada queda como consulta nueva
    demos, (hin, hout) = pares[:-1], pares[-1]
    seed = int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 1024**2

    set_peft_model_state_dict(MODEL, {kk: v.clone() for kk, v in INIT.items()})
    vtr = build_vistas_train(demos, seed)
    assert len(vtr) == 128, "el TTT debe ver 128 secuencias"
    t0 = time.time()
    try:
        n_seqs = entrenar(demos, vtr)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[s%d] OOM en TTT de %s, salto" % (SHARD, k), flush=True)
        continue
    t_ttt = time.time() - t0

    vdec = build_vistas_dec(demos, seed + 1, N_VISTAS)
    t0 = time.time()
    okA, gA, cA = decodificar(demos, hin, hout, vdec)              # A: par retirado
    okB, gB, cB = decodificar(demos, task["test"][0]["input"], SOLS[k][0], vdec)  # B: test
    t_dec = time.time() - t0

    RES["filas"].append({"task": k, "n_demos": len(demos), "n_seqs": n_seqs,
                         "A_par_retirado": okA, "A_vistas_gt": gA, "A_cands": round(cA, 2),
                         "B_test": okB, "B_vistas_gt": gB, "B_cands": round(cB, 2),
                         "ttt_s": round(t_ttt, 1), "dec_s": round(t_dec, 1)})
    print("[s%d %d/%d] %s: A(par retirado)=%s %d/%d vistas | B(test)=%s %d/%d | ttt %.0fs dec %.0fs"
          % (SHARD, i + 1, len(MIS_KEYS), k, okA, gA, N_VISTAS, okB, gB, N_VISTAS, t_ttt, t_dec),
          flush=True)
    json.dump(RES, open(OUT, "w"))

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
               N_TAREAS="12", N_VISTAS="4", BUDGET_VISTA="45",
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
json.dump({"filas": filas}, open("/kaggle/working/ttt_regla.json", "w"))

for i in range(4):
    print("\\n" + "=" * 18 + " LOG WORKER %%d " %% i + "=" * 18, flush=True)
    try:
        print(open("/kaggle/working/log%%d.txt" %% i).read()[-3000:], flush=True)
    except Exception as e:
        print("sin log: %%s" %% e, flush=True)

A = sum(1 for f in filas if f["A_par_retirado"])
B = sum(1 for f in filas if f["B_test"])
n = len(filas)
print("\\n===== RESULTADO =====", flush=True)
print("%%-10s %%16s %%10s" %% ("tarea", "A par retirado", "B test"), flush=True)
for f in sorted(filas, key=lambda x: x["task"]):
    print("%%-10s %%10s %%2d/4 %%5s %%2d/4" %% (f["task"], f["A_par_retirado"], f["A_vistas_gt"],
                                          f["B_test"], f["B_vistas_gt"]), flush=True)
print("\\nA (aplicar la regla a entrada nueva de la misma distribucion): %%d/%%d" %% (A, n), flush=True)
print("B (la tarea real, entrada de test) .........................: %%d/%%d" %% (B, n), flush=True)
print("\\nPRERREG: A-B >= 3 => la regla SI se aplica, falla la novedad del test => CERRAR rama", flush=True)
print("         A ~ B    => el cuello es APRENDER la regla => el paso intermedio puede ayudar", flush=True)
print("RESULT_JSON:" + json.dumps({"A": A, "B": B, "n": n, "wall_s": round(time.time() - T0)}),
      flush=True)
''' % json.dumps(SUCIAS)

MD = """# ¿El 4B con TTT sabe APLICAR la regla a una entrada nueva?

Paso previo y barato antes de invertir en un SFT de formato nuevo (TTT_PLAN §43). Dos decodes
sobre el **mismo** modelo adaptado:

- **A** — se retira el último par de train, el TTT entrena solo con los N−1 restantes, y se
  decodifica la **entrada retirada** contra su salida conocida. Mide *aplicar la regla a una
  entrada nueva de la misma distribución*.
- **B** — se decodifica la **entrada de test** contra la solución del eval. Mide la tarea real.

**Prerreg**: `A−B ≥ 3` de 12 ⇒ la regla se aplica bien y lo que falla es la novedad del test, con
lo que un paso intermedio en rejilla no desbloquearía nada ⇒ **cerrar**. `A ≈ B` ⇒ el cuello es
aprender la regla ⇒ el paso intermedio puede ayudar y el SFT se justifica.
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
    ruta = os.path.join(AQUI, "notebooks", "ttt_regla.ipynb")
    json.dump(nb, open(ruta, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    import ast
    ast.parse(WORKER)
    ast.parse(LANZADOR)
    assert "KEYS_ALL[SHARD::N_SHARDS]" in WORKER
    assert "CUDA_VISIBLE_DEVICES" in LANZADOR
    assert "assert len(vtr) == 128" in WORKER, "falta el guardarrail del calendario"
    assert "/kaggle/input/competitions/" in WORKER, "el montaje debe buscarse"
    assert WORKER.count("torch.cuda.OutOfMemoryError") == 2, "faltan guardas de OOM"
    assert "pares[:-1], pares[-1]" in WORKER, "el par retirado debe salir del TTT"
    # el par retirado NO puede aparecer en las demos que alimentan el TTT ni el prompt
    assert "entrenar(demos, vtr)" in WORKER and "build_vistas_train(demos" in WORKER

    # inversion D4 verificada localmente antes de gastar GPU
    import numpy as _np, random as _rnd
    ns = {"np": _np}
    for f in ("def d4(", "def d4_inv("):
        i = WORKER.index(f)
        exec(WORKER[i:WORKER.index("\n\n\n", i)], ns)          # noqa: S102
    _r = _rnd.Random(0)
    for _ in range(300):
        h, w = _r.randint(1, 6), _r.randint(1, 6)
        g = [[_r.randint(0, 9) for _ in range(w)] for _ in range(h)]
        for k in range(8):
            assert ns["d4_inv"](ns["d4"](g, k), k) == g, "la inversa D4 falla en k=%d" % k
    print("OK notebooks/ttt_regla.ipynb")
    print("  compila; shard x4, guardarrail de 128, busqueda del dataset y guardas de OOM")
    print("  inversa D4 verificada 300/300 antes de gastar GPU")


if __name__ == "__main__":
    main()
