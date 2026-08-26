#!/usr/bin/env python
"""[Colab T4 / Kaggle L4] Banco de DOSIS DE AUGMENTACIONES: 16 vistas decodificadas UNA vez.

Palanca viva unica tras cerrar pasos, perdidas ponderadas y seleccion (TTT_PLAN §17-§20).
Produccion reparte augmentaciones uniformes a todas las tareas: 128 al entrenamiento del TTT
(`augment(n=16)` = 8 D4 x 16 permutaciones) y 16 al decode (`augment(n=2)`). Los perfiles de
voto medidos muestran dos regimenes muy distintos -- dominio claro (17,51 vs 5,74) frente a
empate tecnico (1,457 / 1,417 / 1,408) -- que probablemente piden presupuestos distintos.

DISENO: la dosis NO se toca en el TTT (recortar adaptacion es el modo de fallo que ya costo
ttt05 = 26,39 y tiene a ttt8 en cuarentena). Se toca en el DECODE, que ademas es el bote mas
grande (57,2% del presupuesto contra 34,8% del TTT).

CLAVE DE COSTE: no se corre un experimento por dosis. Se decodifican las 16 vistas UNA vez y se
persiste QUE produjo cada una; la curva 2/4/8/12/16 y la dispersion salen OFFLINE por
submuestreo (`scripts/augdose_analysis.py`, ya testeado sin GPU). Una pasada -> barrido
completo, reutilizable para cualquier regla de agregacion futura.

MODELO=2b (T4, por defecto): validacion de MECANISMO. El 2B memoriza el eval, asi que sus
numeros NO son veredicto -- sirven para comprobar que la des-transformacion, la persistencia y
el submuestreo funcionan. MODELO=4b (L4): el veredicto.

Uso Colab:  colab --auth=adc new -s dose --gpu T4  +  upload + exec desacoplado (ver CLAUDE.md)
"""
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1] o por entorno"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

MODELO = os.environ.get("MODELO", "2b")
N_TAREAS = int(os.environ.get("N_TAREAS", "6"))
N_VISTAS = int(os.environ.get("N_VISTAS", "16"))
BUDGET = float(os.environ.get("BUDGET_VISTA", "25"))
DS = f"{os.environ.get('KAGGLE_USERNAME', 'juliancamilovilla')}/arcscale-augdose-{MODELO}"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "torchao>=0.16"],
               check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)
_SLUG = ("sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1" if MODELO == "4b"
         else "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1")
subprocess.run(["kaggle", "models", "instances", "versions", "download", _SLUG,
                "-p", "/content/m", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR
print(f"[setup] {MODELO} en {MDIR} ({time.time()-T00:.0f}s)", flush=True)

import numpy as np
import torch
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))
MAX_SEQ = 3500
LR = 5e-5

CHAL = json.load(open("/content/arc-agi_evaluation_challenges.json"))
SOLS = json.load(open("/content/arc-agi_evaluation_solutions.json"))

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))


def coste(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


KEYS = sorted([k for k in CHAL if k not in AGI1], key=lambda k: coste(CHAL[k]))[:N_TAREAS]
print(f"[setup] {len(KEYS)} tareas | {N_VISTAS} vistas | budget {BUDGET}s/vista", flush=True)


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


# --- D4 + permutacion de color, con inversa verificada (300/300 en el test local) ---

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
    """n vistas deterministas. La vista 0 es SIEMPRE la identidad (comparabilidad con los
    bancos anteriores, que decodificaban solo esa)."""
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


base = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=(torch.bfloat16 if MODELO == "4b" else torch.float16),
    device_map="cuda:0", attn_implementation="sdpa")
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
except Exception:                                            # noqa: BLE001
    OPT_CLS = torch.optim.AdamW
print(f"[setup] modelo listo | mem={torch.cuda.memory_allocated()/1e9:.1f}GB "
      f"({time.time()-T00:.0f}s)", flush=True)


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
    """UNA epoca sobre las vistas de entrenamiento (calendario EXACTO de produccion)."""
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


RES = {"tareas": [], "modelo": MODELO, "n_vistas": N_VISTAS, "budget_vista": BUDGET}
for i, k in enumerate(KEYS):
    task = CHAL[k]
    gt = SOLS[k][0]
    pares = [(p["input"], p["output"]) for p in task["train"]]
    seed = int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 1024**2   # estable entre procesos

    set_peft_model_state_dict(MODEL, {kk: v.clone() for kk, v in INIT.items()})
    v_train = build_vistas(pares, seed, 16)          # como produccion: augment(n=16)
    t0 = time.time()
    n_seqs = entrenar_ttt(task, v_train)
    t_ttt = time.time() - t0

    # --- decode de N_VISTAS, persistiendo QUE produjo cada una ---
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
            try:                       # des-transformar al marco original
                cands.append({"grid": invertir(g, v["k"], v["perm"]), "score": round(sc, 4)})
            except Exception:                                # noqa: BLE001
                continue
        vistas_out.append(cands)
        torch.cuda.empty_cache()
    t_dec = time.time() - t0

    n_con_gt = sum(1 for cs in vistas_out if any(c["grid"] == gt for c in cs))
    RES["tareas"].append({"task": k, "gt": gt, "vistas": vistas_out, "n_seqs_ttt": n_seqs,
                          "ttt_s": round(t_ttt, 1), "decode_s": round(t_dec, 1),
                          "vistas_con_gt": n_con_gt})
    print(f"[{i+1}/{len(KEYS)}] {k}: vistas con GT {n_con_gt}/{N_VISTAS} | "
          f"cands/vista {sum(len(c) for c in vistas_out)/max(N_VISTAS,1):.1f} | "
          f"ttt {t_ttt:.0f}s decode {t_dec:.0f}s", flush=True)
    json.dump(RES, open("/content/augdose.json", "w"))

    try:                                   # persistencia incremental a Kaggle
        os.makedirs("/content/persist", exist_ok=True)
        json.dump(RES, open("/content/persist/augdose.json", "w"))
        json.dump({"title": f"arcscale augdose {MODELO}", "id": DS,
                   "licenses": [{"name": "CC0-1.0"}]},
                  open("/content/persist/dataset-metadata.json", "w"))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                           capture_output=True, text=True)
        if "error" in (r.stdout + r.stderr).lower():
            subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                            "-m", k, "-q"], check=False, capture_output=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[persist] fallo: {e}", flush=True)

print("RESULT_JSON:" + json.dumps({
    "n": len(RES["tareas"]), "modelo": MODELO,
    "vistas_con_gt": [t["vistas_con_gt"] for t in RES["tareas"]]}), flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
