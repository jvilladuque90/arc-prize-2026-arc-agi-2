#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] A/B DE FORMATO — ¿produccion alimenta al modelo con el formato equivocado?

Hallazgo del 18-ago. El tokenizador del campeon es **WordLevel con 0 merges y 13 entradas**;
en la secuencia exacta del formateador, `'user'` y `'assistant'` se tokenizan como lista VACIA:

    tk('<|im_start|>user\\n')      -> [14, 10]      (no [14, 11, 10])
    tk('user') -> []   tk('assistant') -> []

Consecuencias:
  A) El collator del TTT (`QwenDataCollatorForCompletionOnlyLM`) busca los ids 11 y 12 para
     marcar los tramos de assistant. Al no aparecer NUNCA, `start_idx` queda vacio y **todas
     las etiquetas quedan en -100**: el TTT de produccion no supervisa ni un solo token.
     (Verificado localmente replicando el collator sobre una secuencia real.)
  B) El vocabulario TIENE 'user'=11 y 'assistant'=12 y el collator depende de ellos, asi que
     el pipeline de entrenamiento original casi con seguridad SI los emitia. Si el modelo se
     entreno con ellos y la inferencia se los quita, produccion esta fuera de distribucion.

Indicio fuerte de (B): midiendo la NLL de la GT, la ruta de produccion (sin 11/12) da ~20 y la
ruta con 11/12 da ~0.2 sobre las mismas tareas y el mismo modelo — factor 100.

Este script lo mide limpio y en el mismo proceso, sobre el modelo ORIGINAL de sorokin:
  - `nll_prod`  : prompt tokenizado como en produccion (sin 11/12)
  - `nll_roles` : identico pero inyectando 11 tras `<|im_start|>` de user y 12 en el de assistant
  - pool del DFS al umbral de produccion en ambos formatos
  - y que devuelve la CE con TODAS las etiquetas en -100 (para saber si el TTT es no-op o NaN)

Lectura pre-registrada:
  - nll_roles << nll_prod y pool_roles > pool_prod en la mayoria de tareas -> el formato de
    produccion es el equivocado: es el hallazgo mas grande del proyecto y va directo a un A/B
    en Kaggle (inyectar los roles es un cambio de dos lineas en el formateador).
  - sin diferencia -> mi harness tenia un bug y se descartan sus cifras; T2a queda como buena.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_format_ab.py <KAGGLE_API_TOKEN>
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

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] descargas listas ({time.time()-T00:.0f}s)", flush=True)

import hashlib

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                             device_map="cuda:0", attn_implementation="sdpa")
MODEL.eval()
_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))
print(f"[setup] modelo cargado | tk('user')={TOK('user').input_ids} "
      f"tk('assistant')={TOK('assistant').input_ids}", flush=True)

# ¿que devuelve la CE si TODAS las etiquetas son -100? (define si el TTT es no-op o NaN)
with torch.no_grad():
    _ids = torch.tensor([[14, 10, 1, 2, 15]], device="cuda:0")
    _lab = torch.full_like(_ids, -100)
    _loss = MODEL(input_ids=_ids, labels=_lab).loss
print(f"[collator] CE con todas las etiquetas en -100 -> loss={_loss.item()} "
      f"(nan={bool(torch.isnan(_loss))})", flush=True)


def g2s(g):
    return "\n".join("".join(str(int(c)) for c in row) for row in g)


def prompt_prod(task):
    """Ruta EXACTA de produccion: texto -> tokenizador (que se come 'user'/'assistant')."""
    txt = "".join(f"<|im_start|>user\n{g2s(p['input'])}<|im_end|>"
                  f"<|im_start|>assistant\n{g2s(p['output'])}<|im_end|>" for p in task["train"])
    txt += (f"<|im_start|>user\n{g2s(task['test'][0]['input'])}<|im_end|>"
            f"<|im_start|>assistant\n")
    return TOK(txt).input_ids


def inject_roles(ids):
    """Inserta 11/12 tras cada <|im_start|>, alternando user/assistant como el formato."""
    out, turn = [], 0
    for t in ids:
        out.append(t)
        if t == IM_START:
            out.append(USER if turn % 2 == 0 else ASSIST)
            turn += 1
    return out


@torch.no_grad()
def nll_total(prompt, ans):
    ids = torch.tensor([prompt + ans], device="cuda:0", dtype=torch.long)
    if ids.shape[1] > 5000:
        return None
    with sdpa_kernel(_SDPA):
        logits = MODEL(ids).logits.float()
    n = len(prompt)
    lp = torch.log_softmax(logits[0, n - 1:-1], -1)
    v = -lp.gather(1, torch.tensor(ans, device="cuda:0").unsqueeze(1)).sum().item()
    del logits, lp
    return v


@torch.no_grad()
def dfs_pool(prompt, budget=60):
    t0 = time.time()
    ids = torch.tensor([prompt], device="cuda:0", dtype=torch.long)
    with sdpa_kernel(_SDPA):
        out = MODEL(ids, use_cache=True, return_dict=True)
    leaves = []

    def rec(logits, score, pos, cache, depth):
        if time.time() - t0 > budget or depth <= 1:
            return
        lf = logits.float()
        nll = (score + torch.logsumexp(lf, dim=-1, keepdim=True)
               - lf.index_select(-1, _TOKID)).cpu()
        cands = []
        for j, t in enumerate(ARC_TOKENS):
            sc = nll[0, j].item()
            if sc < THR:
                if t == EOS:
                    leaves.append(sc)
                else:
                    cands.append((sc, t))
        for sc, t in sorted(cands):
            if time.time() - t0 > budget:
                return
            with sdpa_kernel(_SDPA):
                o = MODEL(input_ids=torch.tensor([[t]], device="cuda:0", dtype=torch.long),
                          past_key_values=cache, use_cache=True, return_dict=True)
            rec(o.logits[:, -1], sc, pos + 1, o.past_key_values, depth - 1)

    rec(out.logits[:, -1], 0.0, ids.size(1), out.past_key_values, 931)
    return len(leaves)


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


TASKS = sorted([k for k in CHAL if is_holdout(k) and k not in AGI1],
               key=lambda k: cost(CHAL[k]))[:14]
print(f"[setup] {len(TASKS)} tareas nuevas | umbral {THR:.3f}", flush=True)

RES = {"loss_todo_ignorado": float(_loss.item()), "tasks": []}
for ti, tk in enumerate(TASKS):
    task = CHAL[tk]
    ans = TOK(g2s(SOLS[tk][0]) + "<|im_end|>").input_ids
    p_prod = prompt_prod(task)
    p_roles = inject_roles(p_prod)
    n_prod = nll_total(p_prod, ans)
    n_roles = nll_total(p_roles, ans)
    if n_prod is None or n_roles is None:
        print(f"[{ti+1}/{len(TASKS)}] {tk} saltada (contexto)", flush=True)
        continue
    pool_prod = dfs_pool(p_prod)
    pool_roles = dfs_pool(p_roles)
    row = {"task": tk, "L": len(ans), "nll_prod": round(n_prod, 3),
           "nll_roles": round(n_roles, 3),
           "pasa_prod": bool(n_prod < THR), "pasa_roles": bool(n_roles < THR),
           "pool_prod": pool_prod, "pool_roles": pool_roles}
    RES["tasks"].append(row)
    print(f"[{ti+1}/{len(TASKS)}] {tk}: L={row['L']:4d} | NLL prod={n_prod:8.3f} "
          f"roles={n_roles:8.3f} (x{n_prod/max(n_roles,1e-6):7.1f}) | pasa {row['pasa_prod']}"
          f"->{row['pasa_roles']} | pool {pool_prod}->{pool_roles}", flush=True)

rows = RES["tasks"]
if rows:
    summ = {"n": len(rows),
            "pasa_prod": sum(r["pasa_prod"] for r in rows),
            "pasa_roles": sum(r["pasa_roles"] for r in rows),
            "nll_prod_mediana": round(float(np.median([r["nll_prod"] for r in rows])), 3),
            "nll_roles_mediana": round(float(np.median([r["nll_roles"] for r in rows])), 3),
            "pool_prod_total": sum(r["pool_prod"] for r in rows),
            "pool_roles_total": sum(r["pool_roles"] for r in rows),
            "loss_todo_ignorado": float(_loss.item())}
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)

json.dump(RES, open("/content/format_ab_result.json", "w"), indent=1)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/format_ab_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale format ab", "id": f"{user}/arcscale-format-ab",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "format ab", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
