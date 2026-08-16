#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Bench del BORRADOR-POR-COPIA para decodificacion especulativa (4B real).

Idea (16-ago): el decode es el 57.2% del presupuesto y es SECUENCIAL (un forward por celda).
La especulacion propone k tokens de un borrador y los verifica en UN solo forward; lo aceptado
es exactamente lo que el modelo habria producido -> LOSSLESS por construccion (no toca pesos,
tokenizador ni calibracion: el fallo de D4).

El borrador aqui no cuesta nada: es la PROPIA GRID DE ENTRADA. Medido en el eval AGI-2:
el 72% de las salidas tienen las mismas dimensiones que su entrada y, de esas, la mediana del
85% de las celdas es identica. O sea que en las regiones que la transformacion no toca el
borrador acierta solo, sin modelo auxiliar ni entrenamiento.

Este bench mide sobre el 4B REAL, por tarea:
  1. tasa de aceptacion del borrador contra el argmax del modelo (lo que el DFS recorre primero)
  2. densidad de BIFURCACIONES del DFS al umbral de produccion -log(0.2) (cuantas posiciones
     tienen >=2 tokens bajo el umbral -> ramas que hay que encolar)
  3. forwards simulados con ventana k in {4,8,16,32} vs el baseline de 1 forward/token
  4. VALIDACION DEL MODELO DE COSTE: ¿un forward de 16 tokens con cache cuesta lo mismo que uno
     de 1 token? Si no, la reduccion de forwards no se traduce en tiempo y la palanca muere.
  5. techo: la misma aceptacion calculada contra la GT (si el modelo fuera perfecto)

Lecturas pre-registradas:
  - aceptacion >=0.7 y coste(16 tok)/coste(1 tok) <=1.3 -> VERDE: forwards /3 o mejor en el
    camino principal -> injerto candidato (implementar especulacion dentro de turbo_dfs).
  - aceptacion 0.4-0.7 -> AMARILLO: ganancia real pero moderada; decide el coste de ingenieria.
  - aceptacion <0.4 o coste(16)/coste(1) >2 -> ROJO: cerrar la rama con dato, como dfsbatch.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_specdraft_bench.py <KAGGLE_API_TOKEN>
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

# las 24 tareas NUEVAS del holdout (no-AGI1): misma poblacion que D1-bis/D2-bis
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
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                             device_map="cuda:0")
MODEL.eval()
print(f"[setup] 4B cargado | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
NL_ID, EOS_ID = 10, 15
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
MAX_SCORE = -float(np.log(0.2))          # umbral del campeon
MAX_NEW = 931                            # fmt_reply(30x30) + 1


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def build_prompt(task):
    txt = "".join(f"<|im_start|>user\n{g2s(p['input'])}<|im_end|>"
                  f"<|im_start|>assistant\n{g2s(p['output'])}<|im_end|>"
                  for p in task["train"])
    txt += (f"<|im_start|>user\n{g2s(task['test'][0]['input'])}<|im_end|>"
            f"<|im_start|>assistant\n")
    return TOK.encode(txt)


def draft_token(r, c, inp):
    """El borrador = reproducir la grid de ENTRADA tal cual, luego EOS. Coste cero."""
    H = len(inp)
    if r >= H:
        return EOS_ID
    row = inp[r]
    if c < len(row):
        return int(row[c])
    return EOS_ID if r == H - 1 else NL_ID


def gt_tokens(grid):
    out = []
    for i, row in enumerate(grid):
        out.extend(int(x) for x in row)
        out.append(NL_ID if i < len(grid) - 1 else EOS_ID)
    return out


@torch.no_grad()
def greedy_trace(prompt_ids, inp):
    """Recorre el camino que el DFS explora PRIMERO (argmax restringido a ARC_TOKENS) y anota,
    posicion a posicion: si el borrador acierta y cuantas ramas quedan bajo el umbral."""
    ids = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    pos = ids.size(1)
    run_nll, r, c = 0.0, 0, 0
    matches, branches, toks = [], [], []
    for _ in range(MAX_NEW):
        lf = logits.float()
        nll = (torch.logsumexp(lf, dim=-1, keepdim=True) - lf.index_select(-1, _TOKID))[0]
        j = int(nll.argmin())
        tok = ARC_TOKENS[j]
        branches.append(int(((run_nll + nll) < MAX_SCORE).sum().item()))
        matches.append(tok == draft_token(r, c, inp))
        toks.append(tok)
        run_nll += float(nll[j].item())
        if tok == EOS_ID:
            break
        if tok == NL_ID:
            r, c = r + 1, 0
        else:
            c += 1
        out = MODEL(input_ids=torch.tensor([[tok]], device="cuda:0", dtype=torch.long),
                    position_ids=torch.full((1, 1), pos, device="cuda:0"),
                    past_key_values=past, use_cache=True, return_dict=True)
        past, logits, pos = out.past_key_values, out.logits[:, -1], pos + 1
    return matches, branches, toks


def sim_forwards(matches, k, stop_on_branch=None):
    """Forwards necesarios para recorrer el camino principal con ventana k.
    Cada forward verifica k tokens del borrador y regala 1 token del modelo."""
    i, fwd, n = 0, 0, len(matches)
    while i < n:
        fwd += 1
        acc = 0
        while acc < k and i + acc < n and matches[i + acc]:
            if stop_on_branch is not None and stop_on_branch[i + acc]:
                break
            acc += 1
        i += acc + 1
    return fwd


@torch.no_grad()
def cost_model(prompt_ids):
    """¿Cuesta lo mismo un forward de 1 token que uno de 16 (con cache)? Si no, la reduccion
    de forwards NO se traduce en tiempo y la palanca muere aqui mismo."""
    ids = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
    base = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    out = {}
    for width in (1, 4, 8, 16, 32):
        past = base.past_key_values
        chunk = torch.zeros((1, width), device="cuda:0", dtype=torch.long)
        pos = torch.arange(ids.size(1), ids.size(1) + width, device="cuda:0").view(1, -1)
        for _ in range(3):     # warmup
            MODEL(input_ids=chunk, position_ids=pos, past_key_values=past,
                  use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            MODEL(input_ids=chunk, position_ids=pos, past_key_values=past,
                  use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        out[width] = round((time.time() - t0) / 20 * 1000, 2)
    return out


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
NEW = [k for k in HOLD if k not in AGI1]
TASKS = NEW[:14]
print(f"[setup] {len(TASKS)} tareas nuevas | umbral {MAX_SCORE:.3f}", flush=True)

RES = {"tasks": [], "cost_model_ms": None}
for ti, tk in enumerate(TASKS):
    task = CHAL[tk]
    inp = task["test"][0]["input"]
    gt = SOLS[tk][0]
    prompt = build_prompt(task)
    if len(prompt) > 6000:
        print(f"[{ti+1}/{len(TASKS)}] {tk} saltada (ctx {len(prompt)})", flush=True)
        continue
    if RES["cost_model_ms"] is None:
        RES["cost_model_ms"] = cost_model(prompt)
        print(f"[coste] ms por forward segun ancho: {RES['cost_model_ms']}", flush=True)
    t0 = time.time()
    matches, branches, toks = greedy_trace(prompt, inp)
    secs = time.time() - t0
    n = len(matches)
    acc = float(np.mean(matches)) if n else 0.0
    brs = [b >= 2 for b in branches]
    # techo: aceptacion si el modelo produjera la GT
    gts = gt_tokens(gt)
    r, c, gm = 0, 0, []
    for t in gts:
        gm.append(t == draft_token(r, c, inp))
        if t == NL_ID:
            r, c = r + 1, 0
        elif t != EOS_ID:
            c += 1
    row = {"task": tk, "ctx": len(prompt), "n_steps": n, "secs": round(secs, 1),
           "aceptacion": round(acc, 3),
           "aceptacion_techo_gt": round(float(np.mean(gm)), 3),
           "bifurcaciones": round(float(np.mean(brs)), 3),
           "salida_ok": bool(len(toks) < MAX_NEW),
           "fwd_base": n}
    for k in (4, 8, 16, 32):
        row[f"fwd_k{k}"] = sim_forwards(matches, k)
        row[f"fwd_k{k}_conserv"] = sim_forwards(matches, k, stop_on_branch=brs)
    RES["tasks"].append(row)
    print(f"[{ti+1}/{len(TASKS)}] {tk}: {n} pasos {secs:.0f}s | acepta {acc:.2f} "
          f"(techo GT {row['aceptacion_techo_gt']:.2f}) | ramas {row['bifurcaciones']:.2f} | "
          f"fwd k16 {row['fwd_k16']} vs {n} = x{n/max(row['fwd_k16'],1):.1f}", flush=True)

rows = RES["tasks"]
if rows:
    tot = sum(r["fwd_base"] for r in rows)
    summ = {"n_tareas": len(rows),
            "aceptacion_media": round(float(np.mean([r["aceptacion"] for r in rows])), 3),
            "aceptacion_techo_gt": round(float(np.mean([r["aceptacion_techo_gt"] for r in rows])), 3),
            "bifurcaciones_media": round(float(np.mean([r["bifurcaciones"] for r in rows])), 3),
            "pasos_totales": tot,
            "coste_ms": RES["cost_model_ms"]}
    for k in (4, 8, 16, 32):
        summ[f"speedup_k{k}"] = round(tot / max(sum(r[f"fwd_k{k}"] for r in rows), 1), 2)
        summ[f"speedup_k{k}_conserv"] = round(
            tot / max(sum(r[f"fwd_k{k}_conserv"] for r in rows), 1), 2)
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)

json.dump(RES, open("/content/specdraft_result.json", "w"), indent=1)

# persistir en Kaggle (leccion de la VM fantasma: el cliente puede morir antes del final)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/specdraft_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale specdraft bench", "id": f"{user}/arcscale-specdraft",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "bench specdraft", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
