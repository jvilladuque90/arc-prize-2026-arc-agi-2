#!/usr/bin/env python
"""[Colab T4] ¿Existe un umbral de CONFIANZA que retire tareas sin equivocarse?

Idea del usuario (19-ago): un TTT AUDITADO en cascada -- una pasada barata retira las tareas
faciles, y el presupuesto liberado se concentra en las dificiles. Para que el tier 1 sea seguro
hace falta un disparador OBSERVABLE (sin conocer la respuesta) que separe "candidato confiable"
de "candidato equivocado".

Contraejemplo que ya tenemos y que obliga a medirlo: 9caf5b84 produce un candidato BAJO umbral
que es INCORRECTO (su verdad esta en NLL 2,625, fuera del umbral 1,609). Asi que el disparador
NO puede ser "el pool no esta vacio".

Lo que mide, sin TTT y sobre la base original:
  - score acumulado (NLL) de cada hoja del DFS, no solo la hoja
  - si el MEJOR candidato es correcto, y si alguno de los DOS mejores lo es (= pass@2 real)
  - NLL de la verdad, tamaño del pool, segundos de decode

Con eso se puede barrer offline el umbral tau: retirar la tarea si best_score < tau. La pregunta
es si existe tau con precision 1,0 (nunca retira una equivocada) y cobertura alta.

Aviso de calibracion: `lb33-cap` fallo (-2,5 pt) porque su umbral se valido en una poblacion y
se aplico en otra. Aqui la poblacion es holdout no-AGI1 de AGI-2, la mas parecida al test.

Uso: sesion persistente + lanzamiento desacoplado (ver colab_lora_epochs.py).
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
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1] o por entorno"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json",
          "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)
# MODELO=4b usa el modelo DEL CAMPEON. Toda nuestra evidencia de PRECISION (no hay errores
# confiados bajo el umbral) es del 2B; el log del campeon solo imprime lineas ALL_CORRECT, asi
# que valida su recall pero no su precision. Si el 4B produce errores confiados, la cascada y
# el cambio de umbral se caen. Es el unico hueco que queda en el argumento.
MODELO = os.environ.get("MODELO", "2b")
_SLUG = ("sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1" if MODELO == "4b"
         else "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1")
subprocess.run(["kaggle", "models", "instances", "versions", "download", _SLUG,
                "-p", "/content/m2b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m2b"):
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
from transformers import AutoModelForCausalLM

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = float(os.environ.get("THR_DFS", 0) or 0) or -float(np.log(0.2))
N_TAREAS = int(os.environ.get("N_TAREAS", "24"))
# OFFSET recorta una FRANJA por coste. El tramo 0-24 (las mas baratas) dio separacion perfecta,
# pero ahi el modelo resuelve 16/24 = 67% frente al ~30% de produccion: poblacion demasiado
# facil. El tramo 24-48 prueba lo unico que puede romper el umbral — en tareas dificiles los
# candidatos CORRECTOS son menos confiados y se acercan a la banda de los erroneos. Es el mismo
# modo de fallo que hundio a lb33-cap (umbral validado en una poblacion, aplicado en otra).
OFFSET = int(os.environ.get("OFFSET", "0"))
DS = (f"{os.environ.get('KAGGLE_USERNAME', 'juliancamilovilla')}/arcscale-tier1"
      + (f"-{os.environ['POBLACION']}" if os.environ.get("POBLACION") else "")
      + (f"-{MODELO}" if MODELO != "2b" else "")
      + (f"-thr{os.environ['THR_DFS'].replace('.', '')}" if os.environ.get("THR_DFS") else "")
      + (f"-{OFFSET}" if OFFSET else ""))


def grid_tokens(g):
    out = []
    for i, row in enumerate(g):
        out.extend(int(c) for c in row)
        if i < len(g) - 1:
            out.append(NL)
    return out


def pair_tokens(inp, out):
    t = [IM_START, USER, NL] + grid_tokens(inp) + [EOS]
    head = [IM_START, ASSIST, NL]
    return t + head + grid_tokens(out) + [EOS]


def query_tokens(inp):
    return [IM_START, USER, NL] + grid_tokens(inp) + [EOS] + [IM_START, ASSIST, NL]


# El intento del 21-ago con el 4B en fp16 sobre T4 dio 0/16 y gt_nll de 5-28 donde el 2B saca
# 0,01-0,7: medicion NULA. Los checkpoints vienen en bfloat16 y la T4 no lo soporta nativo
# (capacidad 7.5, hace falta 8.0). Por defecto el 4B se carga en float32.
_DT = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[
    os.environ.get("DTYPE", "fp32" if MODELO == "4b" else "fp16")]
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=_DT,
                                             device_map="cuda:0", attn_implementation="sdpa")
MODEL.eval()
MODEL.config.use_cache = True
print(f"[setup] {MODELO} base {_DT} | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)


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


@torch.no_grad()
def nll_gt(prompt, ans):
    last, cache = prefill(prompt)
    parts = [last.float()]
    if len(ans) > 1:
        with sdpa_kernel(_SDPA):
            o = MODEL(input_ids=torch.tensor([ans[:-1]], device="cuda:0", dtype=torch.long),
                      past_key_values=cache, use_cache=True, return_dict=True)
        parts.append(o.logits[0].float())
    lg = torch.cat([x.view(-1, x.shape[-1]) for x in parts], 0)
    v = -torch.log_softmax(lg, -1).gather(
        1, torch.tensor(ans, device="cuda:0").unsqueeze(1)).sum().item()
    del parts, lg, cache
    torch.cuda.empty_cache()
    return v


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
def dfs_scored(prompt, depth, budget=45):
    """Identico al DFS de produccion salvo que la hoja se guarda CON su score acumulado.
    Ese score es lo unico observable en test, y es el candidato a disparador del tier 1."""
    t0 = time.time()
    last, cache0 = prefill(prompt)
    leaves, path = [], []

    def rec(logits, score, pos, cache, left):
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
                    leaves.append((sc, list(path)))       # <-- score, no solo la hoja
                else:
                    cands.append((sc, t))
        for sc, t in sorted(cands):
            if time.time() - t0 > budget:
                return
            with sdpa_kernel(_SDPA):
                o = MODEL(input_ids=torch.tensor([[t]], device="cuda:0", dtype=torch.long),
                          past_key_values=cache, use_cache=True, return_dict=True)
            path.append(t)
            rec(o.logits[:, -1], sc, pos + 1, o.past_key_values, left - 1)
            path.pop()

    rec(last, 0.0, len(prompt), cache0, depth)
    return leaves, time.time() - t0


# POBLACION: el holdout no-AGI1 del TRAIN tiene solo 24 tareas -- las medimos todas, y el
# modelo resuelve 16 (67%) con NLL de 0,0008-0,03. Eso no parece confianza sino MEMORIZACION:
# son tareas del conjunto de entrenamiento de AGI-2 y el modelo base es de terceros. Si los
# aciertos son memorizaciones, su confianza extrema es un artefacto y la separacion tambien.
# El eval no-AGI1 (114 tareas) es la poblacion menos contaminada disponible. Lo que decide:
# ¿aparecen aciertos con confianza INTERMEDIA (0,6-1,5), donde viven los errores?
POB = os.environ.get("POBLACION", "train")
_pre = "training" if POB == "train" else "evaluation"
CHAL = json.load(open(f"/content/arc-agi_{_pre}_challenges.json"))
SOLS = json.load(open(f"/content/arc-agi_{_pre}_solutions.json"))


def is_holdout(tid):
    if POB != "train":          # el eval entero es holdout: nunca se uso para nada nuestro
        return True
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


KEYS = sorted([k for k in CHAL if is_holdout(k) and k not in AGI1],
              key=lambda k: cost(CHAL[k]))[OFFSET:OFFSET + N_TAREAS]
print(f"[setup] {len(KEYS)} tareas (franja {OFFSET}-{OFFSET+N_TAREAS} por coste) | "
      f"umbral {THR:.3f}", flush=True)

GATE = os.environ.get("GATE_TAREA", "")
if GATE:
    # Failsafe barato: si el modelo no puntua bien UNA tarea que sabemos facil, aborta antes de
    # gastar la sesion. El intento del 4B del 21-ago se comio una sesion entera para producir
    # 16 medidas nulas; esto lo habria cazado en 30 segundos.
    _t = CHAL[GATE]
    _p = []
    for _x in _t["train"]:
        _p += pair_tokens(_x["input"], _x["output"])
    _p += query_tokens(_t["test"][0]["input"])
    _n = nll_gt(_p, grid_tokens(SOLS[GATE][0]) + [EOS])
    print(f"[gate] {GATE}: NLL de la verdad {_n:.4f}", flush=True)
    assert _n < 1.0, f"gate FALLA ({_n:.3f} >= 1.0): el modelo esta mal cargado, se aborta"

RES = {"tareas": [], "umbral": THR}
if os.environ.get("REANUDAR", "0") == "1":
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", DS, "-p", "/content/prev",
                        "--unzip", "-q"], check=True)
        RES["tareas"] = json.load(open("/content/prev/tier1.json"))["tareas"]
        print(f"[reanudar] {len(RES['tareas'])} tareas ya medidas", flush=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"[reanudar] sin parcial previo ({e})", flush=True)
HECHAS = {r["task"] for r in RES["tareas"]}


def persistir(etiqueta):
    try:
        os.makedirs("/content/persist", exist_ok=True)
        json.dump(RES, open("/content/persist/tier1.json", "w"), indent=1)
        json.dump({"title": "arcscale tier1", "id": DS, "licenses": [{"name": "CC0-1.0"}]},
                  open("/content/persist/dataset-metadata.json", "w"))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                           capture_output=True, text=True)
        if "error" in (r.stdout + r.stderr).lower():
            subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                            "-m", etiqueta, "-q"], check=False, capture_output=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"[persist] fallo: {e}", flush=True)


for i, k in enumerate(KEYS):
    if k in HECHAS:
        print(f"[{i+1}/{len(KEYS)}] {k}: ya medida", flush=True)
        continue
    t = CHAL[k]
    gt = SOLS[k][0]
    prompt = []
    for p in t["train"]:
        prompt += pair_tokens(p["input"], p["output"])
    prompt += query_tokens(t["test"][0]["input"])
    gt_toks = grid_tokens(gt) + [EOS]

    try:
        hojas, secs = dfs_scored(prompt, min(931, len(gt_toks) + 40))
    except torch.cuda.OutOfMemoryError:               # el 4B en T4 va justo; se anota y sigue
        print(f"[{i+1}/{len(KEYS)}] {k}: OOM, se salta", flush=True)
        torch.cuda.empty_cache()
        continue
    hojas.sort(key=lambda x: x[0])                    # menor NLL acumulada = mas confiable
    grids = [(sc, toks_to_grid(p)) for sc, p in hojas]
    best_sc = grids[0][0] if grids else None
    best_ok = bool(grids and grids[0][1] == gt)
    dos_ok = bool(any(g == gt for _, g in grids[:2]))  # pass@2, como produccion
    algun_ok = bool(any(g == gt for _, g in grids))
    n = nll_gt(prompt, gt_toks)

    RES["tareas"].append({
        "task": k, "L": len(gt_toks), "pool": len(grids),
        "best_score": None if best_sc is None else round(best_sc, 4),
        "best_ok": best_ok, "pass2_ok": dos_ok, "algun_ok": algun_ok,
        "gt_nll": round(n, 4), "decode_s": round(secs, 1),
        "scores": [round(s, 4) for s, _ in grids[:5]]})
    print(f"[{i+1}/{len(KEYS)}] {k}: pool {len(grids)} | best_score "
          f"{'-' if best_sc is None else f'{best_sc:.4f}'} | best_ok {best_ok} | "
          f"pass2 {dos_ok} | gt_nll {n:.3f} | {secs:.0f}s", flush=True)
    persistir(k)
    torch.cuda.empty_cache()

# Barrido del umbral: ¿existe tau que retire tareas sin equivocarse nunca?
con = [r for r in RES["tareas"] if r["best_score"] is not None]
barrido = []
for tau in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, THR):
    ret = [r for r in con if r["best_score"] < tau]
    ok = sum(r["best_ok"] for r in ret)
    barrido.append({"tau": tau, "retiradas": len(ret), "correctas": ok,
                    "precision": round(ok / len(ret), 3) if ret else None})
RES["barrido_tau"] = barrido
RES["n"] = len(RES["tareas"])
print("RESULT_JSON:" + json.dumps({"barrido_tau": barrido, "n": len(RES["tareas"]),
                                   "pass2_total": sum(r["pass2_ok"] for r in RES["tareas"])}),
      flush=True)
persistir("final")
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
