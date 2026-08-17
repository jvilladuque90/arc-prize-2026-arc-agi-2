#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] ¿Sobrevive la especulacion al LOTE SINCRONIZADO de produccion?

El gate de exactitud paso (15/15 pools identicos) y el bench de aceptacion dio x1.81 en
forwards... pero medido sobre UNA vista. Produccion decodifica 4 vistas por lote en paso
sincronizado (mismo `pos` para las 4, tensor (4,1)). Con especulacion cada fila acepta un
numero distinto de tokens:

  - LOCKSTEP: para no desincronizar, el lote avanza min_i(aceptado_i)+1 por forward. Si la
    aceptacion por fila es p, la del minimo de 4 es ~p^4 -> con p=0.57 cae a 0.10.
  - RAGGED: cada fila avanza lo suyo; exige position_ids por fila + mascaras de atencion +
    slots extra de cache (la maquinaria cuya memoria mato a dfsbatch: 39.3GB > 22GB en L4).

Este bench mide los TRES regimenes sobre las 4 vistas REALES de un lote de produccion
(2 geometrias x 2 permutaciones de color, como arma el campeon):
    forwards_baseline = max_i(len_i)              (lo que hace hoy)
    forwards_lockstep = simulacion avanzando min  (especulacion sin maquinaria nueva)
    forwards_ragged   = max_i(forwards_i)         (especulacion con mascaras por fila)

Lectura pre-registrada:
  - lockstep >= 1.4x  -> se puede integrar SIN mascaras: bajo riesgo, se hace hoy.
  - lockstep < 1.4x y ragged >= 1.6x -> la ganancia existe pero SOLO con la maquinaria de
    riesgo; se decide aparte, con un bench de memoria antes de tocar Kaggle.
  - ragged < 1.6x -> la rama se cierra con dato: la especulacion no sobrevive al lote.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_lockstep_bench.py <TOKEN>
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
NL_ID, EOS_ID = 10, 15
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
MAX_NEW = 931
K = 8


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


# el campeon agrupa por forma: un lote = 2 geometrias x 2 permutaciones de color
GEOS = [lambda a: a, lambda a: np.rot90(np.asarray(a), 2)]


def make_batch_views(task, seed=2):
    rnd = _rnd.Random(seed)
    perms = []
    for _ in range(2):
        p = list(range(10))
        tail = p[1:]
        rnd.shuffle(tail)
        p[1:] = tail
        perms.append(np.array(p))
    views = []
    for geo in GEOS:
        for pm in perms:
            def tr(x, geo=geo, pm=pm):
                return pm[geo(np.asarray(x))]
            txt = "".join(f"<|im_start|>user\n{g2s(tr(p['input']))}<|im_end|>"
                          f"<|im_start|>assistant\n{g2s(tr(p['output']))}<|im_end|>"
                          for p in task["train"])
            inp = tr(task["test"][0]["input"])
            txt += f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n"
            views.append({"ids": TOK.encode(txt), "inp": np.asarray(inp).tolist()})
    return views


def draft_token(r, c, inp):
    H = len(inp)
    if r >= H:
        return EOS_ID
    row = inp[r]
    if c < len(row):
        return int(row[c])
    return EOS_ID if r == H - 1 else NL_ID


@torch.no_grad()
def trace(view):
    """Camino argmax de UNA vista + si el borrador (su grid de entrada) acierta en cada paso."""
    ids = torch.tensor([view["ids"]], device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    past, logits, pos = out.past_key_values, out.logits[:, -1], ids.size(1)
    r, c, matches = 0, 0, []
    for _ in range(MAX_NEW):
        lf = logits.float()
        nll = (torch.logsumexp(lf, dim=-1, keepdim=True) - lf.index_select(-1, _TOKID))[0]
        tok = ARC_TOKENS[int(nll.argmin())]
        matches.append(tok == draft_token(r, c, view["inp"]))
        if tok == EOS_ID:
            break
        r, c = (r + 1, 0) if tok == NL_ID else (r, c + 1)
        out = MODEL(input_ids=torch.tensor([[tok]], device="cuda:0", dtype=torch.long),
                    position_ids=torch.full((1, 1), pos, device="cuda:0"),
                    past_key_values=past, use_cache=True, return_dict=True)
        past, logits, pos = out.past_key_values, out.logits[:, -1], pos + 1
    return matches


def fwd_single(m, k=K):
    """Forwards para UNA fila con ventana k (acepta racha + 1 token de regalo)."""
    i, f, n = 0, 0, len(m)
    while i < n:
        f += 1
        a = 0
        while a < k and i + a < n and m[i + a]:
            a += 1
        i += a + 1
    return f


def fwd_lockstep(ms, k=K):
    """El lote avanza min_i(aceptado_i)+1: las filas NO se desincronizan."""
    pos = [0] * len(ms)
    f = 0
    while any(pos[i] < len(ms[i]) for i in range(len(ms))):
        f += 1
        accs = []
        for i, m in enumerate(ms):
            if pos[i] >= len(m):
                continue
            a = 0
            while a < k and pos[i] + a < len(m) and m[pos[i] + a]:
                a += 1
            accs.append(a)
        step = min(accs) + 1 if accs else 1
        for i, m in enumerate(ms):
            if pos[i] < len(m):
                pos[i] += step
    return f


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
NEW = [k for k in HOLD if k not in AGI1]
TASKS = NEW[:10]

RES = {"tasks": []}
for ti, tk in enumerate(TASKS):
    task = CHAL[tk]
    views = make_batch_views(task)
    L = len(views[0]["ids"])
    if any(len(v["ids"]) != L for v in views) or L > 6000:
        print(f"[{ti+1}/{len(TASKS)}] {tk} saltada (ctx irregular/largo)", flush=True)
        continue
    t0 = time.time()
    ms = [trace(v) for v in views]
    base = max(len(m) for m in ms)
    rag = max(fwd_single(m) for m in ms)
    lock = fwd_lockstep(ms)
    row = {"task": tk, "len_por_vista": [len(m) for m in ms],
           "acept_por_vista": [round(float(np.mean(m)), 3) for m in ms],
           "fwd_base": base, "fwd_lockstep": lock, "fwd_ragged": rag,
           "x_lockstep": round(base / max(lock, 1), 2),
           "x_ragged": round(base / max(rag, 1), 2), "secs": round(time.time() - t0, 1)}
    RES["tasks"].append(row)
    print(f"[{ti+1}/{len(TASKS)}] {tk}: len {row['len_por_vista']} acept "
          f"{row['acept_por_vista']} | fwd base {base} lock {lock} rag {rag} | "
          f"LOCKSTEP x{row['x_lockstep']} RAGGED x{row['x_ragged']} ({row['secs']:.0f}s)",
          flush=True)

rows = RES["tasks"]
if rows:
    b = sum(r["fwd_base"] for r in rows)
    summ = {"n": len(rows), "fwd_base": b,
            "x_lockstep": round(b / max(sum(r["fwd_lockstep"] for r in rows), 1), 2),
            "x_ragged": round(b / max(sum(r["fwd_ragged"] for r in rows), 1), 2),
            "acept_media": round(float(np.mean([a for r in rows
                                                for a in r["acept_por_vista"]])), 3)}
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)

json.dump(RES, open("/content/lockstep_result.json", "w"), indent=1)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/lockstep_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale lockstep bench", "id": f"{user}/arcscale-lockstep",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "lockstep", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
