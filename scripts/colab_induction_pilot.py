#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Piloto de INDUCCION propia: ¿un modelo chico SFT'd en (grids -> programa) produce
programas VERIFICADOS en tareas nuevas de AGI-2?

La apuesta estructural (SCALE_PLAN): ARC-Heavy trae el programa fuente de cada tarea — un
dataset de induccion gratis que no estamos usando. Hibrido ganador 2025 = transduccion +
induccion-verificada. Nuestro intento con BARC-8B congelado dio 6.25% y era caro/ajeno;
aqui: modelo CHICO propio (Qwen2.5-1.5B-Instruct, full vocab) SFT'd con LoRA en 2000 pares
(prompt de grids -> programa BARC), y eval de verifiable-hit en las 24 tareas NUEVAS.

Metrica por tarea (K=6 muestras, temp 0.8): programa VERIFICADO = ejecuta y reproduce TODOS
los train pairs; si ademas acierta el test -> hit generalizado. Referencias: BARC-8B congelado
6.25% verificado (en eval AGI-2); azar ~0.

Lectura pre-registrada: >=3/24 tareas con programa verificado -> la via de induccion propia
VIVE -> escalar (mas data, modelo 4B, hibrido con el campeon). 0-1/24 -> el 1.5B no alcanza
(probar 4B antes de descartar). Senal relativa (carril proxy).

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_induction_pilot.py <KAGGLE_API_TOKEN>
"""
import json
import os
import signal
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
KUSER = sys.argv[2] if len(sys.argv) > 2 else ""
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "kaggle", "datasets",
                "torchao>=0.16"], check=True)

os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)
subprocess.run(["kaggle", "datasets", "download", "-d", "juliancamilovilla/arcscale-barc-common",
                "-p", "/content/bc", "--unzip", "-q"], check=True)
COMMON = None
for root, dirs, files in os.walk("/content/bc"):
    if "common.py" in files:
        COMMON = os.path.join(root, "common.py")
assert COMMON
import shutil
shutil.copy(COMMON, "/content/common.py")
sys.path.insert(0, "/content")
print(f"[setup] common.py listo ({time.time()-T00:.0f}s)", flush=True)

import urllib.request

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))

import hashlib
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-1.5B-Instruct"
N_TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
TOK = AutoTokenizer.from_pretrained(MODEL_ID)


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def make_prompt(pairs):
    ex = "\n\n".join(f"Example {i+1}\nInput:\n{g2s(a)}\nOutput:\n{g2s(b)}"
                     for i, (a, b) in enumerate(pairs))
    return (f"You are given input/output grid pairs of an ARC puzzle (digits 0-9 are colors).\n"
            f"{ex}\n\nWrite a Python function `main(input_grid)` (numpy in, numpy out) "
            f"that implements the transformation. You may use `from common import *`.\n"
            f"```python\n")


def valid_grid(g):
    return (g and g[0] and len(g) <= 30 and len(g[0]) <= 30
            and all(len(r) == len(g[0]) for r in g)
            and all(0 <= int(c) <= 9 for r in g for c in r))


# ---- data de entrenamiento: (prompt grids -> programa) desde ARC-Heavy con `source` ----
MAX_LEN = 3072
STREAM = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                      split="train", streaming=True)
TRAIN, seen = [], 0
for row in STREAM:
    seen += 1
    try:
        src = row["source"]
        # convencion ARC-Heavy: def main(input_grid) + def generate_input() (el generador de la
        # tarea — irrelevante para resolver: se recorta, igual que los imports del header)
        if "def main" not in src or len(src) > 4500:
            continue
        src = src[src.index("def main"):]
        if "def generate_input" in src:
            src = src[:src.index("def generate_input")]
        src = src.rstrip()
        pairs = []
        for exp in row["examples"][:3]:
            gi = [[int(c) for c in r] for r in exp[0]]
            go = [[int(c) for c in r] for r in exp[1]]
            if not (valid_grid(gi) and valid_grid(go)):
                pairs = []
                break
            pairs.append((gi, go))
        if len(pairs) < 3:
            continue
        text = make_prompt(pairs) + src.strip() + "\n```" + TOK.eos_token
        ids = TOK(text, return_tensors="pt").input_ids[0]
        if ids.numel() > MAX_LEN:
            continue
        plen = TOK(make_prompt(pairs), return_tensors="pt").input_ids[0].numel()
        TRAIN.append((ids, plen))
        if len(TRAIN) >= N_TARGET:
            break
    except Exception:
        continue
random.Random(7).shuffle(TRAIN)
print(f"[pool] {len(TRAIN)} items de induccion de {seen} filas ({time.time()-T00:.0f}s)",
      flush=True)

# ---- eval: las 24 nuevas de AGI-2 ----
def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
KEYS = [k for k in HOLD if k not in AGI1]
print(f"[setup] {len(KEYS)} tareas de eval", flush=True)

MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map="cuda:0")
LCFG = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
MODEL = get_peft_model(MODEL, LCFG)
MODEL.enable_input_require_grads()
MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
MODEL.config.use_cache = False
print(f"[setup] {MODEL_ID} + LoRA listo ({time.time()-T00:.0f}s)", flush=True)

# ---- SFT (loss solo en el programa: mask del prompt) ----
OPT = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=1e-4)
SCALER = torch.cuda.amp.GradScaler()
ACCUM = 8
MODEL.train()
t0 = time.time()
OPT.zero_grad()
for n, (ids, plen) in enumerate(TRAIN):
    ids = ids.unsqueeze(0).to("cuda:0")
    labels = ids.clone()
    labels[0, :plen] = -100
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = MODEL(ids, labels=labels)
    SCALER.scale(out.loss / ACCUM).backward()
    if (n + 1) % ACCUM == 0:
        SCALER.step(OPT)
        SCALER.update()
        OPT.zero_grad()
    if (n + 1) % 100 == 0:   # cada ~3 min: el cliente CLI aborta con ~10 min de silencio
        print(f"  [sft] {n+1}/{len(TRAIN)} loss={out.loss.item():.3f} ({time.time()-t0:.0f}s)",
              flush=True)
if len(TRAIN) % ACCUM != 0:
    SCALER.step(OPT)
    SCALER.update()
    OPT.zero_grad()
MODEL.eval()
print(f"[sft] completo en {time.time()-t0:.0f}s", flush=True)


# ---- ejecucion verificada ----
def run_program(src, grid, timeout=4):
    def handler(signum, frame):
        raise TimeoutError

    import contextlib
    import io

    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec("from common import *\nimport numpy as np\nfrom typing import *\n" + src, ns)
            fn = ns.get("main")
            if fn is None:
                return None
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            try:
                out = fn(np.array(grid))
            finally:
                signal.alarm(0)
        out = np.array(out)
        if out.ndim != 2 or out.shape[0] > 30 or out.shape[1] > 30:
            return None
        return out.astype(int)
    except Exception:
        return None


def extract_code(txt):
    if "```" in txt:
        txt = txt.split("```")[0]
    return txt.strip()


K = 6
RES = {"tasks": []}
MODEL.config.use_cache = True
for i, k in enumerate(KEYS):
    t = CHAL[k]
    pairs = [(p["input"], p["output"]) for p in t["train"][:3]]
    prompt = make_prompt(pairs)
    ids = TOK(prompt, return_tensors="pt").input_ids.to("cuda:0")
    if ids.shape[1] > 4000:
        print(f"[{i+1}/{len(KEYS)}] {k}: prompt largo, saltada", flush=True)
        continue
    verified, test_ok = 0, 0
    with torch.no_grad():
        outs = MODEL.generate(ids.repeat(K, 1), max_new_tokens=700, do_sample=True,
                              temperature=0.8, top_p=0.95, pad_token_id=TOK.eos_token_id)
    for j in range(K):
        src = extract_code(TOK.decode(outs[j, ids.shape[1]:], skip_special_tokens=True))
        ok = all((lambda o, b: o is not None and o.shape == np.array(b).shape
                  and (o == np.array(b)).all())(run_program(src, a), b)
                 for a, b in [(p["input"], p["output"]) for p in t["train"]])
        if ok:
            verified += 1
            o = run_program(src, t["test"][0]["input"])
            gt = np.array(SOLS[k][0])
            if o is not None and o.shape == gt.shape and (o == gt).all():
                test_ok += 1
    RES["tasks"].append({"task": k, "verificados": verified, "test_ok": test_ok})
    print(f"[{i+1}/{len(KEYS)}] {k}: verificados={verified}/{K} test_ok={test_ok}", flush=True)

vt = sum(1 for r in RES["tasks"] if r["verificados"] > 0)
gt_ok = sum(1 for r in RES["tasks"] if r["test_ok"] > 0)
RES["resumen"] = {"tareas_con_programa_verificado": vt,
                  "tareas_con_test_correcto": gt_ok, "n": len(RES["tasks"]),
                  "modelo": MODEL_ID, "n_train": len(TRAIN)}
json.dump(RES, open("/content/induction_pilot_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(RES["resumen"]), flush=True)

# persistir en Kaggle (best-effort): si el cliente CLI murio, el resultado sobrevive igual
try:
    if KUSER:
        OUTD = "/content/piloutd"
        os.makedirs(OUTD, exist_ok=True)
        shutil.copy("/content/induction_pilot_result.json", OUTD)
        MODEL.save_pretrained(os.path.join(OUTD, "adapter_induction_1p5b"))
        json.dump({"title": "arcscale-induction-pilot", "id": f"{KUSER}/arcscale-induction-pilot",
                   "licenses": [{"name": "CC0-1.0"}]},
                  open(os.path.join(OUTD, "dataset-metadata.json"), "w"))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", OUTD, "--dir-mode", "zip"],
                           capture_output=True, text=True)
        if r.returncode != 0 or "already exists" in ((r.stdout or "") + (r.stderr or "")):
            r = subprocess.run(["kaggle", "datasets", "version", "-p", OUTD, "-m", "pilot",
                                "--dir-mode", "zip"], capture_output=True, text=True)
        print(f"[persist] rc={r.returncode}", flush=True)
except Exception as e:
    print(f"[persist] fallo (ok): {e}", flush=True)
