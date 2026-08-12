#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] D1: curva de escala de datos sinteticos (ARC-Heavy) sobre el 2B campeon.

Prueba la HIPOTESIS DE ESCALA (docs/SCALE_PLAN.md): ¿SFT con corpus sintetico publico grande
y diverso (BARC ARC-Heavy, 139K tareas generadas por GPT-4o) mejora al modelo campeon en
tareas REALES de AGI-2? Curva acumulativa N = 0 (control) -> 500 -> 2000 en una sesion T4.

Por punto de la curva:
- REAL: 12 tareas holdout AGI-2 (las mismas de la ablacion ttt8) -> NLL teacher-forced de la
  respuesta verdadera, cell-acc greedy, exactas.
- SINTETICO held-out: 40 tareas ARC-Heavy no vistas -> NLL. Separa "aprende la fuente" de
  "transfiere a AGI-2" (la leccion de v1/v2: pueden divergir).

Senal RELATIVA (pendiente), no absoluta — regla del carril proxy (docs/COLAB.md).
SFT: LoRA r=64, lr 1e-4 constante, grad-accum 8, secuencias completas <=2048 tok (se filtra,
no se trunca). Al final guarda el adapter y lo sube como dataset PRIVADO de Kaggle
(best-effort) para que D2 (N=8000) pueda reanudar sin reentrenar.

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_data_scaling.py <KAGGLE_API_TOKEN> [<KAGGLE_USERNAME>]
"""
import json
import os
import subprocess
import sys
import time

# antes de importar torch: la T4 (14.5GB) fragmenta con eval+train alternados
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

subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1",
                "-p", "/content/m2b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m2b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR, "modelo no encontrado"
print(f"[setup] modelo en {MDIR} ({time.time()-T00:.0f}s)", flush=True)

# ---------------------------------------------------------------- utilidades
import hashlib
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def fmt_pair(inp, out):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n{g2s(out)}<|im_end|>"


def fmt_query(inp):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n"


# ---------------------------------------------------------------- pool sintetico (streaming)
from datasets import load_dataset

MAX_LEN = 2048


def parse_row(row):
    """ARC-Heavy: examples = lista de pares [input, output], grids de floats 0-9."""
    try:
        pairs = []
        for ex in row["examples"]:
            gi = [[int(c) for c in r] for r in ex[0]]
            go = [[int(c) for c in r] for r in ex[1]]
            for g in (gi, go):
                if not g or not g[0] or len(g) > 30 or len(g[0]) > 30:
                    return None
                if any(len(r) != len(g[0]) for r in g):
                    return None
                if any(c < 0 or c > 9 for r in g for c in r):
                    return None
            pairs.append((gi, go))
            if len(pairs) == 4:
                break
        return pairs if len(pairs) >= 3 else None
    except Exception:
        return None


def build_item(pairs):
    """Contexto = todos los pares menos el ultimo; objetivo = ultimo par. Filtra por longitud."""
    prompt = "".join(fmt_pair(i, o) for i, o in pairs[:-1]) + fmt_query(pairs[-1][0])
    ans = g2s(pairs[-1][1]) + "<|im_end|>"
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(ans, return_tensors="pt").input_ids[0]
    if pi.numel() + ai.numel() > MAX_LEN:
        return None
    return {"prompt_ids": pi, "ans_ids": ai, "full_ids": torch.cat([pi, ai])}


t0 = time.time()
STREAM = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                      split="train", streaming=True)
SYN_HOLD, TRAIN = [], []
N_TRAIN_TARGET, N_HOLD = 2000, 40
seen = 0
for row in STREAM:
    seen += 1
    pairs = parse_row(row)
    if pairs is None:
        continue
    item = build_item(pairs)
    if item is None:
        continue
    if len(SYN_HOLD) < N_HOLD:
        SYN_HOLD.append(item)
    else:
        TRAIN.append(item)
        if len(TRAIN) >= N_TRAIN_TARGET:
            break
random.Random(7).shuffle(TRAIN)
print(f"[pool] {seen} filas leidas -> {len(SYN_HOLD)} holdout + {len(TRAIN)} train "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- tareas reales (AGI-2)
CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
KEYS = HOLD[8:20]   # las MISMAS 12 de la ablacion ttt8 (comparabilidad)

# ---------------------------------------------------------------- modelo + LoRA
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL.config.use_cache = False
LCFG = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
MODEL = get_peft_model(MODEL, LCFG)
# sin checkpointing, el backward de 2048 tokens no cabe en la T4 (OOM medido con 13.9GB)
MODEL.enable_input_require_grads()
MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print(f"[setup] 2B + LoRA r=64 + grad-ckpt listo | mem={torch.cuda.memory_allocated()/1e9:.1f}GB "
      f"({time.time()-T00:.0f}s)", flush=True)


@torch.no_grad()
def nll_ids(prompt_ids, ans_ids):
    ids = torch.cat([prompt_ids, ans_ids]).unsqueeze(0).to("cuda:0")
    logits = MODEL(ids).logits.float()
    lp = torch.log_softmax(logits[0, prompt_ids.numel() - 1:-1], -1)
    return -lp.gather(1, ans_ids.unsqueeze(1).to("cuda:0")).mean().item()


def real_item(key):
    t = CHAL[key]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    ans = g2s(SOLS[key][0]) + "<|im_end|>"
    return (TOK(prompt, return_tensors="pt").input_ids[0], TOK(ans, return_tensors="pt").input_ids[0])


REAL_ITEMS = {k: real_item(k) for k in KEYS}


def greedy_cell_acc(key, max_new=900):
    pi, _ = REAL_ITEMS[key]
    ids = pi.unsqueeze(0).to("cuda:0")
    MODEL.config.use_cache = True
    with torch.no_grad():
        out = MODEL.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=TOK.eos_token_id)
    MODEL.config.use_cache = False
    txt = TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    try:
        pred = [[int(c) for c in line if c.isdigit()] for line in txt.strip().split("\n") if line.strip()]
        gt = SOLS[key][0]
        if not pred or len(pred) != len(gt) or any(len(r) != len(gt[0]) for r in pred):
            return 0.0, False
        p, g = np.array(pred), np.array(gt)
        return float((p == g).mean()), bool((p == g).all())
    except Exception:
        return 0.0, False


def eval_point(n_seen):
    MODEL.eval()
    torch.cuda.empty_cache()
    t0 = time.time()
    real = []
    for k in KEYS:
        pi, ai = REAL_ITEMS[k]
        nll = nll_ids(pi, ai)
        acc, exact = greedy_cell_acc(k)
        real.append({"task": k, "nll": round(nll, 4), "cell_acc": round(acc, 4), "exact": exact})
    syn_nll = [nll_ids(it["prompt_ids"], it["ans_ids"]) for it in SYN_HOLD]
    point = {"n": n_seen,
             "real": {"nll_media": round(float(np.mean([r["nll"] for r in real])), 4),
                      "cell_acc_media": round(float(np.mean([r["cell_acc"] for r in real])), 4),
                      "exactas": sum(r["exact"] for r in real), "detalle": real},
             "syn": {"nll_media": round(float(np.mean(syn_nll)), 4)},
             "eval_secs": round(time.time() - t0, 1)}
    print("RESULT_POINT:" + json.dumps({k: v for k, v in point.items() if k != "real"}
                                       | {"real": {k2: v2 for k2, v2 in point["real"].items()
                                                   if k2 != "detalle"}}), flush=True)
    return point


ACCUM = 8
OPT = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=1e-4)
SCALER = torch.cuda.amp.GradScaler()


def train_chunk(items):
    MODEL.config.use_cache = False
    MODEL.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    OPT.zero_grad()
    for n, it in enumerate(items):
        ids = it["full_ids"].unsqueeze(0).to("cuda:0")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = MODEL(ids, labels=ids)
        SCALER.scale(out.loss / ACCUM).backward()
        if (n + 1) % ACCUM == 0:
            SCALER.step(OPT)
            SCALER.update()
            OPT.zero_grad()
        if (n + 1) % 250 == 0:
            print(f"  [train] {n+1}/{len(items)} ({time.time()-t0:.0f}s, "
                  f"pico {torch.cuda.max_memory_allocated()/1e9:.1f}GB)", flush=True)
    if len(items) % ACCUM != 0:   # flush SOLO si quedan gradientes (si no, GradScaler asserta)
        SCALER.step(OPT)
        SCALER.update()
        OPT.zero_grad()
    MODEL.eval()
    return time.time() - t0


# ---------------------------------------------------------------- curva acumulativa
RES = {"points": []}
RES["points"].append(eval_point(0))
for lo, hi in ((0, 500), (500, 1250), (1250, 2000)):
    secs = train_chunk(TRAIN[lo:hi])
    print(f"[curva] entrenado {lo}->{hi} en {secs:.0f}s (total {time.time()-T00:.0f}s)", flush=True)
    RES["points"].append(eval_point(hi))

p0 = RES["points"][0]
RES["resumen"] = {
    "curva_real_nll": [p["real"]["nll_media"] for p in RES["points"]],
    "curva_real_cellacc": [p["real"]["cell_acc_media"] for p in RES["points"]],
    "curva_real_exactas": [p["real"]["exactas"] for p in RES["points"]],
    "curva_syn_nll": [p["syn"]["nll_media"] for p in RES["points"]],
    "delta_real_nll_vs_control": [round(p0["real"]["nll_media"] - p["real"]["nll_media"], 4)
                                  for p in RES["points"]],
    "delta_syn_nll_vs_control": [round(p0["syn"]["nll_media"] - p["syn"]["nll_media"], 4)
                                 for p in RES["points"]],
}
json.dump(RES, open("/content/data_scaling_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(RES["resumen"]), flush=True)

# ---------------------------------------------------------------- adapter -> Kaggle (best-effort)
try:
    ADIR = "/content/adapter_n2000"
    MODEL.save_pretrained(ADIR)
    json.dump(RES, open(os.path.join(ADIR, "data_scaling_result.json"), "w"), indent=1)
    if KUSER:
        json.dump({"title": "arcscale-adapter-2b", "id": f"{KUSER}/arcscale-adapter-2b",
                   "licenses": [{"name": "CC0-1.0"}]},
                  open(os.path.join(ADIR, "dataset-metadata.json"), "w"))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", ADIR], capture_output=True, text=True)
        if "already exists" in (r.stdout + r.stderr) or r.returncode != 0:
            r = subprocess.run(["kaggle", "datasets", "version", "-p", ADIR, "-m", "n2000"],
                               capture_output=True, text=True)
        print(f"[adapter] subida: rc={r.returncode} {r.stdout[-200:]}", flush=True)
except Exception as e:
    print(f"[adapter] no subido (ok, best-effort): {e}", flush=True)
