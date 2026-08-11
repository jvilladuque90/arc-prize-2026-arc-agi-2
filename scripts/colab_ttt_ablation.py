#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4 via `colab run`] Ablacion proxy PAREADA: TTT n_aug 16 vs 8 — metricas densas.

Guia el injerto `ttt8` del pipeline campeon SIN gastar quota de Kaggle. Modelo proxy con
competencia ARC real: NVARC Qwen3-2B (18.33% LB; fp16 cabe en T4). Por tarea y brazo:
TTT LoRA fresco -> tiempo de TTT, delta-NLL teacher-forced de la respuesta verdadera (densa)
y cell-accuracy del greedy. Los scores absolutos de un 2B no transfieren (medido: 0 exactas
en <4B); la senal RELATIVA si.

Uso:  colab --auth=adc run --gpu T4 scripts/colab_ttt_ablation.py <KAGGLE_API_TOKEN>
Salida: resumen JSON por stdout + /content/ttt_ablation_result.json en la VM.
"""
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------- setup
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "kaggle"], check=True)

os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)

t0 = time.time()
subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1",
                "-p", "/content/m2b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m2b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR, "modelo no encontrado"
print(f"[setup] modelo en {MDIR} ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- utilidades
import hashlib
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def fmt_pair(inp, out):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n{g2s(out)}<|im_end|>"


def fmt_query(inp):
    return f"<|im_start|>user\n{g2s(inp)}<|im_end|><|im_start|>assistant\n"


def d4(grid, k):
    g = np.array(grid)
    if k >= 4:
        g = g.T
        k -= 4
    return np.rot90(g, k)


def make_ttt_examples(task, n_aug, seed=1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n_aug):
        k = rnd.randrange(8)
        perm = list(range(10))
        tail = perm[1:]
        rnd.shuffle(tail)
        perm[1:] = tail
        pm = np.array(perm)
        pairs = list(task["train"])
        rnd.shuffle(pairs)
        out.append("".join(fmt_pair(pm[d4(p["input"], k)], pm[d4(p["output"], k)]) for p in pairs))
    return out


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
KEYS = HOLD[8:20]
print(f"[setup] {len(HOLD)} holdout; usamos {len(KEYS)}", flush=True)

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16, device_map="cuda:0")
MODEL.config.use_cache = False
LCFG = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
MODEL = get_peft_model(MODEL, LCFG)
INIT = {k: v.clone() for k, v in get_peft_model_state_dict(MODEL).items()}
print("[setup] modelo listo (fp16 + LoRA r=64)", flush=True)


def nll_of_answer(key):
    t = CHAL[key]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    ans = g2s(SOLS[key][0]) + "<|im_end|>"
    pi = TOK(prompt, return_tensors="pt").input_ids
    ai = TOK(ans, return_tensors="pt").input_ids
    ids = torch.cat([pi, ai], 1).to("cuda:0")
    with torch.no_grad():
        logits = MODEL(ids).logits.float()
    lp = torch.log_softmax(logits[0, pi.shape[1] - 1:-1], -1)
    return -lp.gather(1, ai[0].unsqueeze(1).to("cuda:0")).mean().item()


def greedy_cell_acc(key, max_new=900):
    t = CHAL[key]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    ids = TOK(prompt, return_tensors="pt").input_ids.to("cuda:0")
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
        p = np.array(pred)
        g = np.array(gt)
        return float((p == g).mean()), bool((p == g).all())
    except Exception:
        return 0.0, False


def run_ttt(key, n_aug, lr=5e-5):
    set_peft_model_state_dict(MODEL, {k: v.clone() for k, v in INIT.items()})
    texts = make_ttt_examples(CHAL[key], n_aug)
    enc = [TOK(t, return_tensors="pt", truncation=True, max_length=3072).input_ids[0] for t in texts]
    opt = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    MODEL.train()
    t0 = time.time()
    for ids in enc:
        ids = ids.unsqueeze(0).to("cuda:0")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = MODEL(ids, labels=ids)
        scaler.scale(out.loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
    MODEL.eval()
    return time.time() - t0


# ---------------------------------------------------------------- ablacion pareada
RES = {"arms": {}, "tasks": KEYS}
for n_aug in (16, 8):
    arm = []
    for i, k in enumerate(KEYS):
        set_peft_model_state_dict(MODEL, {kk: v.clone() for kk, v in INIT.items()})
        MODEL.eval()
        nll0 = nll_of_answer(k)
        secs = run_ttt(k, n_aug)
        nll1 = nll_of_answer(k)
        acc, exact = greedy_cell_acc(k)
        arm.append({"task": k, "ttt_secs": round(secs, 1), "nll_pre": round(nll0, 4),
                    "nll_post": round(nll1, 4), "delta_nll": round(nll0 - nll1, 4),
                    "cell_acc": round(acc, 4), "exact": exact})
        print(f"[n{n_aug} {i+1}/{len(KEYS)}] {k}: ttt={secs:.0f}s nll {nll0:.3f}->{nll1:.3f} "
              f"acc={acc:.2f} exact={exact}", flush=True)
    RES["arms"][f"n{n_aug}"] = arm

summ = {}
for name, arm in RES["arms"].items():
    summ[name] = {"ttt_secs_media": round(float(np.mean([a["ttt_secs"] for a in arm])), 1),
                  "delta_nll_media": round(float(np.mean([a["delta_nll"] for a in arm])), 4),
                  "cell_acc_media": round(float(np.mean([a["cell_acc"] for a in arm])), 4),
                  "exactas": sum(a["exact"] for a in arm)}
a16 = {a["task"]: a for a in RES["arms"]["n16"]}
a8 = {a["task"]: a for a in RES["arms"]["n8"]}
pares = [(a16[k]["delta_nll"] - a8[k]["delta_nll"], a16[k]["ttt_secs"] - a8[k]["ttt_secs"]) for k in a16]
summ["pareado"] = {
    "ventaja_adaptacion_n16_vs_n8": round(float(np.mean([p[0] for p in pares])), 4),
    "ahorro_ttt_de_n8_secs_media": round(float(np.mean([p[1] for p in pares])), 1),
    "tareas_donde_n8_adapta_igual_o_mejor": sum(1 for p in pares if p[0] <= 0),
}
RES["resumen"] = summ
json.dump(RES, open("/content/ttt_ablation_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(summ), flush=True)
