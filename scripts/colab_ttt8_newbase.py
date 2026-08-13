#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] ttt8 sobre la BASE NUEVA: ¿con mejor prior bastan 8 augs de TTT?

D1-bis confirmo transferencia genuina del SFT con ARC-Heavy (0/24 -> 7/24 exactas en tareas
NUEVAS de AGI-2, NLL mejora 24/24). Pero el pipeline campeon hace TTT por tarea: si el TTT ya
captura lo mismo, el SFT no mueve el LB. Este script decide el EV de D4 (SFT del 4B en Kaggle):

Brazos PAREADOS sobre 12 tareas nuevas (cada tarea, TTT fresco n_aug=16 como el campeon):
  A: base 2B + TTT
  B: base 2B con adapter ARC-Heavy n=2000 FUSIONADO (merge) + TTT
Metricas post-TTT: NLL teacher-forced, cell-acc greedy, exactas.

Lectura pre-registrada: si B > A claramente en exactas/cell-acc post-TTT -> el SFT compone
con TTT -> luz verde D4. Si B ~= A -> el TTT ya lo captura -> el SFT no paga en produccion.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_ttt_adapter.py <KAGGLE_API_TOKEN>
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

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "kaggle",
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
assert MDIR

subprocess.run(["kaggle", "datasets", "download", "-d", "juliancamilovilla/arcscale-adapter-2b",
                "-p", "/content/adapter", "--unzip", "-q"], check=True)
ADIR = "/content/adapter"
assert os.path.exists(os.path.join(ADIR, "adapter_model.safetensors"))

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] descargas listas | AGI-1 ids: {len(AGI1)} ({time.time()-T00:.0f}s)", flush=True)

import hashlib
import random

import numpy as np
import torch
from peft import (LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(MDIR)


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


def make_ttt_examples(task, n_aug=16, seed=1):
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
NEW = [k for k in HOLD if k not in AGI1]
KEYS = NEW[::2][:12]   # 12 nuevas por el espectro de coste
print(f"[setup] {len(NEW)} nuevas; brazos sobre {len(KEYS)}", flush=True)

ITEMS = {}
for k in KEYS:
    t = CHAL[k]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(g2s(SOLS[k][0]) + "<|im_end|>", return_tensors="pt").input_ids[0]
    ITEMS[k] = (pi, ai)

LCFG = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])


def run_arm(name, with_adapter, n_aug=16):
    global MODEL
    base = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                                device_map="cuda:0")
    if with_adapter:
        base = PeftModel.from_pretrained(base, ADIR).merge_and_unload()
    MODEL = get_peft_model(base, LCFG)
    MODEL.enable_input_require_grads()
    MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    MODEL.config.use_cache = False
    init = {k: v.clone() for k, v in get_peft_model_state_dict(MODEL).items()}

    def nll_ids(pi, ai):
        with torch.no_grad():
            ids = torch.cat([pi, ai]).unsqueeze(0).to("cuda:0")
            logits = MODEL(ids).logits.float()
            lp = torch.log_softmax(logits[0, pi.numel() - 1:-1], -1)
            return -lp.gather(1, ai.unsqueeze(1).to("cuda:0")).mean().item()

    def greedy(key, max_new=960):
        pi, _ = ITEMS[key]
        ids = pi.unsqueeze(0).to("cuda:0")
        MODEL.config.use_cache = True
        with torch.no_grad():
            out = MODEL.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=TOK.eos_token_id)
        MODEL.config.use_cache = False
        txt = TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        try:
            pred = [[int(c) for c in line if c.isdigit()]
                    for line in txt.strip().split("\n") if line.strip()]
            gt = SOLS[key][0]
            if not pred or len(pred) != len(gt) or any(len(r) != len(gt[0]) for r in pred):
                return 0.0, False
            p, g = np.array(pred), np.array(gt)
            return float((p == g).mean()), bool((p == g).all())
        except Exception:
            return 0.0, False

    rows = []
    for i, k in enumerate(KEYS):
        set_peft_model_state_dict(MODEL, {kk: v.clone() for kk, v in init.items()})
        MODEL.eval()
        pi, ai = ITEMS[k]
        nll0 = nll_ids(pi, ai)
        # --- TTT fresco (config del campeon proxy: n_aug=16, lr 5e-5) ---
        texts = make_ttt_examples(CHAL[k], n_aug=n_aug)
        enc = [TOK(t, return_tensors="pt", truncation=True, max_length=3072).input_ids[0]
               for t in texts]
        opt = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=5e-5)
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
        ttt_secs = time.time() - t0
        MODEL.eval()
        nll1 = nll_ids(pi, ai)
        acc, exact = greedy(k)
        rows.append({"task": k, "nll_pre": round(nll0, 4), "nll_post": round(nll1, 4),
                     "cell_acc": round(acc, 4), "exact": exact, "ttt_secs": round(ttt_secs, 1)})
        print(f"[{name} {i+1}/{len(KEYS)}] {k}: nll {nll0:.3f}->{nll1:.3f} acc={acc:.2f} "
              f"exact={exact} ttt={ttt_secs:.0f}s", flush=True)
    del MODEL, base
    torch.cuda.empty_cache()
    return rows


RES = {"arms": {}}
RES["arms"]["n16"] = run_arm("n16", with_adapter=True, n_aug=16)
RES["arms"]["n8"] = run_arm("n8", with_adapter=True, n_aug=8)

summ = {}
for name, rows in RES["arms"].items():
    summ[name] = {"nll_post_media": round(float(np.mean([r["nll_post"] for r in rows])), 4),
                  "cell_acc_media": round(float(np.mean([r["cell_acc"] for r in rows])), 4),
                  "exactas": sum(r["exact"] for r in rows)}
a = {r["task"]: r for r in RES["arms"]["n16"]}
b = {r["task"]: r for r in RES["arms"]["n8"]}
summ["pareado"] = {
    "tareas_donde_n8_mejora_nll_post": sum(1 for k in a if b[k]["nll_post"] < a[k]["nll_post"]),
    "exactas_ganadas_por_n8": sum(1 for k in a if b[k]["exact"] and not a[k]["exact"]),
    "exactas_perdidas_por_n8": sum(1 for k in a if a[k]["exact"] and not b[k]["exact"]),
}
RES["resumen"] = summ
json.dump(RES, open("/content/ttt8_newbase_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(summ), flush=True)
