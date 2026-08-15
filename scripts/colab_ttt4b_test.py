#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Test de interaccion TTT x SFT en el 4B: ¿el TTT del campeon ROMPE la base fusionada?

Contexto (SCALE_PLAN): el A/B del 4B fusionado dio 27.08 (-3.06) pese a que su generacion de
candidatos MEJORO (pools 1.6 vs 0.8; GT-en-pool 8/12 vs 5/12, sin TTT). El sospechoso restante
es el TTT de produccion: r=256 + rslora + targets con embed_tokens/lm_head, lr 5e-5 — intenso
sobre una base ya afilada podria sobreescribirla por tarea.

Brazos pareados en 8 tareas nuevas, TTT con la CONFIG EXACTA del campeon (adaptada a T4 con
grad checkpointing): A) 4B original + TTT   B) 4B fusionado + TTT.
Metricas post-TTT: NLL teacher-forced, cell-acc greedy, exactas.

Lectura: B <= A pese a que B-sin-TTT >> A-sin-TTT (11/24 vs 0/24 greedy) -> mecanismo
CONFIRMADO: el TTT necesita re-tunearse para la base nueva (lr menor / menos augs / sin
embed-lm_head) -> graft de un numero. B > A -> el TTT no es el problema; siguiente sospechoso.

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_ttt4b_test.py <KAGGLE_API_TOKEN>
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "torchao>=0.16",
                "bitsandbytes"], check=True)

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
subprocess.run(["kaggle", "kernels", "output", "juliancamilovilla/arc-agi2-scale-sft-4b",
                "-p", "/content/ad", "--file-pattern", "adapter_4b/.*"], check=True)
ADIR = None
for root, dirs, files in os.walk("/content/ad"):
    if "adapter_model.safetensors" in files:
        ADIR = root
        break
assert ADIR

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] descargas listas ({time.time()-T00:.0f}s)", flush=True)

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
KEYS = NEW[2:18:2]   # 8 tareas nuevas chico-medianas
ITEMS = {}
for k in KEYS:
    t = CHAL[k]
    prompt = "".join(fmt_pair(p["input"], p["output"]) for p in t["train"]) + fmt_query(t["test"][0]["input"])
    pi = TOK(prompt, return_tensors="pt").input_ids[0]
    ai = TOK(g2s(SOLS[k][0]) + "<|im_end|>", return_tensors="pt").input_ids[0]
    ITEMS[k] = (pi, ai)
print(f"[setup] brazos sobre {len(KEYS)} tareas", flush=True)

# config del TTT del CAMPEON (lb33_graft cell 5), adaptada a T4 solo en grad checkpointing
LCFG = LoraConfig(r=256, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  use_rslora=True,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj",
                                  "embed_tokens", "lm_head"])


def run_arm(name, with_adapter):
    base = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                                device_map="cuda:0")
    if with_adapter:
        base = PeftModel.from_pretrained(base, ADIR).merge_and_unload()
    model = get_peft_model(base, LCFG)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    init = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}

    def nll_ids(pi, ai):
        with torch.no_grad():
            ids = torch.cat([pi, ai]).unsqueeze(0).to("cuda:0")
            logits = model(ids).logits.float()
            lp = torch.log_softmax(logits[0, pi.numel() - 1:-1], -1)
            return -lp.gather(1, ai.unsqueeze(1).to("cuda:0")).mean().item()

    def greedy(key, max_new=960):
        pi, _ = ITEMS[key]
        ids = pi.unsqueeze(0).to("cuda:0")
        model.config.use_cache = True
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=TOK.eos_token_id)
        model.config.use_cache = False
        txt = TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        try:
            pred = [[int(c) for c in line if c.isdigit()]
                    for line in txt.strip().split("\n") if line.strip()]
            gt = SOLS[key][0]
            ok = pred and len(pred) == len(gt) and all(len(r) == len(gt[0]) for r in pred)
            acc = float((np.array(pred) == np.array(gt)).mean()) if ok else 0.0
            return acc, bool(ok and (np.array(pred) == np.array(gt)).all())
        except Exception:
            return 0.0, False

    rows = []
    for i, k in enumerate(KEYS):
        set_peft_model_state_dict(model, {kk: v.clone() for kk, v in init.items()})
        model.eval()
        pi, ai = ITEMS[k]
        nll0 = nll_ids(pi, ai)
        texts = make_ttt_examples(CHAL[k])
        enc = [TOK(t, return_tensors="pt", truncation=True, max_length=3072).input_ids[0]
               for t in texts]
        # lr 5e-5 cosine 1 epoca como el campeon (por-ejemplo, accum 1). AdamW de 8 bits:
        # los momentos fp32 de r=256 en el 4B no caben en la T4 (OOM medido); identico en
        # ambos brazos -> la lectura DIFERENCIAL se preserva.
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(enc))
        scaler = torch.cuda.amp.GradScaler()
        model.train()
        t0 = time.time()
        for ids in enc:
            ids = ids.unsqueeze(0).to("cuda:0")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(ids, labels=ids)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
        ttt_secs = time.time() - t0
        model.eval()
        nll1 = nll_ids(pi, ai)
        acc, exact = greedy(k)
        rows.append({"task": k, "nll_pre": round(nll0, 4), "nll_post": round(nll1, 4),
                     "cell_acc": round(acc, 4), "exact": exact})
        print(f"[{name} {i+1}/{len(KEYS)}] {k}: nll {nll0:.3f}->{nll1:.3f} acc={acc:.2f} "
              f"exact={exact} ttt={ttt_secs:.0f}s", flush=True)
    summ = {"nll_pre_media": round(float(np.mean([r["nll_pre"] for r in rows])), 4),
            "nll_post_media": round(float(np.mean([r["nll_post"] for r in rows])), 4),
            "cell_acc_media": round(float(np.mean([r["cell_acc"] for r in rows])), 4),
            "exactas": sum(r["exact"] for r in rows), "n": len(rows)}
    print(f"RESULT_ARM:{name}:" + json.dumps(summ), flush=True)
    del model, base
    torch.cuda.empty_cache()
    return {"resumen": summ, "detalle": rows}


RES = {"arms": {}}
RES["arms"]["original_ttt"] = run_arm("original_ttt", with_adapter=False)
RES["arms"]["merged_ttt"] = run_arm("merged_ttt", with_adapter=True)
json.dump(RES, open("/content/ttt4b_result.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps({k: v["resumen"] for k, v in RES["arms"].items()}), flush=True)
