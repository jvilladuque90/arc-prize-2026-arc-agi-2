#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] ¿Mas EPOCAS de TTT llegan al objetivo, o solo profundizan el sobreajuste?

Pregunta del usuario (18-ago): entrenar, equivocarse, actualizar y volver a entrenar -- ¿con
mas pasos se alcanza la respuesta? Nota: en el test NO hay señal de error que realimentar (no
tenemos la respuesta), asi que "volver a entrenar" solo puede significar MAS pasos sobre las
mismas demostraciones.

Prediccion pre-registrada (antes de correr): EMPEORA. En 1 epoca el TTT ya lleva la perdida a
0.0001 y pone 27/28 pares de train bajo el umbral -- no queda nada por aprender de ellos. Y en
4/10 tareas la GT ya empeoraba tras la primera epoca.

Diseño: entrenamiento continuo de 3 epocas (128 secuencias cada una) y medicion en CADA epoca:
NLL de la GT, pool del DFS al umbral de produccion, y si la GT esta dentro. Dos bases: original
y fusionada con el LoRA de ARC-Heavy. Caveat declarado: el coseno se programa sobre las 3
epocas, asi que el punto de la epoca 1 no es identico a la config del campeon (1 epoca con su
propio coseno); lo que se lee es la TENDENCIA.

Lectura:
  - resueltas suben con las epocas -> mas TTT paga y hay que re-tunearlo hacia arriba.
  - planas o bajando           -> confirma que el TTT esta pasado de vueltas y refuerza
                                  quitarlo (o recortarlo) para recuperar el 34.8% del tiempo.

Uso: colab --auth=adc run --gpu T4 --timeout 14400 scripts/colab_2b_epochs.py <TOKEN>
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes",
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
subprocess.run(["kaggle", "datasets", "download", "-d",
                "juliancamilovilla/arcscale-adapter-2b", "-p", "/content/ad",
                "--unzip", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m2b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR
ADIR = None
for root, dirs, files in os.walk("/content/ad"):
    if "adapter_model.safetensors" in files:
        ADIR = root
        break
assert ADIR, "adapter 2B no encontrado"

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] listo | base={MDIR} adapter={ADIR} ({time.time()-T00:.0f}s)", flush=True)

import hashlib
import random as _rnd

import bitsandbytes as bnb
import numpy as np
import torch
from peft import (LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))
MAX_LEN = 2048
N_AUG = 16


def grid_tokens(g):
    out = []
    for i, row in enumerate(g):
        out.extend(int(c) for c in row)
        if i < len(g) - 1:
            out.append(NL)
    return out


def pair_tokens(inp, out):
    t = [IM_START, USER, NL] + grid_tokens(inp) + [EOS]
    m = [0] * len(t)
    head = [IM_START, ASSIST, NL]
    body = grid_tokens(out) + [EOS]
    return t + head + body, m + [0] * len(head) + [1] * len(body)


def query_tokens(inp):
    return [IM_START, USER, NL] + grid_tokens(inp) + [EOS] + [IM_START, ASSIST, NL]


def d4(g, k):
    a = np.array(g)
    if k >= 4:
        a = a.T
        k -= 4
    return np.rot90(a, k).tolist()


def build_ttt(task, seed=1):
    rnd = _rnd.Random(seed)
    seqs = []
    for k in range(8):
        for _ in range(N_AUG):
            perm = list(range(10))
            tail = perm[1:]
            rnd.shuffle(tail)
            perm[1:] = tail
            pm = np.array(perm)
            pairs = list(task["train"])
            rnd.shuffle(pairs)
            while True:
                toks, mask = [], []
                for p in pairs:
                    t, m = pair_tokens(pm[np.array(d4(p["input"], k))].tolist(),
                                       pm[np.array(d4(p["output"], k))].tolist())
                    toks += t
                    mask += m
                if len(toks) <= MAX_LEN or len(pairs) <= 1:
                    break
                pairs = pairs[1:]
            if len(toks) <= MAX_LEN:
                seqs.append((toks, mask))
    rnd.shuffle(seqs)
    return seqs


@torch.no_grad()
def prefill(model, toks, chunk=512):
    cache, out = None, None
    for i in range(0, len(toks), chunk):
        with sdpa_kernel(_SDPA):
            out = model(input_ids=torch.tensor([toks[i:i + chunk]], device="cuda:0",
                                               dtype=torch.long),
                        past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
    return out.logits[:, -1], cache


@torch.no_grad()
def nll_gt(model, prompt, ans):
    last, cache = prefill(model, prompt)
    parts = [last.float()]
    if len(ans) > 1:
        with sdpa_kernel(_SDPA):
            o = model(input_ids=torch.tensor([ans[:-1]], device="cuda:0", dtype=torch.long),
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
def dfs(model, prompt, depth, budget=60):
    t0 = time.time()
    last, cache0 = prefill(model, prompt)
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
                    leaves.append(list(path))
                else:
                    cands.append((sc, t))
        for sc, t in sorted(cands):
            if time.time() - t0 > budget:
                return
            with sdpa_kernel(_SDPA):
                o = model(input_ids=torch.tensor([[t]], device="cuda:0", dtype=torch.long),
                          past_key_values=cache, use_cache=True, return_dict=True)
            path.append(t)
            rec(o.logits[:, -1], sc, pos + 1, o.past_key_values, left - 1)
            path.pop()

    rec(last, 0.0, len(prompt), cache0, depth)
    return leaves, time.time() - t0


LCFG = LoraConfig(r=256, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  use_rslora=True,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                  "up_proj", "down_proj", "embed_tokens", "lm_head"])

CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


TASKS = sorted([k for k in CHAL if is_holdout(k) and k not in AGI1],
               key=lambda k: cost(CHAL[k]))[:6]
print(f"[setup] {len(TASKS)} tareas nuevas (no-AGI1) | umbral {THR:.3f}", flush=True)

DS = f"{os.environ.get('KAGGLE_USERNAME', 'juliancamilovilla')}/arcscale-2b-epocas"


def persistir(RES, etiqueta):
    """Sube el parcial tras CADA tarea. Colab libre reclama la VM a los ~65 min (medido
    19-ago) y esta corrida pide ~2h: sin esto se pierde entera, como paso con la curva de
    epocas del LoRA."""
    try:
        os.makedirs("/content/persist", exist_ok=True)
        json.dump(RES, open("/content/persist/lora_ttt_2x2.json", "w"), indent=1)
        json.dump({"title": "arcscale 2b epocas ttt", "id": DS,
                   "licenses": [{"name": "CC0-1.0"}]},
                  open("/content/persist/dataset-metadata.json", "w"))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                           capture_output=True, text=True)
        if "error" in (r.stdout + r.stderr).lower():
            subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                            "-m", etiqueta, "-q"], check=False, capture_output=True)
    except Exception as e:                                # noqa: BLE001
        print(f"[persist] fallo: {e}", flush=True)


RES = {"celdas": {}, "tasks": []}
# Reanudacion: si el dataset ya trae tareas medidas, se saltan (cada tarea es independiente,
# el TTT se reinicia desde INIT en cada una, asi que trocear no altera el resultado).
if os.environ.get("REANUDAR", "0") == "1":
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", DS, "-p", "/content/prev",
                        "--unzip", "-q"], check=True)
        RES["tasks"] = json.load(open("/content/prev/lora_ttt_2x2.json"))["tasks"]
        print(f"[reanudar] {len(RES['tasks'])} tareas ya medidas", flush=True)
    except Exception as e:                                # noqa: BLE001
        print(f"[reanudar] sin parcial previo ({e})", flush=True)
HECHAS = {(r["base"], r["task"]) for r in RES["tasks"]}

for base_kind in ("original", "fusionado"):
    if all((base_kind, t) in HECHAS for t in TASKS):
        print(f"[{base_kind}] ya completo, se salta", flush=True)
        continue
    base = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                                device_map="cuda:0",
                                                attn_implementation="sdpa")
    if base_kind == "fusionado":
        base = PeftModel.from_pretrained(base, ADIR).merge_and_unload()
    model = get_peft_model(base, LCFG)
    for _, p_ in model.named_parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    INIT = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    print(f"[{base_kind}] modelo listo | mem={torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)

    for ti, tk in enumerate(TASKS):
        if (base_kind, tk) in HECHAS:
            print(f"[{base_kind} {ti+1}/{len(TASKS)}] {tk}: ya medida, se salta", flush=True)
            continue
        task = CHAL[tk]
        gt = SOLS[tk][0]
        prompt = []
        for p in task["train"]:
            t, _ = pair_tokens(p["input"], p["output"])
            prompt += t
        prompt += query_tokens(task["test"][0]["input"])
        gt_toks = grid_tokens(gt) + [EOS]
        depth = min(931, len(gt_toks) + 40)

        set_peft_model_state_dict(model, {k: v.clone() for k, v in INIT.items()})
        model.config.use_cache = True
        model.eval()
        n0 = nll_gt(model, prompt, gt_toks)
        lv0, s0 = dfs(model, prompt, depth)
        ok0 = any(toks_to_grid(l) == gt for l in lv0)

        model.config.use_cache = False
        model.train()
        seqs = build_ttt(task)
        EPOCAS = 3
        opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5,
                                                  total_steps=len(seqs) * EPOCAS,
                                                  pct_start=0.1, anneal_strategy="cos")
        por_epoca = []
        t0 = time.time()
        losses = []
        try:
            for ep in range(EPOCAS):
                for si, (toks, mask) in enumerate(seqs):
                    ids = torch.tensor([toks], device="cuda:0", dtype=torch.long)
                    lab = torch.tensor([[t if m else -100 for t, m in zip(toks, mask)]],
                                       device="cuda:0", dtype=torch.long)
                    with sdpa_kernel(_SDPA):
                        out = model(input_ids=ids, labels=lab)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    sch.step()
                    opt.zero_grad(set_to_none=True)
                    losses.append(out.loss.item())
                # medicion AL FINAL DE CADA EPOCA
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                model.eval()
                model.config.use_cache = True
                ne = nll_gt(model, prompt, gt_toks)
                lve, _se = dfs(model, prompt, depth)
                oke = any(toks_to_grid(l) == gt for l in lve)
                por_epoca.append({"epoca": ep + 1, "nll": round(ne, 3), "pool": len(lve),
                                  "ok": bool(oke),
                                  "loss": round(float(np.mean(losses[-32:])), 5)})
                print(f"    [{tk}] epoca {ep+1}: NLL {ne:8.3f} pool {len(lve)} "
                      f"RESUELVE {oke} (loss {np.mean(losses[-32:]):.5f})", flush=True)
                model.config.use_cache = False
                model.train()
        except torch.cuda.OutOfMemoryError:
            print(f"  [{tk}] OOM en paso {len(losses)}", flush=True)
        ttt_s = time.time() - t0
        del opt, sch
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        model.eval()
        model.config.use_cache = True
        n1 = nll_gt(model, prompt, gt_toks)
        lv1, s1 = dfs(model, prompt, depth)
        ok1 = any(toks_to_grid(l) == gt for l in lv1)

        RES["tasks"].append({"base": base_kind, "task": tk, "L": len(gt_toks),
                             "nll_sin": round(n0, 3), "nll_con": round(n1, 3),
                             "pool_sin": len(lv0), "pool_con": len(lv1),
                             "ok_sin": bool(ok0), "ok_con": bool(ok1),
                             "seg_sin": round(s0, 1), "seg_con": round(s1, 1),
                             "por_epoca": por_epoca,
                             "ttt_s": round(ttt_s, 1), "pasos": len(losses),
                             "loss_fin": round(float(np.mean(losses[-16:])), 5) if losses else None})
        print(f"[{base_kind} {ti+1}/{len(TASKS)}] {tk}: NLL {n0:8.3f}->{n1:8.3f} | "
              f"pool {len(lv0)}->{len(lv1)} | RESUELVE {ok0}->{ok1} | "
              f"decode {s0:.0f}s->{s1:.0f}s | ttt {ttt_s:.0f}s", flush=True)
        persistir(RES, f"{base_kind}-{tk}")
        torch.cuda.empty_cache()

    del model, base
    torch.cuda.empty_cache()

for bk in ("original", "fusionado"):
    rs = [r for r in RES["tasks"] if r["base"] == bk]
    if not rs:
        continue
    RES["celdas"][bk] = {
        "sin_TTT": {"resueltas": sum(r["ok_sin"] for r in rs),
                    "pool": sum(r["pool_sin"] for r in rs),
                    "nll_mediana": round(float(np.median([r["nll_sin"] for r in rs])), 3),
                    "decode_s": round(sum(r["seg_sin"] for r in rs), 1)},
        "con_TTT": {"resueltas": sum(r["ok_con"] for r in rs),
                    "pool": sum(r["pool_con"] for r in rs),
                    "nll_mediana": round(float(np.median([r["nll_con"] for r in rs])), 3),
                    "decode_s": round(sum(r["seg_con"] for r in rs), 1)},
        "n": len(rs)}
for bk in ("original", "fusionado"):
    rs = [r for r in RES["tasks"] if r["base"] == bk]
    if rs and rs[0].get("por_epoca"):
        curva = {}
        for e in range(1, 4):
            ok = sum(1 for r in rs for x in r["por_epoca"] if x["epoca"] == e and x["ok"])
            nl = [x["nll"] for r in rs for x in r["por_epoca"] if x["epoca"] == e]
            curva[f"epoca{e}"] = {"resueltas": ok,
                                  "nll_mediana": round(float(np.median(nl)), 3) if nl else None}
        RES["celdas"][bk]["curva_epocas"] = curva
        RES["celdas"][bk]["sin_TTT_resueltas"] = sum(r["ok_sin"] for r in rs)
print("RESULT_JSON:" + json.dumps(RES["celdas"]), flush=True)

json.dump(RES, open("/content/lora_ttt_2x2.json", "w"), indent=1)
persistir(RES, "final")
print("[persist] subido a Kaggle", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
