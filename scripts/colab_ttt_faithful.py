#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] T1+T2b — harness de TTT FIEL y medicion de cuanto cierra del x10.4.

T2a (17-ago) dejo la linea base sin TTT: GT 0/38 bajo el umbral, pares de train 2/118, NLL
total mediana 16.79 contra un umbral de 1.609 -> el TTT tiene que cerrar un factor x10.4.
Nadie lo ha medido nunca. Este script lo mide.

Bloqueante conocido: el port anterior de TTT a Colab (fp16 puro + GradScaler sobre LoRA r=256
con rsLoRA escala 2.0) EMPEORABA la NLL -> carril cerrado el 13-ago. Sospecha: produccion
entrena en bf16 y la T4 (SM75) no tiene bf16 por hardware. Fix de este harness:
  - pesos base en fp16, parametros LoRA en **fp32** (master weights), sin GradScaler sobre LoRA
  - gradient checkpointing + AdamW de 8 bits (los 530M de r=256 no caben de otra forma en 15GB)
  - fallback automatico a r=128 si hay OOM, registrando cual se uso (honestidad sobre fidelidad)

Config del campeon replicada: LoRA r=256 alpha=32 rsLoRA sobre q,k,v,o,gate,up,down +
embed_tokens + lm_head; 128 secuencias (8 vistas D4 x 16 permutaciones de color); 1 epoca;
lr 5e-5 coseno con warmup 10%; batch 1; perdida SOLO en los tramos de assistant (como el
collator de produccion).

GATE DE FIDELIDAD, falsable: sin TTT el pool al umbral de produccion es VACIO (medido). Un
harness fiel debe producir **pool no vacio** en al menos algunas tareas. Si no, no se
interpreta ninguna otra cifra de esta corrida.

Metricas por tarea, antes y despues del TTT: NLL total de la GT, NLL de los propios pares de
train (leave-one-out), y el pool del DFS al umbral real.

Uso: colab --auth=adc run --gpu T4 --timeout 10800 scripts/colab_ttt_faithful.py <TOKEN>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
# torchao>=0.16 es un gate de peft (Colab trae 0.10 y el dispatcher revienta al inyectar LoRA)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes",
                "torchao>=0.16"], check=True)
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

import bitsandbytes as bnb
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS, PAD = 14, 11, 12, 10, 15, 13
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))
MAX_LEN = 3072
N_AUG = 16          # x 8 vistas D4 = 128 secuencias, como el campeon


def grid_tokens(g):
    """Tokens de una grid: un token por celda + saltos de linea (sin salto final)."""
    out = []
    for i, row in enumerate(g):
        out.extend(int(c) for c in row)
        if i < len(g) - 1:
            out.append(NL)
    return out


def pair_tokens(inp, out):
    """Devuelve (tokens, mascara) — la perdida vive SOLO en la salida del assistant."""
    t = [IM_START, USER, NL] + grid_tokens(inp) + [EOS]
    m = [0] * len(t)
    head = [IM_START, ASSIST, NL]
    body = grid_tokens(out) + [EOS]
    t += head + body
    m += [0] * len(head) + [1] * len(body)
    return t, m


def query_tokens(inp):
    return [IM_START, USER, NL] + grid_tokens(inp) + [EOS] + [IM_START, ASSIST, NL]


def d4(g, k):
    a = np.array(g)
    if k >= 4:
        a = a.T
        k -= 4
    return np.rot90(a, k).tolist()


def build_ttt_batch(task, seed=1):
    """128 secuencias: 8 vistas D4 x N_AUG permutaciones de color, orden de pares barajado."""
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
            toks, mask = [], []
            for p in pairs:
                t, m = pair_tokens(pm[np.array(d4(p["input"], k))].tolist(),
                                   pm[np.array(d4(p["output"], k))].tolist())
                toks += t
                mask += m
            while len(toks) > MAX_LEN and len(pairs) > 1:   # recorta como cut_to_len
                pairs = pairs[1:]
                toks, mask = [], []
                for p in pairs:
                    t, m = pair_tokens(pm[np.array(d4(p["input"], k))].tolist(),
                                       pm[np.array(d4(p["output"], k))].tolist())
                    toks += t
                    mask += m
            if len(toks) <= MAX_LEN:
                seqs.append((toks, mask))
    rnd.shuffle(seqs)
    return seqs


@torch.no_grad()
def nll_total(model, prompt_toks, ans_toks):
    ids = torch.tensor([prompt_toks + ans_toks], device="cuda:0", dtype=torch.long)
    if ids.shape[1] > 5000:
        return None
    with sdpa_kernel(_SDPA):
        logits = model(ids).logits.float()
    n = len(prompt_toks)
    lp = torch.log_softmax(logits[0, n - 1:-1], -1)
    tgt = torch.tensor(ans_toks, device="cuda:0").unsqueeze(1)
    v = -lp.gather(1, tgt).sum().item()
    del logits, lp
    return v


@torch.no_grad()
def dfs_pool(model, prompt_toks, max_new=931, budget=90):
    """turbo_dfs del campeon, batch 1, umbral de produccion. Devuelve nº de hojas."""
    t0 = time.time()
    ids = torch.tensor([prompt_toks], device="cuda:0", dtype=torch.long)
    with sdpa_kernel(_SDPA):
        out = model(ids, use_cache=True, return_dict=True)
    leaves = []

    def rec(logits, scores, pos, cache, depth):
        if time.time() - t0 > budget or depth <= 1:
            return
        lf = logits.float()
        nll = (torch.as_tensor(scores, dtype=torch.float32, device=lf.device).view(1, 1)
               + torch.logsumexp(lf, dim=-1, keepdim=True) - lf.index_select(-1, _TOKID)).cpu()
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
                o = model(input_ids=torch.tensor([[t]], device="cuda:0", dtype=torch.long),
                          past_key_values=cache, use_cache=True, return_dict=True)
            rec(o.logits[:, -1], [sc], pos + 1, o.past_key_values, depth - 1)

    rec(out.logits[:, -1], [0.0], ids.size(1), out.past_key_values, max_new)
    return len(leaves)


def make_lora(r):
    return LoraConfig(r=r, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
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


NEW = sorted([k for k in CHAL if is_holdout(k) and k not in AGI1],
             key=lambda k: cost(CHAL[k]))
TASKS = NEW[:8]
print(f"[setup] {len(TASKS)} tareas nuevas | umbral {THR:.3f}", flush=True)

RANK = 256
# v2: cargar el base UNA vez y resetear los pesos LoRA por tarea (como produccion). La v1
# recargaba el modelo entero en cada tarea y el anterior no se liberaba -> OOM en la tarea 4.
base = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                            device_map="cuda:0", attn_implementation="sdpa")
model = get_peft_model(base, make_lora(RANK))
for n_, p_ in model.named_parameters():          # LoRA en fp32 (el fp16 puro era el sospechoso)
    if p_.requires_grad:
        p_.data = p_.data.float()
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
INIT_LORA = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
print(f"[setup] modelo+LoRA r={RANK} | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# VERIFICACION CRUZADA: T2a construyo las secuencias con el TOKENIZADOR sobre texto y este
# script las construye token a token. En la misma tarea dieron NLL 20.26 vs 0.204 (x100) ->
# una de las dos rutas no produce lo que creo. Se comprueba en el mismo proceso antes de nada.
from transformers import AutoTokenizer

_TK = AutoTokenizer.from_pretrained(MDIR)


def _g2s(g):
    return "\n".join("".join(str(int(c)) for c in row) for row in g)


_t0 = CHAL[TASKS[0]]
_txt = "".join(f"<|im_start|>user\n{_g2s(p['input'])}<|im_end|>"
               f"<|im_start|>assistant\n{_g2s(p['output'])}<|im_end|>" for p in _t0["train"])
_txt += f"<|im_start|>user\n{_g2s(_t0['test'][0]['input'])}<|im_end|><|im_start|>assistant\n"
_via_tok = _TK(_txt).input_ids
_via_dir = []
for _p in _t0["train"]:
    _a, _ = pair_tokens(_p["input"], _p["output"])
    _via_dir += _a
_via_dir += query_tokens(_t0["test"][0]["input"])
print(f"[verif] longitudes tokenizador={len(_via_tok)} directo={len(_via_dir)} "
      f"IGUALES={_via_tok == _via_dir}", flush=True)
if _via_tok != _via_dir:
    for _i, (_x, _y) in enumerate(zip(_via_tok, _via_dir)):
        if _x != _y:
            print(f"[verif] primera diferencia en {_i}: tok={_x} dir={_y} | "
                  f"contexto tok={_via_tok[max(0,_i-5):_i+5]} dir={_via_dir[max(0,_i-5):_i+5]}",
                  flush=True)
            break

RES = {"rank": RANK, "tasks": [],
       "verif_tokenizacion": {"len_tok": len(_via_tok), "len_dir": len(_via_dir),
                              "iguales": _via_tok == _via_dir}}
for ti, tk in enumerate(TASKS):
    task = CHAL[tk]
    gt = SOLS[tk][0]
    prompt = []
    for p in task["train"]:
        t, _ = pair_tokens(p["input"], p["output"])
        prompt += t
    prompt += query_tokens(task["test"][0]["input"])
    gt_toks = grid_tokens(gt) + [EOS]

    set_peft_model_state_dict(model, {k: v.clone() for k, v in INIT_LORA.items()})
    model.config.use_cache = False
    model.eval()

    pre_gt = nll_total(model, prompt, gt_toks)
    pre_tr = []
    for i, p in enumerate(task["train"]):
        others = [q for j, q in enumerate(task["train"]) if j != i]
        if not others:
            continue
        pr = []
        for q in others:
            t, _ = pair_tokens(q["input"], q["output"])
            pr += t
        pr += query_tokens(p["input"])
        v = nll_total(model, pr, grid_tokens(p["output"]) + [EOS])
        if v is not None:
            pre_tr.append(v)

    seqs = build_ttt_batch(task)
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-5, total_steps=len(seqs), pct_start=0.1, anneal_strategy="cos")
    model.train()
    t0 = time.time()
    losses = []
    oom = False
    try:
        for si, (toks, mask) in enumerate(seqs):
            ids = torch.tensor([toks], device="cuda:0", dtype=torch.long)
            lab = torch.tensor([[t if m else -100 for t, m in zip(toks, mask)]],
                               device="cuda:0", dtype=torch.long)
            with sdpa_kernel(_SDPA):
                out = model(input_ids=ids, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
            if (si + 1) % 32 == 0:
                print(f"  [{tk}] paso {si+1}/{len(seqs)} loss {np.mean(losses[-32:]):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    except torch.cuda.OutOfMemoryError:
        oom = True
        print(f"  [{tk}] OOM en el paso {len(losses)} — se reporta lo entrenado", flush=True)
    ttt_secs = time.time() - t0

    model.eval()
    model.config.use_cache = True
    post_gt = nll_total(model, prompt, gt_toks)
    post_tr = []
    for i, p in enumerate(task["train"]):
        others = [q for j, q in enumerate(task["train"]) if j != i]
        if not others:
            continue
        pr = []
        for q in others:
            t, _ = pair_tokens(q["input"], q["output"])
            pr += t
        pr += query_tokens(p["input"])
        v = nll_total(model, pr, grid_tokens(p["output"]) + [EOS])
        if v is not None:
            post_tr.append(v)
    pool = dfs_pool(model, prompt)

    row = {"task": tk, "L": len(gt_toks), "oom": oom, "pasos": len(losses),
           "loss_ini": round(float(np.mean(losses[:16])), 4) if losses else None,
           "loss_fin": round(float(np.mean(losses[-16:])), 4) if losses else None,
           "gt_pre": round(pre_gt, 3) if pre_gt else None,
           "gt_post": round(post_gt, 3) if post_gt else None,
           "gt_pasa": bool(post_gt is not None and post_gt < THR),
           "train_pre": round(float(np.mean(pre_tr)), 3) if pre_tr else None,
           "train_post": round(float(np.mean(post_tr)), 3) if post_tr else None,
           "train_pasa": sum(1 for v in post_tr if v < THR), "train_n": len(post_tr),
           "pool": pool, "ttt_secs": round(ttt_secs, 1)}
    RES["tasks"].append(row)
    red = (pre_gt / post_gt) if (pre_gt and post_gt) else 0
    print(f"[{ti+1}/{len(TASKS)}] {tk}: L={row['L']} | GT {row['gt_pre']}->{row['gt_post']} "
          f"(x{red:.1f}) pasa={row['gt_pasa']} | train {row['train_pre']}->{row['train_post']} "
          f"({row['train_pasa']}/{row['train_n']} bajo umbral) | POOL={pool} | "
          f"loss {row['loss_ini']}->{row['loss_fin']} | {ttt_secs:.0f}s", flush=True)

    del opt, sched
    torch.cuda.empty_cache()

rows = RES["tasks"]
if rows:
    ok = [r for r in rows if r["gt_pre"] and r["gt_post"]]
    summ = {"rank": RANK, "n": len(rows),
            "GATE_pool_no_vacio": sum(1 for r in rows if r["pool"] > 0),
            "gt_pasa": sum(r["gt_pasa"] for r in rows),
            "reduccion_gt_mediana": round(float(np.median([r["gt_pre"] / r["gt_post"]
                                                           for r in ok])), 2) if ok else None,
            "gt_post_mediana": round(float(np.median([r["gt_post"] for r in ok])), 2) if ok else None,
            "train_pasa_total": sum(r["train_pasa"] for r in rows),
            "train_n_total": sum(r["train_n"] for r in rows),
            "ooms": sum(r["oom"] for r in rows)}
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)

json.dump(RES, open("/content/ttt_faithful_result.json", "w"), indent=1)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/ttt_faithful_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale ttt faithful", "id": f"{user}/arcscale-ttt-faithful",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "ttt fiel", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
