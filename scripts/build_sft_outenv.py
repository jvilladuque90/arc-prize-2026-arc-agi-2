import io, json, ast, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

WORKER = r'''
import json, math, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
WALL_TRAIN = float(os.environ.get("WALL_TRAIN", "3600"))
N_AUG = int(os.environ.get("N_AUG", "16"))
LR = float(os.environ.get("LR", "1e-4"))
R_LORA = int(os.environ.get("R_LORA", "64"))

torch.cuda.set_device(LOCAL)
if WORLD > 1:
    dist.init_process_group("nccl")
DEV = "cuda:%d" % LOCAL

def log(*a):
    if RANK == 0:
        print("[sft]", *a, flush=True)

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
MAX_SEQ = 3500

# ------------------------------------------------------------------ datos
FICH = "arc-agi_training_challenges.json"
DATA = None
for cand in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
             "/kaggle/input/arc-prize-2026-arc-agi-2"):
    if os.path.exists(os.path.join(cand, FICH)):
        DATA = cand
        break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if FICH in files:
            DATA = root
            break
assert DATA, "sin dataset de la competicion"
CHAL = json.load(open(os.path.join(DATA, FICH)))
SOLS = json.load(open(os.path.join(DATA, "arc-agi_training_solutions.json")))

RASGOS = ("H", "W", "logA", "ncol", "dens")
PISO = {"H": 1.0, "W": 1.0, "logA": 0.10, "ncol": 1.0, "dens": 0.05}

def rasgos(g):
    H, W = len(g), len(g[0])
    plano = [c for f in g for c in f]
    nz = sum(1 for c in plano if c != 0)
    return {"H": float(H), "W": float(W), "logA": math.log(H * W),
            "ncol": float(len(set(plano))), "dens": nz / (H * W)}

def d_env(demos, obj):
    peor = 0.0
    for f in RASGOS:
        vs = [r[f] for r in demos]
        lo, hi, t = min(vs), max(vs), obj[f]
        R = max(hi - lo, PISO[f])
        e = 0.0 if lo <= t <= hi else (t - hi) / R if t > hi else (lo - t) / R
        peor = max(peor, e)
    return peor

def pares_de(k):
    ps = [(p["input"], p["output"]) for p in CHAL[k]["train"]]
    for i, t in enumerate(CHAL[k]["test"]):
        try:
            ps.append((t["input"], SOLS[k][i]))
        except Exception:
            pass
    return ps

def resplits():
    """Re-particiones REALES: demos = pares de salida pequena, objetivo = salida grande
    FUERA de la envolvente de las salidas demo. Es el regimen en el que el campeon
    acierta 11% (frente al 75% dentro de la envolvente)."""
    out = []
    for k in sorted(CHAL):
        ps = pares_de(k)
        if len(ps) < 4:
            continue
        idx = sorted(range(len(ps)), key=lambda i: len(ps[i][1]) * len(ps[i][1][0]))
        for j in range(len(idx) - 1, max(len(idx) - 4, 2), -1):
            obj = idx[j]
            demos = [i for i in idx if i != obj
                     and len(ps[i][1]) * len(ps[i][1][0]) <= len(ps[obj][1]) * len(ps[obj][1][0])]
            if len(demos) < 2:
                continue
            if d_env([rasgos(ps[i][1]) for i in demos], rasgos(ps[obj][1])) > 0:
                out.append((k, [ps[i] for i in demos], ps[obj]))
    return out

# ------------------------------------------------------------------ tokenizacion (formato de produccion)
def grid_tokens(g):
    o = []
    for i, row in enumerate(g):
        o.extend(int(c) for c in row)
        if i < len(g) - 1:
            o.append(NL)
    return o

def par_tokens(inp, out):
    t = [IM_START, USER, NL] + grid_tokens(inp) + [EOS]
    head = [IM_START, ASSIST, NL]
    body = grid_tokens(out) + [EOS]
    return t + head + body, len(t) + len(head), len(body)

def d4(g, k):
    a = np.array(g)
    if k >= 4:
        a = a.T
    for _ in range(k % 4):
        a = np.rot90(a)
    return a.tolist()

def aplicar(g, k, perm):
    return [[perm[c] for c in fila] for fila in d4(g, k)]

def construye(rs, n_aug, seed=0):
    """Secuencia = demos (salidas pequenas) + par objetivo (salida FUERA de envolvente).
    La perdida se aplica SOLO al cuerpo de la salida objetivo: cada gradiente empuja
    hacia producir una salida fuera del rango que las demos mostraron."""
    rng = random.Random(seed)
    seqs = []
    for tarea, demos, obj in rs:
        for a in range(n_aug):
            k = rng.randrange(8) if a else 0
            perm = rng.sample(range(10), 10) if a else list(range(10))
            orden = list(range(len(demos)))
            rng.shuffle(orden)
            toks, lab = [], []
            for i in orden:
                di, do = demos[i]
                t, _, _ = par_tokens(aplicar(di, k, perm), aplicar(do, k, perm))
                toks += t
                lab += [-100] * len(t)
            t, ini, nb = par_tokens(aplicar(obj[0], k, perm), aplicar(obj[1], k, perm))
            lab += [-100] * ini + t[ini:]
            toks += t
            if len(toks) <= MAX_SEQ and nb > 0:
                seqs.append((tarea, toks, lab))
    return seqs

RS = resplits()
tareas = sorted({r[0] for r in RS})
random.Random(12345).shuffle(tareas)
n_val = max(20, len(tareas) // 10)
VAL_T = set(tareas[:n_val])
rs_tr = [r for r in RS if r[0] not in VAL_T]
rs_va = [r for r in RS if r[0] in VAL_T]
SEQ_TR = construye(rs_tr, N_AUG, seed=1)
SEQ_VA = construye(rs_va, 2, seed=2)
random.Random(7).shuffle(SEQ_TR)
log("re-particiones %d (train %d / val %d por TAREA) | secuencias train %d val %d"
    % (len(RS), len(rs_tr), len(rs_va), len(SEQ_TR), len(SEQ_VA)))

# ------------------------------------------------------------------ modelo
MDIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR, "sin modelo"
log("modelo en", MDIR)

base = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa")
CFG = LoraConfig(r=R_LORA, lora_alpha=int(2 * math.sqrt(R_LORA)), use_rslora=True,
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
MODEL = get_peft_model(base, CFG)
for _, p in MODEL.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()
MODEL.config.use_cache = False
log("adaptador: %.1fM parametros entrenables"
    % (sum(p.numel() for p in MODEL.parameters() if p.requires_grad) / 1e6))

@torch.no_grad()
def val_nll(seqs):
    """NLL media por token de SALIDA objetivo sobre el conjunto de validacion (tareas
    nunca vistas en el entrenamiento). Mide justo lo que queremos mover."""
    MODEL.eval()
    tot, n = 0.0, 0
    for _, toks, lab in seqs[RANK::WORLD]:
        ids = torch.tensor([toks], device=DEV, dtype=torch.long)
        lb = torch.tensor([lab], device=DEV, dtype=torch.long)
        with sdpa_kernel(_SDPA):
            out = MODEL(input_ids=ids, labels=lb)
        m = (lb[0, 1:] != -100).sum().item()
        tot += out.loss.item() * m
        n += m
    if WORLD > 1:
        t = torch.tensor([tot, n], device=DEV, dtype=torch.float64)
        dist.all_reduce(t)
        tot, n = t[0].item(), t[1].item()
    MODEL.train()
    return tot / max(n, 1)

v0 = val_nll(SEQ_VA)
log("VAL NLL/token ANTES  = %.4f  (si ~0 => sft139 ya lo tiene memorizado y el SFT es vacuo)" % v0)

# ------------------------------------------------------------------ entrenamiento (DDP manual)
if WORLD > 1:
    MODEL = torch.nn.parallel.DistributedDataParallel(
        MODEL, device_ids=[LOCAL], output_device=LOCAL, find_unused_parameters=False)
params = [p for p in MODEL.parameters() if p.requires_grad]
try:
    import bitsandbytes as bnb
    OPT = bnb.optim.AdamW8bit(params, lr=LR)
except Exception:
    OPT = torch.optim.AdamW(params, lr=LR)
MIS = SEQ_TR[RANK::WORLD]
sch = torch.optim.lr_scheduler.CosineAnnealingLR(OPT, T_max=max(len(MIS), 1))
log("pasos por rango: %d | lr %g | r %d" % (len(MIS), LR, R_LORA))

MODEL.train()
t0, hechos, perdida = time.time(), 0, 0.0
for i, (_, toks, lab) in enumerate(MIS):
    if time.time() - t0 > WALL_TRAIN:
        log("WALL_TRAIN alcanzado en el paso %d/%d" % (i, len(MIS)))
        break
    ids = torch.tensor([toks], device=DEV, dtype=torch.long)
    lb = torch.tensor([lab], device=DEV, dtype=torch.long)
    try:
        with sdpa_kernel(_SDPA):
            out = MODEL(input_ids=ids, labels=lb)
        out.loss.backward()
    except torch.cuda.OutOfMemoryError:
        OPT.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        continue
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    OPT.step()
    sch.step()
    OPT.zero_grad(set_to_none=True)
    perdida += out.loss.item()
    hechos += 1
    if hechos % 200 == 0:
        log("paso %d/%d | loss media %.4f | %.0fs" % (hechos, len(MIS), perdida / hechos, time.time() - t0))

log("entrenados %d pasos por rango en %.0fs | loss media %.4f" % (hechos, time.time() - t0, perdida / max(hechos, 1)))

nucleo = MODEL.module if WORLD > 1 else MODEL
globals()["MODEL"] = nucleo
v1 = val_nll(SEQ_VA)
log("VAL NLL/token DESPUES = %.4f  (antes %.4f, cambio %+.1f%%)" % (v1, v0, 100 * (v1 - v0) / max(v0, 1e-9)))

if RANK == 0:
    sd = {k: v.detach().cpu().to(torch.float32) for k, v in get_peft_model_state_dict(nucleo).items()}
    torch.save(sd, "/kaggle/working/sft_outenv_adapter.pt")
    CFG.save_pretrained("/kaggle/working/")
    json.dump({"val_nll_antes": v0, "val_nll_despues": v1, "pasos_por_rango": hechos,
               "n_resplits": len(RS), "n_seq_train": len(SEQ_TR), "n_seq_val": len(SEQ_VA),
               "r": R_LORA, "lr": LR, "wall_s": round(time.time() - T00)},
              open("/kaggle/working/sft_outenv.json", "w"))
    log("adaptador guardado (%.1f MB)" % (sum(v.numel() * 4 for v in sd.values()) / 1e6))
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
log("fin %.0fs" % (time.time() - T00))
'''

LANZADOR = r'''import os, subprocess, sys, time
T0 = time.time()
env = dict(os.environ)
env["WALL_TRAIN"] = "3300"
env["N_AUG"] = "16"
env["LR"] = "1e-4"
env["R_LORA"] = "64"
env["OMP_NUM_THREADS"] = "2"
env["TOKENIZERS_PARALLELISM"] = "false"
cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
       "--nproc_per_node=4", "/kaggle/working/_worker_src.py"]
print("lanzando DDP en 4 GPUs:", " ".join(cmd), flush=True)
p = subprocess.run(cmd, env=env)
print("torchrun rc=%d tras %.0fs" % (p.returncode, time.time() - T0), flush=True)

# respaldo: si DDP no produjo resultado, reintentar en UNA sola GPU (mas lento pero seguro)
if not os.path.exists("/kaggle/working/sft_outenv.json"):
    print("DDP no dejo resultado -> respaldo mono-GPU", flush=True)
    env2 = dict(env)
    env2.update({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
                 "CUDA_VISIBLE_DEVICES": "0",
                 "WALL_TRAIN": str(max(600, 3300 - int(time.time() - T0)))})
    for k in ("MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        env2.pop(k, None)
    q = subprocess.run([sys.executable, "/kaggle/working/_worker_src.py"], env=env2)
    print("respaldo rc=%d tras %.0fs" % (q.returncode, time.time() - T0), flush=True)

import json
try:
    d = json.load(open("/kaggle/working/sft_outenv.json"))
    print("")
    print("=" * 78, flush=True)
    print("SFT fuera-de-envolvente: resultado", flush=True)
    print("=" * 78, flush=True)
    for k, v in d.items():
        print("   %-18s %s" % (k, v), flush=True)
    a, b = d["val_nll_antes"], d["val_nll_despues"]
    print("")
    print("VAL NLL/token en TAREAS NUNCA VISTAS: %.4f -> %.4f (%+.1f%%)" % (a, b, 100 * (b - a) / a), flush=True)
    print("PRERREG: mejora >=10%% => el adaptador aprende algo transferible => medir en el LB", flush=True)
    print("         mejora <10%% o empeora => no hay senal => no gastar un slot", flush=True)
    print("         val_nll_antes ~0 => sft139 ya memoriza estos pares => el banco es VACUO", flush=True)
except Exception as e:
    print("sin resultado:", e, flush=True)
    for f in ("/kaggle/working/sft_outenv.json",):
        print(f, os.path.exists(f), flush=True)
'''

MD = """# SFT fuera-de-envolvente: entrenar a producir salidas que las demos no mostraron

Medido el 9-sep sobre 23 tareas de eval con resultado del campeon: la `d` de la **entrada** de test
**no predice nada** (Spearman +0,004), pero la de la **salida** si (-0,234). Tasa condicionada:
**75%** de GT-en-pool cuando la salida cae DENTRO de la envolvente de las salidas demo (18% de la
poblacion) frente a **11%** cuando cae fuera (82%).

**Datos: re-particiones REALES, no sinteticos.** La puerta del DSL demostro que las
transformaciones globales de rejilla que yo generaria no aparecen en AGI-2 (0 de 353), asi que
generar sinteticos entrenaria la distribucion equivocada y con riesgo de atajo. En su lugar se
re-parten las 1000 tareas de entrenamiento por **tamano de salida**: demos = pares de salida
pequena, objetivo = par cuya salida cae FUERA de la envolvente. 803 re-particiones de 625 tareas.

**La perdida se aplica SOLO al cuerpo de la salida objetivo**: cada gradiente empuja a producir una
salida fuera del rango que las demos mostraron — exactamente lo contrario de lo que hace el TTT hoy,
donde las salidas demo son mas pequenas en el 94,7% de las tareas.

Split de validacion **por TAREA** (no por secuencia): mide transferencia, no memorizacion.
Formato de tokens identico a produccion, con marcadores de rol. DDP en las 4 L4.

**PRERREG**: mejora de VAL NLL >=10% => hay senal transferible => medir en el LB. <10% o peor =>
no gastar un slot. Y si `val_nll_antes` sale ~0, `sft139` ya memoriza estos pares y el banco es vacuo.
"""

def celda(src, tipo="code"):
    d = {"cell_type": tipo, "metadata": {}, "source": src.splitlines(keepends=True)}
    if tipo == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d

ast.parse(WORKER)
ast.parse(LANZADOR)
nb = {"cells": [celda(MD, "markdown"),
                celda("%%writefile /kaggle/working/_worker_src.py\n" + WORKER.lstrip("\n")),
                celda(LANZADOR)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/sft_outenv.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

assert "nproc_per_node=4" in LANZADOR
assert "lab += [-100] * ini + t[ini:]" in WORKER      # perdida solo en la salida objetivo
assert "VAL_T" in WORKER and "r[0] not in VAL_T" in WORKER   # split por TAREA
assert "IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15" in WORKER
assert "arc-agi_training_challenges.json" in WORKER and "evaluation" not in WORKER

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # r-buyer (9-sep)'
entry = '''    # SFT fuera-de-envolvente (9-sep), L4x4 con DDP en las 4 GPUs. Entrena sft139 a producir
    # salidas cuyas dimensiones/estadistica caen FUERA del rango que mostraron las demos, que es
    # el regimen donde el campeon pasa de 75% a 11% de GT-en-pool y que cubre el 82% del eval.
    # Datos: re-particiones REALES del training set por tamano de salida (803 de 625 tareas), NO
    # sinteticos -- la puerta del DSL probo que los sinteticos que yo generaria no existen en AGI-2.
    # Perdida solo en el cuerpo de la salida objetivo. Val por TAREA. PRERREG: >=10% de mejora.
    "sft-outenv": {"notebook": "notebooks/sft_outenv.ipynb", "slug": "arc-agi2-sft-outenv",
                   "title": "arc agi2 sft outenv", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
'''
if '"sft-outenv"' not in s:
    assert anchor in s, "ancla no encontrada"
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado sft-outenv")
else:
    print("sft-outenv ya estaba")
print("OK notebooks/sft_outenv.ipynb (worker %d B, lanzador %d B)" % (len(WORKER), len(LANZADOR)))
