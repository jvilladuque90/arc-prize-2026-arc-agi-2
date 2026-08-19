#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] ¿Mas EPOCAS del LoRA global llegan al objetivo, o solo saturan?

Pregunta del usuario (18-ago): entrenar el LoRA, ver que no acierta, seguir entrenando --
¿con mas pasos se alcanza el objetivo? Es sobre el LoRA GLOBAL (el SFT con ARC-Heavy), no
sobre el TTT.

Lo ya medido y lo que falta:
  - D1/D2 barrieron el VOLUMEN de datos (N = 2000 -> 4000 -> 8000): exactas 7 -> 8 -> 8. Los
    datos SATURAN. Pero eso es mas datos, no mas pasadas sobre los mismos.
  - Nunca se midio la curva de EPOCAS sobre un corpus fijo. Es lo que hace este script.

Diseño: 2B original + LoRA r=64 (config del SFT: alpha 32, lr 1e-4, accum 8) sobre 2000
ejemplos fijos de ARC-Heavy, 3 epocas, midiendo en la epoca 0 (base), 1, 2 y 3 sobre 10 tareas
NUEVAS del holdout de AGI-2 (no-AGI1): NLL de la GT y si el DFS la encuentra al umbral de
produccion -log(0.2).

Correccion importante respecto a D1/D2: alli las secuencias se construian con el TOKENIZADOR
sobre texto, y el tokenizador de Colab DESCARTA los ids 11/12 ('user'/'assistant'). Aqui los
tokens se construyen directamente CON los marcadores, que es lo que ve produccion.

Prediccion pre-registrada: satura en la epoca 1-2. Los datos ya saturaban con mas volumen, y
mas pasadas sobre el mismo corpus aportan menos diversidad que mas ejemplos.

Lectura:
  - resueltas suben con las epocas -> hay que re-tunear el SFT hacia arriba (mas epocas).
  - planas o bajando               -> el SFT esta en su punto y el limite es de otra cosa.

REANUDABLE (19-ago): Colab libre reclama la VM a los ~65 min, asi que la corrida se parte en
sesiones. Variables de entorno: N_TRAIN (600), EPOCA_INI (0 = de cero; k = carga el adapter de
la epoca k desde el dataset de Kaggle y sigue), EPOCAS (cuantas hacer en esta sesion).

Uso (NO usar `colab run` para esto: ata la corrida al canal de salida del cliente y una caida
se lleva 4h). Sesion persistente + lanzamiento desacoplado:
  colab --auth=adc new -s epocas --gpu T4
  colab --auth=adc upload -s epocas scripts/colab_lora_epochs.py /content/run.py
  colab --auth=adc exec -s epocas -f lanzar.py     # Popen(start_new_session=True) -> log.txt
"""
import json
import os
import random
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
                "datasets"], check=True)
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
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
IM_START, USER, ASSIST, NL, EOS = 14, 11, 12, 10, 15
ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
THR = -float(np.log(0.2))
MAX_LEN = 2048

# Colab libre da sesiones de ~1h (medido 19-ago: la VM murio a los 65 min a mitad de la epoca 1
# con N=2000, que cuesta 62 min/epoca a 1,85 s/ejemplo). El corpus se recorta para que
# epoca+medida quepan holgadamente, y la corrida se hace REANUDABLE: cada sesion carga el
# adapter de la anterior desde Kaggle y sigue. Eso es ademas lo que pide la decision sobre
# saturacion: continuar desde los mismos pesos hasta que deje de mejorar.
N_TRAIN = int(os.environ.get("N_TRAIN", "600"))
EPOCA_INI = int(os.environ.get("EPOCA_INI", "0"))     # 0 = arranca de cero; k = reanuda tras k
EPOCAS = int(os.environ.get("EPOCAS", "2"))           # cuantas epocas hacer en ESTA sesion
DS = os.environ.get("DS_ID") or \
    f"{os.environ.get('KAGGLE_USERNAME', 'juliancamilovilla')}/arcscale-lora-epocas"
# CONTROL DE FORMATO (19-ago): D1/D2 tokenizaban por texto y el `tokenizers` de Colab se comia
# los ids 11/12 ('user'/'assistant'). Su base marco NLL 0,327 y 1/12 exactas; la base de aqui,
# con marcadores, marca 0,142 y 6/10. SIN_ROLES=1 mide la MISMA base sobre las MISMAS tareas
# pero quitando 11/12 del prompt: separa el efecto del formato del de la poblacion de tareas.
SIN_ROLES = os.environ.get("SIN_ROLES", "0") == "1"
ETIQUETA = os.environ.get("ETIQUETA", "epoca0_base")


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


def parse_row(row):
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
    except Exception:                                        # noqa: BLE001
        return None


def build_item(pairs):
    toks, mask = [], []
    for i, o in pairs:
        t, m = pair_tokens(i, o)
        toks += t
        mask += m
    if len(toks) > MAX_LEN:
        return None
    return (toks, mask)


t0 = time.time()
STREAM = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                      split="train", streaming=True)
TRAIN = []
for row in STREAM:
    p = parse_row(row)
    if p is None:
        continue
    it = build_item(p)
    if it is not None:
        TRAIN.append(it)
        if len(TRAIN) >= N_TRAIN:
            break
random.Random(7).shuffle(TRAIN)
print(f"[pool] {len(TRAIN)} ejemplos de ARC-Heavy ({time.time()-t0:.0f}s)", flush=True)

MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                             device_map="cuda:0", attn_implementation="sdpa")
LCFG = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
MODEL = get_peft_model(MODEL, LCFG)
for _, p_ in MODEL.named_parameters():          # LoRA en fp32: sin GradScaler, mas estable
    if p_.requires_grad:
        p_.data = p_.data.float()
MODEL.enable_input_require_grads()
MODEL.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

PREVIAS = []
if EPOCA_INI > 0:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    subprocess.run(["kaggle", "datasets", "download", "-d", DS, "-p", "/content/prev",
                    "--unzip", "-q"], check=True)
    sd = load_file("/content/prev/adapter/adapter_model.safetensors")
    r_ = set_peft_model_state_dict(MODEL, {k: v.float() for k, v in sd.items()})
    prev = json.load(open("/content/prev/lora_epochs.json"))
    assert prev.get("adapter_epoca") == EPOCA_INI, \
        f"el adapter guardado es de la epoca {prev.get('adapter_epoca')}, no de {EPOCA_INI}"
    PREVIAS = prev["medidas"]
    print(f"[reanudar] adapter de la epoca {EPOCA_INI} cargado "
          f"(faltantes {len(getattr(r_, 'unexpected_keys', []) or [])}) | "
          f"{len(PREVIAS)} medidas previas", flush=True)
print(f"[setup] 2B + LoRA r=64 | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)


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
def dfs(prompt, depth, budget=45):
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
                    leaves.append(list(path))
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
    return leaves


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


KEYS = sorted([k for k in CHAL if is_holdout(k) and k not in AGI1],
              key=lambda k: cost(CHAL[k]))[:10]
EVAL = {}
for k in KEYS:
    t = CHAL[k]
    pr = []
    for p in t["train"]:
        a, _ = pair_tokens(p["input"], p["output"])
        pr += a
    pr += query_tokens(t["test"][0]["input"])
    if SIN_ROLES:                      # replica lo que hacia el tokenizador de Colab
        pr = [t_ for t_ in pr if t_ not in (USER, ASSIST)]
    EVAL[k] = (pr, grid_tokens(SOLS[k][0]) + [EOS], SOLS[k][0])
print(f"[setup] {len(KEYS)} tareas nuevas de evaluacion | umbral {THR:.3f}", flush=True)


def persistir(RES, etiqueta):
    """Sube la curva PARCIAL tras cada medida. La corrida del 18-ago se perdio entera
    porque solo se persistia al final: 4h de T4 y cero resultados. Ahora cada epoca
    queda guardada en cuanto se mide."""
    try:
        os.makedirs("/content/persist", exist_ok=True)
        json.dump(RES, open("/content/persist/lora_epochs.json", "w"), indent=1)
        user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
        json.dump({"title": "arcscale lora epocas", "id": f"{user}/arcscale-lora-epocas",
                   "licenses": [{"name": "CC0-1.0"}]},
                  open("/content/persist/dataset-metadata.json", "w"))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                           capture_output=True, text=True)
        if "error" in (r.stdout + r.stderr).lower():
            subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                            "-m", etiqueta, "-q"], check=False, capture_output=True)
        print(f"[persist] {etiqueta} subido a Kaggle", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"[persist] fallo: {e}", flush=True)


def medir(etiqueta):
    MODEL.eval()
    MODEL.config.use_cache = True
    res = []
    for k in KEYS:
        pr, gt_toks, gt = EVAL[k]
        n = nll_gt(pr, gt_toks)
        lv = dfs(pr, min(931, len(gt_toks) + 40))
        ok = any(toks_to_grid(x) == gt for x in lv)
        res.append({"task": k, "nll": round(n, 3), "pool": len(lv), "ok": bool(ok)})
        torch.cuda.empty_cache()
    r = {"etapa": etiqueta, "resueltas": sum(x["ok"] for x in res),
         "nll_mediana": round(float(np.median([x["nll"] for x in res])), 3),
         "pool_total": sum(x["pool"] for x in res), "detalle": res}
    print(f"[MEDIDA {etiqueta}] resueltas {r['resueltas']}/{len(KEYS)} | "
          f"NLL mediana {r['nll_mediana']} | pool {r['pool_total']}", flush=True)
    MODEL.config.use_cache = False
    return r


ACCUM = 8
OPT = torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad], lr=1e-4)
if EPOCA_INI == 0 and os.environ.get("SALTAR_BASE", "0") == "1":
    # la base ya esta medida dos veces identicas (6/10, NLL 0,142): no se repaga
    RES = {"medidas": [], "n_train": N_TRAIN, "sin_roles": SIN_ROLES}
elif EPOCA_INI == 0:
    RES = {"medidas": [medir(ETIQUETA)], "n_train": N_TRAIN, "sin_roles": SIN_ROLES}
    persistir(RES, ETIQUETA)
else:
    RES = {"medidas": list(PREVIAS), "n_train": N_TRAIN, "adapter_epoca": EPOCA_INI}

def una_epoca(OPT, etq):
    """Una pasada sobre TRAIN. El orden NO se rebaraja entre epocas: por eso la media de las
    ultimas 200 perdidas es comparable ejemplo a ejemplo entre pasadas, y fue lo que delato
    que a lr 1e-4 la segunda pasada EMPEORA la perdida sobre los mismos datos (0,1197 ->
    0,1247) en vez de bajarla."""
    MODEL.train()
    MODEL.config.use_cache = False
    OPT.zero_grad(set_to_none=True)
    t0, losses = time.time(), []
    for n, (toks, mask) in enumerate(TRAIN):
        ids = torch.tensor([toks], device="cuda:0", dtype=torch.long)
        lab = torch.tensor([[t if m else -100 for t, m in zip(toks, mask)]],
                           device="cuda:0", dtype=torch.long)
        with sdpa_kernel(_SDPA):
            out = MODEL(input_ids=ids, labels=lab)
        (out.loss / ACCUM).backward()
        losses.append(out.loss.item())
        if (n + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in MODEL.parameters() if p.requires_grad], 1.0)
            OPT.step()
            OPT.zero_grad(set_to_none=True)
        if (n + 1) % 250 == 0:
            print(f"  [{etq}] {n+1}/{len(TRAIN)} loss {np.mean(losses[-250:]):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if len(TRAIN) % ACCUM != 0:
        OPT.step()
        OPT.zero_grad(set_to_none=True)
    MODEL.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return losses, time.time() - t0


# BARRIDO DE LR (19-ago). La curva de epocas a lr 1e-4 destruye monotonamente (6/10 -> 3 -> 2)
# Y la perdida de entrenamiento sube sobre datos identicos ⇒ el paso es demasiado grande, no es
# solo desplazamiento de distribucion. Pregunta: ¿existe ALGUN lr al que una pasada de ARC-Heavy
# MEJORE sobre la base (>6/10 o NLL<0,142)? Si ninguno lo hace, el corpus no aporta a este
# modelo y la rama se cierra con una razon, no con una sospecha.
LRS = os.environ.get("LRS", "")
if LRS:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    INIT = {k: v.detach().clone() for k, v in get_peft_model_state_dict(MODEL).items()}
    for lr_s in LRS.split(","):
        set_peft_model_state_dict(MODEL, {k: v.clone() for k, v in INIT.items()})
        losses, secs = una_epoca(
            torch.optim.AdamW([p for p in MODEL.parameters() if p.requires_grad],
                              lr=float(lr_s)), f"lr{lr_s}")
        m = medir(f"lr{lr_s}")
        m["loss_final"] = round(float(np.mean(losses[-200:])), 5)
        m["lr"] = float(lr_s)
        m["secs"] = round(secs, 1)
        RES["medidas"].append(m)
        RES["curva"] = [{"etapa": x["etapa"], "resueltas": x["resueltas"],
                         "nll_mediana": x["nll_mediana"], "loss_final": x.get("loss_final")}
                        for x in RES["medidas"]]
        persistir(RES, f"lr{lr_s}")
        print("RESULT_JSON:" + json.dumps(RES["curva"]), flush=True)
    print(f"[fin] {time.time()-T00:.0f}s", flush=True)
    sys.exit(0)

for ep in range(EPOCA_INI + 1, EPOCA_INI + EPOCAS + 1):
    losses, secs = una_epoca(OPT, f"epoca {ep}")
    t0 = time.time() - secs
    m = medir(f"epoca{ep}")
    m["loss_media"] = round(float(np.mean(losses)), 5)
    m["loss_final"] = round(float(np.mean(losses[-200:])), 5)
    m["secs"] = round(time.time() - t0, 1)
    RES["medidas"].append(m)
    RES["curva"] = [{"etapa": x["etapa"], "resueltas": x["resueltas"],
                     "nll_mediana": x["nll_mediana"], "loss_final": x.get("loss_final")}
                    for x in RES["medidas"]]
    # Los pesos entrenados se guardan para poder CONTINUAR desde aqui (epoca 4, 5...)
    # sin repetir las anteriores, que es justo lo que pide la decision sobre saturacion.
    # Se sobrescribe siempre el mismo directorio: solo hace falta el ultimo, y versionar
    # tres adapters de ~250MB en el dataset costaria mas subida que valor.
    try:
        subprocess.run(["rm", "-rf", "/content/persist/adapter"], check=False)
        MODEL.save_pretrained("/content/persist/adapter")
        RES["adapter_epoca"] = ep
    except Exception as e:                                   # noqa: BLE001
        print(f"[save] fallo ep{ep}: {e}", flush=True)
    persistir(RES, f"epoca{ep}")
    print("RESULT_JSON:" + json.dumps(RES["curva"]), flush=True)

json.dump(RES, open("/content/lora_epochs.json", "w"), indent=1)
print("RESULT_JSON:" + json.dumps(RES.get("curva", RES["medidas"])), flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
