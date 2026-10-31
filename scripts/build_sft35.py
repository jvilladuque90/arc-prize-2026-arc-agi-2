"""Genera notebooks/sft35.ipynb: SFT PAREADO de Qwen3-4B-Base vs Qwen3.5-4B-Base sobre los MISMOS
puzzles NVARC (muestra de sft35-data), misma receta (vocab recortado a 16 tokens como NVARC, LoRA
sobre todas las lineales + embeddings/cabeza completas), x2 semillas de orden/aumentacion.
4 GPUs = 4 corridas independientes (sin DDP): q3_s1, q35_s1, q3_s2, q35_s2.
Medida al final (misma poblacion que gate35b: 24 baratas + 12 medias del eval publico, que NINGUN
brazo ha visto): NLL/celda de la GT en vista canonica y mediana de 3 vistas, exactas greedy, antes y
despues del SFT, mas la curva de loss a tokens iguales y tok/s.
PRERREG (docs/PREREG_sft35.md): la via 3.5 se REABRE si post-SFT q35/q3 <= 0,85 en AMBAS semillas y
gana >= 2/3 de tareas (signo p<0,05) con la banda semilla-a-semilla (q3_s1 vs q3_s2) mas estrecha
que el efecto. Empate => via cerrada definitivamente (el SFT no cambia el orden del prior).
Lanzar: PYTHONUTF8=1 python scripts/push_kernels.py sft35 --accelerator NvidiaL4
"""
import json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

WORKER = r'''
import gc, json, os, sys, time, math, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()
ARM = os.environ["ARM"]; SEED = int(os.environ["SEED"]); SHARD = int(os.environ["SHARD"])
N_SEQ = int(os.environ.get("N_SEQ", "9000")); MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", "10800"))
ACCUM = int(os.environ.get("ACCUM", "8")); LR = float(os.environ.get("LR", "1e-4"))
LORA_R = int(os.environ.get("LORA_R", "64")); LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
N_BARATAS = 24; N_MEDIAS = 12
SUCIAS = set(json.loads(os.environ["SUCIAS"]))
TAG = "%s_s%d" % (ARM, SEED)
OUTDIR = "/kaggle/working/%s" % TAG; os.makedirs(OUTDIR, exist_ok=True)
def log(*a):
    print("[%s %5.0fs]" % (TAG, time.time() - T00), *a, flush=True)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import transformers
from transformers.utils.import_utils import is_flash_linear_attention_available
log("transformers", transformers.__version__, "FLA", is_flash_linear_attention_available(), "torch", torch.__version__)

# ---------------- datos ----------------
def buscar(nombre):
    for root, dirs, files in os.walk("/kaggle/input"):
        if nombre in files:
            return os.path.join(root, nombre)
    return None
def buscar_modelo(*marcas):
    mejor = None
    for root, dirs, files in os.walk("/kaggle/input"):
        low = root.lower().replace("\\", "/")
        if "config.json" in files and all(m in low for m in marcas):
            if any(x.endswith(".safetensors") for x in files): return root
            mejor = mejor or root
    return mejor
MDIR = buscar_modelo("qwen-3/", "/4b-base") if ARM == "q3" else buscar_modelo("q35_4b_base")
F_MUESTRA = buscar("nvarc_muestra.json")
F_CHAL = buscar("arc-agi_evaluation_challenges.json")
assert MDIR and F_MUESTRA and F_CHAL, (MDIR, F_MUESTRA, F_CHAL)
log("modelo", MDIR); log("datos", F_MUESTRA)
PUZZLES = json.load(open(F_MUESTRA))
CHAL = json.load(open(F_CHAL)); SOLS = json.load(open(os.path.join(os.path.dirname(F_CHAL), "arc-agi_evaluation_solutions.json")))
def coste(t):
    return sum(len(g["input"])*len(g["input"][0]) + len(g["output"])*len(g["output"][0]) for g in t["train"])
ORD = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))
n = len(ORD); medio = ORD[int(0.40*n):int(0.60*n)]
KEYS_EVAL = ORD[:N_BARATAS] + medio[::max(1, len(medio)//N_MEDIAS)][:N_MEDIAS]
ESTRATO = {k: ("barata" if k in ORD[:N_BARATAS] else "media") for k in KEYS_EVAL}
log("eval:", len(KEYS_EVAL), "tareas; puzzles:", len(PUZZLES))

# ---------------- vocabulario de 16 tokens (como NVARC) ----------------
# 0-9 digitos, 10 '\n', 11 user, 12 assistant, 13 <|endoftext|>, 14 <|im_start|>, 15 <|im_end|>
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
def uno(s):
    ids = tok.encode(s, add_special_tokens=False); assert len(ids) == 1, (s, ids); return ids[0]
IDS16 = [uno(str(d)) for d in range(10)] + [uno("\n"), uno("user"), uno("assistant"),
         tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")]
assert all(isinstance(i, int) and i >= 0 for i in IDS16), IDS16
log("IDS16", IDS16)
NL, USER, ASSIST, PAD, IMS, IME = 10, 11, 12, 13, 14, 15

def grid_ids(g):
    out = []
    for i, fila in enumerate(g):
        if i: out.append(NL)
        out.extend(int(c) for c in fila)
    return out
def par_ids(inp, out, con_salida=True):
    """ids y mascara (1 = token entrenable/evaluable) de un par; formato de produccion."""
    ids = [IMS, USER, NL] + grid_ids(inp) + [IME, IMS, ASSIST, NL]
    m = [0] * len(ids)
    if con_salida:
        o = grid_ids(out) + [IME]
        ids += o; m += [1] * len(o)
    return ids, m
def seq_ids(pares, con_salida_final=True):
    ids, m = [], []
    for j, p in enumerate(pares):
        a, b = par_ids(p["input"], p["output"], con_salida=(con_salida_final or j < len(pares) - 1))
        ids += a; m += b
    return ids, m
def len_par(p):
    return 8 + sum(len(p[k]) * len(p[k][0]) + len(p[k]) - 1 for k in ("input", "output"))

# ---------------- aumentacion (D4 x permutacion de 10 colores) ----------------
def d4(g, k):
    a = np.asarray(g)
    if k >= 4: a = a.T
    return np.rot90(a, k % 4)
def aug(pares, rng):
    k = rng.randrange(8); perm = list(range(10)); rng.shuffle(perm); perm = np.asarray(perm)
    return [{"input": perm[d4(p["input"], k)].tolist(), "output": perm[d4(p["output"], k)].tolist()} for p in pares]

def construir_secuencias(n_seq, seed):
    rng = random.Random(1000 + seed)
    seqs = []; i = 0; vueltas = 0
    while len(seqs) < n_seq and vueltas < 3:
        if i >= len(PUZZLES): i = 0; vueltas += 1
        pares = list(PUZZLES[i]["pares"]); i += 1
        rng.shuffle(pares)
        pares = aug(pares, rng)
        while len(pares) >= 2 and sum(len_par(p) for p in pares) > MAX_LEN: pares.pop(0)
        if len(pares) < 2: continue
        ids, m = seq_ids(pares)
        if len(ids) > MAX_LEN: continue
        seqs.append((ids, m))
    return seqs
SEQS = construir_secuencias(N_SEQ, SEED)
n_tok = sum(len(s[0]) for s in SEQS); n_lab = sum(sum(s[1]) for s in SEQS)
log("secuencias", len(SEQS), "tokens", n_tok, "etiquetas", n_lab, "media", round(n_tok / max(1, len(SEQS))))

# ---------------- modelo ----------------
CLASES = ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForVision2Seq", "AutoModel")
def cargar(dir_):
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    for nombre in CLASES:
        try:
            cls = getattr(transformers, nombre)
            try:
                m, info = cls.from_pretrained(dir_, **kw)
            except TypeError:
                kw2 = dict(kw); kw2["torch_dtype"] = kw2.pop("dtype"); m, info = cls.from_pretrained(dir_, **kw2)
        except Exception as e:
            log("  ", nombre, "error", str(e)[:160]); continue
        falt = list(info.get("missing_keys") or [])
        log("  ", nombre, "faltan", len(falt), "inesperadas", len(info.get("unexpected_keys") or []), "params", sum(p.numel() for p in m.parameters()) // 10**6, "M")
        if len(falt) == 0: return m, nombre
        del m; gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError("sin clase que cargue " + dir_)
model, clase = cargar(MDIR)
model.config.use_cache = False

# recorte de vocabulario: embeddings y cabeza a 16 filas (entrenables completas, no LoRA)
H = model.get_input_embeddings().weight.shape[1]
emb_old = model.get_input_embeddings().weight.data
head_old = model.get_output_embeddings().weight.data
tied = bool(emb_old.data_ptr() == head_old.data_ptr())
emb = nn.Embedding(16, H, dtype=torch.bfloat16, device=emb_old.device)
emb.weight.data.copy_(emb_old[IDS16])
head = nn.Linear(H, 16, bias=False, dtype=torch.bfloat16, device=emb_old.device)
head.weight.data.copy_(head_old[IDS16])
if tied: head.weight = emb.weight
model.set_input_embeddings(emb); model.set_output_embeddings(head)
for c in (model.config, getattr(model.config, "text_config", None)):
    if c is not None:
        try: c.vocab_size = 16
        except Exception: pass
if hasattr(model, "vocab_size"): model.vocab_size = 16
log("vocab recortado a 16; H =", H, "atados =", tied, "clase =", clase)
del emb_old, head_old; gc.collect(); torch.cuda.empty_cache()

# LoRA sobre TODAS las lineales del decodificador (trato igual a ambas arquitecturas)
from peft import LoraConfig, get_peft_model
lineales = sorted({n.split(".")[-1] for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and mod is not head and "visual" not in n and "vision" not in n})
log("modulos lineales objetivo:", lineales)
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none", use_rslora=True, target_modules=lineales))
for p_ in list(emb.parameters()) + list(head.parameters()): p_.requires_grad_(True)
params = [p for p in model.parameters() if p.requires_grad]
n_tr = sum(p.numel() for p in params)
log("parametros entrenables: %.1fM" % (n_tr / 1e6))
model.train()

def forward_logits(ids):
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    return out.logits

def nll_eval(ids, m):
    """suma de NLL sobre los tokens con mascara 1 y numero de celdas (digitos)."""
    with torch.no_grad():
        t = torch.tensor([ids], device="cuda")
        lg = forward_logits(t)[0, :-1].float()
        lp = lg.log_softmax(-1).gather(-1, t[0, 1:].unsqueeze(-1)).squeeze(-1)
        mm = torch.tensor(m[1:], device="cuda", dtype=torch.bool)
        tgt = t[0, 1:]
        celdas = int(((tgt < 10) & mm).sum().item())
        return float(-lp[mm].sum().item()), celdas

def greedy(ids, max_new):
    with torch.no_grad():
        t = torch.tensor([ids], device="cuda")
        out = model(input_ids=t, use_cache=True, return_dict=True)
        pkv = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); gen = [nxt]
        while nxt != IME and len(gen) < max_new:
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=pkv, use_cache=True, return_dict=True)
            pkv = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); gen.append(nxt)
        return gen
def ids_a_grid(gen):
    filas, fila = [], []
    for x in gen:
        if x == IME: break
        if x == NL: filas.append(fila); fila = []
        elif x < 10: fila.append(x)
        else: return None
    if fila: filas.append(fila)
    if not filas or any(len(f) != len(filas[0]) for f in filas) or len(filas[0]) == 0: return None
    return filas

VISTAS = [(0, None), (1, [0, 5, 3, 8, 1, 9, 2, 7, 4, 6]), (6, [9, 0, 4, 1, 7, 2, 8, 3, 5, 6])]
def vista(pares, k, perm):
    pp = np.asarray(perm) if perm else None
    f = lambda g: (pp[d4(g, k)] if pp is not None else d4(g, k)).tolist()
    return [{"input": f(p["input"]), "output": f(p["output"])} for p in pares]

def evaluar(etiqueta):
    model.eval(); filas = []; t0 = time.time()
    for key in KEYS_EVAL:
        task = CHAL[key]; gt = SOLS[key][0]
        pares = task["train"] + [{"input": task["test"][0]["input"], "output": gt}]
        fila = {"tarea": key, "estrato": ESTRATO[key]}
        nlls = []
        for vi, (k, perm) in enumerate(VISTAS):
            pv = vista(pares, k, perm)
            ids, m = seq_ids(pv)
            if len(ids) > 8192: fila["largo"] = len(ids); break
            s, c = nll_eval(ids, m)
            nlls.append(s / max(1, c))
            if vi == 0:
                fila["celdas"] = c; fila["n_tok"] = len(ids)
                ids_q, _ = seq_ids(pv, con_salida_final=False)
                try:
                    g = ids_a_grid(greedy(ids_q, 31 * 30 + 2))
                except Exception as e:
                    log("greedy fallo en", key, str(e)[:120]); g = None; fila["greedy_error"] = str(e)[:120]
                fila["exacta"] = bool(g == pv[-1]["output"])
        if nlls:
            fila["nll_celda"] = nlls[0]; fila["nll_celda_med3"] = float(np.median(nlls)); fila["nll_vistas"] = nlls
        filas.append(fila)
    model.train()
    ex = sum(1 for f in filas if f.get("exacta")); v = [f["nll_celda"] for f in filas if "nll_celda" in f]
    log("EVAL %s: exactas %d/%d | NLL/celda mediana %.3f media %.3f | %.0fs" % (etiqueta, ex, len(filas), float(np.median(v)), float(np.mean(v)), time.time() - t0))
    return filas

res = {"arm": ARM, "seed": SEED, "modelo": MDIR, "clase": clase, "ids16": IDS16, "tied": tied, "H": H,
       "n_seq": len(SEQS), "n_tok": n_tok, "n_lab": n_lab, "params_tr_M": round(n_tr / 1e6, 1), "lineales": lineales,
       "cfg": {"lr": LR, "accum": ACCUM, "r": LORA_R, "alpha": LORA_ALPHA, "max_len": MAX_LEN, "train_seconds": TRAIN_SECONDS},
       "fla": bool(is_flash_linear_attention_available()), "transformers": transformers.__version__}
res["eval_pre"] = evaluar("pre-SFT")

# ---------------- entrenamiento ----------------
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.99))
pasos_plan = max(1, len(SEQS) // ACCUM); warm = max(1, int(0.03 * pasos_plan))
def lr_en(paso):
    if paso < warm: return LR * (paso + 1) / warm
    pr = (paso - warm) / max(1, pasos_plan - warm)
    return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
curva = []; t_tr = time.time(); tok_vistos = 0; lab_vistos = 0; paso = 0; acc_loss = 0.0; acc_n = 0
ck = False
i = 0
while i < len(SEQS):
    if time.time() - t_tr > TRAIN_SECONDS:
        log("tope de tiempo de entrenamiento alcanzado en la secuencia", i); break
    ids, m = SEQS[i]
    t = torch.tensor([ids], device="cuda"); mm = torch.tensor(m[1:], device="cuda", dtype=torch.bool)
    try:
        lg = forward_logits(t)[0, :-1].float()
        loss = F.cross_entropy(lg[mm], t[0, 1:][mm])
        (loss / ACCUM).backward()
    except torch.cuda.OutOfMemoryError:
        if ck: raise
        log("OOM: activo gradient checkpointing y reintento"); ck = True
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        continue
    acc_loss += float(loss.item()); acc_n += 1; tok_vistos += len(ids); lab_vistos += int(mm.sum().item()); i += 1
    if i % ACCUM == 0 or i == len(SEQS):
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups: g["lr"] = lr_en(paso)
        opt.step(); opt.zero_grad(set_to_none=True); paso += 1
        if paso % 25 == 0 or paso == 1:
            el = time.time() - t_tr
            curva.append({"paso": paso, "seq": i, "tok": tok_vistos, "loss": acc_loss / max(1, acc_n), "lr": lr_en(paso), "s": round(el)})
            log("paso %d seq %d tok %d loss %.4f lr %.2e | %.0f tok/s | mem %.1f GB" % (paso, i, tok_vistos, acc_loss / max(1, acc_n), lr_en(paso), tok_vistos / el, torch.cuda.max_memory_allocated() / 1e9))
            acc_loss = 0.0; acc_n = 0
el = time.time() - t_tr
res.update({"train": {"seq_hechas": i, "pasos": paso, "tok": tok_vistos, "lab": lab_vistos, "s": round(el), "tok_s": round(tok_vistos / max(1, el)),
                      "ckpt": ck, "mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1), "curva": curva}})
log("entrenamiento: %d seq, %d pasos, %d tok, %.0f s, %.0f tok/s" % (i, paso, tok_vistos, el, tok_vistos / max(1, el)))

# ---------------- evaluacion post y guardado ----------------
res["eval_post"] = evaluar("post-SFT")
from peft import get_peft_model_state_dict
sd = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
sd["__emb16__"] = emb.weight.detach().cpu(); sd["__head16__"] = head.weight.detach().cpu()
torch.save(sd, os.path.join(OUTDIR, "adapter.pt"))
json.dump(res, open(os.path.join(OUTDIR, "res.json"), "w"))
log("guardado", OUTDIR, "wall total %.0f s" % (time.time() - T00))
'''

LANZADOR = r'''
import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = __SUCIAS__

def hallar_wheels(patron):
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.startswith(patron) and f.endswith(".whl") for f in files):
            return root
    return None
W_TF = hallar_wheels("transformers-"); W_FLA = hallar_wheels("flash_linear_attention")
print("wheels transformers:", W_TF, "| wheels fla:", W_FLA, flush=True)
if W_TF:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--upgrade", "--find-links", W_TF,
                        "transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "regex"], capture_output=True, text=True)
    print("pip tf rc=%d" % r.returncode, (r.stdout + r.stderr)[-500:], flush=True)
if W_FLA:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", W_FLA,
                        "flash-linear-attention", "fla-core", "einops"], capture_output=True, text=True)
    print("pip fla rc=%d" % r.returncode, (r.stdout + r.stderr)[-500:], flush=True)
r = subprocess.run([sys.executable, "-c", "import transformers, peft;from transformers.utils.import_utils import is_flash_linear_attention_available as f;print(transformers.__version__, 'peft', peft.__version__, 'FLA', f())"], capture_output=True, text=True)
print("entorno:", (r.stdout + r.stderr).strip()[-400:], flush=True)

BRAZOS = [("q3", 1), ("q35", 1), ("q3", 2), ("q35", 2)]
procs = []
for i, (arm, seed) in enumerate(BRAZOS):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "SHARD": str(i), "ARM": arm, "SEED": str(seed), "SUCIAS": json.dumps(SUCIAS),
                "N_SEQ": os.environ.get("N_SEQ", "9000"), "TRAIN_SECONDS": os.environ.get("TRAIN_SECONDS", "10800")})
    fh = open("/kaggle/working/log_%s_s%d.txt" % (arm, seed), "w")
    procs.append((arm, seed, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"], env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
for arm, seed, p, fh in procs:
    p.wait(); fh.close()
    print("worker %s_s%d rc=%d a los %.0fs" % (arm, seed, p.returncode, time.time() - T0), flush=True)

R = {}
for arm, seed, p, fh in procs:
    f = "/kaggle/working/%s_s%d/res.json" % (arm, seed)
    if os.path.exists(f): R["%s_s%d" % (arm, seed)] = json.load(open(f))
json.dump(R, open("/kaggle/working/sft35.json", "w"))
for arm, seed, p, fh in procs:
    print("=" * 12 + (" LOG %s_s%d " % (arm, seed)) + "=" * 12, flush=True)
    try: print(open("/kaggle/working/log_%s_s%d.txt" % (arm, seed)).read()[-3000:], flush=True)
    except Exception as e: print("sin log", e, flush=True)

import math
def med(xs):
    xs = sorted(xs); n = len(xs)
    return float("nan") if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))
def signo_p(k, n):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else float("nan")
def tabla(a, b, campo="nll_celda", fase="eval_post"):
    if a not in R or b not in R: return None
    A = {f["tarea"]: f for f in R[a][fase]}; B = {f["tarea"]: f for f in R[b][fase]}
    out = {}
    for estr in ("barata", "media", "todas"):
        ks = [k for k in A if k in B and campo in A[k] and campo in B[k] and (estr == "todas" or A[k]["estrato"] == estr)]
        if not ks: continue
        ratios = [B[k][campo] / max(1e-6, A[k][campo]) for k in ks]
        gana = sum(1 for r in ratios if r < 1)
        out[estr] = {"n": len(ks), "ratio_med": round(med(ratios), 3), "ratio_media": round(sum(ratios) / len(ks), 3), "gana_b": gana, "p_signo": round(signo_p(gana, len(ks)), 4),
                     "exactas_a": sum(1 for k in ks if A[k].get("exacta")), "exactas_b": sum(1 for k in ks if B[k].get("exacta"))}
    return out
print("\n" + "=" * 70, flush=True)
for tag, r in R.items():
    tr = r.get("train", {})
    pre = [f["nll_celda"] for f in r["eval_pre"] if "nll_celda" in f]; post = [f["nll_celda"] for f in r["eval_post"] if "nll_celda" in f]
    print("%-7s %-28s seq %5d tok %9d %5d tok/s loss_fin %.4f ckpt=%s | NLL/celda med pre %.3f -> post %.3f | exactas pre %d post %d" % (
        tag, r["clase"][:28], tr.get("seq_hechas", 0), tr.get("tok", 0), tr.get("tok_s", 0), (tr.get("curva") or [{}])[-1].get("loss", float("nan")), tr.get("ckpt"),
        med(pre), med(post), sum(1 for f in r["eval_pre"] if f.get("exacta")), sum(1 for f in r["eval_post"] if f.get("exacta"))), flush=True)
print("\nPAREADO post-SFT (ratio = q35 / q3 de NLL/celda; <1 = gana 3.5):", flush=True)
for a, b in (("q3_s1", "q35_s1"), ("q3_s2", "q35_s2"), ("q3_s1", "q3_s2"), ("q35_s1", "q35_s2")):
    print("  %s vs %s:" % (a, b), json.dumps(tabla(a, b)), flush=True)
print("  mediana de 3 vistas, s1:", json.dumps(tabla("q3_s1", "q35_s1", campo="nll_celda_med3")), flush=True)
print("  mediana de 3 vistas, s2:", json.dumps(tabla("q3_s2", "q35_s2", campo="nll_celda_med3")), flush=True)
print("PAREADO pre-SFT (enlace con gate35b):", json.dumps(tabla("q3_s1", "q35_s1", fase="eval_pre")), flush=True)
print("\nPRERREG: la via 3.5 se REABRE si ratio post <= 0,85 en AMBAS semillas y gana >= 2/3 (p<0,05), con la banda s1-vs-s2 del mismo brazo mas estrecha que el efecto.", flush=True)
print("RESULT_JSON:" + json.dumps({"brazos": list(R), "wall_s": round(time.time() - T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS))

MD = """# sft35: SFT pareado Qwen3-4B-Base vs Qwen3.5-4B-Base sobre los mismos puzzles NVARC (vocab 16)

`gate35b` (31-oct) midio el PRIOR de rejilla: empate (1,044). Este kernel mide lo que falta: si la generacion
3.5 **convierte el SFT de NVARC en habilidad de rejilla** mejor que la 3. Misma muestra de puzzles
(`sft35-data`), misma receta (vocab recortado a 16 tokens como NVARC, LoRA r=64 rsLoRA sobre todas las
lineales + embeddings/cabeza completas, lr 1e-4 coseno, ~9K secuencias <= 2048 tok, 1 epoca), dos
semillas de orden/aumentacion por brazo. 4 GPUs = 4 corridas independientes.

Medida (misma poblacion que gate35b, 24 baratas + 12 medias del eval publico, no vista por ningun brazo):
NLL/celda de la GT (canonica y mediana de 3 vistas), exactas greedy, pre y post SFT; curva de loss; tok/s.

**PRERREG**: la via 3.5 se REABRE si post-SFT ratio q35/q3 <= 0,85 en AMBAS semillas y gana >= 2/3
(signo p<0,05) con la banda semilla-a-semilla mas estrecha que el efecto. Empate => cerrada definitivamente."""

def celda(src, tipo="code"):
    c = {"cell_type": tipo, "metadata": {}, "source": src.splitlines(keepends=True)}
    if tipo == "code":
        c["execution_count"] = None; c["outputs"] = []
    return c

nb = {"cells": [celda(MD, "markdown"),
                celda("%%writefile /kaggle/working/_worker_src.py\n" + WORKER.lstrip("\n")),
                celda(LANZADOR.lstrip("\n"))],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/sft35.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("OK notebooks/sft35.ipynb (worker %d B, lanzador %d B)" % (len(WORKER), len(LANZADOR)))
