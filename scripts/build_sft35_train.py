"""Genera notebooks/sft35_train.ipynb: entrena UN Qwen3.5-4B-Base con la receta de sft35 (vocab 16,
LoRA r=64 rsLoRA en todas las lineales + embeddings/cabeza completas, lr 1e-4 coseno, 1 epoca) usando
las 4 GPUs como datos-paralelo (gloo, gradientes promediados cada paso: 4 x 8 = 32 secuencias/paso) y
lo GUARDA fusionado (bf16, vocab de 16 filas, tokenizador de sft139 => mismos ids 0-15 que el campeon).
Validacion del artefacto: NLL/celda en memoria vs recargado desde disco en 6 tareas (debe coincidir).
Pedido del usuario (1-nov): enviar el 3.5 entrenado con el TTT y todo el pipeline del campeon al LB.
Lanzar: PYTHONUTF8=1 python scripts/push_kernels.py sft35-train --accelerator NvidiaL4
"""
import json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

WORKER = r'''
import gc, json, os, sys, time, math, random, shutil, datetime
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()
RANK = int(os.environ["RANK"]); WORLD = int(os.environ["WORLD_SIZE"])
N_SEQ = int(os.environ.get("N_SEQ", "4700")); MAX_LEN = 2048
TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", "11400"))
ACCUM = 8; LR = 1e-4; LORA_R = 64; LORA_ALPHA = 32
OUT = "/kaggle/working/q35_merged"
def log(*a):
    print("[r%d %6.0fs]" % (RANK, time.time() - T00), *a, flush=True)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers.utils.import_utils import is_flash_linear_attention_available
dist.init_process_group("gloo", init_method="env://", rank=RANK, world_size=WORLD, timeout=datetime.timedelta(hours=3))
log("transformers", transformers.__version__, "FLA", is_flash_linear_attention_available(), "gloo ok, world", WORLD)

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
MDIR = buscar_modelo("q35_4b_base")
TOKDIR = buscar_modelo("qwen3_4b_grids15_sft139")
F_MUESTRA = buscar("nvarc_muestra.json"); F_CHAL = buscar("arc-agi_evaluation_challenges.json")
assert MDIR and TOKDIR and F_MUESTRA and F_CHAL, (MDIR, TOKDIR, F_MUESTRA, F_CHAL)
log("modelo", MDIR, "| tokenizador sft139", TOKDIR)
PUZZLES = json.load(open(F_MUESTRA))[RANK::WORLD]   # cada rango ve puzzles DISTINTOS
CHAL = json.load(open(F_CHAL)); SOLS = json.load(open(os.path.join(os.path.dirname(F_CHAL), "arc-agi_evaluation_solutions.json")))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
def uno(s):
    ids = tok.encode(s, add_special_tokens=False); assert len(ids) == 1, (s, ids); return ids[0]
IDS16 = [uno(str(d)) for d in range(10)] + [uno("\n"), uno("user"), uno("assistant"),
         tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")]
NL, USER, ASSIST, PAD, IMS, IME = 10, 11, 12, 13, 14, 15

def grid_ids(g):
    out = []
    for i, fila in enumerate(g):
        if i: out.append(NL)
        out.extend(int(c) for c in fila)
    return out
def par_ids(inp, out, con_salida=True):
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
    while len(seqs) < n_seq and vueltas < 60:
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
SEQS = construir_secuencias(N_SEQ, RANK)
assert len(SEQS) == N_SEQ, len(SEQS)
log("secuencias", len(SEQS), "tokens", sum(len(s[0]) for s in SEQS))

CLASES = ("AutoModelForImageTextToText", "AutoModelForCausalLM")
def cargar(dir_):
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    for nombre in CLASES:
        try:
            m, info = getattr(transformers, nombre).from_pretrained(dir_, **kw)
        except Exception as e:
            log("  ", nombre, "error", str(e)[:200]); continue
        falt = list(info.get("missing_keys") or []); ines = list(info.get("unexpected_keys") or [])
        log("  ", nombre, "faltan", len(falt), falt[:4], "inesperadas", len(ines), ines[:4])
        if len(falt) == 0: return m, nombre
        del m; gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError("sin clase que cargue " + dir_)
model, clase = cargar(MDIR)
model.config.use_cache = False
H = model.get_input_embeddings().weight.shape[1]
emb_old = model.get_input_embeddings().weight.data; head_old = model.get_output_embeddings().weight.data
tied = bool(emb_old.data_ptr() == head_old.data_ptr())
emb = nn.Embedding(16, H, dtype=torch.bfloat16, device=emb_old.device); emb.weight.data.copy_(emb_old[IDS16])
head = nn.Linear(H, 16, bias=False, dtype=torch.bfloat16, device=emb_old.device); head.weight.data.copy_(head_old[IDS16])
if tied: head.weight = emb.weight
model.set_input_embeddings(emb); model.set_output_embeddings(head)
tc = getattr(model.config, "text_config", None) or model.config
tc.vocab_size = 16
log("vocab 16; atados", tied, "text_config.vocab_size", tc.vocab_size)
del emb_old, head_old; gc.collect(); torch.cuda.empty_cache()

from peft import LoraConfig, get_peft_model
lineales = sorted({n.split(".")[-1] for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and mod is not head and "visual" not in n and "vision" not in n})
torch.manual_seed(0)
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none", use_rslora=True, target_modules=lineales))
for p_ in list(emb.parameters()) + list(head.parameters()): p_.requires_grad_(True)
params = [p for p in model.parameters() if p.requires_grad]
# todos los rangos arrancan de los MISMOS pesos (difundidos desde el rango 0)
for p in params:
    buf = p.data.detach().float().cpu(); dist.broadcast(buf, 0); p.data.copy_(buf.to(p.device, p.dtype))
log("entrenables %.1fM, modulos %s" % (sum(p.numel() for p in params) / 1e6, lineales))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()

def logits_de(ids):
    return model(input_ids=ids, use_cache=False, return_dict=True).logits

def nll_tareas(fwd, keys, codificar):
    """NLL/celda de la GT (vista canonica) con la funcion de forward dada."""
    res = {}
    with torch.no_grad():
        for k in keys:
            t = CHAL[k]; pares = t["train"] + [{"input": t["test"][0]["input"], "output": SOLS[k][0]}]
            ids, m = codificar(pares)
            x = torch.tensor([ids], device="cuda")
            lg = fwd(x)[0, :-1].float()
            lp = lg.log_softmax(-1).gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)
            mm = torch.tensor(m[1:], device="cuda", dtype=torch.bool)
            cel = int(((x[0, 1:] < 10) & mm).sum().item())
            res[k] = float(-lp[mm].sum().item()) / max(1, cel)
    return res
def coste(t):
    return sum(len(g["input"])*len(g["input"][0]) + len(g["output"])*len(g["output"][0]) for g in t["train"])
KV = sorted(CHAL, key=lambda k: coste(CHAL[k]))[6:12]

opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.99))
pasos_plan = max(1, len(SEQS) // ACCUM); warm = max(1, int(0.03 * pasos_plan))
def lr_en(paso):
    if paso < warm: return LR * (paso + 1) / warm
    pr = (paso - warm) / max(1, pasos_plan - warm)
    return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
t_tr = time.time(); tok_v = 0; paso = 0; acc_loss = 0.0; acc_n = 0; i = 0; curva = []
SAVE_CADA = 2700; ult_save = time.time()
while i < len(SEQS):
    ids, m = SEQS[i]
    t = torch.tensor([ids], device="cuda"); mm = torch.tensor(m[1:], device="cuda", dtype=torch.bool)
    lg = logits_de(t)[0, :-1].float()
    loss = F.cross_entropy(lg[mm], t[0, 1:][mm])
    (loss / ACCUM).backward()
    acc_loss += float(loss.item()); acc_n += 1; tok_v += len(ids); i += 1
    if i % ACCUM == 0 or i == len(SEQS):
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params]).cpu()
        dist.all_reduce(flat); flat /= WORLD
        off = 0
        for p in params:
            n_ = p.numel(); g = flat[off:off + n_].view_as(p).to(p.device, p.dtype); off += n_
            if p.grad is None: p.grad = g
            else: p.grad.copy_(g)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g_ in opt.param_groups: g_["lr"] = lr_en(paso)
        opt.step(); opt.zero_grad(set_to_none=True); paso += 1
        stop = torch.tensor([1.0 if time.time() - t_tr > TRAIN_SECONDS else 0.0]); dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if paso % 25 == 0 or paso == 1:
            el = time.time() - t_tr
            lt = torch.tensor([acc_loss / max(1, acc_n)]); dist.all_reduce(lt); lt /= WORLD
            curva.append({"paso": paso, "seq_global": i * WORLD, "loss": float(lt), "s": round(el)})
            if RANK == 0:
                log("paso %d/%d seq_global %d loss %.4f lr %.2e | %.0f tok/s por GPU | mem %.1f GB" % (paso, pasos_plan, i * WORLD, float(lt), lr_en(paso), tok_v / el, torch.cuda.max_memory_allocated() / 1e9))
            acc_loss = 0.0; acc_n = 0
        if RANK == 0 and time.time() - ult_save > SAVE_CADA:
            sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(sd, "/kaggle/working/q35_lora_parcial.pt"); ult_save = time.time()
            log("checkpoint parcial guardado en el paso", paso)
        if float(stop) > 0:
            log("tope de tiempo en el paso", paso); break
log("entrenamiento: %d pasos, %d seq globales, %.0f s" % (paso, i * WORLD, time.time() - t_tr))
if RANK != 0:
    log("fin del rango"); sys.exit(0)

# ---------------- rango 0: guardar fusionado y VALIDAR el artefacto ----------------
model.eval()
cod_mem = lambda pares: seq_ids(pares)
nll_mem = nll_tareas(logits_de, KV, cod_mem)
log("NLL/celda en memoria:", {k: round(v, 4) for k, v in nll_mem.items()})
sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
torch.save(sd, "/kaggle/working/q35_lora_final.pt")
base = model.merge_and_unload()
tc = getattr(base.config, "text_config", None) or base.config
tc.vocab_size = 16
base.config.use_cache = True
os.makedirs(OUT, exist_ok=True)
base.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
# tokenizador de sft139: sus ids 0-15 son EXACTAMENTE nuestro orden (0-9, salto, user, assistant, eot, im_start, im_end)
for f in os.listdir(TOKDIR):
    if f.startswith(("tokenizer", "special_tokens", "vocab", "merges", "added_tokens")) or f.endswith(".jinja"):
        shutil.copy(os.path.join(TOKDIR, f), os.path.join(OUT, f))
json.dump({"curva": curva, "pasos": paso, "seq_global": i * WORLD, "nll_mem": nll_mem, "ids16_fuente": IDS16, "clase": clase},
          open("/kaggle/working/sft35_train.json", "w"))
log("guardado:", sorted(os.listdir(OUT)))
del base, model, opt, params; gc.collect(); torch.cuda.empty_cache()

m2, c2 = cargar(OUT)
m2.eval()
tk2 = AutoTokenizer.from_pretrained(OUT)
prueba = tk2.encode("<|im_start|>user\n01\n2<|im_end|><|im_start|>assistant\n3<|im_end|>")
log("tokenizador recargado:", prueba)
assert prueba == [14, 11, 10, 0, 1, 10, 2, 15, 14, 12, 10, 3, 15], prueba
def txt(g): return "\n".join("".join(str(c) for c in r) for r in g)
def cod_txt(pares):
    ids, m = [], []
    for p in pares:
        a = tk2.encode("<|im_start|>user\n" + txt(p["input"]) + "<|im_end|><|im_start|>assistant\n")
        b = tk2.encode(txt(p["output"]) + "<|im_end|>")
        ids += a + b; m += [0] * len(a) + [1] * len(b)
    return ids, m
lm = m2.model.language_model; hd = m2.lm_head
nll_disco = nll_tareas(lambda x: hd(lm(input_ids=x, use_cache=False).last_hidden_state), KV, cod_txt)
rat = [nll_disco[k] / max(1e-6, nll_mem[k]) for k in KV]
ok = all(0.97 < r < 1.03 for r in rat)
log("NLL/celda recargado:", {k: round(v, 4) for k, v in nll_disco.items()})
log("VERIFICACION %s: ratios disco/memoria %s" % ("OK" if ok else "FALLA", [round(r, 4) for r in rat]))
d = json.load(open("/kaggle/working/sft35_train.json")); d.update({"nll_disco": nll_disco, "verificacion": ok}); json.dump(d, open("/kaggle/working/sft35_train.json", "w"))
'''

LANZADOR = r'''
import json, os, subprocess, sys, time, socket
T0 = time.time()
def hallar_wheels(patron):
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.startswith(patron) and f.endswith(".whl") for f in files):
            return root
    return None
W_TF = hallar_wheels("transformers-"); W_FLA = hallar_wheels("flash_linear_attention")
print("wheels transformers:", W_TF, "| wheels fla:", W_FLA, flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--upgrade", "--find-links", W_TF,
                    "transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "regex"], capture_output=True, text=True)
print("pip tf rc=%d" % r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", W_FLA,
                    "flash-linear-attention", "fla-core", "einops"], capture_output=True, text=True)
print("pip fla rc=%d" % r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], capture_output=True, text=True)
print("pip uninstall torchao rc=%d" % r.returncode, flush=True)
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
procs = []
for i in range(4):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "RANK": str(i), "WORLD_SIZE": "4", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    fh = open("/kaggle/working/log_r%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"], env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
for i, p, fh in procs:
    p.wait(); fh.close()
    print("rango %d rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)
for i, p, fh in procs:
    print("=" * 12 + (" LOG r%d " % i) + "=" * 12, flush=True)
    txt = open("/kaggle/working/log_r%d.txt" % i).read()
    print("\n".join(l for l in txt.splitlines() if "Loading weights" not in l)[-4000:], flush=True)
print("RESULT_JSON:" + json.dumps({"rc": [p.returncode for _, p, _ in procs], "wall_s": round(time.time() - T0)}), flush=True)
'''

MD = """# sft35-train: Qwen3.5-4B-Base + SFT NVARC (receta sft35) en 4 GPUs datos-paralelo, guardado fusionado

Pedido del usuario (1-nov): enviar al LB el 3.5 entrenado con el TTT y el resto del pipeline del campeon.
Salida: `/kaggle/working/q35_merged` (bf16, vocab 16 con el tokenizador de sft139) para `lb35`."""

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
json.dump(nb, open("notebooks/sft35_train.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("OK notebooks/sft35_train.ipynb (worker %d B)" % len(WORKER))
