"""Genera notebooks/lb35.ipynb: submission = pipeline COMPLETO del campeon lb33-det (TTT por tarea LoRA r=256
lr 5e-5 1 epoca 128 secuencias, DFS con umbral -ln 0,2 en 16 vistas, re-scoring en 8 augmentaciones, seleccion
del decoder) sobre el Qwen3.5-4B-Base entrenado por sft35-train (fusionado, vocab 16, tokenizador sft139).
Pedido del usuario (1-nov): "envia la version mas actual entrenada con el TTT y demas arandelas".

Cambios OBLIGADOS respecto a lb33-det (unsloth no soporta Gated DeltaNet):
  1. Carga con transformers 5.16 (wheels) + FLA; el decoder de texto se envuelve en TextLM (sin torre visual).
  2. LoRA del TTT inyectada con peft.inject_adapter_in_model en todas las lineales del decodificador
     (atencion, MLP y las proyecciones de las capas lineales). Sin LoRA en embed/lm_head (atados, 16 filas).
  3. Bucle de TTT propio = mismo calendario que el Trainer del campeon (bs1, 1 epoca, AdamW, lr 5e-5, warmup
     10 %, coseno, clip 1,0, perdida solo en las respuestas). Checkpointing solo en secuencias > 2048 tokens.
  4. DFS: la cache de las capas lineales se actualiza IN PLACE (no se puede "rebobinar" como la KV), asi que
     en cada nodo con >= 2 ramas se guarda una instantanea y se restaura antes de cada rama siguiente.
  5. Cola de tareas de barata a cara (con el reloj justo, las que quedan sin hacer son las caras).
Lanzar: PYTHONUTF8=1 python scripts/push_kernels.py lb35 --accelerator NvidiaL4
"""
import json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

det = json.load(open("notebooks/lb33_det.ipynb", encoding="utf-8"))["cells"]
k4 = json.load(open("notebooks/lb33_k4np32.ipynb", encoding="utf-8"))["cells"]
src = lambda c: "".join(c["source"])

RUTA = src(k4[5]).split("import hashlib")[0].replace("%%writefile arc_solver.py\n", "")
assert "def _ruta_arc" in RUTA

def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)

# ------------------------------------------------------------------ arc_solver.py
S = src(det[5])
S = rep(S, "%%writefile arc_solver.py\n", "%%writefile arc_solver.py\n" + RUTA)
S = rep(S, "from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer\n", "")
S = rep(S, "from datasets import Dataset\n", "")
S = rep(S, "import gc\n", "import gc\nimport math\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom types import SimpleNamespace\n")
i0 = S.index("class UnslothFixedTrainer"); i1 = S.index("class QwenDataCollatorForCompletionOnlyLM")
AYUDA = r'''
class TextLM(nn.Module):
    """Decodificador de texto de Qwen3.5 + cabeza de 16 filas, sin la torre visual ni el mrope multimodal."""
    def __init__(self, lm, head):
        super().__init__()
        self.lm = lm
        self.head = head
    @property
    def device(self):
        return self.head.weight.device
    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=False, return_dict=True, **kw):
        out = self.lm(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache)
        return SimpleNamespace(logits=self.head(out.last_hidden_state), past_key_values=out.past_key_values)


def _cache_snap(cache):
    """Instantanea de la cache hibrida: la KV de atencion crece por torch.cat (basta guardar referencias);
    los estados conv/recurrentes de las capas lineales se sobrescriben in place (hay que clonarlos)."""
    snap = []
    for l in cache.layers:
        d = {}
        if hasattr(l, "keys") and not callable(getattr(l, "keys")):
            d["kv"] = (l.keys, l.values)
        if hasattr(l, "recurrent_states"):
            d["conv"] = {k: (v.clone() if v is not None else None) for k, v in l.conv_states.items()}
            d["rec"] = {k: (v.clone() if v is not None else None) for k, v in l.recurrent_states.items()}
        snap.append(d)
    return snap


def _cache_restore(cache, snap):
    for l, d in zip(cache.layers, snap):
        if "kv" in d:
            l.keys, l.values = d["kv"]
        for nombre in ("conv", "rec"):
            if nombre in d:
                dest = l.conv_states if nombre == "conv" else l.recurrent_states
                for k, v in d[nombre].items():
                    if v is None:
                        continue
                    if dest[k] is not None and dest[k].shape == v.shape:
                        dest[k].copy_(v)
                    else:
                        dest[k] = v.clone()


def _labels_mask(ids):
    """Misma regla que QwenDataCollatorForCompletionOnlyLM: perdida solo en las respuestas del assistant."""
    a = np.asarray(ids)
    starts = sorted(np.where(a == USER_TOKEN_ID)[0].tolist() + np.where(a == ASSISTANT_TOKEN_ID)[0].tolist())
    ends = np.where(a == EOS_ID)[0]
    m = np.zeros(len(a), dtype=bool)
    for j, (s, e) in enumerate(zip(starts, ends)):
        if j % 2 == 1:
            m[s + 2:e + 1] = True
    return m


def _es_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _set_ckpt(lm, on):
    for layer in lm.layers:
        layer.gradient_checkpointing = on


'''
S = S[:i0] + AYUDA + S[i1:]

# DFS con instantaneas de la cache
_SORT = "        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]" + chr(10)
S = rep(S, _SORT, _SORT + chr(10).join([
    "",
    "    # Qwen3.5: >= 2 ramas en este nodo => instantanea para restaurar antes de cada rama siguiente",
    "    n_ramas = max([len(candidates[i]) for i in range(n)] + [0])",
    "    snap = _cache_snap(cache) if n_ramas >= 2 else None",
    "    primera = True",
    ""]))
S = rep(S, """        if num_alive_beams == 0:
            break

        outputs = model(""", """        if num_alive_beams == 0:
            break

        if not primera and snap is not None:
            _cache_restore(cache, snap)
        primera = False

        outputs = model(""")

# carga del modelo
i0 = S.index("    peft_params = dict(")
i1 = S.index("    formatter = QwenFormatter(tokenizer=tokenizer)")
CARGA = r'''    import transformers
    from transformers import AutoTokenizer
    from peft import LoraConfig, inject_adapter_in_model

    max_seq_length = 8192
    CK_UMBRAL = 2048

    mdir = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "q35_merged" in root and "config.json" in files and any(f.endswith(".safetensors") for f in files):
            mdir = root
            break
    assert mdir, "sin q35_merged"
    print(f"[Rank {rank}] modelo {mdir} | transformers {transformers.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(mdir)
    _p = tokenizer.encode("<|im_start|>user\n01\n2<|im_end|><|im_start|>assistant\n3<|im_end|>")
    assert _p == [14, 11, 10, 0, 1, 10, 2, 15, 14, 12, 10, 3, 15], _p

    try:
        full = transformers.AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    except Exception as e:
        print(f"[Rank {rank}] sdpa no disponible ({str(e)[:120]}); carga por defecto")
        full = transformers.AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.bfloat16, device_map={"": 0})
    lm = full.model.language_model
    head = full.lm_head
    full.model.visual = None
    gc.collect()
    torch.cuda.empty_cache()
    model = TextLM(lm, head)

    nombres = sorted({n.split(".")[-1] for n, mod in lm.named_modules() if isinstance(mod, nn.Linear)})
    objetivo = [x for x in nombres if x not in ("in_proj_a", "in_proj_b")]
    print(f"[Rank {rank}] LoRA del TTT en {objetivo}")
    torch.manual_seed(42)
    model = inject_adapter_in_model(LoraConfig(r=256, lora_alpha=32, lora_dropout=0.0, bias="none", use_rslora=True,
                                               target_modules=objetivo), model)
    lora_params = []
    for n_, p_ in model.named_parameters():
        es = "lora_" in n_
        p_.requires_grad_(es)
        if es:
            p_.data = p_.data.to(torch.bfloat16)
            lora_params.append(p_)
    default_weights = [p_.detach().clone() for p_ in lora_params]
    print(f"[Rank {rank}] LoRA TTT: {sum(p_.numel() for p_ in lora_params) / 1e6:.0f}M parametros")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _set_ckpt(lm, False)

'''
S = S[:i0] + CARGA + S[i1:]

# TTT
i0 = S.index("        load_result = set_peft_model_state_dict(")
i1 = S.index("        model = FastLanguageModel.for_inference(model)")
i1 = i1 + len("        model = FastLanguageModel.for_inference(model)")
TTT = r'''        for p_, w_ in zip(lora_params, default_weights):
            p_.data.copy_(w_)
        model.train()

        puzzle_ds = arc_test_set.change_keys([key])

        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        seqs = [tokenizer.encode(x["text"]) for x in train_ds.as_list(formatter)]
        orden = np.random.RandomState(42).permutation(len(seqs))
        opt = torch.optim.AdamW(lora_params, lr=5e-5, weight_decay=0.0)
        n_pasos = len(seqs)
        warm = math.ceil(0.1 * n_pasos)
        perdidas = []
        for paso, j in enumerate(orden):
            if paso < warm:
                f_lr = paso / max(1, warm)
            else:
                f_lr = 0.5 * (1.0 + math.cos(math.pi * (paso - warm) / max(1, n_pasos - warm)))
            for g_ in opt.param_groups:
                g_["lr"] = 5e-5 * f_lr
            ids = seqs[j]
            mask = torch.tensor(_labels_mask(ids)[1:], device=model.device)
            x = torch.tensor([ids], device=model.device, dtype=torch.long)
            hecho = False
            for intento in range(2):
                _set_ckpt(lm, intento == 1 or len(ids) > CK_UMBRAL)
                try:
                    lg = model(input_ids=x, use_cache=False).logits[0, :-1].float()
                    loss = F.cross_entropy(lg[mask], x[0, 1:][mask])
                    loss.backward()
                    hecho = True
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if not _es_oom(e):
                        raise
                    opt.zero_grad(set_to_none=True)
                    lg = None
                    loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[Rank {rank}] OOM en TTT ({len(ids)} tok), reintento con checkpointing")
            if hecho:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                opt.step()
                perdidas.append(float(loss.item()))
            opt.zero_grad(set_to_none=True)
        del opt
        _set_ckpt(lm, False)
        model.eval()
        stats = {"pasos": n_pasos, "loss_media": float(np.mean(perdidas)) if perdidas else None, "s": round(time.time() - start_time, 1)}
'''
S = S[:i0] + TTT + S[i1:]
S = rep(S, '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"',
        '        test_path = _ruta_arc("arc-agi_test_challenges.json")')
S = rep(S, '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"',
        '        test_path = _ruta_arc("arc-agi_evaluation_challenges.json")')
for prohibido in ("FastLanguageModel", "Unsloth", "set_peft_model_state_dict(", "Dataset.from_list"):
    assert prohibido not in S, prohibido

# ------------------------------------------------------------------ starter.py (cola de barata a cara)
ST = src(k4[6])
ST = rep(ST, """    for key in sorted(data.keys()):""", """    def _coste(t):
        return sum(len(g["input"]) * len(g["input"][0]) + (len(g["output"]) * len(g["output"][0]) if "output" in g else 0)
                   for g in t["train"] + t["test"])
    for key in sorted(data.keys(), key=lambda k: (_coste(data[k]), k)):""")
ST = rep(ST, "    # Fix Unsloth patching issue\n", "    # importacion escalonada (heredada del campeon)\n")

INSTALL = r'''import os, sys, subprocess
def _wheels(patron):
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.startswith(patron) and f.endswith(".whl") for f in files):
            return root
    return None
W_TF = _wheels("transformers-"); W_FLA = _wheels("flash_linear_attention")
print("wheels:", W_TF, W_FLA, flush=True)
for args in (["install", "-q", "--no-index", "--no-deps", "--upgrade", "--find-links", W_TF, "transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "regex"],
             ["install", "-q", "--no-index", "--no-deps", "--find-links", W_FLA, "flash-linear-attention", "fla-core", "einops"],
             ["uninstall", "-y", "-q", "torchao", "tensorflow"]):
    r = subprocess.run([sys.executable, "-m", "pip"] + args, capture_output=True, text=True)
    print(args[0], args[-1], "rc=%d" % r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
r = subprocess.run([sys.executable, "-c", "import transformers, peft; from transformers.utils.import_utils import is_flash_linear_attention_available as f; print(transformers.__version__, peft.__version__, 'FLA', f())"], capture_output=True, text=True)
print("entorno:", (r.stdout + r.stderr).strip()[-300:], flush=True)
'''

MD = """# lb35: pipeline del campeon (TTT + DFS + re-scoring + seleccion) sobre Qwen3.5-4B-Base + SFT NVARC propio

Pedido del usuario (1-nov). Modelo: `sft35-train` (Qwen3.5-4B-Base, SFT de ~19K secuencias NVARC, vocab 16).
Cambios obligados frente a `lb33-det`: sin unsloth (no soporta Gated DeltaNet), TTT con bucle propio y el mismo
calendario, instantaneas de cache en el DFS, cola de barata a cara. Resto identico."""

def celda(s, tipo="code"):
    c = {"cell_type": tipo, "metadata": {}, "source": s.splitlines(keepends=True)}
    if tipo == "code":
        c["execution_count"] = None; c["outputs"] = []
    return c

RUN = "!OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}"
cells = [celda(MD, "markdown"), celda(src(det[1])), celda(INSTALL), celda(src(det[3])), celda(src(det[4])),
         celda(S), celda(ST), celda(RUN), celda(src(k4[8]))]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/lb35.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(os.path.join(os.environ.get("LB35_DUMP", "."), "_lb35_arc_solver.py"), "w", encoding="utf-8").write(S.split("\n", 1)[1])
print("OK notebooks/lb35.ipynb (arc_solver %d B)" % len(S))
