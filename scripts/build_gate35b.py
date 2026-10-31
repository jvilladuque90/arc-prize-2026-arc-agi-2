"""Genera notebooks/gate35b.ipynb: re-medicion LIMPIA del prior de rejilla de la generacion 3.5.

Corrige los tres huecos de gate35 (auditoria 31-oct, docs/PLAN_AB.md):
  1. variante: compara BASE contra BASE (Qwen3.5-4B-Base / Qwen3.5-2B-Base vs Qwen3-4B-Base);
     el post-entrenado Qwen3-4B va de ancla para enlazar con la medicion vieja.
  2. kernels: instala flash-linear-attention (wheels puras del kernel q35base-dl) y comprueba
     is_flash_linear_attention_available() ANTES de cargar; mide fwd con los kernels activos.
  3. control absoluto: NLL/token sobre texto plano (ingles + codigo) por modelo, para ver que la
     ruta de carga produce un modelo sano, no solo que dos espejos son iguales.
Poblacion: 24 tareas mas baratas (comparables con gate35) + 12 del tramo medio (p40-p60).
L4x4 shard x4 (cada worker carga los 4 modelos en serie, uno a la vez).
PRERREG: el 3.5-4B-Base vive si ratio mediano <= 0,85 frente a Qwen3-4B-Base Y gana >= 2/3;
fwd se reporta pero NO decide (el cuello es el SFT, no el forward de stock).
"""
import json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

WORKER = r'''
import gc, json, os, sys, time, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()
SHARD = int(os.environ.get("SHARD", "0")); N_SHARDS = int(os.environ.get("N_SHARDS", "4"))
N_BARATAS = int(os.environ.get("N_BARATAS", "24")); N_MEDIAS = int(os.environ.get("N_MEDIAS", "12"))
SUCIAS = set(json.loads(os.environ["SUCIAS"]))
OUT = "/kaggle/working/shard%d.json" % SHARD

import torch
import transformers
from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
FLA = bool(is_flash_linear_attention_available()); CONV = bool(is_causal_conv1d_available())
print("[s%d] transformers %s | FLA=%s | causal_conv1d=%s" % (SHARD, transformers.__version__, FLA, CONV), flush=True)
if FLA:
    try:
        import fla; print("[s%d] fla %s" % (SHARD, getattr(fla, "__version__", "?")), flush=True)
    except Exception as e:
        print("[s%d] fla import error: %s" % (SHARD, e), flush=True)

FICH = "arc-agi_evaluation_challenges.json"
DATA = None
for cand in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"):
    if os.path.exists(os.path.join(cand, FICH)):
        DATA = cand; break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if FICH in files:
            DATA = root; break
assert DATA, "sin dataset de la competicion"
CHAL = json.load(open(os.path.join(DATA, FICH)))
SOLS = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))

def buscar_modelo(*marcas):
    mejor = None
    for root, dirs, files in os.walk("/kaggle/input"):
        low = root.lower().replace("\\", "/")
        if "config.json" in files and all(m in low for m in marcas):
            if any(x.endswith(".safetensors") for x in files):
                return root
            mejor = mejor or root
    return mejor

MODELOS = [("q3_4b_base",  buscar_modelo("qwen-3/", "/4b-base")),   # TITULAR justo
           ("q35_4b_base", buscar_modelo("q35_4b_base")),           # candidato
           ("q35_2b_base", buscar_modelo("q35_2b_base")),           # candidato con descuento
           ("q3_4b_post",  buscar_modelo("qwen-3/", "/4b/"))]       # ancla: enlaza con gate35
for e, d in MODELOS:
    print("[s%d] %-12s -> %s" % (SHARD, e, d), flush=True)
assert MODELOS[0][1], "sin titular 4b-base"

def coste(t):
    return sum(len(g["input"])*len(g["input"][0]) + len(g["output"])*len(g["output"][0]) for g in t["train"])
ORD = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))
n = len(ORD); medio = ORD[int(0.40*n):int(0.60*n)]
KEYS_ALL = ORD[:N_BARATAS] + medio[::max(1, len(medio)//N_MEDIAS)][:N_MEDIAS]
ESTRATO = {k: ("barata" if k in ORD[:N_BARATAS] else "media") for k in KEYS_ALL}
MIS = KEYS_ALL[SHARD::N_SHARDS]
print("[s%d] %d tareas" % (SHARD, len(MIS)), flush=True)

def rejilla_txt(g):
    return "\n".join(" ".join(str(c) for c in fila) for fila in g)

def prompt_y_gt(task, gt):
    p = "Solve the puzzle. Each example maps an input grid to an output grid by the same rule.\n\n"
    for i, par in enumerate(task["train"]):
        p += "Example %d\nInput:\n%s\nOutput:\n%s\n\n" % (i+1, rejilla_txt(par["input"]), rejilla_txt(par["output"]))
    p += "Test\nInput:\n%s\nOutput:\n" % rejilla_txt(task["test"][0]["input"])
    return p, rejilla_txt(gt) + "\n"

# CONTROL ABSOLUTO: texto plano. Un modelo sano de 2-4B da ~0,6-1,2 nats/token en ingles de
# wikipedia y menos en codigo. Si un modelo da >3, la carga/ruta esta rota aunque "no lance".
CTRL_EN = ("The Nile is a major north-flowing river in northeastern Africa. It flows into the Mediterranean Sea. "
           "The Nile is the longest river in Africa and has historically been considered the longest river in the world, "
           "though this has been contested by research suggesting that the Amazon River is slightly longer. "
           "The Nile is among the smallest of the world's major rivers as measured by annual flow in cubic metres of water. "
           "About 6,650 km long, its drainage basin covers eleven countries.")
CTRL_PY = ("def fibonacci(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n"
           "        a, b = b, a + b\n    return b\n\n\nfor i in range(10):\n    print(i, fibonacci(i))\n")

@torch.no_grad()
def nll_texto(model, tok, txt):
    ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids.to(next(model.parameters()).device)
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    lg = out.logits[0, :-1].float()
    lp = lg.log_softmax(-1).gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(-lp.mean().item()), int(ids.shape[1])

@torch.no_grad()
def nll_gt(model, tok, prompt, gt_txt):
    ids_p = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    ids_g = tok(gt_txt, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([ids_p, ids_g], dim=1).to(next(model.parameters()).device)
    if ids.shape[1] > 12000:
        return None
    torch.cuda.synchronize(); t0 = time.time()
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    lg = out.logits[0, :-1].float()
    lp = lg.log_softmax(-1).gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    n_p = ids_p.shape[1]
    return {"nll": float(-(lp[n_p-1:]).sum().item()), "n_tok_gt": int(ids_g.shape[1]),
            "n_tok_total": int(ids.shape[1]), "fwd_s": dt}

CLASES = ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForVision2Seq", "AutoModel")

def cargar(dir_):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dir_, trust_remote_code=True)
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True,
              dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    diag, mejor = [], None
    for nombre in CLASES:
        try:
            cls = getattr(transformers, nombre)
            m, info = cls.from_pretrained(dir_, **kw)
        except TypeError:
            kw2 = dict(kw); kw2["torch_dtype"] = kw2.pop("dtype")
            try:
                m, info = getattr(transformers, nombre).from_pretrained(dir_, **kw2)
            except Exception as e:
                diag.append({"clase": nombre, "error": str(e)[:140]}); continue
        except Exception as e:
            diag.append({"clase": nombre, "error": str(e)[:140]}); continue
        falt = list(info.get("missing_keys") or []); inesp = list(info.get("unexpected_keys") or [])
        pars = sum(p.numel() for p in m.parameters())
        diag.append({"clase": nombre, "faltan": len(falt), "inesperadas": len(inesp),
                     "params_M": round(pars / 1e6, 1), "ej_faltan": falt[:5], "ej_inesperadas": inesp[:5]})
        print("[s%d]   %s: faltan %d | inesperadas %d | %.0fM params" % (SHARD, nombre, len(falt), len(inesp), pars / 1e6), flush=True)
        m.eval()
        if mejor is None or len(falt) < mejor[2]:
            if mejor is not None:
                del mejor; gc.collect(); torch.cuda.empty_cache()
            mejor = (m, nombre, len(falt))
        else:
            del m; gc.collect(); torch.cuda.empty_cache()
        if len(falt) == 0:
            break
    if mejor is None:
        raise RuntimeError(json.dumps(diag)[:600])
    return mejor[0], tok, mejor[1], diag

def ruta_lineal(model):
    """Que implementacion usan las capas Gated DeltaNet: kernel FLA o fallback de torch."""
    for mod in model.modules():
        f = getattr(mod, "chunk_gated_delta_rule", None)
        if f is not None:
            return getattr(f, "__module__", str(f))
    return "sin capas lineales"

RES = {"filas": [], "diag": {}, "control": {}, "shard": SHARD, "errores": {},
       "entorno": {"transformers": transformers.__version__, "fla": FLA, "causal_conv1d": CONV}}
tareas = {k: (CHAL[k], SOLS[k][0]) for k in MIS}

for etiq, dir_ in MODELOS:
    if dir_ is None:
        RES["errores"][etiq] = "sin directorio"; print("[s%d] %s SIN DIRECTORIO" % (SHARD, etiq), flush=True); continue
    torch.cuda.empty_cache(); t0 = time.time()
    print("[s%d] === %s ===" % (SHARD, etiq), flush=True)
    try:
        model, tok, clsn, diag = cargar(dir_)
    except Exception as e:
        RES["errores"][etiq] = str(e)[:600]; print("[s%d] %s NO CARGA: %s" % (SHARD, etiq, str(e)[:250]), flush=True)
        json.dump(RES, open(OUT, "w")); continue
    rl = ruta_lineal(model)
    RES["diag"][etiq] = {"elegida": clsn, "intentos": diag, "ruta_lineal": rl, "vocab": int(getattr(tok, "vocab_size", 0) or 0)}
    print("[s%d] %s listo con %s: %.1f GB, %.0fs | ruta lineal: %s" % (SHARD, etiq, clsn, torch.cuda.memory_allocated(0)/1e9, time.time()-t0, rl), flush=True)
    try:
        en, n_en = nll_texto(model, tok, CTRL_EN); py, n_py = nll_texto(model, tok, CTRL_PY)
        RES["control"][etiq] = {"nll_tok_en": round(en, 4), "n_en": n_en, "nll_tok_py": round(py, 4), "n_py": n_py}
        print("[s%d] %s CONTROL texto: ingles %.3f nats/tok (%d tok) | python %.3f nats/tok (%d tok)" % (SHARD, etiq, en, n_en, py, n_py), flush=True)
    except Exception as e:
        RES["control"][etiq] = {"error": str(e)[:200]}
    for k, (task, gt) in tareas.items():
        p, g = prompt_y_gt(task, gt)
        try:
            r = nll_gt(model, tok, p, g)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); r = None
        except Exception as e:
            print("[s%d] %s %s error: %s" % (SHARD, etiq, k, str(e)[:120]), flush=True); r = None
        celdas = len(gt) * len(gt[0])
        fila = {"task": k, "modelo": etiq, "celdas": celdas, "clase": clsn, "estrato": ESTRATO[k]}
        if r:
            fila.update(r); fila["nll_por_celda"] = round(r["nll"] / celdas, 4); fila["tok_por_celda"] = round(r["n_tok_gt"] / celdas, 3)
        RES["filas"].append(fila)
        print("[s%d] %-12s %s [%s]: nll/celda %s | tok/celda %s | fwd %ss" % (SHARD, etiq, k, ESTRATO[k],
              fila.get("nll_por_celda", "-"), fila.get("tok_por_celda", "-"), ("%.2f" % r["fwd_s"]) if r else "-"), flush=True)
        json.dump(RES, open(OUT, "w"))
    del model; gc.collect(); torch.cuda.empty_cache()

json.dump(RES, open(OUT, "w"))
print("[s%d] fin %.0fs" % (SHARD, time.time() - T00), flush=True)
'''

LANZADOR = r'''
import json, os, subprocess, sys, time, glob
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
r = subprocess.run([sys.executable, "-c", "import transformers;from transformers.utils.import_utils import is_flash_linear_attention_available as f;print(transformers.__version__, 'FLA', f())"], capture_output=True, text=True)
print("entorno:", (r.stdout + r.stderr).strip(), flush=True)

N_SHARDS = 4
procs = []
for i in range(N_SHARDS):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "SHARD": str(i), "N_SHARDS": str(N_SHARDS),
                "N_BARATAS": "24", "N_MEDIAS": "12", "SUCIAS": json.dumps(SUCIAS)})
    fh = open("/kaggle/working/log%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"], env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
for i, p, fh in procs:
    p.wait(); fh.close()
    print("worker %d rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)

filas, diag, errores, control, entorno = [], {}, {}, {}, {}
for i in range(N_SHARDS):
    f = "/kaggle/working/shard%d.json" % i
    if os.path.exists(f):
        d = json.load(open(f))
        filas += d.get("filas", []); diag.update(d.get("diag", {})); errores.update(d.get("errores", {}))
        control.update(d.get("control", {})); entorno = d.get("entorno", entorno)
json.dump({"filas": filas, "diag": diag, "errores": errores, "control": control, "entorno": entorno}, open("/kaggle/working/gate35b.json", "w"))

for i in range(N_SHARDS):
    print("=" * 12 + (" LOG WORKER %d " % i) + "=" * 12, flush=True)
    try: print(open("/kaggle/working/log%d.txt" % i).read()[-2500:], flush=True)
    except Exception as e: print("sin log:", e, flush=True)

def med(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

print("\n" + "=" * 80); print("ENTORNO:", json.dumps(entorno)); print("=" * 80, flush=True)
print("CARGA y RUTA DE KERNELS", flush=True)
for e, d in diag.items():
    print("%-12s clase %s | vocab %s | ruta lineal: %s" % (e, d["elegida"], d.get("vocab"), d.get("ruta_lineal")), flush=True)
    for it in d["intentos"]:
        if "error" in it: print("     %-32s ERROR %s" % (it["clase"], it["error"][:80]), flush=True)
        else: print("     %-32s faltan %4d | inesperadas %4d | %.0fM params" % (it["clase"], it["faltan"], it["inesperadas"], it["params_M"]), flush=True)
print("\nCONTROL ABSOLUTO (texto plano, nats/token; sano ~0,6-1,5 en ingles, <1 en codigo)", flush=True)
for e, c in control.items(): print("%-12s %s" % (e, json.dumps(c)), flush=True)

por = {}
for f in filas:
    if "nll_por_celda" in f: por.setdefault(f["task"], {})[f["modelo"]] = f
print("\n" + "=" * 80); print("PRIOR frente al TITULAR JUSTO (Qwen3-4B-Base)"); print("=" * 80, flush=True)
def signo_p(k, n):
    from math import comb
    return sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n   # P(X <= k) con X~Bin(n, 1/2)
for cand in ("q35_4b_base", "q35_2b_base", "q3_4b_post"):
    for estrato in ("barata", "media", None):
        pr = [(t, v["q3_4b_base"], v[cand]) for t, v in por.items() if "q3_4b_base" in v and cand in v and (estrato is None or v[cand]["estrato"] == estrato)]
        if not pr: continue
        r = [b["nll_por_celda"] / max(a["nll_por_celda"], 1e-9) for _, a, b in pr]
        gana = sum(1 for x in r if x < 1); n = len(r)
        print("%-12s [%s] n=%2d | ratio MEDIANO %.3f | gana %d/%d | p(signo) %.3f | fwd x%.2f | tok/celda %.2f vs %.2f"
              % (cand, estrato or "todas", n, med(r), gana, n, signo_p(n - gana, n),
                 med([b["fwd_s"] for _, _, b in pr]) / max(med([a["fwd_s"] for _, a, _ in pr]), 1e-9),
                 med([b["tok_por_celda"] for _, _, b in pr]), med([a["tok_por_celda"] for _, a, _ in pr])), flush=True)
pr = [(t, v["q3_4b_base"], v["q3_4b_post"]) for t, v in por.items() if "q3_4b_base" in v and "q3_4b_post" in v]
if pr:
    r = [b["nll_por_celda"] / max(a["nll_por_celda"], 1e-9) for _, a, b in pr]
    print("\nENLACE con gate35: Qwen3-4B post/base ratio mediano %.3f (si >1, el post-training ya castiga al completado crudo en Qwen3)" % med(r), flush=True)
print("\nPRERREG: q35_4b_base VIVE si ratio <= 0,85 Y gana >= 2/3 frente a q3_4b_base (todas). fwd se reporta, no decide.", flush=True)
print("Si el control de texto da >3 nats/token en algun modelo, su fila es INVALIDA (ruta de carga rota).", flush=True)
if errores: print("ERRORES:", json.dumps(errores)[:1200], flush=True)
print("RESULT_JSON:" + json.dumps({"filas": len(filas), "wall_s": round(time.time()-T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS))

MD = """# gate35b: prior de rejilla de la generacion 3.5, medido LIMPIO (Base vs Base, kernels FLA, control de texto)

Corrige la auditoria de `gate35` (31-oct): alli se midio el **post-entrenado** con el **fallback de torch**
para las 24/32 capas Gated DeltaNet. Aqui: `Qwen3.5-4B-Base` y `Qwen3.5-2B-Base` (kernel `q35base-dl`)
contra `Qwen3-4B-Base` (titular justo), con `flash-linear-attention` instalado y verificado, control
absoluto de NLL en texto plano, y `Qwen3-4B` post-entrenado como ancla para enlazar con la medicion vieja.
24 tareas baratas (comparables) + 12 del tramo medio. L4x4, shard x4.

**PRERREG**: `q35_4b_base` vive si ratio mediano <= 0,85 Y gana >= 2/3 frente a `q3_4b_base`.
`fwd` se reporta pero no decide (el cuello real es rehacer el SFT, no el forward de stock)."""

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
json.dump(nb, open("notebooks/gate35b.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("OK notebooks/gate35b.ipynb (worker %d B, lanzador %d B)" % (len(WORKER), len(LANZADOR)))
