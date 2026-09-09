import io, json, ast, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

WORKER = r'''
import gc, json, os, sys, time, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T00 = time.time()
SHARD = int(os.environ.get("SHARD", "0")); N_SHARDS = int(os.environ.get("N_SHARDS", "4"))
N_TAREAS = int(os.environ.get("N_TAREAS", "24"))
SUCIAS = set(json.loads(os.environ["SUCIAS"]))
OUT = "/kaggle/working/shard%d.json" % SHARD

import torch
import torch.nn as nn
import transformers as _tf
print("[s%d] transformers %s" % (SHARD, _tf.__version__), flush=True)

FICH = "arc-agi_evaluation_challenges.json"
DATA = None
for cand in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"):
    if os.path.exists(os.path.join(cand, FICH)):
        DATA = cand
        break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if FICH in files:
            DATA = root
            break
assert DATA, "sin dataset; /kaggle/input: %s" % os.listdir("/kaggle/input")
CHAL = json.load(open(os.path.join(DATA, FICH)))
SOLS = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))

def buscar_modelo(*marcas):
    for root, dirs, files in os.walk("/kaggle/input/models"):
        low = root.lower().replace("\\", "/")
        if "config.json" in files and all(m in low for m in marcas):
            return root
    return None

MODELOS = [("q3_4b",   buscar_modelo("qwen-3/", "/4b")),        # TITULAR: base de sft139
           ("q35_4b",  buscar_modelo("qwen3.5-4b")),            # candidato mismo tamano
           ("q35_2b",  buscar_modelo("qwen-3-5-2b")),           # candidato con descuento
           ("q3_1.7b", buscar_modelo("qwen-3/", "/1.7b"))]      # ancla de calibracion
for e, d in MODELOS:
    print("[s%d] %-8s -> %s" % (SHARD, e, d), flush=True)
assert MODELOS[0][1], "no se encontro Qwen3-4B de stock (el titular)"

def coste(t):
    return sum(len(g["input"])*len(g["input"][0]) + len(g["output"])*len(g["output"][0]) for g in t["train"])
KEYS_ALL = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))[:N_TAREAS]
MIS = KEYS_ALL[SHARD::N_SHARDS]
print("[s%d] %d tareas: %s" % (SHARD, len(MIS), ",".join(MIS)), flush=True)

def rejilla_txt(g):
    return "\n".join(" ".join(str(c) for c in fila) for fila in g)

def prompt_y_gt(task, gt):
    p = "Solve the puzzle. Each example maps an input grid to an output grid by the same rule.\n\n"
    for i, par in enumerate(task["train"]):
        p += "Example %d\nInput:\n%s\nOutput:\n%s\n\n" % (i+1, rejilla_txt(par["input"]), rejilla_txt(par["output"]))
    p += "Test\nInput:\n%s\nOutput:\n" % rejilla_txt(task["test"][0]["input"])
    return p, rejilla_txt(gt) + "\n"

@torch.no_grad()
def nll_gt(model, tok, prompt, gt_txt):
    ids_p = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    ids_g = tok(gt_txt, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([ids_p, ids_g], dim=1).to(next(model.parameters()).device)
    if ids.shape[1] > 12000:
        return None
    t0 = time.time()
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    lg = out.logits[0, :-1].float()
    tgt = ids[0, 1:]
    lp = lg.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    n_p = ids_p.shape[1]
    nll = float(-(lp[n_p-1:]).sum().item())      # solo los tokens de la GT
    return {"nll": nll, "n_tok_gt": int(ids_g.shape[1]), "n_tok_total": int(ids.shape[1]), "fwd_s": dt}

# ------------------------------------------------------------------ contabilidad de features
VIS = ("visual", "vision", "vit", "image", "merger", "patch_embed", "projector", "mm_proj",
       "multi_modal", "audio", "video")

def censo(model):
    """Reparto de parametros por bloque funcional: que sobra si solo hacemos texto/rejillas."""
    c = {"vision": 0, "embed": 0, "lm_head": 0, "tronco": 0}
    for n, p in model.named_parameters():
        low = n.lower()
        k = p.numel()
        if any(v in low for v in VIS):
            c["vision"] += k
        elif "embed_tokens" in low or low.endswith("embed_tokens.weight"):
            c["embed"] += k
        elif "lm_head" in low:
            c["lm_head"] += k
        else:
            c["tronco"] += k
    c["total"] = sum(c[x] for x in ("vision", "embed", "lm_head", "tronco"))
    return c

def recorta_lm_head(model, n=16):
    """Sustituye la cabeza por una de n filas (lo que hizo Sorokin: 151.936 -> 16).
    Rompe el tie con embed_tokens, que se deja intacto para no romper el lookup de entrada.
    Las salidas dejan de tener sentido: aqui solo se mide el TIEMPO."""
    lh = model.get_output_embeddings()
    if lh is None or not hasattr(lh, "weight"):
        return False
    W = lh.weight.data[:n].clone()
    nueva = nn.Linear(W.shape[1], n, bias=False, dtype=W.dtype, device=W.device)
    nueva.weight.data = W
    try:
        model.set_output_embeddings(nueva)
    except Exception:
        return False
    return True

@torch.no_grad()
def cronometra(model, tok, prompt, reps=3):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(next(model.parameters()).device)
    model(input_ids=ids, use_cache=False)         # calentar
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps

def cargar(dir_):
    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dir_, trust_remote_code=True)
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True,
              torch_dtype=torch.bfloat16, device_map={"": 0})
    errores = []
    for nombre in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
        try:
            cls = getattr(transformers, nombre)
            m = cls.from_pretrained(dir_, **kw)
            m.eval()
            print("[s%d] cargado con %s" % (SHARD, nombre), flush=True)
            return m, tok, nombre
        except Exception as e:
            errores.append("%s: %s" % (nombre, str(e)[:160]))
    raise RuntimeError(" | ".join(errores))

RES = {"filas": [], "censo": {}, "ablacion": {}, "shard": SHARD, "errores": {}}
tareas = {k: (CHAL[k], SOLS[k][0]) for k in MIS}

for etiq, dir_ in MODELOS:
    if dir_ is None:
        RES["errores"][etiq] = "directorio no encontrado"
        print("[s%d] %s SIN DIRECTORIO, salto" % (SHARD, etiq), flush=True)
        continue
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        model, tok, clsn = cargar(dir_)
    except Exception as e:
        RES["errores"][etiq] = str(e)[:600]
        print("[s%d] %s NO CARGA: %s" % (SHARD, etiq, str(e)[:300]), flush=True)
        json.dump(RES, open(OUT, "w"))
        continue
    mem = torch.cuda.memory_allocated(0) / 1e9
    c = censo(model)
    RES["censo"][etiq] = c
    print("[s%d] %s listo: %.1f GB, carga %.0fs (%s)" % (SHARD, etiq, mem, time.time()-t0, clsn), flush=True)
    print("[s%d]   censo (M params): total %.0f | tronco %.0f | vision %.0f | embed %.0f | head %.0f"
          % (SHARD, c["total"]/1e6, c["tronco"]/1e6, c["vision"]/1e6, c["embed"]/1e6, c["lm_head"]/1e6), flush=True)

    # ---- NLL pareada
    for k, (task, gt) in tareas.items():
        p, g = prompt_y_gt(task, gt)
        try:
            r = nll_gt(model, tok, p, g)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[s%d] %s %s OOM" % (SHARD, etiq, k), flush=True)
            r = None
        except Exception as e:
            print("[s%d] %s %s error: %s" % (SHARD, etiq, k, str(e)[:120]), flush=True)
            r = None
        celdas = len(gt) * len(gt[0])
        fila = {"task": k, "modelo": etiq, "celdas": celdas, "mem_gb": round(mem, 2)}
        if r:
            fila.update(r)
            fila["nll_por_celda"] = round(r["nll"] / celdas, 4)
            fila["tok_por_celda"] = round(r["n_tok_gt"] / celdas, 3)
        RES["filas"].append(fila)
        print("[s%d] %-8s %s: nll/celda %s | tok/celda %s | fwd %ss" % (SHARD, etiq, k,
              fila.get("nll_por_celda", "-"), fila.get("tok_por_celda", "-"),
              ("%.2f" % r["fwd_s"]) if r else "-"), flush=True)
        json.dump(RES, open(OUT, "w"))

    # ---- ABLACION DE FEATURES: cuanto cuesta el vocabulario completo (la vision no
    #      entra en el forward de texto, asi que ahi el ahorro es de memoria, no de reloj)
    try:
        kdemo = MIS[0]
        pdemo = prompt_y_gt(CHAL[kdemo], SOLS[kdemo][0])[0]
        t_full = cronometra(model, tok, pdemo)
        ok = recorta_lm_head(model, 16)
        t_trim = cronometra(model, tok, pdemo) if ok else None
        ab = {"fwd_stock_s": round(t_full, 4),
              "fwd_vocab16_s": round(t_trim, 4) if t_trim else None,
              "ahorro_vocab": round(1 - t_trim / t_full, 4) if t_trim else None,
              "params_muertos_M": round((c["vision"] + max(c["embed"], c["lm_head"])) / 1e6, 1),
              "frac_muerta": round((c["vision"] + max(c["embed"], c["lm_head"])) / max(c["total"], 1), 4)}
        RES["ablacion"][etiq] = ab
        print("[s%d]   ablacion: stock %.3fs -> vocab16 %s | ahorro %s | peso muerto %.0fM (%.1f%%)"
              % (SHARD, t_full, ("%.3fs" % t_trim) if t_trim else "n/a",
                 ("%.1f%%" % (100*ab["ahorro_vocab"])) if t_trim else "n/a",
                 ab["params_muertos_M"], 100*ab["frac_muerta"]), flush=True)
    except Exception as e:
        RES["errores"][etiq + "_ablacion"] = str(e)[:300]
        print("[s%d] %s ablacion fallo: %s" % (SHARD, etiq, str(e)[:200]), flush=True)
    json.dump(RES, open(OUT, "w"))

    del model
    gc.collect()
    torch.cuda.empty_cache()

json.dump(RES, open(OUT, "w"))
print("[s%d] fin %.0fs" % (SHARD, time.time() - T00), flush=True)
'''

LANZADOR = r'''import json, os, subprocess, sys, time, glob
T0 = time.time()
SUCIAS = __SUCIAS__

# transformers>=5.2 offline desde el dataset de wheels (Qwen3.5 no existe en la imagen de Kaggle)
WDIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if any(f.startswith("transformers-") and f.endswith(".whl") for f in files):
        WDIR = root
        break
print("wheels en:", WDIR, flush=True)
if WDIR:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                        "--upgrade", "--find-links", WDIR,
                        "transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "regex"],
                       capture_output=True, text=True)
    print("pip rc=%d" % r.returncode, (r.stdout + r.stderr)[-1200:], flush=True)
    v = subprocess.run([sys.executable, "-c", "import transformers, tokenizers; print(transformers.__version__, tokenizers.__version__)"],
                       capture_output=True, text=True)
    print("versiones instaladas:", (v.stdout + v.stderr).strip()[:200], flush=True)
else:
    print("AVISO: no se encontro el dataset de wheels; los Qwen3.5 fallaran", flush=True)

N_SHARDS = 4
procs = []
for i in range(N_SHARDS):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    env["SHARD"] = str(i)
    env["N_SHARDS"] = str(N_SHARDS)
    env["N_TAREAS"] = "24"
    env["SUCIAS"] = json.dumps(SUCIAS)
    fh = open("/kaggle/working/log%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"],
                                      env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    print("lanzado worker %d en GPU %d" % (i, i), flush=True)
for i, p, fh in procs:
    p.wait()
    fh.close()
    print("worker %d termino rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)

filas, censo, ablacion, errores = [], {}, {}, {}
for i in range(N_SHARDS):
    f = "/kaggle/working/shard%d.json" % i
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    filas += d.get("filas", [])
    censo.update(d.get("censo", {}))
    ablacion.update(d.get("ablacion", {}))
    errores.update(d.get("errores", {}))
json.dump({"filas": filas, "censo": censo, "ablacion": ablacion, "errores": errores},
          open("/kaggle/working/gate_base.json", "w"))

for i in range(N_SHARDS):
    print("=" * 14 + (" LOG WORKER %d " % i) + "=" * 14, flush=True)
    try:
        print(open("/kaggle/working/log%d.txt" % i).read()[-2500:], flush=True)
    except Exception as e:
        print("sin log: %s" % e, flush=True)

por = {}
for f in filas:
    if "nll_por_celda" in f:
        por.setdefault(f["task"], {})[f["modelo"]] = f

def mediana(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

print("")
print("=" * 78, flush=True)
print("CENSO DE PARAMETROS (M) — que parte del modelo NO nos sirve (solo texto/rejillas)", flush=True)
print("=" * 78, flush=True)
print("%-9s %9s %9s %9s %9s %9s" % ("modelo", "total", "tronco", "vision", "embed", "lm_head"), flush=True)
for e in ("q3_4b", "q35_4b", "q35_2b", "q3_1.7b"):
    c = censo.get(e)
    if not c:
        continue
    print("%-9s %9.0f %9.0f %9.0f %9.0f %9.0f"
          % (e, c["total"]/1e6, c["tronco"]/1e6, c["vision"]/1e6, c["embed"]/1e6, c["lm_head"]/1e6), flush=True)
print("")
print("%-9s %14s %14s %10s %16s" % ("modelo", "fwd stock", "fwd vocab16", "ahorro", "peso muerto"), flush=True)
for e in ("q3_4b", "q35_4b", "q35_2b", "q3_1.7b"):
    a = ablacion.get(e)
    if not a:
        continue
    print("%-9s %13.3fs %13s %9s %13.0fM (%.1f%%)"
          % (e, a["fwd_stock_s"],
             ("%.3fs" % a["fwd_vocab16_s"]) if a.get("fwd_vocab16_s") else "n/a",
             ("%.1f%%" % (100*a["ahorro_vocab"])) if a.get("ahorro_vocab") else "n/a",
             a["params_muertos_M"], 100*a["frac_muerta"]), flush=True)

print("")
print("=" * 78, flush=True)
print("GATE: prior de ARC frente al TITULAR (Qwen3-4B stock, base de sft139)", flush=True)
print("=" * 78, flush=True)
for cand in ("q35_4b", "q35_2b", "q3_1.7b"):
    pares = [(t, d["q3_4b"], d[cand]) for t, d in por.items() if "q3_4b" in d and cand in d]
    if not pares:
        print("%s: SIN PARES" % cand, flush=True)
        continue
    ratios = [b["nll_por_celda"] / max(a["nll_por_celda"], 1e-9) for _, a, b in pares]
    gana = sum(1 for r in ratios if r < 1)
    f0 = mediana([a["fwd_s"] for _, a, _ in pares])
    f1 = mediana([b["fwd_s"] for _, _, b in pares])
    print("")
    print("--- %s vs q3_4b : n=%d | ratio MEDIANO %.3f | gana %d/%d | fwd x%.2f"
          % (cand, len(pares), mediana(ratios), gana, len(pares), f1 / max(f0, 1e-9)), flush=True)
    for t, a, b in sorted(pares):
        print("    %-10s %8.4f -> %8.4f  x%.3f" % (t, a["nll_por_celda"], b["nll_por_celda"],
              b["nll_por_celda"] / max(a["nll_por_celda"], 1e-9)), flush=True)

print("")
print("PRERREG (registrado antes de correr, un liston por banda):", flush=True)
print("  q35_4b  (mismo tamano, SIN descuento de reloj): vive solo si", flush=True)
print("          ratio mediano <= 0,85  Y  gana >= 2/3  Y  fwd <= 1,05x  ->  si no, MUERTO", flush=True)
print("  q35_2b  (descuento ~x0,56): le basta ratio mediano <= 1,05 (empatar, no ganar)", flush=True)
print("  q3_1.7b ANCLA, no es criterio: traduce delta-NLL a puntos de LB (el 2B de Sorokin", flush=True)
print("          costo 2-3 pt). Aproximada: su 2B no es exactamente Qwen3-1.7B.", flush=True)
if errores:
    print("ERRORES:", json.dumps(errores)[:1500], flush=True)
print("RESULT_JSON:" + json.dumps({"n_filas": len(filas), "censo": list(censo),
                                   "errores": len(errores), "wall_s": round(time.time()-T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS))

MD = """# GATE de modelo base a 4 bandas + ablacion de features

Dos preguntas en un kernel (L4x4, shard x4, ~0,4-0,5 h-cuota).

**1. Prior.** NLL por celda de la rejilla correcta, demos en el prompt como texto, modelos **de
stock** (sin SFT de ARC, sin TTT, sin cirugia de vocabulario), pareado sobre las mismas 24 tareas
de eval limpias de AGI-1. Normalizado por celda porque los tokenizadores difieren.

- `q3_4b` = **titular**, la base de `sft139` (nuestro campeon).
- `q35_4b` = candidato del **mismo tamano**: sin descuento de reloj, tiene que ser mejor.
- `q35_2b` = candidato con **descuento** (~x0,56 de reloj): le basta empatar.
- `q3_1.7b` = **ancla** de calibracion, no criterio.

El `q35_4b` gatea tambien al `q35_2b`: la tesis del 2B exige que la generacion 3.5 valga ~2x en
eficiencia. Si el 3.5-4B no le gana al 3-4B al mismo tamano, esa generacion no vale y el 2B muere.

**2. Ablacion de features.** Solo hacemos texto y rejillas: sobra todo lo multimodal y sobra el
99,99% del vocabulario. Se mide (a) el **censo de parametros** por bloque (tronco / vision /
embeddings / cabeza) y (b) el **tiempo de forward con la cabeza recortada a 16 filas**, que es lo
que hizo Sorokin (151.936 -> 16). La vision no entra en el forward de texto, asi que ahi el ahorro
es de memoria; el vocabulario si entra, y en decode el `lm_head` es un matmul `hidden x vocab` por
token. Esto decide si el +16% de parametros del 3.5-4B lo paga su torre de vision, en cuyo caso el
tronco de texto cabe en el reloj.

**PRERREG**: `q35_4b` vive solo con ratio mediano <= 0,85 **y** gana >= 2/3 **y** fwd <= 1,05x.
`q35_2b`: ratio mediano <= 1,05. `q3_1.7b`: ancla, informativa.
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
json.dump(nb, open("notebooks/gate_base.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

assert "KEYS_ALL[SHARD::N_SHARDS]" in WORKER
assert "N_SHARDS = 4" in LANZADOR and 'CUDA_VISIBLE_DEVICES' in LANZADOR
assert "recorta_lm_head" in WORKER and "censo" in WORKER
assert "/kaggle/input/competitions/" in WORKER and "OutOfMemoryError" in WORKER

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # GATE MoE (8-sep)'
entry = '''    # GATE de modelo base a 4 bandas + ablacion de features (9-sep), L4x4 shardeado x4.
    # Prior (NLL/celda de la GT, pareado, 24 tareas eval) del titular Qwen3-4B contra
    # Qwen3.5-4B (mismo tamano), Qwen3.5-2B (con descuento) y Qwen3-1.7B (ancla). Ademas
    # censo de parametros por bloque y forward con lm_head recortado a 16 filas: cuanto del
    # modelo es peso muerto si solo hacemos texto/rejillas. PRERREG: q35_4b vive solo con
    # ratio <= 0,85 Y gana >= 2/3 Y fwd <= 1,05x; q35_2b le basta ratio <= 1,05.
    "gate-base": {"notebook": "notebooks/gate_base.ipynb", "slug": "arc-agi2-gate-base",
                  "title": "arc agi2 gate base", "gpu": True, "docker": "",
                  "defaults": {"dataset": ["juliancamilovilla/tf52-wheels"],
                               "model": ["qwen-lm/qwen-3/transformers/4b/1",
                                         "qwen-lm/qwen-3/transformers/1.7b/1",
                                         "barnobarno/qwen3.5-4b/transformers/unsloth/1",
                                         "akzsh5100/qwen-3-5-2b/transformers/transformers/1"]}},
'''
if '"gate-base"' not in s:
    assert anchor in s, "ancla no encontrada en push_kernels.py"
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado gate-base")
else:
    print("gate-base ya estaba registrado")
print("OK notebooks/gate_base.ipynb (worker %d B, lanzador %d B)" % (len(WORKER), len(LANZADOR)))
