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
import transformers
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
assert DATA
CHAL = json.load(open(os.path.join(DATA, FICH)))
SOLS = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))

def buscar_modelo(*marcas):
    mejor = None
    for root, dirs, files in os.walk("/kaggle/input/models"):
        low = root.lower().replace("\\", "/")
        if "config.json" in files and all(m in low for m in marcas):
            if any(x.endswith(".safetensors") for x in files):
                return root
            mejor = mejor or root
    return mejor

MODELOS = [("q3_4b",  buscar_modelo("qwen-3/", "/4b")),      # TITULAR
           ("m_lima", buscar_modelo("limamateus")),          # Qwen3.5-4B espejo A
           ("m_thanh", buscar_modelo("thanhduc1102"))]       # Qwen3.5-4B espejo B
for e, d in MODELOS:
    print("[s%d] %-8s -> %s" % (SHARD, e, d), flush=True)
assert MODELOS[0][1], "sin titular"

def coste(t):
    return sum(len(g["input"])*len(g["input"][0]) + len(g["output"])*len(g["output"][0]) for g in t["train"])
KEYS_ALL = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))[:N_TAREAS]
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
    lp = lg.log_softmax(-1).gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    n_p = ids_p.shape[1]
    return {"nll": float(-(lp[n_p-1:]).sum().item()), "n_tok_gt": int(ids_g.shape[1]),
            "n_tok_total": int(ids.shape[1]), "fwd_s": dt}

CLASES = ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForVision2Seq", "AutoModel")

def cargar(dir_):
    """LA PREGUNTA CENTRAL: cuantos pesos se quedan SIN CARGAR. Si faltan claves del tronco
    de texto, esos tensores quedan ALEATORIOS y el modelo parece 'casi bien con reventones'
    -- exactamente el patron que vimos. Se prueba cada clase y se elige la de MENOS
    claves ausentes, no la primera que no lance excepcion."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dir_, trust_remote_code=True)
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True,
              torch_dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    diag, mejor = [], None
    for nombre in CLASES:
        try:
            cls = getattr(transformers, nombre)
            m, info = cls.from_pretrained(dir_, **kw)
        except Exception as e:
            diag.append({"clase": nombre, "error": str(e)[:140]})
            continue
        falt = list(info.get("missing_keys") or [])
        inesp = list(info.get("unexpected_keys") or [])
        pars = sum(p.numel() for p in m.parameters())
        d = {"clase": nombre, "faltan": len(falt), "inesperadas": len(inesp),
             "params_M": round(pars / 1e6, 1),
             "ej_faltan": falt[:5], "ej_inesperadas": inesp[:5]}
        diag.append(d)
        print("[s%d]   %s: faltan %d | inesperadas %d | %.0fM params"
              % (SHARD, nombre, len(falt), len(inesp), pars / 1e6), flush=True)
        if falt[:3]:
            print("[s%d]     ejemplo de clave AUSENTE: %s" % (SHARD, falt[0]), flush=True)
        m.eval()
        if mejor is None or len(falt) < mejor[2]:
            if mejor is not None:
                del mejor
                gc.collect(); torch.cuda.empty_cache()
            mejor = (m, nombre, len(falt))
        else:
            del m
            gc.collect(); torch.cuda.empty_cache()
        if len(falt) == 0:
            break
    if mejor is None:
        raise RuntimeError(json.dumps(diag)[:600])
    return mejor[0], tok, mejor[1], diag

RES = {"filas": [], "diag": {}, "shard": SHARD, "errores": {}}
tareas = {k: (CHAL[k], SOLS[k][0]) for k in MIS}

for etiq, dir_ in MODELOS:
    if dir_ is None:
        RES["errores"][etiq] = "sin directorio"
        print("[s%d] %s SIN DIRECTORIO" % (SHARD, etiq), flush=True)
        continue
    torch.cuda.empty_cache()
    t0 = time.time()
    print("[s%d] === %s ===" % (SHARD, etiq), flush=True)
    try:
        model, tok, clsn, diag = cargar(dir_)
    except Exception as e:
        RES["errores"][etiq] = str(e)[:600]
        print("[s%d] %s NO CARGA: %s" % (SHARD, etiq, str(e)[:250]), flush=True)
        json.dump(RES, open(OUT, "w"))
        continue
    RES["diag"][etiq] = {"elegida": clsn, "intentos": diag}
    mem = torch.cuda.memory_allocated(0) / 1e9
    print("[s%d] %s listo con %s: %.1f GB, %.0fs" % (SHARD, etiq, clsn, mem, time.time()-t0), flush=True)
    for k, (task, gt) in tareas.items():
        p, g = prompt_y_gt(task, gt)
        try:
            r = nll_gt(model, tok, p, g)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); r = None
        except Exception as e:
            print("[s%d] %s %s error: %s" % (SHARD, etiq, k, str(e)[:120]), flush=True); r = None
        celdas = len(gt) * len(gt[0])
        fila = {"task": k, "modelo": etiq, "celdas": celdas, "clase": clsn}
        if r:
            fila.update(r)
            fila["nll_por_celda"] = round(r["nll"] / celdas, 4)
            fila["tok_por_celda"] = round(r["n_tok_gt"] / celdas, 3)
        RES["filas"].append(fila)
        print("[s%d] %-8s %s: nll/celda %s | tok/celda %s | fwd %ss" % (SHARD, etiq, k,
              fila.get("nll_por_celda", "-"), fila.get("tok_por_celda", "-"),
              ("%.2f" % r["fwd_s"]) if r else "-"), flush=True)
        json.dump(RES, open(OUT, "w"))
    del model
    gc.collect(); torch.cuda.empty_cache()

json.dump(RES, open(OUT, "w"))
print("[s%d] fin %.0fs" % (SHARD, time.time() - T00), flush=True)
'''

LANZADOR = r'''import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = __SUCIAS__

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
    print("pip rc=%d" % r.returncode, (r.stdout + r.stderr)[-800:], flush=True)

N_SHARDS = 4
procs = []
for i in range(N_SHARDS):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "SHARD": str(i), "N_SHARDS": str(N_SHARDS),
                "N_TAREAS": "24", "SUCIAS": json.dumps(SUCIAS)})
    fh = open("/kaggle/working/log%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"],
                                      env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
for i, p, fh in procs:
    p.wait(); fh.close()
    print("worker %d rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)

filas, diag, errores = [], {}, {}
for i in range(N_SHARDS):
    f = "/kaggle/working/shard%d.json" % i
    if os.path.exists(f):
        d = json.load(open(f))
        filas += d.get("filas", []); diag.update(d.get("diag", {})); errores.update(d.get("errores", {}))
json.dump({"filas": filas, "diag": diag, "errores": errores}, open("/kaggle/working/gate35.json", "w"))

for i in range(N_SHARDS):
    print("=" * 12 + (" LOG WORKER %d " % i) + "=" * 12, flush=True)
    try:
        print(open("/kaggle/working/log%d.txt" % i).read()[-3000:], flush=True)
    except Exception as e:
        print("sin log:", e, flush=True)

def med(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

print("")
print("=" * 80, flush=True)
print("DIAGNOSTICO DE CARGA — cuantos pesos quedaron SIN cargar (=> tensores aleatorios)", flush=True)
print("=" * 80, flush=True)
for e, d in diag.items():
    print("%-8s elegida: %s" % (e, d["elegida"]), flush=True)
    for it in d["intentos"]:
        if "error" in it:
            print("     %-32s ERROR %s" % (it["clase"], it["error"][:80]), flush=True)
        else:
            print("     %-32s faltan %4d | inesperadas %4d | %.0fM params  %s"
                  % (it["clase"], it["faltan"], it["inesperadas"], it["params_M"],
                     ("<- AUSENTE: " + it["ej_faltan"][0]) if it["ej_faltan"] else ""), flush=True)

por = {}
for f in filas:
    if "nll_por_celda" in f:
        por.setdefault(f["task"], {})[f["modelo"]] = f
print("")
print("=" * 80, flush=True)
print("PRIOR frente al TITULAR (Qwen3-4B stock)", flush=True)
print("=" * 80, flush=True)
for cand in ("m_lima", "m_thanh"):
    pr = [(t, v["q3_4b"], v[cand]) for t, v in por.items() if "q3_4b" in v and cand in v]
    if not pr:
        print("%s: SIN PARES" % cand, flush=True)
        continue
    r = [b["nll_por_celda"] / max(a["nll_por_celda"], 1e-9) for _, a, b in pr]
    print("%-8s n=%2d | ratio MEDIANO %.3f | gana %d/%d | fwd x%.2f"
          % (cand, len(r), med(r), sum(1 for x in r if x < 1), len(r),
             med([b["fwd_s"] for _, _, b in pr]) / max(med([a["fwd_s"] for _, a, _ in pr]), 1e-9)), flush=True)
if "m_lima" in [f["modelo"] for f in filas] and "m_thanh" in [f["modelo"] for f in filas]:
    pr = [(t, v["m_lima"], v["m_thanh"]) for t, v in por.items() if "m_lima" in v and "m_thanh" in v]
    if pr:
        r = [b["nll_por_celda"] / max(a["nll_por_celda"], 1e-9) for _, a, b in pr]
        print("")
        print("CONTROL entre espejos (lima vs thanh): ratio mediano %.3f | identicos si ~1,000" % med(r), flush=True)
        print("  -> si discrepan, el problema son los ESPEJOS; si coinciden, es el MODELO o la CARGA", flush=True)
print("")
print("PRERREG: (1) si algun espejo carga con 0 claves ausentes y su ratio <= 0,85 con fwd <= 1,05x,", flush=True)
print("    la generacion 3.5 SI mejora a igual tamano => el proyecto de modernizar arranca.", flush=True)
print("(2) si el mejor ratio se queda ~1 o peor con carga limpia, la rama muere DE VERDAD.", flush=True)
print("(3) si todas las cargas dejan claves ausentes, el veredicto de gate-base era un ARTEFACTO", flush=True)
print("    y hay que traer el checkpoint oficial en vez de espejos.", flush=True)
if errores:
    print("ERRORES:", json.dumps(errores)[:1200], flush=True)
print("RESULT_JSON:" + json.dumps({"filas": len(filas), "wall_s": round(time.time()-T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS))

MD = """# gate35: la generacion 3.5 mejora a igual tamano, o el veredicto anterior fue un artefacto?

`gate-base` dio que Qwen3.5-4B es 1,299x PEOR que el titular **y 1,300x peor que el Qwen3.5-2B de
su propia familia**, con reventones de x5,7 en tareas sueltas. Un 4B peor que el 2B de su misma
generacion no es plausible. Y hay una pista dura: se cargaron **4.206M parametros de los 4.660M
del disco**. Lo atribui a la torre de vision, pero si lo que falto fueron pesos del TRONCO DE
TEXTO, esos tensores quedan **aleatorios** y producen exactamente ese patron.

Este banco lo mide directamente con `output_loading_info=True`: **cuantas claves quedan ausentes**
en cada clase de carga, y elige la de MENOS ausentes en vez de la primera que no lance excepcion.

Ademas trae un **control entre espejos**: dos mirrors independientes del mismo Qwen3.5-4B
(`limamateus` en pyTorch y `thanhduc1102` en transformers). Si discrepan entre si, el problema son
los espejos; si coinciden, es el modelo o la carga.

**PRERREG**: (1) carga limpia (0 ausentes) + ratio <= 0,85 + fwd <= 1,05x => la generacion mejora a
igual tamano y el proyecto de modernizar arranca. (2) carga limpia y ratio ~1 o peor => la rama
muere de verdad. (3) todas las cargas con claves ausentes => el veredicto de `gate-base` era un
ARTEFACTO y hay que traer el checkpoint oficial.
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
json.dump(nb, open("notebooks/gate35.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

assert "output_loading_info=True" in WORKER
assert "missing_keys" in WORKER and "len(falt) < mejor[2]" in WORKER
assert "KEYS_ALL[SHARD::N_SHARDS]" in WORKER and "N_SHARDS = 4" in LANZADOR

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # GATE de modelo base a 4 bandas'
entry = '''    # gate35 (10-sep), L4x4 shard x4: re-mide Qwen3.5-4B contra el titular con DIAGNOSTICO DE
    # CARGA (output_loading_info: cuantas claves quedan ausentes => tensores aleatorios) y CONTROL
    # ENTRE ESPEJOS (limamateus vs thanhduc1102). En gate-base se cargaron 4.206M de 4.660M params
    # y el 3.5-4B salio peor que el 3.5-2B de su propia familia: hay que descartar artefacto de
    # carga antes de enterrar la rama. PRERREG: carga limpia + ratio <=0,85 + fwd <=1,05x => arranca.
    "gate35": {"notebook": "notebooks/gate35.ipynb", "slug": "arc-agi2-gate35",
               "title": "arc agi2 gate35", "gpu": True, "docker": "",
               "defaults": {"dataset": ["juliancamilovilla/tf52-wheels"],
                            "model": ["qwen-lm/qwen-3/transformers/4b/1",
                                      "limamateus/qwen3-5-4b/pyTorch/default/1",
                                      "thanhduc1102/qwen_35_4b/transformers/default/1"]}},
'''
if '"gate35"' not in s:
    assert anchor in s
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado gate35")
else:
    print("gate35 ya estaba")
print("OK notebooks/gate35.ipynb (worker %d B, lanzador %d B)" % (len(WORKER), len(LANZADOR)))
