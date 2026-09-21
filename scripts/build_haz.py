import io, json, ast, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

# las 18 tareas de strat (produccion exacta, eval estratificado p5-p95), sin la contaminada
T = [t for t in json.load(open("scripts/_strat_analizado.json", encoding="utf-8"))
     if t.get("vistas_hechas", 16) > 0 and t["task"] != "36a08778"]
VACIAS = [t["task"] for t in T if sum(1 for v in t["vistas"] if v) == 0]
CON = [t["task"] for t in T if sum(1 for v in t["vistas"] if v) > 0]
est = {t["task"]: t["ttt_s"] for t in T}
W = [[], [], [], []]; L = [0.0] * 4
for k in sorted(est, key=lambda k: -est[k]):
    j = L.index(min(L)); W[j].append(k); L[j] += est[k]
for j in range(4):
    W[j].sort(key=lambda k: est[k])
print("18 tareas (%d con pool vacio, %d con candidatos) | TTT estimado por worker %s s"
      % (len(VACIAS), len(CON), [round(x) for x in L]))

# ---------------------------------------------------------------- worker: strat + decode nuevo
nb = json.load(open("notebooks/strat.ipynb", encoding="utf-8"))
w = "".join(nb["cells"][1]["source"]).split("\n", 1)[1]

def rep(old, new):
    global w
    assert old in w, "no encontrado: " + old[:70]
    w = w.replace(old, new, 1)

rep('''MIS_KEYS = json.loads(os.environ["MIS_KEYS"])''',
    '''MIS_KEYS = json.loads(os.environ["MIS_KEYS"])
VACIAS = set(json.loads(os.environ.get("VACIAS", "[]")))
ANCHOS = json.loads(os.environ.get("ANCHOS", "[1, 8]"))''')

# --- instrumentos nuevos: rango de la GT con teacher forcing, greedy y haz
rep('''@torch.no_grad()
def nll_de_gt(task, gt):''',
'''def _prompt_canonico(task):
    """Vista canonica (k=0, permutacion identidad), igual que nll_de_gt."""
    p = []
    for par in task["train"]:
        t, _ = pair_tokens_mask(par["input"], par["output"])
        p += t
    return p + query_tokens(task["test"][0]["input"])


@torch.no_grad()
def rangos_de_gt(task, gt):
    """EL INSTRUMENTO CENTRAL, y cuesta UN forward: fuerza la GT y mide, en cada posicion,
    el RANGO del token correcto entre los 12 tokens ARC y su NLL. De ahi sale analiticamente
    que ancho de haz haria falta:
      - rango 1 en todas las posiciones  => el greedy YA produce la GT (ancho 1 basta)
      - deficit acumulado frente al camino greedy => cota del ancho necesario (exp(deficit))"""
    prompt = _prompt_canonico(task)
    gtoks = grid_tokens(gt) + [EOS]
    ids = torch.tensor([prompt + gtoks], device="cuda:0", dtype=torch.long)
    with sdpa_kernel(_SDPA):
        out = MODEL(input_ids=ids, use_cache=False, return_dict=True)
    lg = out.logits[0, len(prompt) - 1:len(prompt) - 1 + len(gtoks)].float()
    lg = lg.index_select(-1, _TOKID)                      # solo los 12 tokens ARC
    lp = lg.log_softmax(-1)
    idx = {t: i for i, t in enumerate(ARC_TOKENS)}
    col = torch.tensor([idx[t] for t in gtoks], device="cuda:0")
    nll_gt = -lp.gather(1, col.view(-1, 1)).squeeze(1)     # NLL del token correcto
    nll_best = -lp.max(dim=1).values                       # NLL del argmax
    rango = (lp > lp.gather(1, col.view(-1, 1))).sum(1) + 1
    deficit = float((nll_gt - nll_best).sum().item())      # cuanto peor es la GT que el greedy
    r = rango.tolist()
    return {"n_tok": len(gtoks), "n_no_argmax": int(sum(1 for x in r if x > 1)),
            "rango_max": int(max(r)), "rangos_altos": sorted([x for x in r if x > 1], reverse=True)[:8],
            "deficit_vs_greedy": round(deficit, 3),
            "ancho_implicado": round(float(np.exp(min(deficit, 20))), 1),
            "nll_total": round(float(nll_gt.sum().item()), 3)}


def _reordena(cache, idx):
    """Reordena/duplica las filas de la cache KV para que coincidan con los padres elegidos.
    Sin esto el haz mezcla la historia de unas secuencias con los tokens de otras (fallo
    silencioso). Se prueban las tres APIs que ha tenido transformers; si ninguna existe se
    LANZA, no se ignora."""
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(idx)
        return
    if getattr(cache, "key_cache", None) is not None:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].index_select(0, idx)
            cache.value_cache[i] = cache.value_cache[i].index_select(0, idx)
        return
    capas = getattr(cache, "layers", None)
    if capas:
        for c in capas:
            c.keys = c.keys.index_select(0, idx)
            c.values = c.values.index_select(0, idx)
        return
    raise RuntimeError("no se como reordenar esta cache: %s" % type(cache))


@torch.no_grad()
def haz(task, gt, ancho, tope_tok):
    """Busqueda en HAZ de ancho fijo sobre la vista canonica. Coste acotado y predecible:
    siempre `ancho` secuencias vivas, sin explosion de ramas (el modo de fallo del umbral
    lineal). Devuelve si la GT aparece en el haz y el mejor candidato."""
    prompt = _prompt_canonico(task)
    gtoks = grid_tokens(gt) + [EOS]
    ids = torch.tensor([prompt] * ancho, device="cuda:0", dtype=torch.long)
    with sdpa_kernel(_SDPA):
        out = MODEL(input_ids=ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    lp = out.logits[:, -1].float().index_select(-1, _TOKID).log_softmax(-1)
    vivos = [([], 0.0) for _ in range(ancho)]
    acabados = []
    activo = [True] * ancho
    primero = True
    for paso in range(tope_tok):
        cands = []
        for b in range(ancho):
            if not activo[b]:
                continue
            if primero and b > 0:
                break                                     # todas las filas son iguales al inicio
            cam, sc = vivos[b]
            for j, t in enumerate(ARC_TOKENS):
                cands.append((sc - float(lp[b, j].item()), b, t))
        primero = False
        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        nuevos, filas, toks = [], [], []
        for sc, b, t in cands:
            if len(nuevos) >= ancho:
                break
            cam = vivos[b][0] + [t]
            if t == EOS:
                acabados.append((sc, cam))
                continue
            nuevos.append((cam, sc)); filas.append(b); toks.append(t)
        if not nuevos:
            break
        idxf = torch.tensor(filas, device="cuda:0")
        _reordena(cache, idxf)          # SIN esto las filas de la cache quedan desalineadas
        with sdpa_kernel(_SDPA):
            o = MODEL(input_ids=torch.tensor(toks, device="cuda:0", dtype=torch.long).view(-1, 1),
                      past_key_values=cache, use_cache=True, return_dict=True)
        cache = o.past_key_values
        lp = o.logits[:, -1].float().index_select(-1, _TOKID).log_softmax(-1)
        vivos = nuevos + [([], 1e9)] * (ancho - len(nuevos))
        activo = [True] * len(nuevos) + [False] * (ancho - len(nuevos))
        if len(acabados) >= ancho:
            break
    acabados.sort(key=lambda x: x[0])
    en_haz = any(c == gtoks for _, c in acabados)
    mejor = None
    for _, c in acabados[:1]:
        mejor = toks_to_grid(c[:-1]) if c and c[-1] == EOS else toks_to_grid(c)
    return {"ancho": ancho, "n_acabados": len(acabados), "gt_en_haz": bool(en_haz),
            "mejor_ok": bool(mejor is not None and mejor == gt)}


@torch.no_grad()
def nll_de_gt(task, gt):''')

# --- sustituir el bloque de decode por las medidas nuevas
old_dec = w[w.index("    v_dec = build_vistas(pares, seed + 1, N_VISTAS)"):w.index("    n_con_gt = sum(")]
rep(old_dec, '''    # === MEDIDAS NUEVAS (sin las 16 vistas: aqui la pregunta es el ANCHO, no la vista) ===
    t0 = time.time()
    try:
        rg = rangos_de_gt(task, gt)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); rg = None
    except Exception as e:
        print("[s%d] rangos fallo en %s: %s" % (SHARD, k, str(e)[:120]), flush=True); rg = None
    t_rang = time.time() - t0
    hazes = []
    tope = min(931, len(grid_tokens(gt)) + 40)
    for anc in ANCHOS:
        t1 = time.time()
        try:
            h = haz(task, gt, anc, tope)
            h["s"] = round(time.time() - t1, 1)
            hazes.append(h)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hazes.append({"ancho": anc, "oom": True, "s": round(time.time() - t1, 1)})
        except Exception as e:
            print("[s%d] haz %d fallo en %s: %s" % (SHARD, anc, k, str(e)[:120]), flush=True)
            hazes.append({"ancho": anc, "error": str(e)[:120]})
        torch.cuda.empty_cache()
    vistas_out = []
    t_dec = time.time() - t0
''')

rep('''    n_con_gt = sum(1 for cs in vistas_out if any(c["grid"] == gt for c in cs))''',
    '''    n_con_gt = 0''')

rep('''                          "truncado": truncado, "vistas_hechas": len(vistas_out)})''',
    '''                          "pool_vacio_en_strat": k in VACIAS,
                          "rangos": rg, "hazes": hazes, "t_rangos": round(t_rang, 1)})''')

rep('''    print("[s%d %d/%d] %s coste %d: GT en %d/%d vistas | nll/celda pre %s -> post %s | seqs %d | ttt %.0fs dec %.0fs%s"
          % (SHARD, i + 1, len(MIS_KEYS), k, coste(task), n_con_gt, len(vistas_out),
             ("%.4f" % (_pre / celdas)) if _pre is not None else "-",
             ("%.4f" % (gt_nll / celdas)) if gt_nll is not None else "-",
             n_seqs, t_ttt, t_dec, " TRUNCADO" if truncado else ""), flush=True)''',
'''    print("[s%d %d/%d] %s coste %d %s | tok %s no-argmax %s rango_max %s deficit %s ancho~%s | haz %s | ttt %.0fs dec %.0fs"
          % (SHARD, i + 1, len(MIS_KEYS), k, coste(task), "VACIA" if k in VACIAS else "con-pool",
             rg["n_tok"] if rg else "-", rg["n_no_argmax"] if rg else "-",
             rg["rango_max"] if rg else "-", rg["deficit_vs_greedy"] if rg else "-",
             rg["ancho_implicado"] if rg else "-",
             " ".join("w%d:%s" % (h["ancho"], "GT" if h.get("gt_en_haz") else ("oom" if h.get("oom") else "no")) for h in hazes),
             t_ttt, t_dec), flush=True)''')

# el truncado ya no aplica igual, pero la variable se usa: dejarla definida
rep('''    vistas_out, t0 = [], time.time()
    truncado = False''', '''    truncado = False''') if '''    vistas_out, t0 = [], time.time()
    truncado = False''' in w else None

WORKER = w
ast.parse(WORKER)
assert "rangos_de_gt" in WORKER and "def haz(" in WORKER
assert "len(v_train) == 128" in WORKER and "_prompt_canonico" in WORKER

LANZADOR = r'''import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = __SUCIAS__
PLAN = __PLAN__
VACIAS = __VACIAS__

procs = []
for i in range(4):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "SHARD": str(i), "N_SHARDS": "4",
                "N_VISTAS": "0", "BUDGET_VISTA": "45", "WALL_MAX": "5100",
                "ANCHOS": "[1, 8]", "VACIAS": json.dumps(VACIAS),
                "SUCIAS": json.dumps(SUCIAS), "MIS_KEYS": json.dumps(PLAN[i])})
    fh = open("/kaggle/working/log%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"],
                                      env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    print("worker %d: %d tareas" % (i, len(PLAN[i])), flush=True)
for i, p, fh in procs:
    p.wait(); fh.close()
    print("worker %d rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)

tareas = []
for i in range(4):
    f = "/kaggle/working/shard%d.json" % i
    if os.path.exists(f):
        tareas += json.load(open(f)).get("tareas", [])
json.dump({"tareas": tareas}, open("/kaggle/working/haz.json", "w"))
for i in range(4):
    print("=" * 12 + (" LOG WORKER %d " % i) + "=" * 12, flush=True)
    try:
        print(open("/kaggle/working/log%d.txt" % i).read()[-2500:], flush=True)
    except Exception as e:
        print("sin log:", e, flush=True)

tareas.sort(key=lambda t: (not t.get("pool_vacio_en_strat"), t["coste"]))
print("")
print("=" * 104, flush=True)
print("HAZ: en las tareas donde el DFS no produce NADA, que ancho haria falta para tener la GT?", flush=True)
print("=" * 104, flush=True)
print("%-10s %9s %6s %10s %9s %9s %10s | %s" % ("tarea", "pool strat", "tok", "no-argmax", "rango max", "deficit", "ancho~", "haz 1/8"), flush=True)
for t in tareas:
    r = t.get("rangos") or {}
    hs = {h["ancho"]: h for h in (t.get("hazes") or [])}
    def cel(a):
        h = hs.get(a)
        if not h: return "-"
        if h.get("oom"): return "oom"
        return "GT" if h.get("gt_en_haz") else "no"
    print("%-10s %9s %6s %10s %9s %9s %10s | %s" % (
        t["task"], "VACIO" if t.get("pool_vacio_en_strat") else "con-pool",
        r.get("n_tok", "-"), r.get("n_no_argmax", "-"), r.get("rango_max", "-"),
        r.get("deficit_vs_greedy", "-"), r.get("ancho_implicado", "-"),
        "  ".join("%s:%s" % (a, cel(a)) for a in (1, 8))), flush=True)

vac = [t for t in tareas if t.get("pool_vacio_en_strat") and t.get("rangos")]
con = [t for t in tareas if not t.get("pool_vacio_en_strat") and t.get("rangos")]
def med(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
print("")
for nom, g in (("POOL VACIO", vac), ("con pool", con)):
    if not g: continue
    print("%-11s n=%d | no-argmax mediana %.0f de %.0f tokens (%.2f%%) | deficit mediana %.2f | ancho implicado mediana %.0f"
          % (nom, len(g), med([t["rangos"]["n_no_argmax"] for t in g]), med([t["rangos"]["n_tok"] for t in g]),
             100.0 * med([t["rangos"]["n_no_argmax"] for t in g]) / max(med([t["rangos"]["n_tok"] for t in g]), 1),
             med([t["rangos"]["deficit_vs_greedy"] for t in g]),
             med([t["rangos"]["ancho_implicado"] for t in g])), flush=True)
    for a in (1, 8):
        n = sum(1 for t in g if any(h.get("ancho") == a and h.get("gt_en_haz") for h in (t.get("hazes") or [])))
        print("      ancho %2d: GT en el haz en %d/%d" % (a, n, len(g)), flush=True)
print("")
print("LECTURA: no-argmax = 0 => el GREEDY ya produce la GT y basta ancho 1.", flush=True)
print("         deficit = cuanto peor puntua la GT que el camino greedy; ancho~ = exp(deficit)", flush=True)
print("         es la cota del ancho necesario para que la GT sobreviva en el haz.", flush=True)
print("RESULT_JSON:" + json.dumps({"tareas": len(tareas), "wall_s": round(time.time() - T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS)).replace("__PLAN__", json.dumps(W)).replace("__VACIAS__", json.dumps(VACIAS))
ast.parse(LANZADOR)

MD = """# haz: el 33% que no produce NADA, es recuperable con busqueda de ancho fijo?

Auditoria del 21-sep (`docs/AUDITORIA_perdida.md`): el 33% de las tareas no produce **ningun**
candidato y cae al placeholder `[[0]]`. En cinco de esas seis, el modelo esta al **96,3-98,5% por
celda** tras el TTT: la NLL total supera el umbral 4-18 veces **solo por longitud**.

La familia de reglas de aceptacion se cerro el 22-ago, pero **sin TTT y con una sola vista**
(verificado: `dfs_budget.ipynb` no tiene LoRA ni augmentaciones), y el propio plan dejo escrito
"pendiente: re-medir con TTT". Esto es esa re-medicion.

**Instrumento central (UN forward por tarea):** forzar la GT y medir, en cada posicion, el RANGO
del token correcto y su NLL. De ahi sale **analiticamente** el ancho de haz necesario:
`no-argmax = 0` => el greedy ya produce la GT; `ancho ~ exp(deficit frente al camino greedy)`.
Ademas se corre busqueda en haz real de ancho 1 y 8 (coste acotado, sin explosion de ramas,
que fue el modo de fallo del umbral lineal).

TTT de produccion exacto (128 secuencias). 18 tareas de `strat`: 6 con pool vacio y 12 con pool,
estas ultimas como control.
"""

def celda(src, tipo="code"):
    d = {"cell_type": tipo, "metadata": {}, "source": src.splitlines(keepends=True)}
    if tipo == "code":
        d["execution_count"] = None; d["outputs"] = []
    return d

nb2 = {"cells": [celda(MD, "markdown"),
                 celda("%%writefile /kaggle/working/_worker_src.py\n" + WORKER.lstrip("\n")),
                 celda(LANZADOR)],
       "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
json.dump(nb2, open("notebooks/haz.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # rb2 (12-sep)'
entry = '''    # haz (21-sep): re-medicion CON TTT de la familia de reglas de aceptacion, cerrada el 22-ago
    # sin TTT y con una sola vista. La auditoria encontro que el 33% de tareas no produce NINGUN
    # candidato aunque el modelo este al 96-98% por celda: impuesto al tamano del umbral absoluto.
    # Instrumento: rango del token correcto con teacher forcing (1 forward) => ancho de haz
    # necesario ~ exp(deficit vs greedy). Mas haz real de ancho 1/8/32. 18 tareas de strat.
    "haz": {"notebook": "notebooks/haz.ipynb", "slug": "arc-agi2-haz",
            "title": "arc agi2 haz", "gpu": True,
            "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                       "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
            "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                         "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
'''
if '"haz"' not in s:
    assert anchor in s
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado haz")
print("OK notebooks/haz.ipynb | vacias:", VACIAS)
