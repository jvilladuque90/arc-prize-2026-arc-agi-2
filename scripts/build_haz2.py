import io, json, ast, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

T = [t for t in json.load(open("scripts/_strat_analizado.json", encoding="utf-8"))
     if t.get("vistas_hechas", 16) > 0 and t["task"] != "36a08778"]
VACIAS = [t["task"] for t in T if sum(1 for v in t["vistas"] if v) == 0]
est = {t["task"]: t["ttt_s"] for t in T}
W = [[], [], [], []]; L = [0.0] * 4
for k in sorted(est, key=lambda k: -est[k]):
    j = L.index(min(L)); W[j].append(k); L[j] += est[k]
for j in range(4):
    W[j].sort(key=lambda k: est[k])
print("18 tareas | TTT estimado por worker %s s" % [round(x) for x in L])

nb = json.load(open("notebooks/haz.ipynb", encoding="utf-8"))
w = "".join(nb["cells"][1]["source"]).split("\n", 1)[1]

def rep(old, new):
    global w
    assert old in w, "no encontrado: " + old[:80]
    w = w.replace(old, new, 1)

# ---------------------------------------------------------------- haz VECTORIZADO
ini = w.index("@torch.no_grad()\ndef haz(task, gt, ancho, tope_tok):")
fin = w.index("@torch.no_grad()\ndef nll_de_gt(task, gt):")
viejo = w[ini:fin]
nuevo = '''@torch.no_grad()
def haz(task, gt, ancho, tope_tok):
    """Haz de ancho fijo, VECTORIZADO. La version anterior hacia 96 llamadas a .item() por
    paso (96 sincronizaciones GPU->CPU, hasta 89.000 por tarea larga) y costaba 181 s en la
    tarea mas larga contra un suelo teorico de ~19 s. Aqui: una ordenacion en GPU y DOS
    .tolist() por paso. Ademas el prefill va con lote 1 y la cache se expande, en vez de
    recomputar el prompt `ancho` veces."""
    prompt = _prompt_canonico(task)
    gtoks = grid_tokens(gt) + [EOS]
    NT = len(ARC_TOKENS)
    IEOS = ARC_TOKENS.index(EOS)
    with sdpa_kernel(_SDPA):
        out = MODEL(input_ids=torch.tensor([prompt], device="cuda:0", dtype=torch.long),
                    use_cache=True, return_dict=True)
    cache = out.past_key_values
    lp = out.logits[:, -1].float().index_select(-1, _TOKID).log_softmax(-1)     # [1, NT]
    sc = torch.zeros(1, device="cuda:0")
    caminos = [[]]
    acabados = []
    for _ in range(tope_tok):
        tot = (sc.view(-1, 1) - lp).reshape(-1)               # [k*NT] NLL acumulada
        orden = torch.argsort(tot)
        vals = tot.index_select(0, orden).tolist()            # 1 sincronizacion
        idxs = orden.tolist()                                 # 1 sincronizacion
        nuevos, padres, toks, nsc = [], [], [], []
        for v, ix in zip(vals, idxs):
            if len(nuevos) >= ancho:
                break
            p, j = ix // NT, ix % NT
            cam = caminos[p] + [ARC_TOKENS[j]]
            if j == IEOS:
                acabados.append((v, cam))
                continue
            if len(cam) >= tope_tok:
                continue
            nuevos.append(cam); padres.append(p); toks.append(ARC_TOKENS[j]); nsc.append(v)
        if not nuevos or len(acabados) >= ancho:
            break
        _reordena(cache, torch.tensor(padres, device="cuda:0", dtype=torch.long))
        with sdpa_kernel(_SDPA):
            o = MODEL(input_ids=torch.tensor(toks, device="cuda:0", dtype=torch.long).view(-1, 1),
                      past_key_values=cache, use_cache=True, return_dict=True)
        cache = o.past_key_values
        lp = o.logits[:, -1].float().index_select(-1, _TOKID).log_softmax(-1)
        sc = torch.tensor(nsc, device="cuda:0")
        caminos = nuevos
    acabados.sort(key=lambda x: x[0])
    rejillas = []
    for v, c in acabados[:ancho]:
        g = toks_to_grid(c[:-1] if c and c[-1] == EOS else c)
        if g is not None:
            rejillas.append({"grid": g, "score": round(v, 4)})
    return {"ancho": ancho, "n_acabados": len(acabados), "gt_en_haz": any(c == gtoks for _, c in acabados),
            "cands": rejillas[:8],                      # se guardan para re-jugar la seleccion offline
            "mejor_ok": bool(rejillas and rejillas[0]["grid"] == gt)}


'''
rep(viejo, nuevo)
WORKER = w
ast.parse(WORKER)
_seg = WORKER[WORKER.index("def haz(task"):WORKER.index("def nll_de_gt")]
print("   haz nuevo: %d .item() | %d .tolist() | argsort:%s" % (_seg.count(".item()"), _seg.count(".tolist()"), "torch.argsort" in _seg))
assert "cands.append((sc - float(lp[b, j].item())" not in WORKER, "el bucle viejo sigue ahi"
_cuerpo = _seg.split('"""')[2]          # fuera el docstring, que MENCIONA .item() en prosa
print("   cuerpo ejecutable: %d .item() | %d .tolist()" % (_cuerpo.count(".item()"), _cuerpo.count(".tolist()")))
assert _cuerpo.count(".item()") == 0, "queda una sincronizacion por candidato"
assert _cuerpo.count(".tolist()") == 2, "deben ser exactamente 2 sincronizaciones por paso"
assert '"cands": rejillas[:8]' in WORKER and "_reordena(cache" in WORKER
assert "len(v_train) == 128" in WORKER

LANZADOR = "".join(nb["cells"][2]["source"])
LANZADOR = LANZADOR.replace('"ANCHOS": "[1, 8]"', '"ANCHOS": "[8, 16]"')
LANZADOR = LANZADOR.replace('for a in (1, 8)', 'for a in (8, 16)')
LANZADOR = LANZADOR.replace('"%s:%s" % (a, cel(a)) for a in (1, 8)', '"%s:%s" % (a, cel(a)) for a in (8, 16)')
LANZADOR = LANZADOR.replace('"haz 1/8"', '"haz 8/16"')
LANZADOR = LANZADOR.replace('json.dump({"tareas": tareas}, open("/kaggle/working/haz.json", "w"))',
                            'json.dump({"tareas": tareas}, open("/kaggle/working/haz2.json", "w"))')
# resumen de COSTE, que es lo que decide
LANZADOR = LANZADOR.replace('''print("LECTURA: no-argmax = 0 => el GREEDY ya produce la GT y basta ancho 1.", flush=True)''',
'''seg = {}
for t in tareas:
    for h in (t.get("hazes") or []):
        if h.get("s") is not None:
            seg.setdefault(h["ancho"], []).append((t.get("rangos", {}).get("n_tok", 0), h["s"]))
print("")
print("COSTE VECTORIZADO (lo que decide si cabe en el reloj):", flush=True)
for a, v in sorted(seg.items()):
    tot = sum(s for _, s in v); m = tot / len(v)
    ext = m * 240
    print("   ancho %2d: media %5.1f s/tarea | max %5.1f s | 240 tareas = %.2f h = %+.1f%% de reloj (techo +4,8%%)"
          % (a, m, max(s for _, s in v), ext / 3600.0, 100.0 * ext / (11.83 * 3600)), flush=True)
    vac = [s for n, s in v if n >= 400]
    if vac:
        print("            solo tareas largas (>=400 tok): media %.1f s | si solo se aplica al 33%% con pool vacio = %+.1f%% de reloj"
              % (sum(vac) / len(vac), 100.0 * (sum(vac) / len(vac) * 80) / (11.83 * 3600)), flush=True)
print("")
print("LECTURA: no-argmax = 0 => el GREEDY ya produce la GT y basta ancho 1.", flush=True)''')
ast.parse(LANZADOR)

MD = """# haz2: el haz VECTORIZADO — cabe en el reloj?

`haz` (21-sep) midio que el haz-8 sobre una vista recupera **2 tareas de 18** que produccion
falla, pero costaba **59 s/tarea = +33,5% de reloj**, fatal. Ese coste era **artefacto**: el bucle
hacia 96 llamadas a `.item()` por paso de token (hasta 89.000 sincronizaciones GPU->CPU en una
tarea de 930 tokens) contra un suelo teorico de ~19 s.

Aqui el haz esta vectorizado: **una ordenacion en GPU y dos `.tolist()` por paso**, y el prefill
va con lote 1 expandiendo la cache en vez de recomputar el prompt `ancho` veces.

Dos preguntas: (1) cuanto cuesta de verdad, y (2) se conservan las 2 tareas. Se prueban anchos
**8 y 16** (el 16 por si compra alguna de deficit 4-5). Ademas se **guardan los candidatos del
haz con su score**, para poder re-jugar offline si anadirlos rompe la seleccion, que hoy es sin
perdidas.

Mismas 18 tareas de `strat`, TTT de produccion exacto.
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
json.dump(nb2, open("notebooks/haz2.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # haz (21-sep)'
entry = '''    # haz2 (21-sep): el haz VECTORIZADO. haz midio +2/18 tareas recuperadas pero 59 s/tarea
    # (+33,5% de reloj), artefacto de 96 .item() por paso. Aqui una ordenacion en GPU y dos
    # .tolist() por paso, prefill con lote 1 + expansion de cache. Anchos 8 y 16. Guarda los
    # candidatos del haz con score para re-jugar la seleccion offline. Decide si cabe en el techo.
    "haz2": {"notebook": "notebooks/haz2.ipynb", "slug": "arc-agi2-haz2",
             "title": "arc agi2 haz2", "gpu": True,
             "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                        "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
             "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                          "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
'''
if '"haz2"' not in s:
    assert anchor in s
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado haz2")
print("OK notebooks/haz2.ipynb")
