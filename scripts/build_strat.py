import io, json, ast, os, sys
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")
sys.path.insert(0, "scripts")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

# ---------------------------------------------------------------- plan LPT con datos locales
ev = json.load(open("data/arc-agi_evaluation_challenges.json", encoding="utf-8"))
def coste(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0]) for g in t["train"])
ks = sorted([k for k in ev if k not in SUCIAS], key=lambda k: coste(ev[k]))
lo, hi = int(0.05 * (len(ks) - 1)), int(0.95 * (len(ks) - 1))
N = 20
sel = [ks[int(round(lo + i * (hi - lo) / (N - 1)))] for i in range(N)]
sel = list(dict.fromkeys(sel))
est = {k: 161 + 0.348 * coste(ev[k]) for k in sel}
W = [[] for _ in range(4)]; carga = [0.0] * 4
for k in sorted(sel, key=lambda k: -est[k]):
    j = carga.index(min(carga)); W[j].append(k); carga[j] += est[k]
for j in range(4):                       # dentro de cada worker, barata -> cara (si trunca, trunca la cola)
    W[j].sort(key=lambda k: coste(ev[k]))
print("plan: %d tareas, coste %d-%d | estimado por worker %s s" % (len(sel), coste(ev[sel[0]]), coste(ev[sel[-1]]), [round(c) for c in carga]))

# ---------------------------------------------------------------- parchear el worker de rbuyer
nb = json.load(open("notebooks/rbuyer.ipynb", encoding="utf-8"))
w = "".join(nb["cells"][1]["source"])
w = w.split("\n", 1)[1]                                   # quita %%writefile

def rep(old, new, cnt=1):
    global w
    assert old in w, "no encontrado: " + old[:70]
    w = w.replace(old, new, cnt)

rep('''# DOS BRAZOS EN GPUs SEPARADAS: shards 0-1 = r256 (campeon), shards 2-3 = r512.
# Cada worker carga UN modelo con UN rango: cero fuga de estado entre brazos.
# Mismo orden de tareas en los dos brazos -> si el deadline trunca la cola, trunca
# igual en ambos y luego se emparejan por interseccion.
BRAZO = "r256" if SHARD < 2 else "r512"
R_LORA = 256 if BRAZO == "r256" else 512
ALPHA = 32 if BRAZO == "r256" else 45       # rsLoRA: alpha/sqrt(r) = 2,0 en los dos
SUB = SHARD % 2''',
'''# UN SOLO BRAZO = produccion exacta. Las tareas las asigna el lanzador (plan LPT por coste).
BRAZO = "r256"
R_LORA = 256
ALPHA = 32''')

rep('''_ORD = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))
_L = len(_ORD)
_IDX = sorted(set(int(round(i * (_L - 1) / max(N_TAREAS - 1, 1))) for i in range(N_TAREAS)))
KEYS_ALL = [_ORD[i] for i in _IDX]
MIS_KEYS = KEYS_ALL[SUB::2]          # 2 GPUs por brazo
print("[s%d] brazo %s (r=%d alpha=%d) | %d tareas de %d | costes %s"
      % (SHARD, BRAZO, R_LORA, ALPHA, len(MIS_KEYS), len(KEYS_ALL),
         [coste(CHAL[k]) for k in MIS_KEYS]), flush=True)''',
'''MIS_KEYS = json.loads(os.environ["MIS_KEYS"])
print("[s%d] produccion (r=%d alpha=%d) | %d tareas | costes %s"
      % (SHARD, R_LORA, ALPHA, len(MIS_KEYS), [coste(CHAL[k]) for k in MIS_KEYS]), flush=True)''')

# NLL de la GT ANTES del TTT (modelo base) y comprobacion de deadline DENTRO de la tarea
rep('''    v_train = build_vistas_train(pares, seed)
    assert len(v_train) == 128, "el TTT debe ver 128 secuencias como produccion"
    t0 = time.time()
    n_seqs = entrenar_ttt(task, v_train)''',
'''    try:
        _pre, _ = nll_de_gt(task, gt)
    except Exception:
        torch.cuda.empty_cache(); _pre = None
    v_train = build_vistas_train(pares, seed)
    assert len(v_train) == 128, "el TTT debe ver 128 secuencias como produccion"
    t0 = time.time()
    n_seqs = entrenar_ttt(task, v_train)''')

rep('''    v_dec = build_vistas(pares, seed + 1, N_VISTAS)
    vistas_out, t0 = [], time.time()
    for v in v_dec:''',
'''    v_dec = build_vistas(pares, seed + 1, N_VISTAS)
    vistas_out, t0 = [], time.time()
    truncado = False
    for v in v_dec:
        if time.time() - T00 > WALL_MAX:          # deadline DENTRO de la tarea (leccion rbuyer)
            truncado = True
            print("[s%d] WALL_MAX dentro de %s: %d/%d vistas" % (SHARD, k, len(vistas_out), N_VISTAS), flush=True)
            break''')

rep('''                          "gt_nll": round(gt_nll, 3) if gt_nll is not None else None,
                          "gt_nll_celda": round(gt_nll / celdas, 4) if gt_nll is not None else None})''',
'''                          "gt_nll": round(gt_nll, 3) if gt_nll is not None else None,
                          "gt_nll_celda": round(gt_nll / celdas, 4) if gt_nll is not None else None,
                          "gt_nll_celda_pre": round(_pre / celdas, 4) if _pre is not None else None,
                          "truncado": truncado, "vistas_hechas": len(vistas_out)})''')

# el print por tarea: anadir pre
rep('''    print("[s%d %s %d/%d] %s: GT en %d/%d vistas | nll/celda %s | seqs %d | ttt %.0fs dec %.0fs"
          % (SHARD, BRAZO, i + 1, len(MIS_KEYS), k, n_con_gt, N_VISTAS,
             ("%.4f" % (gt_nll / celdas)) if gt_nll is not None else "-",
             n_seqs, t_ttt, t_dec), flush=True)''',
'''    print("[s%d %d/%d] %s coste %d: GT en %d/%d vistas | nll/celda pre %s -> post %s | seqs %d | ttt %.0fs dec %.0fs%s"
          % (SHARD, i + 1, len(MIS_KEYS), k, coste(task), n_con_gt, len(vistas_out),
             ("%.4f" % (_pre / celdas)) if _pre is not None else "-",
             ("%.4f" % (gt_nll / celdas)) if gt_nll is not None else "-",
             n_seqs, t_ttt, t_dec, " TRUNCADO" if truncado else ""), flush=True)''')

WORKER = w
ast.parse(WORKER)

LANZADOR = r'''import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = __SUCIAS__
PLAN = __PLAN__

procs = []
for i in range(4):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "SHARD": str(i), "N_SHARDS": "4",
                "N_VISTAS": "16", "BUDGET_VISTA": "45", "WALL_MAX": "4200",
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
json.dump({"tareas": tareas}, open("/kaggle/working/strat.json", "w"))
for i in range(4):
    print("=" * 12 + (" LOG WORKER %d " % i) + "=" * 12, flush=True)
    try:
        print(open("/kaggle/working/log%d.txt" % i).read()[-2500:], flush=True)
    except Exception as e:
        print("sin log:", e, flush=True)

tareas.sort(key=lambda t: t["coste"])
print("")
print("=" * 90, flush=True)
print("strat: produccion exacta sobre 20 tareas estratificadas p5-p95 (mapa del terreno)", flush=True)
print("=" * 90, flush=True)
print("%-10s %6s | %6s %7s %8s %8s | %5s %5s %s" % ("tarea", "coste", "gt/16", "seqs", "nll pre", "nll post", "ttt", "dec", ""), flush=True)
for t in tareas:
    print("%-10s %6d | %6d %7d %8s %8s | %5.0f %5.0f %s" % (
        t["task"], t["coste"], t["vistas_con_gt"], t["n_seqs_ttt"],
        ("%.4f" % t["gt_nll_celda_pre"]) if t.get("gt_nll_celda_pre") is not None else "-",
        ("%.4f" % t["gt_nll_celda"]) if t.get("gt_nll_celda") is not None else "-",
        t["ttt_s"], t["decode_s"], "TRUNCADO" if t.get("truncado") else ""), flush=True)
n = len(tareas)
if n:
    tercio = max(1, n // 3)
    for nom, sub in (("barato", tareas[:tercio]), ("medio", tareas[tercio:2 * tercio]), ("caro", tareas[2 * tercio:])):
        con = sum(1 for t in sub if t["vistas_con_gt"] > 0)
        print("P4 %-7s n=%d | GT en algun pool %d (%.0f%%) | coste %d-%d" % (nom, len(sub), con, 100 * con / max(len(sub), 1), sub[0]["coste"], sub[-1]["coste"]), flush=True)
    print("truncadas: %d de %d | wall %.0fs" % (sum(1 for t in tareas if t.get("truncado")), n, time.time() - T0), flush=True)
print("P1 (envolvente), P2 (proxy) y P3 (seleccion) se calculan offline sobre strat.json", flush=True)
print("RESULT_JSON:" + json.dumps({"tareas": n, "wall_s": round(time.time() - T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS)).replace("__PLAN__", json.dumps(W))
ast.parse(LANZADOR)

MD = """# strat: el mapa del terreno que no hemos mirado

Prerregistro en `docs/PREREG_strat.md`. Toda la evidencia por tarea de la campana venia del decil
mas barato del eval (percentil 11 de coste). Aqui: **produccion exacta, un solo brazo**, 20 tareas
estratificadas por coste **p5-p95** repartidas por LPT en 4 workers, deadline dimensionado sobre
la cola y comprobado dentro de la tarea. Registra GT-en-pool, pools con scores, NLL/celda de la
GT antes y despues del TTT.

No prueba ningun injerto. Produce hechos: P1 envolvente de salida en dificultad real, P2 validez
del proxy NLL, P3 'la seleccion es sin perdidas', P4 GT-en-pool por tramo de coste.
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
json.dump(nb2, open("notebooks/strat.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

assert "BRAZO = \"r256\"" in WORKER and "R_LORA = 256" in WORKER and "ALPHA = 32" in WORKER and "r512" not in WORKER
assert 'MIS_KEYS = json.loads(os.environ["MIS_KEYS"])' in WORKER
assert "gt_nll_celda_pre" in WORKER and "truncado = False" in WORKER
assert "len(v_train) == 128" in WORKER and '"BUDGET_VISTA": "45"' in LANZADOR and '"WALL_MAX": "4200"' in LANZADOR

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # r-buyer (9-sep)'
entry = '''    # strat (10-sep): produccion exacta, UN brazo, 20 tareas de eval estratificadas p5-p95 por
    # coste, LPT en 4 workers, deadline sobre la cola y dentro de la tarea. Cierra el punto ciego de
    # la auditoria: toda la evidencia por tarea venia del decil mas barato. Registra GT-en-pool,
    # pools con scores y NLL/celda pre/post TTT. No prueba injertos: produce hechos (P1-P4).
    "strat": {"notebook": "notebooks/strat.ipynb", "slug": "arc-agi2-strat",
              "title": "arc agi2 strat", "gpu": True,
              "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                         "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
              "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                           "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
'''
if '"strat"' not in s:
    assert anchor in s
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado strat")
print("OK notebooks/strat.ipynb | workers:", [len(x) for x in W])
