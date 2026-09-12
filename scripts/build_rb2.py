import io, json, ast, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]
P = "C:/Users/Usuario/AppData/Local/Temp/claude/c--Users-Usuario-Documents-arc-prize-2026-arc-agi-2/bf2b527b-1480-4cb8-994e-9db70d95b620/scratchpad/strat"

# ---------------------------------------------------------------- plan: mismas 19 tareas de strat, LPT por ttt_s medido
T = [t for t in json.load(open(P + "/strat.json", encoding="utf-8"))["tareas"] if t["task"] != "36a08778"]
est = {t["task"]: t["ttt_s"] for t in T}
W = [[], []]; L = [0.0, 0.0]
for k in sorted(est, key=lambda k: -est[k]):
    j = L.index(min(L)); W[j].append(k); L[j] += est[k]
for j in range(2):
    W[j].sort(key=lambda k: est[k])              # barata -> cara: si trunca, trunca la cola
PLAN = [W[0], W[1], W[0], W[1]]                  # 0-1 = r256, 2-3 = r512, MISMA particion
print("plan: %d tareas | TTT estimado por worker r256 %s s | r512 x1,15 %s s"
      % (len(est), [round(x) for x in L], [round(x * 1.15) for x in L]))

# ---------------------------------------------------------------- worker: el de strat con el brazo por entorno
nb = json.load(open("notebooks/strat.ipynb", encoding="utf-8"))
w = "".join(nb["cells"][1]["source"]).split("\n", 1)[1]

def rep(old, new):
    global w
    assert old in w, "no encontrado: " + old[:70]
    w = w.replace(old, new, 1)

rep('''# UN SOLO BRAZO = produccion exacta. Las tareas las asigna el lanzador (plan LPT por coste).
BRAZO = "r256"
R_LORA = 256
ALPHA = 32''',
'''# DOS BRAZOS en GPUs separadas, un modelo por worker (cero fuga de estado).
# El lanzador fija BRAZO/R_LORA/ALPHA por entorno: 0-1 = r256/32 (campeon), 2-3 = r512/45
# (rsLoRA alpha/sqrt(r) = 2,0 en ambos). SIN decode: la primaria es la NLL post-TTT (proxy
# validado en strat: Spearman -0,430 con GT-en-pool, separacion x11,6).
BRAZO = os.environ.get("BRAZO", "r256")
R_LORA = int(os.environ.get("R_LORA", "256"))
ALPHA = int(os.environ.get("ALPHA", "32"))''')
rep('''print("[s%d] produccion (r=%d alpha=%d) | %d tareas | costes %s"''',
    '''print("[s%d] brazo %s (r=%d alpha=%d) | %d tareas | costes %s"''')
rep('''      % (SHARD, R_LORA, ALPHA, len(MIS_KEYS), [coste(CHAL[k]) for k in MIS_KEYS]), flush=True)''',
    '''      % (SHARD, BRAZO, R_LORA, ALPHA, len(MIS_KEYS), [coste(CHAL[k]) for k in MIS_KEYS]), flush=True)''')
WORKER = w
ast.parse(WORKER)
assert 'R_LORA = int(os.environ.get("R_LORA", "256"))' in WORKER and "gt_nll_celda_pre" in WORKER
assert "len(v_train) == 128" in WORKER and 'N_VISTAS = int(os.environ.get("N_VISTAS", "16"))' in WORKER

LANZADOR = r'''import json, os, subprocess, sys, time
from math import comb
T0 = time.time()
SUCIAS = __SUCIAS__
PLAN = __PLAN__
BRAZOS = [("r256", 256, 32), ("r256", 256, 32), ("r512", 512, 45), ("r512", 512, 45)]

procs = []
for i in range(4):
    b, r, a = BRAZOS[i]
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(i), "SHARD": str(i), "N_SHARDS": "4",
                "BRAZO": b, "R_LORA": str(r), "ALPHA": str(a),
                "N_VISTAS": "0", "BUDGET_VISTA": "45", "WALL_MAX": "5400",
                "SUCIAS": json.dumps(SUCIAS), "MIS_KEYS": json.dumps(PLAN[i])})
    fh = open("/kaggle/working/log%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"],
                                      env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    print("worker %d: %s, %d tareas" % (i, b, len(PLAN[i])), flush=True)
for i, p, fh in procs:
    p.wait(); fh.close()
    print("worker %d rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)

tareas = []
for i in range(4):
    f = "/kaggle/working/shard%d.json" % i
    if os.path.exists(f):
        tareas += json.load(open(f)).get("tareas", [])
json.dump({"tareas": tareas}, open("/kaggle/working/rb2.json", "w"))
for i in range(4):
    print("=" * 12 + (" LOG WORKER %d " % i) + "=" * 12, flush=True)
    try:
        print(open("/kaggle/working/log%d.txt" % i).read()[-2000:], flush=True)
    except Exception as e:
        print("sin log:", e, flush=True)

def med(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
def signo(mas, menos):
    n = mas + menos
    if n == 0: return 1.0, 1.0
    k = max(mas, menos)
    dos = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)
    una = sum(comb(n, i) for i in range(mas, n + 1)) / 2.0 ** n     # a favor de r512 (mas = ratio<1)
    return dos, una

por = {}
for t in tareas:
    por.setdefault(t["task"], {})[t["brazo"]] = t
pares = [(k, v["r256"], v["r512"]) for k, v in por.items()
         if "r256" in v and "r512" in v and v["r256"].get("gt_nll_celda") is not None and v["r512"].get("gt_nll_celda") is not None]
pares.sort(key=lambda x: x[1]["coste"])
print("")
print("=" * 88, flush=True)
print("rb2: r512 vs r256 SIN presion de reloj, primaria = NLL/celda de la GT post-TTT (proxy)", flush=True)
print("=" * 88, flush=True)
print("%-10s %6s | %9s %9s %7s | %9s %9s | %6s %6s" % ("tarea", "coste", "post r256", "post r512", "ratio", "pre r256", "pre r512", "s 256", "s 512"), flush=True)
ratios, mas, menos = [], 0, 0
for k, a, b in pares:
    r = b["gt_nll_celda"] / max(a["gt_nll_celda"], 1e-6)
    ratios.append(r); mas += r < 1; menos += r > 1
    print("%-10s %6d | %9.4f %9.4f %7.3f | %9s %9s | %6.0f %6.0f" % (
        k, a["coste"], a["gt_nll_celda"], b["gt_nll_celda"], r,
        ("%.4f" % a["gt_nll_celda_pre"]) if a.get("gt_nll_celda_pre") is not None else "-",
        ("%.4f" % b["gt_nll_celda_pre"]) if b.get("gt_nll_celda_pre") is not None else "-",
        a["ttt_s"], b["ttt_s"]), flush=True)
if pares:
    dos, una = signo(mas, menos)
    m = med(ratios)
    print("", flush=True)
    print("PRIMARIA ratio r512/r256: MEDIANA %.3f | media %.3f | r512 mejor en %d, peor en %d, empate %d | signo 2 colas p=%.4f, a favor de r512 p=%.4f"
          % (m, sum(ratios) / len(ratios), mas, menos, len(ratios) - mas - menos, dos, una), flush=True)
    pre_ok = sum(1 for _, a, b in pares if a.get("gt_nll_celda_pre") is not None and b.get("gt_nll_celda_pre") is not None
                 and abs(a["gt_nll_celda_pre"] - b["gt_nll_celda_pre"]) < 1e-3)
    print("CONTROL pre-TTT identico en ambos brazos: %d/%d (debe ser todos: es el modelo base)" % (pre_ok, len(pares)), flush=True)
    print("tiempo TTT total: r256 %.0fs -> r512 %.0fs = x%.2f (banco con grad-ckpt)" % (
        sum(a["ttt_s"] for _, a, _ in pares), sum(b["ttt_s"] for _, _, b in pares),
        sum(b["ttt_s"] for _, _, b in pares) / max(sum(a["ttt_s"] for _, a, _ in pares), 1)), flush=True)
    if m <= 0.85 and una < 0.10:
        ver = "HAY COMPRADOR: el rango acerca a la GT => rama Qwen3.5-2B VIVA"
    elif m >= 1.15 and (1 - una) < 0.10:
        ver = "NO HAY COMPRADOR: mas rango ALEJA (sobreajuste) => rama 2B MUERTA"
    else:
        ver = "EMPATE: el rango no paga => NO hay comprador => rama 2B BLOQUEADA (registrado como negativo)"
    print("", flush=True)
    print("VEREDICTO PRERREG: " + ver, flush=True)
    print("calibracion: +20% de NLL ~ -2,5 pt de LB (ancla gate-base)", flush=True)
else:
    print("SIN PARES", flush=True)
print("RESULT_JSON:" + json.dumps({"pares": len(pares), "wall_s": round(time.time() - T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS)).replace("__PLAN__", json.dumps(PLAN))
ast.parse(LANZADOR)

MD = """# rb2: la pregunta del comprador, con el proxy validado

Prerregistro en `docs/PREREG_rb2.md`. `rbuyer` no pudo contestar si `r512` acerca mas a la GT que
`r256` porque GT-en-pool dispara en ~20% de tareas (5 de 7 pares salieron 0-0). `strat` valido que
la **NLL/celda de la GT post-TTT** predice GT-en-pool (Spearman -0,430, separacion x11,6) e informa
en el 100% de las tareas.

Mismas 19 tareas estratificadas p5-p95 de `strat` (excluida la contaminada `36a08778`). GPUs 0-1
r=256/alpha=32 (campeon), GPUs 2-3 r=512/alpha=45 (rsLoRA a 2,0 identico). Sin decode. Misma
semilla md5 por tarea => mismas 128 secuencias de TTT en los dos brazos.

**PRERREG (n=19 pareado)**: ratio mediano <=0,85 y signo p<0,10 => HAY comprador, rama 2B viva.
>=1,15 con p<0,10 => mas rango aleja, rama muerta. Lo demas => empate = el rango no paga = no hay
comprador (negativo, registrado antes de ver el dato).
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
json.dump(nb2, open("notebooks/rb2.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # strat (10-sep)'
entry = '''    # rb2 (12-sep): la pregunta del comprador con el proxy validado en strat. r512 vs r256 sin
    # presion de reloj sobre las mismas 19 tareas estratificadas (sin la contaminada), 2 GPUs por
    # brazo, SIN decode: primaria = NLL/celda de la GT post-TTT (informa en el 100% de tareas).
    # PRERREG: ratio mediano <=0,85 y signo p<0,10 => hay comprador (rama Qwen3.5-2B viva);
    # >=1,15 => aleja; lo demas empate = negativo.
    "rb2": {"notebook": "notebooks/rb2.ipynb", "slug": "arc-agi2-rb2",
            "title": "arc agi2 rb2", "gpu": True,
            "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                       "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
            "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                         "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
'''
if '"rb2"' not in s:
    assert anchor in s
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado rb2")
print("OK notebooks/rb2.ipynb | workers:", [len(x) for x in PLAN])
