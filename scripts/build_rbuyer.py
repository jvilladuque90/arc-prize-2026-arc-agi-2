import io, json, ast, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

SUCIAS = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]

BASE = io.open(
    r"C:\Users\Usuario\AppData\Local\Temp\claude\c--Users-Usuario-Documents-arc-prize-2026-arc-agi-2\bf2b527b-1480-4cb8-994e-9db70d95b620\scratchpad\augdose\worker.py",
    encoding="utf-8").read()

# ------------------------------------------------------------------ 1) cabecera y brazos
old = '''SHARD = int(os.environ.get("SHARD", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "4"))
N_TAREAS = int(os.environ.get("N_TAREAS", "16"))
N_VISTAS = int(os.environ.get("N_VISTAS", "16"))
BUDGET = float(os.environ.get("BUDGET_VISTA", "45"))
WALL_MAX = float(os.environ.get("WALL_MAX", "4800"))'''
new = '''SHARD = int(os.environ.get("SHARD", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "4"))
N_TAREAS = int(os.environ.get("N_TAREAS", "12"))
N_VISTAS = int(os.environ.get("N_VISTAS", "16"))
BUDGET = float(os.environ.get("BUDGET_VISTA", "45"))
WALL_MAX = float(os.environ.get("WALL_MAX", "2100"))
# DOS BRAZOS EN GPUs SEPARADAS: shards 0-1 = r256 (campeon), shards 2-3 = r512.
# Cada worker carga UN modelo con UN rango: cero fuga de estado entre brazos.
# Mismo orden de tareas en los dos brazos -> si el deadline trunca la cola, trunca
# igual en ambos y luego se emparejan por interseccion.
BRAZO = "r256" if SHARD < 2 else "r512"
R_LORA = 256 if BRAZO == "r256" else 512
ALPHA = 32 if BRAZO == "r256" else 45       # rsLoRA: alpha/sqrt(r) = 2,0 en los dos
SUB = SHARD % 2'''
assert old in BASE
S = BASE.replace(old, new, 1)

# ------------------------------------------------------------------ 2) montaje robusto
old = '''DATA = "/kaggle/input/arc-prize-2026-arc-agi-2"
CHAL = json.load(open(DATA + "/arc-agi_evaluation_challenges.json"))
SOLS = json.load(open(DATA + "/arc-agi_evaluation_solutions.json"))'''
new = '''FICH = "arc-agi_evaluation_challenges.json"
DATA = None
for cand in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
             "/kaggle/input/arc-prize-2026-arc-agi-2"):
    if os.path.exists(os.path.join(cand, FICH)):
        DATA = cand
        break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if FICH in files:
            DATA = root
            break
assert DATA, "sin dataset de la competicion"
CHAL = json.load(open(os.path.join(DATA, FICH)))
SOLS = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))'''
assert old in S
S = S.replace(old, new, 1)

# ------------------------------------------------------------------ 3) poblacion ESTRATIFICADA
old = '''# poblacion GLOBAL ordenada, y de ahi el shard: el indice global no cambia con N_SHARDS
KEYS_ALL = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))[:N_TAREAS]
MIS_KEYS = KEYS_ALL[SHARD::N_SHARDS]
print("[s%d] %d tareas de %d" % (SHARD, len(MIS_KEYS), len(KEYS_ALL)), flush=True)'''
new = '''# poblacion ESTRATIFICADA por coste a lo largo de TODA la distribucion: coger las N mas
# baratas fue el modo de fallo de lb33-cap (-2,5 pt). Indices equiespaciados en el ranking.
_ORD = sorted([k for k in CHAL if k not in SUCIAS], key=lambda k: coste(CHAL[k]))
_L = len(_ORD)
_IDX = sorted(set(int(round(i * (_L - 1) / max(N_TAREAS - 1, 1))) for i in range(N_TAREAS)))
KEYS_ALL = [_ORD[i] for i in _IDX]
MIS_KEYS = KEYS_ALL[SUB::2]          # 2 GPUs por brazo
print("[s%d] brazo %s (r=%d alpha=%d) | %d tareas de %d | costes %s"
      % (SHARD, BRAZO, R_LORA, ALPHA, len(MIS_KEYS), len(KEYS_ALL),
         [coste(CHAL[k]) for k in MIS_KEYS]), flush=True)'''
assert old in S
S = S.replace(old, new, 1)

# ------------------------------------------------------------------ 4) rango del brazo
old = '''LCFG = LoraConfig(r=256, lora_alpha=32, use_rslora=True, lora_dropout=0.0, bias="none",'''
new = '''LCFG = LoraConfig(r=R_LORA, lora_alpha=ALPHA, use_rslora=True, lora_dropout=0.0, bias="none",'''
assert old in S
S = S.replace(old, new, 1)

# ------------------------------------------------------------------ 5) NLL de la GT (secundaria)
old = '''def entrenar_ttt(task, vistas_train):'''
new = '''@torch.no_grad()
def nll_de_gt(task, gt):
    """NLL de la rejilla correcta bajo el modelo YA ADAPTADO, vista canonica (k=0, perm
    identidad). Medida continua: mas potencia que el 0-16 con n=12. Es MECANISMO, no
    veredicto -- no confundir con la loss del TTT, que esta saturada."""
    prompt = []
    for p in task["train"]:
        t, _ = pair_tokens_mask(p["input"], p["output"])
        prompt += t
    prompt += query_tokens(task["test"][0]["input"])
    gtoks = grid_tokens(gt) + [EOS]
    ids = torch.tensor([prompt + gtoks], device="cuda:0", dtype=torch.long)
    with sdpa_kernel(_SDPA):
        out = MODEL(input_ids=ids, use_cache=False, return_dict=True)
    lg = out.logits[0, :-1].float()
    tgt = ids[0, 1:]
    lp = lg.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    n_p = len(prompt)
    return float(-(lp[n_p - 1:]).sum().item()), len(gtoks)


def entrenar_ttt(task, vistas_train):'''
assert old in S
S = S.replace(old, new, 1)

# ------------------------------------------------------------------ 6) registro por tarea
old = '''RES = {"tareas": [], "modelo": "4b", "n_vistas": N_VISTAS, "budget_vista": BUDGET,
       "shard": SHARD}'''
new = '''RES = {"tareas": [], "modelo": "4b", "n_vistas": N_VISTAS, "budget_vista": BUDGET,
       "shard": SHARD, "brazo": BRAZO, "r": R_LORA, "alpha": ALPHA}'''
assert old in S
S = S.replace(old, new, 1)

old = '''    n_con_gt = sum(1 for cs in vistas_out if any(c["grid"] == gt for c in cs))
    RES["tareas"].append({"task": k, "gt": gt, "vistas": vistas_out, "n_seqs_ttt": n_seqs,
                          "ttt_s": round(t_ttt, 1), "decode_s": round(t_dec, 1),
                          "vistas_con_gt": n_con_gt})
    print("[s%d %d/%d] %s: GT en %d/%d vistas | cands/vista %.1f | ttt %.0fs dec %.0fs"
          % (SHARD, i + 1, len(MIS_KEYS), k, n_con_gt, N_VISTAS,
             sum(len(c) for c in vistas_out) / max(N_VISTAS, 1), t_ttt, t_dec), flush=True)'''
new = '''    n_con_gt = sum(1 for cs in vistas_out if any(c["grid"] == gt for c in cs))
    try:
        gt_nll, n_gt_tok = nll_de_gt(task, gt)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gt_nll, n_gt_tok = None, None
    except Exception as e:
        print("[s%d] nll_de_gt fallo en %s: %s" % (SHARD, k, str(e)[:120]), flush=True)
        gt_nll, n_gt_tok = None, None
    celdas = len(gt) * len(gt[0])
    RES["tareas"].append({"task": k, "brazo": BRAZO, "r": R_LORA, "vistas": vistas_out,
                          "n_seqs_ttt": n_seqs, "coste": coste(task), "celdas": celdas,
                          "ttt_s": round(t_ttt, 1), "decode_s": round(t_dec, 1),
                          "vistas_con_gt": n_con_gt,
                          "gt_nll": round(gt_nll, 3) if gt_nll is not None else None,
                          "gt_nll_celda": round(gt_nll / celdas, 4) if gt_nll is not None else None})
    print("[s%d %s %d/%d] %s: GT en %d/%d vistas | nll/celda %s | seqs %d | ttt %.0fs dec %.0fs"
          % (SHARD, BRAZO, i + 1, len(MIS_KEYS), k, n_con_gt, N_VISTAS,
             ("%.4f" % (gt_nll / celdas)) if gt_nll is not None else "-",
             n_seqs, t_ttt, t_dec), flush=True)'''
assert old in S
S = S.replace(old, new, 1)

# el campo "gt" (rejilla completa) inflaba el JSON; ya no se guarda
assert '"gt": gt' not in S
WORKER = S

# ------------------------------------------------------------------ lanzador
LANZADOR = r'''import json, os, subprocess, sys, time
T0 = time.time()
SUCIAS = __SUCIAS__

N_SHARDS = 4
procs = []
for i in range(N_SHARDS):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    env["SHARD"] = str(i)
    env["N_SHARDS"] = str(N_SHARDS)
    env["N_TAREAS"] = "12"
    env["N_VISTAS"] = "16"
    env["BUDGET_VISTA"] = "45"
    env["WALL_MAX"] = "2100"
    env["SUCIAS"] = json.dumps(SUCIAS)
    fh = open("/kaggle/working/log%d.txt" % i, "w")
    procs.append((i, subprocess.Popen([sys.executable, "/kaggle/working/_worker_src.py"],
                                      env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    print("lanzado worker %d en GPU %d (brazo %s)" % (i, i, "r256" if i < 2 else "r512"), flush=True)
for i, p, fh in procs:
    p.wait()
    fh.close()
    print("worker %d termino rc=%d a los %.0fs" % (i, p.returncode, time.time() - T0), flush=True)

tareas = []
for i in range(N_SHARDS):
    f = "/kaggle/working/shard%d.json" % i
    if os.path.exists(f):
        tareas += json.load(open(f)).get("tareas", [])
json.dump({"tareas": tareas}, open("/kaggle/working/rbuyer.json", "w"))

for i in range(N_SHARDS):
    print("=" * 14 + (" LOG WORKER %d " % i) + "=" * 14, flush=True)
    try:
        print(open("/kaggle/working/log%d.txt" % i).read()[-3000:], flush=True)
    except Exception as e:
        print("sin log: %s" % e, flush=True)

def mediana(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

def signo(mas, menos):
    n = mas + menos
    if n == 0:
        return 1.0
    from math import comb
    k = max(mas, menos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)

por = {}
for t in tareas:
    por.setdefault(t["task"], {})[t["brazo"]] = t
pares = [(k, v["r256"], v["r512"]) for k, v in por.items() if "r256" in v and "r512" in v]
pares.sort(key=lambda x: x[1]["coste"])

print("")
print("=" * 84, flush=True)
print("r-buyer: r512 frente a r256 SIN presion de reloj (primaria = GT en el pool)", flush=True)
print("=" * 84, flush=True)
print("%-10s %6s %5s | %7s %7s %6s | %9s %9s | %6s %6s" %
      ("tarea", "coste", "seqs", "gt r256", "gt r512", "delta", "nllc 256", "nllc 512", "s 256", "s 512"), flush=True)
d_gt, mas, menos, nll_mej, nll_peor = [], 0, 0, 0, 0
for k, a, b in pares:
    d = b["vistas_con_gt"] - a["vistas_con_gt"]
    d_gt.append(d)
    mas += (d > 0); menos += (d < 0)
    na, nb = a.get("gt_nll_celda"), b.get("gt_nll_celda")
    if na is not None and nb is not None:
        nll_mej += (nb < na); nll_peor += (nb > na)
    print("%-10s %6d %5d | %7d %7d %+6d | %9s %9s | %6.0f %6.0f" %
          (k, a["coste"], a["n_seqs_ttt"], a["vistas_con_gt"], b["vistas_con_gt"], d,
           ("%.4f" % na) if na is not None else "-", ("%.4f" % nb) if nb is not None else "-",
           a["ttt_s"] + a["decode_s"], b["ttt_s"] + b["decode_s"]), flush=True)

if pares:
    ta = sum(a["vistas_con_gt"] for _, a, _ in pares)
    tb = sum(b["vistas_con_gt"] for _, _, b in pares)
    ra = sum(1 for _, a, _ in pares if a["vistas_con_gt"] > 0)
    rb = sum(1 for _, _, b in pares if b["vistas_con_gt"] > 0)
    sa = sum(a["ttt_s"] + a["decode_s"] for _, a, _ in pares)
    sb = sum(b["ttt_s"] + b["decode_s"] for _, _, b in pares)
    print("")
    print("PRIMARIA  delta vistas_con_gt: mediana %+.1f | media %+.2f | mejora %d, empata %d, empeora %d"
          % (mediana(d_gt), sum(d_gt) / len(d_gt), mas, len(d_gt) - mas - menos, menos), flush=True)
    print("          test de signo (2 colas) p = %.4f" % signo(mas, menos), flush=True)
    print("          vistas con GT totales: r256 %d -> r512 %d | tareas con GT en algun pool: %d -> %d de %d"
          % (ta, tb, ra, rb, len(pares)), flush=True)
    print("SECUNDARIA nll/celda de la GT: r512 mejor en %d, peor en %d (mecanismo, NO veredicto)"
          % (nll_mej, nll_peor), flush=True)
    print("INSTRUMENTAL tiempo total: r256 %.0fs -> r512 %.0fs = x%.2f  (banco CON gradient"
          % (sa, sb, sb / max(sa, 1e-9)), flush=True)
    print("             checkpointing; produccion NO lo usa, su ratio real fue +11,9%)", flush=True)
    print("")
    print("PRERREG: r512 mejor              => el acantilado fue el RELOJ => HAY comprador => rama 2B viva", flush=True)
    print("         r512 peor con p<0,05    => el rango esta SATURADO    => NO hay comprador => rama 2B muerta", flush=True)
    print("         empate                  => el rango NO PAGA          => NO hay comprador => rama 2B bloqueada", flush=True)
    print("         (el empate se registro como NEGATIVO antes de ver el dato)", flush=True)
else:
    print("SIN PARES: revisar logs", flush=True)
print("RESULT_JSON:" + json.dumps({"pares": len(pares), "tareas": len(tareas),
                                   "wall_s": round(time.time() - T0)}), flush=True)
'''.replace("__SUCIAS__", json.dumps(SUCIAS))

MD = """# `r-buyer`: el acantilado de r512, fue el RELOJ o la CAPACIDAD?

Prerregistro completo en `docs/PREREG_rbuyer.md`.

`gate-base` dejo viva una rama con upside de dos digitos (Qwen3.5-2B empata el prior del titular
con 0,378x de tronco => ~50% de reloj libre => ~30 tareas => +12,4 pt). Pero eso exige que el
**reloj liberado tenga comprador**, y las dos formas medidas de gastarlo dan +0 (mas vistas, mas
hipotesis). Queda el **rango**, con un confundido sin resolver: `r512` perdio 15,3 tareas en el LB
**y** anadio +11,9% de coste sobre un pipeline al 95% de su deadline.

Aqui se quita la presion de reloj: ningun deadline global salta tareas, cada tarea recibe su TTT
completo (128 secuencias, calendario de produccion) y sus 16 vistas en **ambos** brazos.

- shards 0-1 => `r=256, alpha=32` (campeon) | shards 2-3 => `r=512, alpha=45` (rsLoRA escala 2,0)
- 12 tareas de eval **estratificadas por coste en toda la distribucion** (no las mas baratas)
- primaria `vistas_con_gt` (0-16), secundaria NLL/celda de la GT ya adaptado, instrumental tiempos

**PRERREG**: r512 mejor => hay comprador, rama 2B viva. r512 peor con p<0,05 => rango saturado,
rama muerta. **Empate => el rango no paga => tampoco hay comprador** (registrado como negativo
antes de ver el dato).
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
json.dump(nb, open("notebooks/rbuyer.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

assert 'BRAZO = "r256" if SHARD < 2 else "r512"' in WORKER
assert "r=R_LORA, lora_alpha=ALPHA" in WORKER and "use_rslora=True" in WORKER
assert "len(v_train) == 128" in WORKER
assert "_IDX = sorted(set(" in WORKER and "MIS_KEYS = KEYS_ALL[SUB::2]" in WORKER
assert "nll_de_gt" in WORKER and "/kaggle/input/competitions/" in WORKER

p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # GATE de modelo base a 4 bandas'
entry = '''    # r-buyer (9-sep), L4x4: r512 vs r256 SIN presion de reloj, pareado sobre 12 tareas de
    # eval estratificadas por coste. Separa el confundido del acantilado de r512 (-15,3 tareas
    # en LB con +11,9% de coste): reloj o capacidad? Bloquea la rama Qwen3.5-2B (+12,4 pt), que
    # necesita que el reloj liberado tenga comprador. Primaria vistas_con_gt; el EMPATE cuenta
    # como negativo (registrado antes de correr). 2 GPUs por brazo, sin fuga de estado.
    "rbuyer": {"notebook": "notebooks/rbuyer.ipynb", "slug": "arc-agi2-rbuyer",
               "title": "arc agi2 rbuyer", "gpu": True,
               "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                          "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
               "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                            "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
'''
if '"rbuyer"' not in s:
    assert anchor in s, "ancla no encontrada"
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado rbuyer")
else:
    print("rbuyer ya estaba")
print("OK notebooks/rbuyer.ipynb (worker %d B, lanzador %d B)" % (len(WORKER), len(LANZADOR)))
