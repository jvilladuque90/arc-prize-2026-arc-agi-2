"""Genera notebooks/sft35_data.ipynb: kernel CPU (0 cuota GPU) que muestrea puzzles del dataset
NVARC (sorokin/nvarc-synthetic-puzzles, 266K ficheros) a UN solo JSON para que el kernel de SFT
(sft35) no tenga que montar 266K ficheros. Muestra estratificada por grupo (tarea semilla), seed fija.
Lanzar: PYTHONUTF8=1 python scripts/push_kernels.py sft35-data
"""
import json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

MD = """# sft35-data: muestra de puzzles NVARC para el SFT pareado Qwen3-4B-Base vs Qwen3.5-4B-Base

CPU, 0 cuota GPU. Recorre `nvarc_full/<grupo>/*.json`, toma hasta `POR_GRUPO` puzzles por grupo con
semilla fija y escribe `/kaggle/working/nvarc_muestra.json` (lista de puzzles; puzzle = lista de pares
`{input, output}`). `sft35` lo consume como `kernel_source`."""

CODE = r'''
import os, json, random, time
T0 = time.time()
POR_GRUPO = int(os.environ.get("POR_GRUPO", "40"))
random.seed(0)
raiz = None
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.basename(root) == "nvarc_full":
        raiz = root; break
assert raiz, "sin nvarc_full"
grupos = sorted(os.listdir(raiz))
print("grupos:", len(grupos), "en", raiz, flush=True)
puzzles, stats = [], {"grupos": len(grupos), "ficheros": 0, "pares": 0, "celdas": 0, "saltados": 0}
for gi, g in enumerate(grupos):
    d = os.path.join(raiz, g)
    try:
        fichs = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    except NotADirectoryError:
        continue
    stats["ficheros"] += len(fichs)
    random.shuffle(fichs)
    for f in fichs[:POR_GRUPO]:
        try:
            p = json.load(open(os.path.join(d, f)))
        except Exception:
            stats["saltados"] += 1; continue
        if not (isinstance(p, list) and len(p) >= 2 and all("input" in x and "output" in x for x in p)):
            stats["saltados"] += 1; continue
        ok = True
        for x in p:
            for k in ("input", "output"):
                gr = x[k]
                if not (1 <= len(gr) <= 30 and 1 <= len(gr[0]) <= 30 and all(len(r) == len(gr[0]) for r in gr)):
                    ok = False
        if not ok:
            stats["saltados"] += 1; continue
        puzzles.append({"grupo": g, "fichero": f, "pares": p})
        stats["pares"] += len(p)
        stats["celdas"] += sum(len(x[k]) * len(x[k][0]) for x in p for k in ("input", "output"))
    if gi % 50 == 0:
        print("  grupo %d/%d, puzzles %d, %.0fs" % (gi, len(grupos), len(puzzles), time.time() - T0), flush=True)
random.shuffle(puzzles)
json.dump(puzzles, open("/kaggle/working/nvarc_muestra.json", "w"))
stats["puzzles"] = len(puzzles); stats["wall_s"] = round(time.time() - T0)
stats["MB"] = round(os.path.getsize("/kaggle/working/nvarc_muestra.json") / 1e6, 1)
print("RESULT_JSON:" + json.dumps(stats), flush=True)
'''

def celda(src, tipo="code"):
    c = {"cell_type": tipo, "metadata": {}, "source": src.splitlines(keepends=True)}
    if tipo == "code":
        c["execution_count"] = None; c["outputs"] = []
    return c

nb = {"cells": [celda(MD, "markdown"), celda(CODE.lstrip("\n"))],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/sft35_data.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("OK notebooks/sft35_data.ipynb")
