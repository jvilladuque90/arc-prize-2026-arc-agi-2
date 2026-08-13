"""[LOCAL] Construye el corpus claudegen v1 (JSONL) + estadísticas + descontaminación.

~66 tareas por regla (~2200 total): 2000 de entrenamiento + holdout. Mismo esquema del
snapshot ARC-Heavy ({"pairs": [...]}) para reusar el builder del carril proxy tal cual.
Descontaminación: 0 pares exactos contra AGI-2 train completo (mismo check que ARC-Heavy).

Uso: python scripts/build_claudegen.py <OUT_DIR> [<KAGGLE_USERNAME>]
(env KAGGLE_API_TOKEN para subir como dataset privado arcscale-claudegen-v1)
"""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc.claudegen import RULES, build_task  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
KUSER = sys.argv[2] if len(sys.argv) > 2 else ""
os.makedirs(OUT, exist_ok=True)

PER_RULE = 66
rng = np.random.default_rng(20260813)

tasks, fails = [], Counter()
for name in RULES:
    got, tries = 0, 0
    while got < PER_RULE and tries < PER_RULE * 12:
        tries += 1
        t = build_task(rng, rule=name)
        if t is None:
            fails[name] += 1
            continue
        tasks.append(t)
        got += 1
    if got < PER_RULE:
        print(f"[warn] {name}: solo {got}/{PER_RULE} (fallos {fails[name]})")

order = rng.permutation(len(tasks))
tasks = [tasks[i] for i in order]

# ---- descontaminación exacta contra AGI-2 train (pares input->output) ----
ROOT = os.path.join(os.path.dirname(__file__), "..")
CHAL = json.load(open(os.path.join(ROOT, "data", "arc-agi_training_challenges.json")))
SOLS = json.load(open(os.path.join(ROOT, "data", "arc-agi_training_solutions.json")))


def h(g):
    return hashlib.md5(json.dumps(g).encode()).hexdigest()


real = set()
for k in CHAL:
    for p in CHAL[k]["train"]:
        real.add(h(p["input"]) + h(p["output"]))
    for i, p in enumerate(CHAL[k]["test"]):
        real.add(h(p["input"]) + h(SOLS[k][i]))

hits = 0
path = os.path.join(OUT, "claudegen_v1.jsonl")
with open(path, "w") as fh:
    for t in tasks:
        for gi, go in t["pairs"]:
            if h(gi) + h(go) in real:
                hits += 1
        fh.write(json.dumps({"rule": t["rule"], "pairs": t["pairs"]}) + "\n")

sizes = [max(len(g), len(g[0]), len(o), len(o[0])) for t in tasks for g, o in t["pairs"]]
shape_change = sum(1 for t in tasks for g, o in t["pairs"]
                   if (len(g), len(g[0])) != (len(o), len(o[0])))
n_pairs = sum(len(t["pairs"]) for t in tasks)
print(f"[corpus] {len(tasks)} tareas ({len(RULES)} reglas x ~{PER_RULE}) | {n_pairs} pares | "
      f"lado max medio {np.mean(sizes):.1f} | pares con cambio de forma {shape_change}/{n_pairs} "
      f"({100*shape_change/n_pairs:.0f}%)")
print(f"[descontaminacion] pares que coinciden con AGI-2 train: {hits} (debe ser 0)")
assert hits == 0
print(f"[out] {path} ({os.path.getsize(path)/1e6:.1f}MB)")

if KUSER:
    json.dump({"title": "arcscale-claudegen-v1", "id": f"{KUSER}/arcscale-claudegen-v1",
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", OUT], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "already exists" in out:
        r = subprocess.run(["kaggle", "datasets", "version", "-p", OUT, "-m", "v1"],
                           capture_output=True, text=True)
    print(f"[upload] rc={r.returncode} {(r.stdout or '')[-200:]}")
