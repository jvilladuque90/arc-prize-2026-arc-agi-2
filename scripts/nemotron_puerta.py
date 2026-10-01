# -*- coding: utf-8 -*-
"""Puerta GRATIS del Plan A con datos Nemotron (nvidia/Nemotron-SFT-ARC-AGI-v1).

Pregunta: los puzzles de ese dataset, sirven como datos de SFT para sft139?
  1. NOVEDAD: cuantos son tareas de ARC (train/eval) que sft139 ya memorizo o que contaminarian
     el eval. Coincidencia exacta de rejillas y bajo las 8 simetrias D4.
  2. ESCALA: se parecen en tamano a AGI-2 eval (donde perdemos) o al AGI-1 facil?
  3. RESPUESTA: cuantas filas traen una rejilla final parseable (\\boxed{...}) => par SFT util.
Sin GPU, sobre trozos descargados del dataset.
"""
import json, os, re, sys, glob
from collections import Counter

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")


def carga(n):
    with open(os.path.join(DATA, n), encoding="utf-8") as fh:
        return json.load(fh)


def rejilla(txt):
    filas = [l.split() for l in txt.strip().splitlines() if l.strip()]
    if not filas or any(len(f) != len(filas[0]) for f in filas):
        return None
    try:
        return [[int(x) for x in f] for f in filas]
    except ValueError:
        return None


def parse_prompt(u):
    """'Train Example k: Input: ... Output: ...' * n + 'Test Input: ...'"""
    pares = re.findall(r"Input:\n(.*?)\n\nOutput:\n(.*?)(?=\n\n)", u, re.S)
    t = re.search(r"Test Input:\n(.*?)(?:\n\n|$)", u, re.S)
    train = [(rejilla(a), rejilla(b)) for a, b in pares]
    if not train or any(a is None or b is None for a, b in train) or not t:
        return None
    ti = rejilla(t.group(1))
    return (train, ti) if ti is not None else None


def respuesta(c):
    m = re.findall(r"\\boxed\{([^{}]*)\}", c or "")
    if not m:
        return None
    s = m[-1].replace("\\ ", " ").replace("\\\\", "\n").replace("\\n", "\n")
    return rejilla(s)


def d4(g):
    a = np.array(g)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, r.T]
    return out


def clave(g):
    return (len(g), len(g[0]), tuple(x for f in g for x in f))


def indice_arc():
    """rejilla (bajo D4) -> origen, para todas las rejillas de ARC train+eval."""
    idx = {}
    tr = carga("arc-agi_training_challenges.json")
    ev = carga("arc-agi_evaluation_challenges.json")
    for nom, ch in (("train", tr), ("eval", ev)):
        for k, t in ch.items():
            for p in t["train"] + t["test"]:
                for g in (p["input"], p.get("output")):
                    if g is None:
                        continue
                    for v in d4(g):
                        idx.setdefault(clave(v.tolist()), set()).add((nom, k))
    return idx, tr, ev


def filas(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for l in fh:
            try:
                yield json.loads(l)
            except Exception:
                pass


def mediana(x):
    s = sorted(x)
    return s[len(s) // 2] if s else float("nan")


def perfil(nombre, areas):
    a = sorted(areas)
    n = len(a)
    if not n:
        return
    print("   %-26s n=%5d | area mediana %4d | p75 %4d | p90 %4d | >=400 celdas %4.1f%%"
          % (nombre, n, a[n // 2], a[3 * n // 4], a[9 * n // 10], 100.0 * sum(1 for x in a if x >= 400) / n))


if __name__ == "__main__":
    carpeta = sys.argv[1]
    idx, tr, ev = indice_arc()
    agi1 = set(json.load(open(os.path.join(DATA, "agi1_keys.json"), encoding="utf-8")))
    print("indice ARC: %d rejillas (x8 simetrias)" % len(idx))

    tot = Counter()
    vistos = set()
    areas_nem_out, areas_nem_in = [], []
    origen = Counter()
    for f in sorted(glob.glob(os.path.join(carpeta, "nem_*"))):
        n0 = tot["filas"]
        for r in filas(f):
            tot["filas"] += 1
            ms = r.get("messages") or []
            u = next((m["content"] for m in ms if m.get("role") == "user"), "")
            a = next((m["content"] for m in reversed(ms) if m.get("role") == "assistant"), "")
            pp = parse_prompt(u)
            if pp is None:
                tot["prompt_no_parseable"] += 1
                continue
            train, ti = pp
            firma = json.dumps([train, ti])
            if firma in vistos:
                tot["duplicada"] += 1
                continue
            vistos.add(firma)
            tot["puzzles_unicos"] += 1
            # novedad: alguna rejilla del puzzle esta en ARC?
            hits = set()
            for g in [x for p in train for x in p] + [ti]:
                hits |= idx.get(clave(g), set())
            if hits:
                tot["toca_ARC"] += 1
                srcs = {s for s, _ in hits}
                if "eval" in srcs:
                    tot["toca_ARC_eval"] += 1
                if any(s == "train" and k in agi1 for s, k in hits):
                    origen["AGI-1 train"] += 1
                elif any(s == "train" for s, _ in hits):
                    origen["AGI-2 train"] += 1
                else:
                    origen["eval"] += 1
            else:
                tot["nuevo"] += 1
            resp = respuesta(a)
            if resp is not None:
                tot["con_respuesta"] += 1
                if not hits:
                    tot["nuevo_y_con_respuesta"] += 1
                    areas_nem_out.append(len(resp) * len(resp[0]))
                    areas_nem_in.append(len(ti) * len(ti[0]))
        print("   %-28s %6d filas" % (os.path.basename(f), tot["filas"] - n0))

    print("")
    print("PUERTA NEMOTRON (0 h-cuota)")
    for k in ("filas", "prompt_no_parseable", "duplicada", "puzzles_unicos", "toca_ARC", "toca_ARC_eval",
              "nuevo", "con_respuesta", "nuevo_y_con_respuesta"):
        print("   %-24s %6d" % (k, tot[k]))
    print("   origen de los que tocan ARC:", dict(origen))
    print("")
    print("ESCALA (area de la rejilla de test; la salida es lo que el modelo debe emitir):")
    perfil("Nemotron nuevo: SALIDA", areas_nem_out)
    perfil("Nemotron nuevo: entrada", areas_nem_in)
    trs = carga("arc-agi_training_solutions.json")
    for nom, ks in (("AGI-1 train: SALIDA", [k for k in tr if k in agi1]),
                    ("AGI-2 train: SALIDA", [k for k in tr if k not in agi1])):
        perfil(nom, [len(trs[k][0]) * len(trs[k][0][0]) for k in ks])
    evs = carga("arc-agi_evaluation_solutions.json")
    perfil("AGI-2 eval: SALIDA", [len(evs[k][0]) * len(evs[k][0][0]) for k in ev])
