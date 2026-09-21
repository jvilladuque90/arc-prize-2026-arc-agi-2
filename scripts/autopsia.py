# -*- coding: utf-8 -*-
"""Autopsia del pipeline: DONDE se pierde el score, etapa por etapa.

Usa los pools completos con scores que guardo `strat` (produccion exacta, 20 tareas de eval
estratificadas p5-p95) y `augdose`. Sin GPU, sin modelo.

Etapas: decode produce candidatos -> el pool contiene la GT -> la seleccion la elige.
"""
import json, os, sys
from collections import Counter

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")


def carga(n):
    with open(os.path.join(DATA, n), encoding="utf-8") as fh:
        return json.load(fh)


def celdas_ok(g, gt):
    """Fraccion de celdas correctas si la forma coincide; None si no."""
    a, b = np.array(g), np.array(gt)
    if a.shape != b.shape:
        return None
    return float((a == b).mean())


def analiza(tareas, sol, nombre):
    print("=" * 96)
    print("AUTOPSIA — %s" % nombre)
    print("=" * 96)
    print("%-10s %6s %5s %6s %6s | %7s %7s | %8s %8s | %s"
          % ("tarea", "coste", "vis+", "cands", "unic", "forma%", "mejor%", "celdas!=", "GT/16", "diagnostico"))
    tot = Counter()
    filas = []
    for t in tareas:
        k = t["task"]
        gt = sol[k][0]
        vistas = t["vistas"]
        n_vis_con = sum(1 for v in vistas if v)
        cands = [c for v in vistas for c in v]
        unic = {}
        for c in cands:
            unic.setdefault(json.dumps(c["grid"]), c["score"])
        # forma correcta
        n_forma = sum(1 for c in cands if len(c["grid"]) == len(gt) and len(c["grid"][0]) == len(gt[0]))
        # mejor candidato por celdas correctas (solo los de forma correcta)
        accs = [celdas_ok(c["grid"], gt) for c in cands]
        accs = [x for x in accs if x is not None]
        mejor = max(accs) if accs else None
        dif = None
        if mejor is not None:
            dif = int(round((1 - mejor) * len(gt) * len(gt[0])))
        tiene_gt = t["vistas_con_gt"] > 0
        if not cands:
            diag = "SIN CANDIDATOS"
        elif tiene_gt:
            diag = "GT EN POOL"
        elif n_forma == 0:
            diag = "forma incorrecta siempre"
        elif mejor is not None and mejor >= 0.95:
            diag = "casi (>=95% celdas)"
        elif mejor is not None and mejor >= 0.80:
            diag = "cerca (80-95%)"
        else:
            diag = "contenido lejos"
        tot[diag] += 1
        print("%-10s %6d %5d %6d %6d | %6.0f%% %s | %8s %8d | %s"
              % (k, t.get("coste", 0), n_vis_con, len(cands), len(unic),
                 100.0 * n_forma / max(len(cands), 1),
                 ("%6.0f%%" % (100 * mejor)) if mejor is not None else "     -",
                 str(dif) if dif is not None else "-", t["vistas_con_gt"], diag))
        filas.append({"task": k, "coste": t.get("coste", 0), "n_cands": len(cands),
                      "n_unic": len(unic), "frac_forma": n_forma / max(len(cands), 1),
                      "mejor_acc": mejor, "celdas_dif": dif, "gt": t["vistas_con_gt"],
                      "diag": diag})
    n = len(tareas)
    print("")
    print("REPARTO DE LA PERDIDA (n=%d):" % n)
    for d, c in tot.most_common():
        print("   %-26s %3d  (%4.1f%%)" % (d, c, 100.0 * c / n))
    # redundancia: candidatos unicos frente a totales
    ru = [f["n_unic"] / max(f["n_cands"], 1) for f in filas if f["n_cands"]]
    fa = [f["frac_forma"] for f in filas if f["n_cands"]]
    ma = [f["mejor_acc"] for f in filas if f["mejor_acc"] is not None]
    df = sorted(f["celdas_dif"] for f in filas if f["celdas_dif"] is not None and f["gt"] == 0)
    def med(x):
        s = sorted(x); m = len(s)
        return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0
    print("")
    print("   candidatos por tarea: mediana %.0f | unicos/total mediana %.2f (redundancia x%.1f)"
          % (med([f["n_cands"] for f in filas]), med(ru), 1 / max(med(ru), 1e-9)))
    print("   candidatos con FORMA correcta: mediana %.0f%%" % (100 * med(fa)))
    if ma:
        print("   mejor candidato (celdas correctas): mediana %.1f%% | max %.1f%%" % (100 * med(ma), 100 * max(ma)))
    if df:
        print("   en las FALLADAS, celdas equivocadas del mejor candidato: mediana %d | min %d | max %d"
              % (med(df), min(df), max(df)))
    return filas


if __name__ == "__main__":
    ev, evs = carga("arc-agi_evaluation_challenges.json"), carga("arc-agi_evaluation_solutions.json")
    T = json.load(open(os.path.join(RAIZ, "scripts", "_strat_analizado.json"), encoding="utf-8"))
    T = [t for t in T if t.get("vistas_hechas", 16) > 0 and t["task"] != "36a08778"]
    T.sort(key=lambda t: t["coste"])
    filas = analiza(T, evs, "strat (produccion exacta, eval estratificado p5-p95)")
    json.dump(filas, open(os.path.join(RAIZ, "scripts", "_autopsia.json"), "w"))
