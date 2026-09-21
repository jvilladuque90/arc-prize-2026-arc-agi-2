# -*- coding: utf-8 -*-
# Pre-chequeo GRATIS de la premisa de la puerta de objetos: las tareas de AGI-2, tienen
# estructura de OBJETOS que se conserva entre entrada y salida? Sin modelo, sin GPU.
import json, os, sys
from collections import Counter

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")
SUCIAS = {"0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"}


def comps(a, color_ciego=True, diag=False):
    """Componentes conexas de celdas != 0. color_ciego: une colores distintos adyacentes."""
    H, W = a.shape
    visto = np.zeros((H, W), bool)
    vec = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diag else [])
    out = []
    for i in range(H):
        for j in range(W):
            if a[i, j] == 0 or visto[i, j]:
                continue
            pila, cel = [(i, j)], []
            visto[i, j] = True
            while pila:
                x, y = pila.pop()
                cel.append((x, y))
                for dx, dy in vec:
                    p, q = x + dx, y + dy
                    if 0 <= p < H and 0 <= q < W and not visto[p, q] and a[p, q] != 0:
                        if color_ciego or a[p, q] == a[x, y]:
                            visto[p, q] = True
                            pila.append((p, q))
            out.append(cel)
    return out


def firma(a, cel):
    """Forma normalizada (mascara relativa a su caja) + multiconjunto de colores."""
    xs = [c[0] for c in cel]; ys = [c[1] for c in cel]
    x0, y0 = min(xs), min(ys)
    m = frozenset((x - x0, y - y0) for x, y in cel)
    col = frozenset(Counter(int(a[x, y]) for x, y in cel).items())
    return m, col


def analiza(ch, sols, claves, nombre):
    n = len(claves)
    cnt = Counter()
    for k in claves:
        pares = [(np.array(p["input"]), np.array(p["output"])) for p in ch[k]["train"]]
        # --- cuantos objetos hay, y se conserva el numero?
        ci = [len(comps(a)) for a, _ in pares]
        co = [len(comps(b)) for _, b in pares]
        if min(ci) >= 2:
            cnt["tiene_objetos(>=2 en todas)"] += 1 if min(ci) >= 2 else 0
        if all(x == y for x, y in zip(ci, co)):
            cnt["n_objetos_conservado"] += 1
        if all(y <= x for x, y in zip(ci, co)) and any(y < x for x, y in zip(ci, co)):
            cnt["objetos_se_filtran(salen menos)"] += 1
        # --- se conservan las FORMAS (mismo multiconjunto de mascaras) ?
        ok_forma = ok_forma_color = True
        for a, b in pares:
            fa = Counter(firma(a, c)[0] for c in comps(a))
            fb = Counter(firma(b, c)[0] for c in comps(b))
            if fa != fb:
                ok_forma = False
            ga = Counter(firma(a, c) for c in comps(a))
            gb = Counter(firma(b, c) for c in comps(b))
            if ga != gb:
                ok_forma_color = False
        if ok_forma:
            cnt["formas_conservadas(solo cambia color/pos)"] += 1
        if ok_forma and not ok_forma_color:
            cnt["formas_iguales_pero_COLOR_cambia"] += 1
        # --- la salida es UNO de los objetos de la entrada (seleccion)?
        sel = True
        for a, b in pares:
            masks = {firma(a, c)[0] for c in comps(a)}
            fb = comps(b)
            if len(fb) != 1 or firma(b, fb[0])[0] not in masks:
                sel = False
                break
        if sel:
            cnt["salida = UN objeto de la entrada"] += 1
    print("%-22s n=%4d" % (nombre, n))
    for k2 in ("n_objetos_conservado", "objetos_se_filtran(salen menos)",
               "formas_conservadas(solo cambia color/pos)", "formas_iguales_pero_COLOR_cambia",
               "salida = UN objeto de la entrada"):
        print("   %-44s %4d  (%4.1f%%)" % (k2, cnt[k2], 100.0 * cnt[k2] / n))
    return cnt


def carga(n):
    with open(os.path.join(DATA, n), encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    ev, evs = carga("arc-agi_evaluation_challenges.json"), carga("arc-agi_evaluation_solutions.json")
    tr, trs = carga("arc-agi_training_challenges.json"), carga("arc-agi_training_solutions.json")
    agi1 = set(json.load(open(os.path.join(DATA, "agi1_keys.json"), encoding="utf-8")))
    print("=" * 74)
    print("PREMISA DE LA PUERTA DE OBJETOS: hay estructura de objetos conservada?")
    print("=" * 74)
    analiza(tr, trs, sorted(k for k in tr if k in agi1), "train AGI-1")
    analiza(tr, trs, sorted(k for k in tr if k not in agi1), "train AGI-2")
    analiza(ev, evs, sorted(k for k in ev if k not in SUCIAS), "eval AGI-2")

    # cuantos objetos hay por rejilla (el DSL global trataba la rejilla como UN bloque)
    print("")
    print("numero de objetos por entrada (mediana sobre las demos de cada tarea):")
    for nom, ch, ks in (("AGI-1", tr, [k for k in tr if k in agi1]),
                        ("AGI-2 train", tr, [k for k in tr if k not in agi1]),
                        ("AGI-2 eval", ev, [k for k in ev if k not in SUCIAS])):
        med = []
        for k in ks:
            c = [len(comps(np.array(p["input"]))) for p in ch[k]["train"]]
            c.sort()
            med.append(c[len(c) // 2])
        med.sort()
        print("   %-12s mediana %3d | p25 %3d | p75 %3d | >=3 objetos: %4.1f%%"
              % (nom, med[len(med) // 2], med[len(med) // 4], med[3 * len(med) // 4],
                 100.0 * sum(1 for x in med if x >= 3) / len(med)))
