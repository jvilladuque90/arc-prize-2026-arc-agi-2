# -*- coding: utf-8 -*-
# Puerta del Plan B (0 h-cuota, CPU, sin modelo): puede un DSL expresar ARC-AGI-2?
# Enumeracion exhaustiva de profundidad <=2 + mapa de color ajustado a las demos.
# Prerregistro en docs/PREREG_dsl.md.
import json, os, sys, time
from collections import Counter

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")

# ------------------------------------------------------------------ operadores del DSL
def _bbox(a):
    nz = np.argwhere(a != 0)
    if nz.size == 0:
        return None
    (i0, j0), (i1, j1) = nz.min(0), nz.max(0)
    return a[i0:i1 + 1, j0:j1 + 1]


def _comps(a):
    """componentes conexas 4-conectividad sobre celdas != 0, sin mirar color."""
    H, W = a.shape
    visto = np.zeros((H, W), bool)
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
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p, q = x + dx, y + dy
                    if 0 <= p < H and 0 <= q < W and not visto[p, q] and a[p, q] != 0:
                        visto[p, q] = True
                        pila.append((p, q))
            out.append(cel)
    return out


def _recorta_comp(a, cual):
    cs = _comps(a)
    if len(cs) < 2:
        return None
    cs.sort(key=len)
    cel = cs[-1] if cual == "max" else cs[0]
    xs = [c[0] for c in cel]; ys = [c[1] for c in cel]
    sub = np.zeros((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1), a.dtype)
    for x, y in cel:
        sub[x - min(xs), y - min(ys)] = a[x, y]
    return sub


def _comp_color_unico(a):
    """el componente cuyo color no se repite en ningun otro (patron ARC clasico)."""
    cs = _comps(a)
    if len(cs) < 2:
        return None
    cols = [frozenset(int(a[x, y]) for x, y in c) for c in cs]
    cnt = Counter()
    for s in cols:
        for c in s:
            cnt[c] += 1
    for cel, s in zip(cs, cols):
        if any(cnt[c] == 1 for c in s):
            xs = [c[0] for c in cel]; ys = [c[1] for c in cel]
            sub = np.zeros((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1), a.dtype)
            for x, y in cel:
                sub[x - min(xs), y - min(ys)] = a[x, y]
            return sub
    return None


def _quita_marco(a):
    if a.shape[0] < 3 or a.shape[1] < 3:
        return None
    borde = set(a[0]) | set(a[-1]) | set(a[:, 0]) | set(a[:, -1])
    return a[1:-1, 1:-1] if len(borde) == 1 else None


def _comprime(a):
    """quita filas y columnas enteramente de fondo."""
    f = a[~np.all(a == 0, axis=1)]
    if f.size == 0:
        return None
    f = f[:, ~np.all(f == 0, axis=0)]
    return f if f.shape != a.shape and f.size else None


def _bloque_reduce(a):
    """si la rejilla son bloques kxk constantes, reduce por el mayor k."""
    H, W = a.shape
    for k in range(min(H, W), 1, -1):
        if H % k or W % k:
            continue
        r = a.reshape(H // k, k, W // k, k)
        if np.all(r == r[:, :1, :, :1]):
            return r[:, 0, :, 0].copy()
    return None


def _destesela(a):
    H, W = a.shape
    for h in range(1, H + 1):
        if H % h:
            continue
        for w in range(1, W + 1):
            if W % w or (h == H and w == W):
                continue
            if np.array_equal(a, np.tile(a[:h, :w], (H // h, W // w))):
                return a[:h, :w].copy()
    return None


def _color_mayoritario(a):
    v = a[a != 0]
    if v.size == 0:
        return None
    return np.array([[Counter(v.tolist()).most_common(1)[0][0]]], a.dtype)


OPS = [
    ("id", lambda a: a),
    ("rot90", lambda a: np.rot90(a, 1)),
    ("rot180", lambda a: np.rot90(a, 2)),
    ("rot270", lambda a: np.rot90(a, 3)),
    ("fliph", lambda a: a[:, ::-1]),
    ("flipv", lambda a: a[::-1, :]),
    ("transp", lambda a: a.T),
    ("antitransp", lambda a: np.rot90(a, 2).T),
    ("bbox", _bbox),
    ("marco", _quita_marco),
    ("comprime", _comprime),
    ("bloque_red", _bloque_reduce),
    ("destesela", _destesela),
    ("comp_max", lambda a: _recorta_comp(a, "max")),
    ("comp_min", lambda a: _recorta_comp(a, "min")),
    ("comp_unico", _comp_color_unico),
    ("up2", lambda a: np.kron(a, np.ones((2, 2), a.dtype))),
    ("up3", lambda a: np.kron(a, np.ones((3, 3), a.dtype))),
    ("tile22", lambda a: np.tile(a, (2, 2))),
    ("tile33", lambda a: np.tile(a, (3, 3))),
    ("tile_h2", lambda a: np.tile(a, (1, 2))),
    ("tile_v2", lambda a: np.tile(a, (2, 1))),
    ("espejo_h", lambda a: np.concatenate([a, a[:, ::-1]], 1)),
    ("espejo_v", lambda a: np.concatenate([a, a[::-1, :]], 0)),
    ("color_may", _color_mayoritario),
]
NOMBRES = [n for n, _ in OPS]
MAXCEL = 3000          # corta explosiones de tile33 sobre rejillas grandes


def aplica(f, a):
    try:
        r = f(a)
    except Exception:
        return None
    if r is None or r.size == 0 or r.size > MAXCEL:
        return None
    return np.ascontiguousarray(r)


# ------------------------------------------------------------------ mapa de color
def fit_colormap(preds, reals):
    """un unico mapa color->color consistente en TODAS las demos, o None."""
    m = {}
    for p, r in zip(preds, reals):
        if p is None or p.shape != r.shape:
            return None
        for x, y in zip(p.ravel().tolist(), r.ravel().tolist()):
            if m.setdefault(x, y) != y:
                return None
    return m


def aplica_mapa(a, m):
    b = a.copy()
    for x, y in m.items():
        b[a == x] = y
    return b


# ------------------------------------------------------------------ busqueda
def resuelve(task, sol):
    """Devuelve (programa, acierta_test) o (None, None). Profundidad <=2 + mapa de color."""
    ins = [np.array(p["input"]) for p in task["train"]]
    outs = [np.array(p["output"]) for p in task["train"]]
    tin = np.array(task["test"][0]["input"])
    tout = np.array(sol)

    def check(res, res_t, nombre):
        if any(r is None for r in res):
            return None
        if all(np.array_equal(r, o) for r, o in zip(res, outs)):
            ok = res_t is not None and np.array_equal(res_t, tout)
            return (nombre, ok)
        m = fit_colormap(res, outs)
        if m is not None:
            pred = [aplica_mapa(r, m) for r in res]
            if all(np.array_equal(p, o) for p, o in zip(pred, outs)):
                ok = res_t is not None and np.array_equal(aplica_mapa(res_t, m), tout)
                return (nombre + "+color", ok)
        return None

    # nivel 1 (memoizado y reutilizado para el nivel 2)
    n1 = []
    for nom, f in OPS:
        res = [aplica(f, a) for a in ins]
        res_t = aplica(f, tin)
        n1.append((nom, res, res_t))
        r = check(res, res_t, nom)
        if r:
            return r
    # nivel 2
    for nom1, res1, rt1 in n1:
        if any(r is None for r in res1):
            continue
        for nom2, f2 in OPS:
            if nom2 == "id" or nom1 == "id":
                continue
            res = [aplica(f2, a) for a in res1]
            res_t = aplica(f2, rt1) if rt1 is not None else None
            r = check(res, res_t, nom1 + ">" + nom2)
            if r:
                return r
    return None, None


# ------------------------------------------------------------------ ejecucion
def carga(n):
    with open(os.path.join(DATA, n), encoding="utf-8") as fh:
        return json.load(fh)


def barrido(chal, sols, claves, etiqueta):
    t0 = time.time()
    res = {}
    for i, k in enumerate(claves):
        prog, ok = resuelve(chal[k], sols[k][0])
        res[k] = (prog, ok)
        if (i + 1) % 200 == 0:
            print("   ... %d/%d (%.0fs)" % (i + 1, len(claves), time.time() - t0), flush=True)
    cub = [k for k in claves if res[k][0]]
    aci = [k for k in cub if res[k][1]]
    print("%-22s n=%4d | cobertura %5.1f%% (%d) | precision %5.1f%% | score_dsl %5.1f%% | %.0fs"
          % (etiqueta, len(claves), 100.0 * len(cub) / len(claves), len(cub),
             (100.0 * len(aci) / len(cub)) if cub else 0.0,
             100.0 * len(aci) / len(claves), time.time() - t0), flush=True)
    return res, cub, aci


if __name__ == "__main__":
    solo_eval = "--eval" in sys.argv
    ev, evs = carga("arc-agi_evaluation_challenges.json"), carga("arc-agi_evaluation_solutions.json")
    print("=" * 88)
    print("PUERTA DEL PLAN B: cobertura de un DSL de juguete (profundidad <=2 + mapa de color)")
    print("=" * 88)
    r_ev, cub_ev, aci_ev = barrido(ev, evs, sorted(ev), "eval ARC-AGI-2")

    if not solo_eval:
        tr, trs = carga("arc-agi_training_challenges.json"), carga("arc-agi_training_solutions.json")
        agi1 = set(json.load(open(os.path.join(DATA, "agi1_keys.json"), encoding="utf-8")))
        k1 = sorted(k for k in tr if k in agi1)
        k2 = sorted(k for k in tr if k not in agi1)
        barrido(tr, trs, k1, "train subset AGI-1")
        barrido(tr, trs, k2, "train subset AGI-2")

    print("")
    print("operadores que aparecen en las soluciones de eval:")
    for nom, c in Counter(r_ev[k][0] for k in cub_ev).most_common(12):
        print("   %-28s %d" % (nom, c))

    # complementariedad: el DSL resuelve las tareas de d_test bajo (lo que ya hacemos)?
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fase0_ood import d_de_tarea, mediana
    d = {k: d_de_tarea(ev[k])[0] for k in ev}
    dentro = [d[k] for k in aci_ev]
    fuera = [d[k] for k in ev if k not in aci_ev]
    print("")
    print("COMPLEMENTARIEDAD (d_test de la Fase 0):")
    if dentro:
        print("   resueltas por el DSL : mediana %.3f (n=%d) | con d>1: %.0f%%"
              % (mediana(dentro), len(dentro), 100.0 * sum(1 for x in dentro if x > 1) / len(dentro)))
    print("   NO resueltas         : mediana %.3f (n=%d) | con d>1: %.0f%%"
          % (mediana(fuera), len(fuera), 100.0 * sum(1 for x in fuera if x > 1) / len(fuera)))
    json.dump({k: list(v) for k, v in r_ev.items()},
              open(os.path.join(RAIZ, "scripts", "_dsl_eval.json"), "w"))
    print("")
    print("PRERREG: score_dsl >=10%% => B plan principal | 3-10%% => decide complementariedad")
    print("         <3%% => B muere en la puerta | precision <60%% => la premisa de B se rompe")
