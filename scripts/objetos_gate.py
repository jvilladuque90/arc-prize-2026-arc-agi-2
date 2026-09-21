# -*- coding: utf-8 -*-
"""Puerta de OBJETOS (0 h-cuota, CPU, sin modelo).

La puerta del DSL (9-sep) trataba la rejilla como UN bloque: 5,1% en AGI-1, 0 de 353 en AGI-2.
Esta la trata como una ESCENA DE OBJETOS. Cuatro etapas: segmentar (varias politicas) ->
emparejar entrada/salida -> ajustar una regla POR OBJETO desde las demos -> verificar EXACTO
contra todas las demostraciones.

Motivacion medida (scripts/objetos_premisa.py): en eval AGI-2 el 58,8% conserva el numero de
objetos, el 30,7% conserva las formas cambiando solo el color y el 28,9% tiene como salida uno
de los objetos de la entrada. Y el campeon resuelve 0 de 9 de esas tareas frente a 8 de 29 del
resto (Fisher p~0,09).
"""
import json, os, sys
from collections import Counter

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")
SUCIAS = {"0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"}
BG = 0

# ===================================================================== segmentacion
def _cc(a, diag, mismo_color):
    H, W = a.shape
    visto = np.zeros((H, W), bool)
    vec = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        vec += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    for i in range(H):
        for j in range(W):
            if a[i, j] == BG or visto[i, j]:
                continue
            pila, cel = [(i, j)], []
            visto[i, j] = True
            while pila:
                x, y = pila.pop()
                cel.append((x, y))
                for dx, dy in vec:
                    p, q = x + dx, y + dy
                    if 0 <= p < H and 0 <= q < W and not visto[p, q] and a[p, q] != BG:
                        if (not mismo_color) or a[p, q] == a[x, y]:
                            visto[p, q] = True
                            pila.append((p, q))
            out.append(cel)
    return out


def _por_color(a):
    out = []
    for c in sorted(set(a.ravel().tolist())):
        if c == BG:
            continue
        cel = [(int(x), int(y)) for x, y in np.argwhere(a == c)]
        if cel:
            out.append(cel)
    return out


def _celdas(a):
    """Si hay filas/columnas uniformes que actuan de separador, trocea en bloques."""
    H, W = a.shape
    fr = [i for i in range(H) if len(set(a[i].tolist())) == 1]
    fc = [j for j in range(W) if len(set(a[:, j].tolist())) == 1]
    if not fr and not fc:
        return []
    def tramos(sep, n):
        t, ini = [], 0
        for s in sorted(sep) + [n]:
            if s > ini:
                t.append((ini, s))
            ini = s + 1
        return t
    TR, TC = tramos(fr, H), tramos(fc, W)
    if len(TR) * len(TC) < 2:
        return []
    out = []
    for r0, r1 in TR:
        for c0, c1 in TC:
            cel = [(i, j) for i in range(r0, r1) for j in range(c0, c1) if a[i, j] != BG]
            if cel:
                out.append(cel)
    return out


def _todo(a):
    cel = [(int(x), int(y)) for x, y in np.argwhere(a != BG)]
    return [cel] if cel else []


POLITICAS = [("cc4", lambda a: _cc(a, False, False)),
             ("cc8", lambda a: _cc(a, True, False)),
             ("cc4col", lambda a: _cc(a, False, True)),
             ("cc8col", lambda a: _cc(a, True, True)),
             ("color", _por_color),
             ("celdas", _celdas),
             ("todo", _todo)]

# ===================================================================== objeto y propiedades
class Obj(object):
    __slots__ = ("cel", "col", "r0", "c0", "h", "w", "mask", "area", "cols")

    def __init__(self, a, cel):
        self.cel = cel
        xs = [c[0] for c in cel]; ys = [c[1] for c in cel]
        self.r0, self.c0 = min(xs), min(ys)
        self.h = max(xs) - self.r0 + 1
        self.w = max(ys) - self.c0 + 1
        self.col = {(x, y): int(a[x, y]) for x, y in cel}
        self.mask = frozenset((x - self.r0, y - self.c0) for x, y in cel)
        self.area = len(cel)
        self.cols = frozenset(self.col.values())

    def render(self):
        g = np.zeros((self.h, self.w), int)
        for (x, y), c in self.col.items():
            g[x - self.r0, y - self.c0] = c
        return g


def props(o, todos):
    """Propiedades escalares/hashables del objeto en su escena."""
    areas = sorted({p.area for p in todos}, reverse=True)
    formas = Counter(p.mask for p in todos)
    colores = Counter(c for p in todos for c in p.cols)
    d = {"area": o.area,
         "h": o.h, "w": o.w, "hw": (o.h, o.w),
         "rank_area_desc": areas.index(o.area) + 1,
         "rank_area_asc": len(areas) - areas.index(o.area),
         "n_colores": len(o.cols),
         "color": min(o.cols) if len(o.cols) == 1 else -1,
         "forma": o.mask,
         "n_misma_forma": formas[o.mask],
         "forma_unica": int(formas[o.mask] == 1),
         "huecos": o.h * o.w - o.area,
         "es_rect": int(o.area == o.h * o.w),
         "es_cuadro": int(o.h == o.w),
         "simetrico_h": int(all((x, o.w - 1 - y) in o.mask for x, y in o.mask)),
         "simetrico_v": int(all((o.h - 1 - x, y) in o.mask for x, y in o.mask)),
         "color_unico_escena": int(any(colores[c] == 1 for c in o.cols))}
    return d


PROPS = ["area", "h", "w", "hw", "rank_area_desc", "rank_area_asc", "n_colores", "color",
         "forma", "n_misma_forma", "forma_unica", "huecos", "es_rect", "es_cuadro",
         "simetrico_h", "simetrico_v", "color_unico_escena"]

# ===================================================================== familias de regla
def _segmenta(a, pol):
    return [Obj(a, c) for c in pol(a)]


def fam_recolorear(demos, pol):
    """A: la salida es la entrada con cada objeto pintado de UN color que depende de una
    propiedad del objeto. Cubre la familia 'formas iguales, cambia el color' (30,7% del eval)."""
    datos = []
    for a, b in demos:
        if a.shape != b.shape:
            return None
        if not np.array_equal(a != BG, b != BG):        # mismas celdas ocupadas
            return None
        objs = _segmenta(a, pol)
        if len(objs) < 1:
            return None
        for o in objs:
            sal = {int(b[x, y]) for x, y in o.cel}
            if len(sal) != 1:
                return None
            datos.append((props(o, objs), sal.pop()))
    if not datos:
        return None
    for p in PROPS:
        m = {}
        ok = True
        for pr, c in datos:
            k = pr[p]
            if m.setdefault(k, c) != c:
                ok = False
                break
        if ok and len(set(m.values())) > 1:            # rechaza la regla trivial 'todo del mismo color'
            return ("recolorear", p, m)
    return None


CRITERIOS = [("max_area", lambda pr: pr["area"], True), ("min_area", lambda pr: pr["area"], False),
             ("max_huecos", lambda pr: pr["huecos"], True),
             ("max_ncol", lambda pr: pr["n_colores"], True),
             ("min_ncol", lambda pr: pr["n_colores"], False),
             ("max_hw", lambda pr: pr["h"] * pr["w"], True),
             ("mas_repetida", lambda pr: pr["n_misma_forma"], True),
             ("menos_repetida", lambda pr: pr["n_misma_forma"], False),
             ("forma_unica", lambda pr: pr["forma_unica"], True),
             ("color_unico", lambda pr: pr["color_unico_escena"], True),
             ("simetrica", lambda pr: pr["simetrico_h"] + pr["simetrico_v"], True),
             ("rectangular", lambda pr: pr["es_rect"], True)]


def fam_seleccion(demos, pol):
    """B: la salida ES uno de los objetos de la entrada (recortado). Cubre el 28,9%."""
    elegidos = []
    for a, b in demos:
        objs = _segmenta(a, pol)
        if len(objs) < 2:
            return None
        idx = [i for i, o in enumerate(objs) if np.array_equal(o.render(), b)]
        if not idx:
            return None
        elegidos.append((objs, idx))
    for nom, f, mayor in CRITERIOS:
        ok = True
        for objs, idx in elegidos:
            vs = [f(props(o, objs)) for o in objs]
            best = max(vs) if mayor else min(vs)
            cand = [i for i, v in enumerate(vs) if v == best]
            if len(cand) != 1 or cand[0] not in idx:
                ok = False
                break
        if ok:
            return ("seleccion", nom, None)
    return None


def fam_filtrar(demos, pol):
    """C: la salida es la entrada quitando los objetos que NO cumplen un predicado."""
    casos = []
    for a, b in demos:
        if a.shape != b.shape:
            return None
        objs = _segmenta(a, pol)
        if len(objs) < 2:
            return None
        keep = []
        for o in objs:
            iguales = all(int(b[x, y]) == c for (x, y), c in o.col.items())
            borrados = all(int(b[x, y]) == BG for x, y in o.cel)
            if iguales:
                keep.append(True)
            elif borrados:
                keep.append(False)
            else:
                return None
        # la salida no puede tener celdas nuevas
        ocup = np.zeros_like(b, bool)
        for o, k in zip(objs, keep):
            if k:
                for x, y in o.cel:
                    ocup[x, y] = True
        if not np.array_equal(b != BG, ocup):
            return None
        casos.append((objs, keep))
    # exige que el predicado discrimine EN ALGUN sitio (si nunca borra nada, no es una regla)
    if not any(any(k for k in keep) for _, keep in casos) or not any(not all(keep) for _, keep in casos):
        return None
    for p in PROPS:
        m = {}
        ok = True
        for objs, keep in casos:
            for o, k in zip(objs, keep):
                v = props(o, objs)[p]
                if m.setdefault(v, k) != k:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return ("filtrar", p, m)
    for nom, f, mayor in CRITERIOS:
        ok = True
        for objs, keep in casos:
            vs = [f(props(o, objs)) for o in objs]
            best = max(vs) if mayor else min(vs)
            if [v == best for v in vs] != keep:
                ok = False
                break
        if ok:
            return ("filtrar_crit", nom, None)
    return None


FAMILIAS = [("recolorear", fam_recolorear), ("seleccion", fam_seleccion), ("filtrar", fam_filtrar)]

# ===================================================================== aplicar al test
def aplica(prog, pol, a):
    fam, clave, mapa = prog
    objs = _segmenta(a, pol)
    if not objs:
        return None
    if fam == "recolorear":
        g = a.copy()
        for o in objs:
            k = props(o, objs)[clave]
            if k not in mapa:
                return None
            for x, y in o.cel:
                g[x, y] = mapa[k]
        return g
    if fam == "seleccion":
        f, mayor = next((f, m) for n, f, m in CRITERIOS if n == clave)
        vs = [f(props(o, objs)) for o in objs]
        best = max(vs) if mayor else min(vs)
        cand = [i for i, v in enumerate(vs) if v == best]
        return objs[cand[0]].render() if len(cand) == 1 else None
    if fam in ("filtrar", "filtrar_crit"):
        g = a.copy()
        if fam == "filtrar":
            for o in objs:
                k = props(o, objs)[clave]
                if k not in mapa:
                    return None
                if not mapa[k]:
                    for x, y in o.cel:
                        g[x, y] = BG
        else:
            f, mayor = next((f, m) for n, f, m in CRITERIOS if n == clave)
            vs = [f(props(o, objs)) for o in objs]
            best = max(vs) if mayor else min(vs)
            for o, v in zip(objs, vs):
                if v != best:
                    for x, y in o.cel:
                        g[x, y] = BG
        return g
    return None


def resuelve(task, sol):
    """Devuelve (etiqueta_programa, acierta_test) o (None, None)."""
    demos = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    tin = np.array(task["test"][0]["input"])
    tout = np.array(sol)
    for pnom, pol in POLITICAS:
        for fnom, fam in FAMILIAS:
            try:
                prog = fam(demos, pol)
            except Exception:
                prog = None
            if prog is None:
                continue
            # VERIFICACION EXACTA sobre TODAS las demos
            if not all(np.array_equal(aplica(prog, pol, a) if aplica(prog, pol, a) is not None
                                      else np.array([[-1]]), b) for a, b in demos):
                continue
            pred = aplica(prog, pol, tin)
            ok = pred is not None and pred.shape == tout.shape and np.array_equal(pred, tout)
            return ("%s/%s:%s" % (pnom, prog[0], prog[1] if not isinstance(prog[1], frozenset) else "forma"), ok)
    return None, None


def carga(n):
    with open(os.path.join(DATA, n), encoding="utf-8") as fh:
        return json.load(fh)


def barrido(ch, sols, claves, etiqueta):
    import time
    t0 = time.time()
    res = {}
    for i, k in enumerate(claves):
        try:
            res[k] = resuelve(ch[k], sols[k][0])
        except Exception as e:
            res[k] = (None, None)
        if (i + 1) % 200 == 0:
            print("   ... %d/%d (%.0fs)" % (i + 1, len(claves), time.time() - t0), flush=True)
    cub = [k for k in claves if res[k][0]]
    aci = [k for k in cub if res[k][1]]
    print("%-20s n=%4d | cobertura %5.1f%% (%3d) | precision %5.1f%% | score %5.1f%% | %.0fs"
          % (etiqueta, len(claves), 100.0 * len(cub) / len(claves), len(cub),
             (100.0 * len(aci) / len(cub)) if cub else 0.0,
             100.0 * len(aci) / len(claves), time.time() - t0), flush=True)
    return res, cub, aci
