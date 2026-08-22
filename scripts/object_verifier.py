"""Prototipo del VERIFICADOR DE OBJETOS (idea de grafos aplicada como juez, no como solver).

Cada invariante se APRENDE de las demostraciones de la tarea (solo si se cumple en TODAS) y
luego se aplica al candidato. Aqui se mide la mitad barata del argumento: la tasa de FALSOS
POSITIVOS, usando las verdades del eval como "candidatos correctos" — si un invariante
aprendido rechaza a la verdad, es peligroso y no puede filtrar pools en produccion.

La otra mitad (tasa de aciertos sobre candidatos INCORRECTOS) necesita los volcados de pools
del proximo kernel.

Uso: python scripts/object_verifier.py [data_dir]
"""
import json
import sys
from collections import Counter
from pathlib import Path

FONDO = 0


def colores(g):
    return {c for fila in g for c in fila}


def shape(g):
    return (len(g), len(g[0]))


def componentes(g):
    """Objetos = componentes conexas 4-vecinas de mismo color, ignorando el fondo."""
    h, w = shape(g)
    visto = [[False] * w for _ in range(h)]
    n = 0
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == FONDO:
                continue
            n += 1
            color = g[i][j]
            pila = [(i, j)]
            visto[i][j] = True
            while pila:
                a, b = pila.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not visto[x][y] and g[x][y] == color:
                        visto[x][y] = True
                        pila.append((x, y))
    return n


def sim_h(g):
    return g == g[::-1]


def sim_v(g):
    return all(fila == fila[::-1] for fila in g)


def fondo_dominante(g):
    return Counter(c for fila in g for c in fila).most_common(1)[0][0]


# Cada invariante: (nombre, aprende(demos)->estado|None, chequea(estado, test_in, cand)->bool)

def inv_size_igual(demos):
    return True if all(shape(i) == shape(o) for i, o in demos) else None


def chk_size_igual(_, tin, cand):
    return shape(cand) == shape(tin)


def inv_size_const(demos):
    s = {shape(o) for _, o in demos}
    return s.pop() if len(s) == 1 else None


def chk_size_const(s, _, cand):
    return shape(cand) == s


def inv_paleta(demos):
    return set().union(*(colores(o) for _, o in demos))


def chk_paleta(pal, tin, cand):
    return colores(cand) <= (pal | colores(tin))


def inv_paleta_estricta(demos):
    return True if all(colores(o) <= colores(i) for i, o in demos) else None


def chk_paleta_estricta(_, tin, cand):
    return colores(cand) <= colores(tin)


def inv_nobj_const(demos):
    n = {componentes(o) for _, o in demos}
    return n.pop() if len(n) == 1 else None


def chk_nobj_const(n, _, cand):
    return componentes(cand) == n


def inv_nobj_igual(demos):
    return True if all(componentes(i) == componentes(o) for i, o in demos) else None


def chk_nobj_igual(_, tin, cand):
    return componentes(cand) == componentes(tin)


def inv_sim_h(demos):
    return True if all(sim_h(o) for _, o in demos) else None


def chk_sim_h(_, __, cand):
    return sim_h(cand)


def inv_sim_v(demos):
    return True if all(sim_v(o) for _, o in demos) else None


def chk_sim_v(_, __, cand):
    return sim_v(cand)


def inv_fondo(demos):
    f = {fondo_dominante(o) for _, o in demos}
    return f.pop() if len(f) == 1 else None


def chk_fondo(f, _, cand):
    return fondo_dominante(cand) == f


INVARIANTES = [
    ("size_igual", inv_size_igual, chk_size_igual),
    ("size_const", inv_size_const, chk_size_const),
    ("paleta", inv_paleta, chk_paleta),
    ("paleta_estricta", inv_paleta_estricta, chk_paleta_estricta),
    ("n_obj_const", inv_nobj_const, chk_nobj_const),
    ("n_obj_igual_in", inv_nobj_igual, chk_nobj_igual),
    ("sim_h", inv_sim_h, chk_sim_h),
    ("sim_v", inv_sim_v, chk_sim_v),
    ("fondo", inv_fondo, chk_fondo),
]


def aprender(demos):
    activos = {}
    for nombre, ap, chk in INVARIANTES:
        est = ap(demos)
        if est is not None:
            activos[nombre] = (est, chk)
    return activos


def violaciones(activos, tin, cand):
    return [n for n, (est, chk) in activos.items() if not chk(est, tin, cand)]


def main():
    ddir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    chal = json.load(open(ddir / "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(ddir / "arc-agi_evaluation_solutions.json"))

    fp_por_inv = Counter()
    activos_por_inv = Counter()
    tareas_fp = []
    for k, t in chal.items():
        demos = [(p["input"], p["output"]) for p in t["train"]]
        activos = aprender(demos)
        for n in activos:
            activos_por_inv[n] += 1
        tin = t["test"][0]["input"]
        gt = sols[k][0]
        v = violaciones(activos, tin, gt)
        for n in v:
            fp_por_inv[n] += 1
        if v:
            tareas_fp.append((k, v))

    n = len(chal)
    print(f"tareas eval: {n}\n")
    print(f"{'invariante':16} {'activo en':>10} {'FP (rechaza GT)':>16} {'tasa FP':>9}")
    for nombre, _, _ in INVARIANTES:
        a, f = activos_por_inv[nombre], fp_por_inv[nombre]
        print(f"{nombre:16} {a:7d}/{n:3d} {f:16d} {f/a if a else 0:9.1%}")
    print(f"\ntareas donde ALGUN invariante rechaza la GT: {len(tareas_fp)}/{n} "
          f"= {len(tareas_fp)/n:.1%}")
    for k, v in tareas_fp[:15]:
        print(f"  {k}: {v}")
    seguros = [nm for nm, _, _ in INVARIANTES
               if activos_por_inv[nm] and fp_por_inv[nm] == 0]
    print(f"\ninvariantes con CERO falsos positivos (usables tal cual): {seguros}")


if __name__ == "__main__":
    main()
