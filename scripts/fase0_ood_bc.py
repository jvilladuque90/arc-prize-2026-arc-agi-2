# -*- coding: utf-8 -*-
# Fase 0, partes B y C. B: correlacion d_test vs GT-en-pool (banco augdose, n=16).
# C: catalogo de re-representaciones, cuantas tareas de eval bajan d_test a 0.
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase0_ood import (carga, d_de_tarea, CATALOGO, aplica, rasgos, d_envolvente,
                       spearman, mediana)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUG = sys.argv[1] if len(sys.argv) > 1 else None


def d_con_lista(task, Ts):
    """aplica una secuencia de re-representaciones a demos y test, y mide d."""
    tr = [p["input"] for p in task["train"]]
    te = task["test"][0]["input"]
    for T in Ts:
        tr = [aplica(T, g) for g in tr]
        te = aplica(T, te)
    return d_envolvente([rasgos(g) for g in tr], rasgos(te))[0]


ev = carga("arc-agi_evaluation_challenges.json")

# ------------------------------------------------------------------ B
print("=" * 78)
print("B. d_test frente a GT-en-pool (banco augdose, 4B, 16 vistas)")
print("=" * 78)
if AUG and os.path.exists(AUG):
    aug = json.load(open(AUG, encoding="utf-8"))
    filas = []
    for r in aug["tareas"]:
        k = r["task"]
        if k not in ev:
            continue
        d = d_de_tarea(ev[k])[0]
        filas.append((k, d, r["vistas_con_gt"]))
    filas.sort(key=lambda t: t[1])
    print("%-10s %8s %s" % ("tarea", "d_test", "vistas_con_gt (de 16)"))
    for k, d, v in filas:
        print("%-10s %8.3f %3d %s" % (k, d, v, "#" * v))
    xs = [d for _, d, _ in filas]
    ys = [float(v) for _, _, v in filas]
    rho = spearman(xs, ys)
    n = len(xs)
    print("")
    print("Spearman rho = %+.3f  (n=%d, prediccion registrada: rho < 0)" % (rho, n))
    dentro = [v for _, d, v in filas if d == 0]
    fuera = [v for _, d, v in filas if d > 0]
    if dentro and fuera:
        print("vistas_con_gt  mediana: d==0 -> %.1f (n=%d) | d>0 -> %.1f (n=%d)"
              % (mediana(dentro), len(dentro), mediana(fuera), len(fuera)))
    res = [v for _, _, v in filas if v > 0]
    print("tareas con GT en algun pool: %d/%d" % (len(res), n))
else:
    print("(sin fichero augdose: pasar la ruta como argv[1])")

# ------------------------------------------------------------------ C
print("")
print("=" * 78)
print("C. CATALOGO: re-representaciones que meten el test dentro de la envolvente")
print("=" * 78)
base = {k: d_de_tarea(ev[k])[0] for k in ev}
fuera = [k for k in ev if base[k] > 0]
print("tareas de eval con d>0: %d de %d" % (len(fuera), len(ev)))

cierra = {}
dispara = Counter()
mejora = Counter()
for k in fuera:
    for nom, T in CATALOGO:
        d2 = d_con_lista(ev[k], [T])
        if d2 < base[k] - 1e-9:
            mejora[nom] += 1
        if d2 == 0:
            cierra.setdefault(k, []).append(nom)
            dispara[nom] += 1
print("")
print("una sola T la cierra (d -> 0): %d de %d  = %.1f%%   (prediccion: >=20%%)"
      % (len(cierra), len(fuera), 100.0 * len(cierra) / len(fuera)))
print("por transformacion (cierra / solo reduce):")
for nom, _ in CATALOGO:
    print("   %-16s cierra %3d | reduce %3d" % (nom, dispara[nom], mejora[nom]))

# exploratorio (NO prerregistrado): composiciones de dos
COMPO = [("strip_frame+crop_bbox", ["strip_frame", "crop_bbox"]),
         ("block_downscale+crop_bbox", ["block_downscale", "crop_bbox"]),
         ("detile+crop_bbox", ["detile", "crop_bbox"]),
         ("transpose+crop_bbox", ["transpose_canon", "crop_bbox"])]
D = dict(CATALOGO)
cierra2 = dict(cierra)
disp2 = Counter()
for k in fuera:
    if k in cierra2:
        continue
    for nom, seq in COMPO:
        if d_con_lista(ev[k], [D[s] for s in seq]) == 0:
            cierra2.setdefault(k, []).append(nom)
            disp2[nom] += 1
            break
print("")
print("[exploratorio, no prerregistrado] anadiendo composiciones de dos:")
print("   cierran: %d de %d = %.1f%%" % (len(cierra2), len(fuera),
                                         100.0 * len(cierra2) / len(fuera)))
for nom, c in disp2.most_common():
    print("   %-28s +%d" % (nom, c))

# cuanto queda sin cerrar y por que
resto = [k for k in fuera if k not in cierra2]
q = Counter()
for k in resto:
    q[d_de_tarea(ev[k])[1]] += 1
print("")
print("sin cerrar: %d tareas | rasgo dominante: %s"
      % (len(resto), ", ".join("%s %d" % kv for kv in q.most_common())))
print("d mediana de las que quedan sin cerrar: %.3f" % mediana([base[k] for k in resto]))
json.dump({"base": base, "cierra": cierra, "cierra2": {k: v for k, v in cierra2.items()}},
          open(os.path.join(RAIZ, "scripts", "_fase0_c.json"), "w"))
