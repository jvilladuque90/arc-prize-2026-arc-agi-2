# -*- coding: utf-8 -*-
# Por que falla C: el hueco lo mandan H/W. La entrada de test es mas grande...
# pero, es mas grande por VACIO (normalizable) o por CONTENIDO (no normalizable)?
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase0_ood import carga, d_de_tarea, rasgos, d_envolvente, mediana

ev = carga("arc-agi_evaluation_challenges.json")
sol = carga("arc-agi_evaluation_solutions.json")

fuera = [k for k in ev if d_de_tarea(ev[k])[0] > 0]

print("=" * 78)
print("POR QUE NO SE CIERRA: vacio frente a contenido")
print("=" * 78)
rel_area, rel_tinta, rel_comp = [], [], []
for k in fuera:
    tr = [p["input"] for p in ev[k]["train"]]
    te = ev[k]["test"][0]["input"]
    a_tr = mediana([len(g) * len(g[0]) for g in tr])
    a_te = len(te) * len(te[0])
    t_tr = mediana([sum(1 for f in g for c in f if c != 0) for g in tr])
    t_te = sum(1 for f in te for c in f if c != 0)
    c_tr = mediana([rasgos(g)["ncomp"] for g in tr])
    c_te = rasgos(te)["ncomp"]
    rel_area.append(a_te / a_tr)
    rel_tinta.append(t_te / max(t_tr, 1))
    rel_comp.append(c_te / max(c_tr, 1))
print("test / mediana(demos), sobre las %d tareas con d>0:" % len(fuera))
print("   area total       mediana x%.2f" % mediana(rel_area))
print("   celdas no-fondo  mediana x%.2f   <- si ~1, el exceso es VACIO (normalizable)"
      % mediana(rel_tinta))
print("   componentes      mediana x%.2f   <- si >1, hay MAS OBJETOS (no normalizable)"
      % mediana(rel_comp))
crece_vacio = sum(1 for a, t in zip(rel_area, rel_tinta) if a > 1.15 and t < 1.15)
crece_todo = sum(1 for a, t in zip(rel_area, rel_tinta) if a > 1.15 and t >= 1.15)
print("")
print("   crece el area pero NO la tinta (vacio) : %d" % crece_vacio)
print("   crece el area Y la tinta   (contenido) : %d" % crece_todo)

print("")
print("=" * 78)
print("LA SALIDA: el test tambien se sale de la envolvente de las salidas demo?")
print("=" * 78)
d_in, d_out = [], []
for k in fuera:
    tin = [p["input"] for p in ev[k]["train"]]
    tou = [p["output"] for p in ev[k]["train"]]
    te = ev[k]["test"][0]["input"]
    go = sol[k][0]
    d_in.append(d_envolvente([rasgos(g) for g in tin], rasgos(te))[0])
    d_out.append(d_envolvente([rasgos(g) for g in tou], rasgos(go))[0])
print("d de la ENTRADA de test : mediana %.3f | fuera %.1f%%"
      % (mediana(d_in), 100.0 * sum(1 for d in d_in if d > 0) / len(d_in)))
print("d de la SALIDA  de test : mediana %.3f | fuera %.1f%%"
      % (mediana(d_out), 100.0 * sum(1 for d in d_out if d > 0) / len(d_out)))
amb = sum(1 for a, b in zip(d_in, d_out) if a > 0 and b > 0)
print("entrada Y salida fuera de envolvente: %d de %d (%.1f%%)"
      % (amb, len(fuera), 100.0 * amb / len(fuera)))

print("")
print("=" * 78)
print("ES 'LA MISMA REGLA A MAYOR ESCALA'? (razon salida/entrada estable)")
print("=" * 78)
estables, inestables, sin = 0, 0, 0
for k in fuera:
    rs = []
    for p in ev[k]["train"]:
        ai = len(p["input"]) * len(p["input"][0])
        ao = len(p["output"]) * len(p["output"][0])
        rs.append(ao / ai)
    if len(rs) < 2:
        sin += 1
        continue
    if max(rs) / max(min(rs), 1e-9) <= 1.10:
        estables += 1
    else:
        inestables += 1
print("razon area_salida/area_entrada constante en las demos (+-10%%): %d de %d (%.1f%%)"
      % (estables, len(fuera), 100.0 * estables / len(fuera)))
print("   no constante: %d | sin datos: %d" % (inestables, sin))
print("")
print("Lectura: si la razon fuese constante, bastaria reescalar. No lo es en la mayoria,")
print("asi que el tamano de la salida de test NO se deduce del de la entrada.")
