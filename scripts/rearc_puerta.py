# -*- coding: utf-8 -*-
"""Puerta re-ARC: falla sft139 cuando una regla que CONOCE se aplica a una escala que NO vio?

Para cada tarea AGI-1 de training (sft139 la tiene memorizada), demos = las ORIGINALES y test = una
instancia NUEVA de re-ARC (verificada con su verificador):
  - rearc_grande: salida >= max(400, 1,3 x la mayor salida demo)  -> fuera de la envolvente de escala
  - rearc_chico : salida <= la mayor salida demo                   -> dentro (control, misma regla)
Escribe nem.json con el mismo esquema que nemvac para reutilizar el kernel.

Uso: python scripts/rearc_puerta.py <dir re-arc> <salida nem.json>
"""
import json, os, random, sys, signal, time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")


def area(g):
    return len(g) * len(g[0])


def main():
    rearc, salida = sys.argv[1], sys.argv[2]
    sys.path.insert(0, rearc)
    os.chdir(rearc)
    import generators, verifiers
    tr = json.load(open(os.path.join(DATA, "arc-agi_training_challenges.json"), encoding="utf-8"))
    agi1 = set(json.load(open(os.path.join(DATA, "agi1_keys.json"), encoding="utf-8")))
    rng = random.Random(0)
    claves = sorted(k for k in agi1 if k in tr and hasattr(generators, "generate_" + k))
    rng.shuffle(claves)
    grandes, chicos, sin = [], [], 0
    t0 = time.time()
    for k in claves:
        gen = getattr(generators, "generate_" + k)
        ver = getattr(verifiers, "verify_" + k)
        demos = tr[k]["train"]
        env = max(area(p["output"]) for p in demos)
        umbral = max(400, int(1.3 * env))
        g_ok = c_ok = None
        for intento in range(40):
            lo, hi = ((0.6, 1.0) if g_ok is None and intento % 2 == 0 else (0.0, 0.5))
            try:
                ex = gen(lo, hi)
                i, o = [list(r) for r in ex["input"]], [list(r) for r in ex["output"]]
                if [list(r) for r in ver(ex["input"])] != o:
                    continue
            except Exception:
                continue
            if max(len(o), len(o[0]), len(i), len(i[0])) > 30:
                continue
            if g_ok is None and area(o) >= umbral:
                g_ok = (i, o)
            elif c_ok is None and area(o) <= env:
                c_ok = (i, o)
            if g_ok and c_ok:
                break
        if g_ok is None:
            sin += 1
        for par, lista, src in ((g_ok, grandes, "rearc_grande"), (c_ok, chicos, "rearc_chico")):
            if par is not None:
                lista.append({"id": "%s_%s" % (k, src[6:]), "src": src, "train": demos,
                              "test_input": par[0], "gt": par[1], "env": env})
        if len(grandes) >= 150 and len(chicos) >= 150:
            break
    print("tareas probadas %d | grandes %d | chicos %d | sin grande alcanzable %d | %.0fs"
          % (claves.index(k) + 1, len(grandes), len(chicos), sin, time.time() - t0))
    pares = {x["id"].split("_")[0] for x in grandes} & {x["id"].split("_")[0] for x in chicos}
    print("tareas con AMBOS brazos (comparacion pareada): %d" % len(pares))
    a_g = sorted(area(x["gt"]) for x in grandes)
    print("area salida grandes: mediana %d | p90 %d ; ratio vs envolvente mediana %.2f"
          % (a_g[len(a_g) // 2], a_g[9 * len(a_g) // 10],
             sorted(area(x["gt"]) / x["env"] for x in grandes)[len(grandes) // 2]))
    ctrl_prev = json.load(open(os.path.join(RAIZ, "scripts", "_nemvac_ctrl.json"), encoding="utf-8"))
    json.dump({"nem": grandes[:150] + chicos[:150], "ctrl": ctrl_prev}, open(salida, "w"), separators=(",", ":"))
    print("escrito", salida)


if __name__ == "__main__":
    main()
