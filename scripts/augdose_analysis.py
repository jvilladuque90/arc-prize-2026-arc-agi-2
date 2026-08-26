"""Analisis OFFLINE de la dosis de augmentaciones: toda la curva sin volver a tocar la GPU.

Idea central (26-ago): no hace falta un experimento por dosis. Si el banco decodifica las 16
vistas UNA vez y persiste QUE produjo cada vista, entonces la curva 2/4/8/12/16 y la dispersion
del voto se calculan aqui por submuestreo. Una pasada de GPU -> barrido completo, reutilizable
para cualquier regla de agregacion futura.

La pregunta que decide la palanca: **¿la dispersion del voto con POCAS vistas predice que mas
vistas ayuden?** Si predice, la reasignacion por tarea es politica de control (barata,
injertable). Si no, la palanca muere.

Agregacion: replica el `getter_full_probmul_3` de produccion en su parte de inferencia --
voto de un candidato = suma de (3 - beam_score) sobre las vistas que lo produjeron. (Produccion
suma ademas un termino de re-scoring que este banco no genera; anotado como aproximacion.)

Uso: python scripts/augdose_analysis.py [ruta_json]   (sin argumento: corre los tests)
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

BASELINE = 3.0
NS = (2, 4, 8, 12, 16)
N_MUESTRAS = 40          # submuestras aleatorias por dosis (reduce el ruido del submuestreo)


def votar(vistas):
    """vistas: lista de listas de {'grid': [[...]], 'score': float} ya DES-TRANSFORMADOS.
    Devuelve [(clave, voto)] ordenado de mayor a menor voto."""
    acc = defaultdict(float)
    for cands in vistas:
        for c in cands:
            acc[json.dumps(c["grid"])] += max(0.0, BASELINE - c["score"])
    return sorted(acc.items(), key=lambda x: -x[1])


def resuelta(vistas, gt, k=2):
    """¿La GT esta entre los k candidatos mas votados? (pass@2 = como puntua produccion)"""
    top = [json.loads(h) for h, _ in votar(vistas)[:k]]
    return any(g == gt for g in top)


def dispersion(vistas):
    """Metricas OBSERVABLES (no usan la GT) del perfil de voto.

    `productividad` = fraccion de vistas que devolvieron ALGUN candidato. El smoke del 26-ago
    mostro que es la observable buena y que `ratio` NO lo es: 7b5033c1 (beneficio +0,60 al
    pasar de 4 a 16 vistas) y e8686506 (beneficio 0,00) tienen ratio identico (0,000) y un solo
    candidato -- con pocas vistas no hay competencia que medir. Lo que las separa es que la
    primera solo produce candidato en la mitad de las vistas: vistas vacias = el modelo apenas
    encuentra nada = mas vistas son mas billetes de loteria.
    """
    v = votar(vistas)
    prod = sum(1 for cs in vistas if cs) / len(vistas) if vistas else 0.0
    if not v:
        return {"n_cands": 0, "ratio": 0.0, "margen": 0.0, "masa_top1": 0.0,
                "productividad": prod}
    v1 = v[0][1]
    v2 = v[1][1] if len(v) > 1 else 0.0
    total = sum(x for _, x in v)
    return {
        "n_cands": len(v),
        "ratio": (v2 / v1) if v1 else 0.0,        # 0 = dominio total; ~1 = empate tecnico
        "margen": v1 - v2,
        "masa_top1": (v1 / total) if total else 0.0,
        "productividad": prod,
    }


def curva(vistas_todas, gt, ns=NS, n_muestras=N_MUESTRAS, seed=0):
    """Tasa de acierto por dosis, promediando submuestras aleatorias de vistas."""
    rng = random.Random(seed)
    out = {}
    n_tot = len(vistas_todas)
    for n in ns:
        if n > n_tot:
            continue
        if n == n_tot:
            out[n] = 1.0 if resuelta(vistas_todas, gt) else 0.0
            continue
        aciertos = sum(resuelta(rng.sample(vistas_todas, n), gt) for _ in range(n_muestras))
        out[n] = aciertos / n_muestras
    return out


def analizar(datos):
    """datos: {'tareas': [{'task', 'gt', 'vistas': [[cand,...], ...]}, ...]}"""
    filas = []
    for t in datos["tareas"]:
        c = curva(t["vistas"], t["gt"])
        d4 = dispersion(t["vistas"][:4])
        ns = sorted(c)
        filas.append({"task": t["task"], "curva": c, "disp4": d4,
                      "beneficio": (c[ns[-1]] - c[ns[0]]) if len(ns) > 1 else 0.0})
    return filas


def _pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else 0.0


def informe(filas):
    print(f"{'tarea':10} " + " ".join(f"n={n:<5}" for n in NS)
          + f" {'benef':>7} {'prod@4':>7} {'ratio@4':>8} {'cands@4':>8}")
    for f in filas:
        cs = " ".join(f"{f['curva'].get(n, float('nan')):<7.2f}" for n in NS)
        print(f"{f['task']:10} {cs} {f['beneficio']:+7.2f} "
              f"{f['disp4']['productividad']:7.2f} {f['disp4']['ratio']:8.3f} "
              f"{f['disp4']['n_cands']:8d}")
    ys = [f["beneficio"] for f in filas]
    for nom, clave, signo in (("productividad@4", "productividad", "<"),
                              ("dispersion@4", "ratio", ">")):
        r = _pearson([f["disp4"][clave] for f in filas], ys)
        if r is None:
            continue
        print(f"\ncorrelacion( {nom} , beneficio de 4->16 ) = {r:+.3f}")
        if clave == "productividad":
            print("  r NEGATIVA fuerte => la ESCASEZ de candidatos predice que mas vistas")
            print("     ayudan -> disparador observable para la reasignacion -> palanca VIVA")
        else:
            print("  (contraste: el empate de votos resulto NO discriminar en el smoke)")


# ----------------------------- tests (corren sin GPU) -----------------------------

def _tests():
    GT = [[1, 2], [3, 4]]
    OTRO = [[5, 5], [5, 5]]
    TERCERO = [[7, 7], [7, 7]]

    # 1) voto: suma (3 - score) sobre vistas que produjeron el mismo grid
    v = votar([[{"grid": GT, "score": 1.0}], [{"grid": GT, "score": 0.0}],
               [{"grid": OTRO, "score": 2.0}]])
    assert json.loads(v[0][0]) == GT and abs(v[0][1] - 5.0) < 1e-9, v
    assert abs(v[1][1] - 1.0) < 1e-9, v

    # 2) pass@2: la GT segunda TAMBIEN cuenta; tercera NO
    vistas = [[{"grid": OTRO, "score": 0.0}, {"grid": GT, "score": 0.5}]]
    assert resuelta(vistas, GT, k=2)
    vistas3 = [[{"grid": OTRO, "score": 0.0}, {"grid": TERCERO, "score": 0.1},
                {"grid": GT, "score": 2.9}]]
    assert not resuelta(vistas3, GT, k=2)

    # 3) dispersion: dominio total vs empate
    dom = dispersion([[{"grid": GT, "score": 0.0}], [{"grid": GT, "score": 0.0}],
                      [{"grid": OTRO, "score": 2.9}]])
    emp = dispersion([[{"grid": GT, "score": 1.5}], [{"grid": OTRO, "score": 1.5}]])
    assert dom["ratio"] < 0.1 and emp["ratio"] > 0.9, (dom, emp)

    # 4) curva: si TODAS las vistas dan la GT, cualquier dosis acierta
    todas = [[{"grid": GT, "score": 0.1}] for _ in range(16)]
    c = curva(todas, GT)
    assert all(abs(x - 1.0) < 1e-9 for x in c.values()), c

    # 5) curva creciente: la GT solo aparece en 3 de 16 vistas y hay un distractor
    #    constante -> con pocas vistas gana el distractor, con muchas gana la GT
    mixtas = ([[{"grid": GT, "score": 0.0}, {"grid": OTRO, "score": 2.0}] for _ in range(3)]
              + [[{"grid": OTRO, "score": 2.0}] for _ in range(13)])
    c = curva(mixtas, GT, seed=7)
    assert c[16] == 1.0, c
    assert c[2] < c[16], c          # con 2 vistas a menudo ni aparece la GT

    # 6) el submuestreo no rompe con n > vistas disponibles
    assert set(curva([[{"grid": GT, "score": 0.0}]] * 4, GT).keys()) <= {2, 4}

    print("tests OK: voto, pass@2, dispersion, curva plana, curva creciente, borde de n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        informe(analizar(json.load(open(sys.argv[1], encoding="utf-8"))))
    else:
        _tests()
