# -*- coding: utf-8 -*-
# Fase 0 (0 h-cuota, CPU local): la entrada de test, cae fuera de la envolvente de las
# entradas de las demostraciones? Prerregistro en docs/PREREG_fase0_ood.md.
import json, math, os, sys
from collections import Counter, deque

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")

# ---------------------------------------------------------------- rasgos
PISO = {"H": 1.0, "W": 1.0, "logA": 0.10, "ncol": 1.0,
        "dens": 0.05, "ncomp": 1.0, "fmax": 0.05}
RASGOS = list(PISO)


def componentes(g):
    # componentes conexas 4-conectividad sobre celdas != 0 (fondo), sin mirar color
    H, W = len(g), len(g[0])
    visto = [[False] * W for _ in range(H)]
    n, mayor = 0, 0
    for i in range(H):
        for j in range(W):
            if g[i][j] == 0 or visto[i][j]:
                continue
            n += 1
            tam = 0
            q = deque([(i, j)])
            visto[i][j] = True
            while q:
                a, b = q.popleft()
                tam += 1
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and not visto[x][y] and g[x][y] != 0:
                        visto[x][y] = True
                        q.append((x, y))
            mayor = max(mayor, tam)
    return n, mayor


def rasgos(g):
    H, W = len(g), len(g[0])
    A = H * W
    plano = [c for f in g for c in f]
    nz = sum(1 for c in plano if c != 0)
    n, mayor = componentes(g)
    return {"H": float(H), "W": float(W), "logA": math.log(A),
            "ncol": float(len(set(plano))), "dens": nz / A,
            "ncomp": float(n), "fmax": (mayor / nz) if nz else 0.0}


def d_envolvente(demos, test):
    # max_f (distancia al borde de la envolvente de las demos) / max(rango, piso)
    peor, quien = 0.0, None
    det = {}
    for f in RASGOS:
        vs = [r[f] for r in demos]
        lo, hi = min(vs), max(vs)
        t = test[f]
        R = max(hi - lo, PISO[f])
        e = 0.0 if lo <= t <= hi else (t - hi) / R if t > hi else (lo - t) / R
        det[f] = e
        if e > peor:
            peor, quien = e, f
    return peor, quien, det


# ---------------------------------------------------------------- re-representaciones T
def crop_bbox(g):
    H, W = len(g), len(g[0])
    ce = [(i, j) for i in range(H) for j in range(W) if g[i][j] != 0]
    if not ce:
        return None
    i0 = min(i for i, _ in ce); i1 = max(i for i, _ in ce)
    j0 = min(j for _, j in ce); j1 = max(j for _, j in ce)
    if (i1 - i0 + 1, j1 - j0 + 1) == (H, W):
        return None
    return [fila[j0:j1 + 1] for fila in g[i0:i1 + 1]]


def strip_frame(g):
    cur = [f[:] for f in g]
    quitados = 0
    while len(cur) > 2 and len(cur[0]) > 2:
        borde = set(cur[0]) | set(cur[-1]) | {f[0] for f in cur} | {f[-1] for f in cur}
        if len(borde) != 1:
            break
        cur = [f[1:-1] for f in cur[1:-1]]
        quitados += 1
    return cur if quitados else None


def block_downscale(g):
    H, W = len(g), len(g[0])
    for k in range(min(H, W), 1, -1):
        if H % k or W % k:
            continue
        ok = True
        for bi in range(H // k):
            for bj in range(W // k):
                v = g[bi * k][bj * k]
                for a in range(k):
                    for b in range(k):
                        if g[bi * k + a][bj * k + b] != v:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return [[g[bi * k][bj * k] for bj in range(W // k)] for bi in range(H // k)]
    return None


def detile(g):
    H, W = len(g), len(g[0])
    for h in range(1, H + 1):
        if H % h:
            continue
        for w in range(1, W + 1):
            if W % w or (h == H and w == W):
                continue
            if all(g[i][j] == g[i % h][j % w] for i in range(H) for j in range(W)):
                return [f[:w] for f in g[:h]]
    return None


def color_canon(g):
    cnt = Counter(c for f in g for c in f if c != 0)
    orden = [c for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    mapa = {c: i + 1 for i, c in enumerate(orden)}
    mapa[0] = 0
    return [[mapa[c] for c in f] for f in g]


def transpose_canon(g):
    if len(g) <= len(g[0]):
        return None
    return [list(col) for col in zip(*g)]


CATALOGO = [("crop_bbox", crop_bbox), ("strip_frame", strip_frame),
            ("block_downscale", block_downscale), ("detile", detile),
            ("color_canon", color_canon), ("transpose_canon", transpose_canon)]


def aplica(T, g):
    r = T(g)
    return r if r else g          # si no aplica a esta rejilla, queda igual


# ---------------------------------------------------------------- estadistica sin scipy
def rangos(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        med = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[idx[k]] = med
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rangos(x), rangos(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def mannwhitney(a, b):
    # U de a frente a b + p de una cola (a > b) por aproximacion normal con empates
    todos = a + b
    r = rangos(todos)
    Ra = sum(r[:len(a)])
    na, nb = len(a), len(b)
    Ua = Ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    cnt = Counter(todos)
    N = na + nb
    corr = sum(t ** 3 - t for t in cnt.values())
    sd = math.sqrt(na * nb / 12.0 * ((N + 1) - corr / (N * (N - 1))))
    z = (Ua - mu) / sd if sd else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))          # cola derecha: a > b
    return Ua, z, p


def mediana(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------- carga
def carga(nombre):
    with open(os.path.join(DATA, nombre), encoding="utf-8") as fh:
        return json.load(fh)


def d_de_tarea(task, T=None):
    tr = [p["input"] for p in task["train"]]
    te = task["test"][0]["input"]
    if T is not None:
        tr = [aplica(T, g) for g in tr]
        te = aplica(T, te)
    return d_envolvente([rasgos(g) for g in tr], rasgos(te))


def analiza(tasks, claves=None):
    ks = claves if claves is not None else list(tasks)
    ds, quien = [], Counter()
    for k in ks:
        d, q, _ = d_de_tarea(tasks[k])
        ds.append(d)
        if d > 0:
            quien[q] += 1
    return ds, quien


if __name__ == "__main__":
    ev = carga("arc-agi_evaluation_challenges.json")
    tr = carga("arc-agi_training_challenges.json")
    agi1 = set(json.load(open(os.path.join(DATA, "agi1_keys.json"), encoding="utf-8")))

    d_ev, q_ev = analiza(ev)
    d_tr, q_tr = analiza(tr)
    k_agi1 = [k for k in tr if k in agi1]
    k_agi2 = [k for k in tr if k not in agi1]
    d_a1, _ = analiza(tr, k_agi1)
    d_a2, _ = analiza(tr, k_agi2)

    print("=" * 78)
    print("A. DISTANCIA DEL TEST A LA ENVOLVENTE DE LAS DEMOS (d = rangos-de-demo fuera)")
    print("=" * 78)
    for nom, ds in (("eval ARC-AGI-2", d_ev), ("train completo", d_tr),
                    ("  -> subset AGI-1", d_a1), ("  -> subset AGI-2", d_a2)):
        if not ds:
            print("%-18s SIN TAREAS" % nom)
            continue
        fuera = sum(1 for d in ds if d > 0)
        print("%-18s n=%4d | mediana %6.3f | media %6.3f | d>0 %5.1f%% | d>1 %5.1f%%"
              % (nom, len(ds), mediana(ds), sum(ds) / len(ds),
                 100.0 * fuera / len(ds), 100.0 * sum(1 for d in ds if d > 1) / len(ds)))
    U, z, p = mannwhitney(d_ev, d_tr)
    print("")
    print("A1  Mann-Whitney eval > train : z = %.3f, p(1 cola) = %.4g" % (z, p))
    if d_a1:
        U2, z2, p2 = mannwhitney(d_ev, d_a1)
        print("A1b Mann-Whitney eval > AGI-1 : z = %.3f, p(1 cola) = %.4g" % (z2, p2))
    print("A2  eval con d>0: %.1f%%  (prediccion: >50%%)"
          % (100.0 * sum(1 for d in d_ev if d > 0) / len(d_ev)))
    print("")
    print("rasgo que dispara (eval) :", ", ".join("%s %d" % kv for kv in q_ev.most_common()))
    print("rasgo que dispara (train):", ", ".join("%s %d" % kv for kv in q_tr.most_common()))
    json.dump({"eval": d_ev, "train": d_tr},
              open(os.path.join(RAIZ, "scripts", "_fase0_d.json"), "w"))
