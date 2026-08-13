"""claudegen v1 — corpus sintético con AUTOR SEMÁNTICO (Claude), piloto de distribución nueva.

Contexto (docs/SCALE_PLAN.md D3): las fuentes públicas (ARC-Heavy, NVARC) saturan en el mismo
plateau del 2B (~7-8/24) → la única palanca de datos restante es una distribución NUEVA. Este
módulo define ~30 REGLAS escritas a mano, cada una una transformación semánticamente coherente
(condicional / compositiva / dependiente de contexto — el sabor de AGI-2), con parámetros
aleatorios por tarea. Una tarea = una regla + params fijos + 4 pares (instancias frescas).

Diferencias con el seedgen v1/v2 (cerrado con dos negativos):
- El modelo NUNCA ve código: data = pares de grids ejecutados (transducción, formato campeón)
  → el modo de fallo "imita el estilo del programa" es imposible por construcción.
- Reglas AUTORADAS una a una (no composición mecánica de seeds ajenas): cada regla es una
  historia coherente e inferible desde 3 ejemplos.
- Verificado por construcción (el generador ES la regla) + guardas por par (rectangular, ≤30,
  no trivial, pares distintos).

Todo numpy+stdlib. Uso: build_task(rng) → {"rule": name, "pairs": [(in, out) x4]}.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ helpers


def _empty(rng, lo=6, hi=16):
    return np.zeros((rng.integers(lo, hi + 1), rng.integers(lo, hi + 1)), dtype=int)


def _colors(rng, k, exclude=()):
    pool = [c for c in range(1, 10) if c not in exclude]
    return list(rng.choice(pool, size=k, replace=False))


def _blob(rng, h, w):
    """Mancha 4-conexa aleatoria dentro de h x w (al menos 1 celda)."""
    m = np.zeros((h, w), dtype=bool)
    r, c = rng.integers(h), rng.integers(w)
    m[r, c] = True
    for _ in range(rng.integers(h * w // 2, h * w + 1)):
        rs, cs = np.where(m)
        i = rng.integers(len(rs))
        r, c = rs[i] + [-1, 1, 0, 0][rng.integers(4)], cs[i] + [0, 0, -1, 1][rng.integers(4)]
        if 0 <= r < h and 0 <= c < w:
            m[r, c] = True
    return m


def _place_objects(rng, g, n, colors, max_side=4, solid=True):
    """Coloca n objetos que no se tocan (ni diagonal). Devuelve lista de máscaras."""
    placed = []
    occupied = np.zeros_like(g, dtype=bool)
    tries = 0
    while len(placed) < n and tries < 200:
        tries += 1
        oh, ow = rng.integers(1, max_side + 1), rng.integers(1, max_side + 1)
        if oh >= g.shape[0] or ow >= g.shape[1]:
            continue
        r = rng.integers(g.shape[0] - oh + 1)
        c = rng.integers(g.shape[1] - ow + 1)
        r0, r1 = max(0, r - 1), min(g.shape[0], r + oh + 1)
        c0, c1 = max(0, c - 1), min(g.shape[1], c + ow + 1)
        if occupied[r0:r1, c0:c1].any():
            continue
        m = np.zeros_like(g, dtype=bool)
        sub = _blob(rng, oh, ow) if not solid else np.ones((oh, ow), dtype=bool)
        m[r:r + oh, c:c + ow] = sub
        if not m.any():
            continue
        g[m] = colors[len(placed) % len(colors)]
        occupied[m] = True
        placed.append(m)
    return placed


def _objects(g):
    """Componentes 4-conexas de celdas no-cero del MISMO color."""
    seen = np.zeros_like(g, dtype=bool)
    out = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] and not seen[r, c]:
                col, stack, cells = g[r, c], [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < g.shape[0] and 0 <= nx < g.shape[1]
                                and g[ny, nx] == col and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append((col, cells))
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


# ------------------------------------------------------------------ reglas
# Cada regla: rule(rng, params) -> (input, output) o None; make_params(rng) -> params fijos
# de la tarea. Registradas en RULES al final.

def p_swap_extremes(rng):
    return {}


def r_swap_extremes(rng, p):
    """El color más frecuente y el menos frecuente (no-cero) se intercambian."""
    g = _empty(rng)
    cols = _colors(rng, 3)
    _place_objects(rng, g, rng.integers(4, 7), cols, solid=False)
    vals, counts = np.unique(g[g > 0], return_counts=True)
    if len(vals) < 2 or counts.min() == counts.max():
        return None
    hi, lo = vals[counts.argmax()], vals[counts.argmin()]
    if (counts == counts.max()).sum() > 1 or (counts == counts.min()).sum() > 1:
        return None
    out = g.copy()
    out[g == hi], out[g == lo] = lo, hi
    return g, out


def p_delete_rare(rng):
    return {"thr": int(rng.integers(3, 6))}


def r_delete_rare(rng, p):
    """Los colores con menos de `thr` celdas desaparecen; el resto queda igual."""
    g = _empty(rng)
    cols = _colors(rng, 4)
    _place_objects(rng, g, rng.integers(4, 8), cols, max_side=3, solid=False)
    vals, counts = np.unique(g[g > 0], return_counts=True)
    if len(vals) < 3 or not ((counts < p["thr"]).any() and (counts >= p["thr"]).any()):
        return None
    out = g.copy()
    for v, n in zip(vals, counts):
        if n < p["thr"]:
            out[g == v] = 0
    return g, out


def p_keep_largest(rng):
    return {"recolor": int(_colors(rng, 1)[0])}


def r_keep_largest(rng, p):
    """Solo sobrevive el objeto más grande, repintado a un color fijo de la tarea."""
    g = _empty(rng)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 3, exclude=(p["recolor"],)),
                   solid=False)
    obs = _objects(g)
    if len(obs) < 2:
        return None
    sizes = [len(c) for _, c in obs]
    if sizes.count(max(sizes)) > 1:
        return None
    out = np.zeros_like(g)
    for r, c in obs[int(np.argmax(sizes))][1]:
        out[r, c] = p["recolor"]
    return g, out


def p_mirror_overlay(rng):
    return {"axis": int(rng.integers(2))}


def r_mirror_overlay(rng, p):
    """El grid se superpone con su espejo (h o v, fijo por tarea); no-cero gana."""
    g = _empty(rng)
    _place_objects(rng, g, rng.integers(2, 5), _colors(rng, 2), solid=False)
    if not g.any():
        return None
    mir = np.flip(g, axis=p["axis"])
    out = g.copy()
    out[out == 0] = mir[out == 0]
    if (out == g).all():
        return None
    return g, out


def p_transpose_swap(rng):
    return {}


def r_transpose_swap(rng, p):
    """Transponer y luego intercambiar los DOS colores más frecuentes."""
    g = _empty(rng, 6, 12)
    _place_objects(rng, g, rng.integers(4, 7), _colors(rng, 3), solid=False)
    vals, counts = np.unique(g[g > 0], return_counts=True)
    if len(vals) < 2:
        return None
    order = np.argsort(-counts)
    if counts[order[0]] == counts[order[1]] or (len(vals) > 2 and counts[order[1]] == counts[order[2]]):
        return None
    a, b = vals[order[0]], vals[order[1]]
    t = g.T.copy()
    out = t.copy()
    out[t == a], out[t == b] = b, a
    return g, out


def p_border_rarest(rng):
    return {}


def r_border_rarest(rng, p):
    """El borde vacío se pinta con el color MENOS frecuente del interior."""
    g = _empty(rng, 7, 14)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 3), max_side=3, solid=False)
    inner = g[1:-1, 1:-1]
    vals, counts = np.unique(inner[inner > 0], return_counts=True)
    if len(vals) < 2 or (counts == counts.min()).sum() > 1:
        return None
    rare = vals[counts.argmin()]
    out = g.copy()
    border = np.zeros_like(g, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    out[border & (g == 0)] = rare
    if (out == g).all():
        return None
    return g, out


def p_count_square(rng):
    return {"fill": int(_colors(rng, 1)[0])}


def r_count_square(rng, p):
    """Salida = cuadrado k x k (k = número de objetos) del color fijo de la tarea."""
    g = _empty(rng)
    n = int(rng.integers(2, 6))
    placed = _place_objects(rng, g, n, _colors(rng, 3), max_side=3)
    if len(placed) != n:
        return None
    return g, np.full((n, n), p["fill"], dtype=int)


def p_row_bars(rng):
    return {"bar": int(_colors(rng, 1)[0])}


def r_row_bars(rng, p):
    """Cada fila -> barra izquierda con tantas celdas como no-ceros tenía la fila."""
    g = _empty(rng, 5, 10)
    _place_objects(rng, g, rng.integers(3, 7), _colors(rng, 3), max_side=3, solid=False)
    if not g.any():
        return None
    out = np.zeros_like(g)
    for i in range(g.shape[0]):
        out[i, :int((g[i] > 0).sum())] = p["bar"]
    if (out == g).all():
        return None
    return g, out


def p_tile_by_colors(rng):
    return {}


def r_tile_by_colors(rng, p):
    """Salida = input tejado k x k, con k = número de colores distintos (2 o 3)."""
    g = _empty(rng, 3, 6)
    k = int(rng.integers(2, 4))
    cols = _colors(rng, k)
    for c in cols:
        g[rng.integers(g.shape[0]), rng.integers(g.shape[1])] = c
    _place_objects(rng, g, 2, cols, max_side=2, solid=False)
    if len(np.unique(g[g > 0])) != k or g.shape[0] * k > 30 or g.shape[1] * k > 30:
        return None
    return g, np.tile(g, (k, k))


def p_percol_column(rng):
    return {"mark": int(_colors(rng, 1)[0])}


def r_percol_column(rng, p):
    """Se añade a la derecha una columna que marca las filas que contienen algún objeto."""
    g = _empty(rng, 5, 12)
    _place_objects(rng, g, rng.integers(2, 5), _colors(rng, 2, exclude=(p["mark"],)),
                   max_side=3, solid=False)
    if not g.any() or (g > 0).all(axis=1).all():
        return None
    col = np.where((g > 0).any(axis=1), p["mark"], 0).reshape(-1, 1)
    return g, np.concatenate([g, col], axis=1)


def p_checker_topcolors(rng):
    return {}


def r_checker_topcolors(rng, p):
    """Salida = tablero k x k (k = nº de colores) alternando los DOS más frecuentes."""
    g = _empty(rng)
    cols = _colors(rng, int(rng.integers(2, 5)))
    _place_objects(rng, g, len(cols) + 2, cols, max_side=3, solid=False)
    vals, counts = np.unique(g[g > 0], return_counts=True)
    k = len(vals)
    if k < 2:
        return None
    order = np.argsort(-counts)
    if counts[order[0]] == counts[order[1]]:
        return None
    a, b = int(vals[order[0]]), int(vals[order[1]])
    out = np.fromfunction(lambda i, j: np.where((i + j) % 2 == 0, a, b), (k, k), dtype=int)
    return g, out.astype(int)


def p_crop_scale(rng):
    return {"k": int(rng.integers(2, 4))}


def r_crop_scale(rng, p):
    """Recortar al bounding box de lo no-cero y escalar x k (k fijo por tarea)."""
    g = _empty(rng, 8, 14)
    _place_objects(rng, g, rng.integers(1, 3), _colors(rng, 2), max_side=3, solid=False)
    if not g.any():
        return None
    rs, cs = np.where(g > 0)
    sub = g[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
    if sub.shape[0] * p["k"] > 30 or sub.shape[1] * p["k"] > 30:
        return None
    return g, np.kron(sub, np.ones((p["k"], p["k"]), dtype=int))


def p_frame_interior(rng):
    return {}


def r_frame_interior(rng, p):
    """Hay un marco rectangular; la salida es su interior rotado 180°."""
    g = _empty(rng, 9, 16)
    fh, fw = rng.integers(4, min(7, g.shape[0])), rng.integers(4, min(9, g.shape[1]))
    r = rng.integers(g.shape[0] - fh + 1)
    c = rng.integers(g.shape[1] - fw + 1)
    fc, ic = _colors(rng, 2)
    g[r:r + fh, c:c + fw] = 0
    g[r, c:c + fw] = g[r + fh - 1, c:c + fw] = fc
    g[r:r + fh, c] = g[r:r + fh, c + fw - 1] = fc
    inner = g[r + 1:r + fh - 1, c + 1:c + fw - 1]
    for _ in range(rng.integers(2, 6)):
        inner[rng.integers(inner.shape[0]), rng.integers(inner.shape[1])] = ic
    noise = _colors(rng, 1, exclude=(fc, ic))[0]
    for _ in range(rng.integers(2, 5)):
        rr, cc = rng.integers(g.shape[0]), rng.integers(g.shape[1])
        if not (r <= rr < r + fh and c <= cc < c + fw) and g[rr, cc] == 0:
            g[rr, cc] = noise
    if not inner.any():
        return None
    return g, np.rot90(inner.copy(), 2)


def p_marker_quadrant(rng):
    return {"marker": int(_colors(rng, 1)[0])}


def r_marker_quadrant(rng, p):
    """El cuadrante que contiene la celda-marcador se devuelve rotado 90°."""
    h, w = int(rng.integers(3, 8)) * 2, int(rng.integers(3, 8)) * 2
    g = np.zeros((h, w), dtype=int)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 2, exclude=(p["marker"],)),
                   max_side=3, solid=False)
    g[g == p["marker"]] = 0
    q = int(rng.integers(4))
    r0, c0 = (0 if q < 2 else h // 2), (0 if q % 2 == 0 else w // 2)
    rr, cc = r0 + rng.integers(h // 2), c0 + rng.integers(w // 2)
    g[rr, cc] = p["marker"]
    quad = g[r0:r0 + h // 2, c0:c0 + w // 2]
    return g, np.rot90(quad.copy(), 1)


def p_densest_quadrant(rng):
    return {}


def r_densest_quadrant(rng, p):
    """Salida = el cuadrante con MÁS celdas no-cero (sin transformar)."""
    h, w = int(rng.integers(3, 8)) * 2, int(rng.integers(3, 8)) * 2
    g = np.zeros((h, w), dtype=int)
    _place_objects(rng, g, rng.integers(4, 8), _colors(rng, 3), max_side=3, solid=False)
    quads = [g[:h // 2, :w // 2], g[:h // 2, w // 2:], g[h // 2:, :w // 2], g[h // 2:, w // 2:]]
    counts = [(q > 0).sum() for q in quads]
    if counts.count(max(counts)) > 1 or max(counts) == 0:
        return None
    return g, quads[int(np.argmax(counts))].copy()


def p_unique_corners(rng):
    return {}


def r_unique_corners(rng, p):
    """La celda del color ÚNICO (aparece 1 vez) se copia a las 4 esquinas."""
    g = _empty(rng, 7, 14)
    cols = _colors(rng, 3)
    _place_objects(rng, g, rng.integers(3, 6), cols[:2], max_side=3, solid=False)
    g[0, 0] = g[0, -1] = g[-1, 0] = g[-1, -1] = 0
    vals, counts = np.unique(g[g > 0], return_counts=True)
    if len(vals) < 1:
        return None
    rr, cc = rng.integers(1, g.shape[0] - 1), rng.integers(1, g.shape[1] - 1)
    if g[rr, cc] != 0:
        return None
    g[rr, cc] = cols[2]
    if (g == cols[2]).sum() != 1:
        return None
    out = g.copy()
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = cols[2]
    return g, out


def p_mask_and(rng):
    return {"sep": int(_colors(rng, 1)[0]), "fill": None}


def r_mask_and(rng, p):
    """Dos paneles separados por línea: salida = AND celda a celda (ambos no-cero)."""
    if p["fill"] is None:
        p["fill"] = int(_colors(rng, 1, exclude=(p["sep"],))[0])
    h, w = int(rng.integers(4, 9)), int(rng.integers(3, 7))
    a = (np.asarray(rng.random((h, w))) < 0.45).astype(int)
    b = (np.asarray(rng.random((h, w))) < 0.45).astype(int)
    ca, cb = _colors(rng, 2, exclude=(p["sep"], p["fill"]))
    g = np.concatenate([a * ca, np.full((h, 1), p["sep"]), b * cb], axis=1)
    out = ((a > 0) & (b > 0)).astype(int) * p["fill"]
    if out.sum() == 0 or (a & b).all():
        return None
    return g, out


def p_mask_xor(rng):
    return {"sep": int(_colors(rng, 1)[0]), "fill": None}


def r_mask_xor(rng, p):
    """Dos paneles separados por línea: salida = XOR celda a celda."""
    if p["fill"] is None:
        p["fill"] = int(_colors(rng, 1, exclude=(p["sep"],))[0])
    h, w = int(rng.integers(4, 9)), int(rng.integers(3, 7))
    a = (np.asarray(rng.random((h, w))) < 0.45).astype(int)
    b = (np.asarray(rng.random((h, w))) < 0.45).astype(int)
    ca, cb = _colors(rng, 2, exclude=(p["sep"], p["fill"]))
    g = np.concatenate([a * ca, np.full((h, 1), p["sep"]), b * cb], axis=1)
    out = ((a > 0) ^ (b > 0)).astype(int) * p["fill"]
    if out.sum() == 0:
        return None
    return g, out


def p_mirror_repair(rng):
    return {}


def r_mirror_repair(rng, p):
    """Patrón simétrico (izq-der) con la mitad derecha dañada (ceros) → repararla."""
    h, w = int(rng.integers(5, 12)), int(rng.integers(3, 8))
    left = np.zeros((h, w), dtype=int)
    _place_objects(rng, left, rng.integers(3, 6), _colors(rng, 3), max_side=3, solid=False)
    if not left.any():
        return None
    full = np.concatenate([left, np.fliplr(left)], axis=1)
    g = full.copy()
    g[:, w:] = 0
    return g, full


def p_periodic_rows(rng):
    return {}


def r_periodic_rows(rng, p):
    """Filas periódicas (periodo 2-3) con una franja borrada → restaurar el patrón."""
    period = int(rng.integers(2, 4))
    h, w = period * int(rng.integers(3, 6)), int(rng.integers(5, 12))
    base = np.zeros((period, w), dtype=int)
    cols = _colors(rng, 2)
    for i in range(period):
        for j in range(w):
            base[i, j] = cols[int(rng.integers(len(cols)))] if rng.random() < 0.6 else 0
    if not base.any():
        return None
    full = np.tile(base, (h // period, 1))
    g = full.copy()
    r0 = int(rng.integers(1, h - 1))
    g[r0:min(h, r0 + rng.integers(1, period + 1)), :] = 0
    if (g == full).all():
        return None
    return g, full


def p_symmetrize180(rng):
    return {}


def r_symmetrize180(rng, p):
    """Superponer el grid con su rotación 180° (no-cero gana; el original prevalece)."""
    g = _empty(rng, 6, 12)
    _place_objects(rng, g, rng.integers(2, 4), _colors(rng, 2), max_side=3, solid=False)
    if not g.any():
        return None
    rot = np.rot90(g, 2)
    out = g.copy()
    out[out == 0] = rot[out == 0]
    if (out == g).all():
        return None
    return g, out


def p_fill_zeros_majority(rng):
    return {}


def r_fill_zeros_majority(rng, p):
    """Cada cero con ≥2 vecinos (4-conn) del MISMO color adopta ese color."""
    g = _empty(rng, 6, 12)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 2), max_side=4)
    out = g.copy()
    changed = False
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == 0:
                neigh = [g[r + dr, c + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1]]
                nz = [v for v in neigh if v > 0]
                for v in set(nz):
                    if nz.count(v) >= 2:
                        out[r, c] = v
                        changed = True
                        break
    if not changed:
        return None
    return g, out


def p_outline_objects(rng):
    return {}


def r_outline_objects(rng, p):
    """Cada objeto sólido se reduce a su contorno (el interior pasa a 0)."""
    g = _empty(rng, 8, 14)
    _place_objects(rng, g, rng.integers(2, 4), _colors(rng, 2), max_side=6)
    obs = _objects(g)
    ok = any(len(cells) >= 9 for _, cells in obs)
    if not ok:
        return None
    out = g.copy()
    for col, cells in obs:
        cellset = set(cells)
        for r, c in cells:
            if all((r + dr, c + dc) in cellset for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out[r, c] = 0
    if (out == g).all():
        return None
    return g, out


def p_remove_isolated(rng):
    return {}


def r_remove_isolated(rng, p):
    """Las celdas sin ningún vecino (4-conn) del mismo color desaparecen (ruido fuera)."""
    g = _empty(rng, 6, 12)
    _place_objects(rng, g, rng.integers(2, 4), _colors(rng, 2), max_side=3)
    noise = _colors(rng, 1)[0]
    n_noise = 0
    for _ in range(rng.integers(3, 7)):
        r, c = rng.integers(g.shape[0]), rng.integers(g.shape[1])
        neigh_zero = all(not (0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1])
                         or g[r + dr, c + dc] == 0
                         for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if g[r, c] == 0 and neigh_zero:
            g[r, c] = noise
            n_noise += 1
    if n_noise == 0:
        return None
    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] > 0:
                alone = all(not (0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1])
                            or g[r + dr, c + dc] != g[r, c]
                            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                if alone:
                    out[r, c] = 0
    if (out == g).all():
        return None
    return g, out


def p_gravity(rng):
    return {}


def r_gravity(rng, p):
    """Todas las celdas caen al fondo de su columna (colores apilados en orden)."""
    g = _empty(rng, 6, 12)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 3), max_side=3, solid=False)
    if not g.any():
        return None
    out = np.zeros_like(g)
    for c in range(g.shape[1]):
        col = g[:, c][g[:, c] > 0]
        if len(col):
            out[-len(col):, c] = col
    if (out == g).all():
        return None
    return g, out


def p_gravity_paint_floor(rng):
    return {"paint": int(_colors(rng, 1)[0])}


def r_gravity_paint_floor(rng, p):
    """Gravedad y además la fila inferior resultante se repinta al color de la tarea."""
    res = r_gravity(rng, {})
    if res is None:
        return None
    g, out = res
    if (g == p["paint"]).any():
        return None
    out = out.copy()
    out[-1, out[-1] > 0] = p["paint"]
    return g, out


def p_left_align(rng):
    return {}


def r_left_align(rng, p):
    """Cada fila se compacta a la izquierda (orden preservado)."""
    g = _empty(rng, 5, 11)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 3), max_side=3, solid=False)
    if not g.any():
        return None
    out = np.zeros_like(g)
    for r in range(g.shape[0]):
        row = g[r][g[r] > 0]
        out[r, :len(row)] = row
    if (out == g).all():
        return None
    return g, out


def p_ray_cross(rng):
    return {}


def r_ray_cross(rng, p):
    """Desde la única celda-semilla salen rayos en cruz hasta los bordes (su color)."""
    g = _empty(rng, 6, 12)
    noise_col, seed_col = _colors(rng, 2)
    for _ in range(rng.integers(2, 5)):
        g[rng.integers(g.shape[0]), rng.integers(g.shape[1])] = noise_col
    r, c = rng.integers(1, g.shape[0] - 1), rng.integers(1, g.shape[1] - 1)
    g[r, c] = seed_col
    if (g == seed_col).sum() != 1:
        return None
    out = g.copy()
    out[r, :], out[:, c] = np.where(g[r, :] == 0, seed_col, g[r, :]), \
        np.where(g[:, c] == 0, seed_col, g[:, c])
    out[r, c] = seed_col
    return g, out


def p_connect_row_pairs(rng):
    return {}


def r_connect_row_pairs(rng, p):
    """Dos puntos del mismo color en una fila se conectan con una línea de ese color."""
    g = _empty(rng, 6, 12)
    cols = _colors(rng, 3)
    used_rows = set()
    n_pairs = 0
    for col in cols[:rng.integers(2, 4)]:
        r = int(rng.integers(g.shape[0]))
        if r in used_rows:
            continue
        used_rows.add(r)
        c1, c2 = sorted(rng.choice(g.shape[1], size=2, replace=False))
        if c2 - c1 < 2:
            continue
        g[r, c1] = g[r, c2] = col
        n_pairs += 1
    if n_pairs < 2:
        return None
    out = g.copy()
    for r in range(g.shape[0]):
        row = np.where(g[r] > 0)[0]
        if len(row) == 2 and g[r, row[0]] == g[r, row[1]]:
            out[r, row[0]:row[1] + 1] = g[r, row[0]]
    if (out == g).all():
        return None
    return g, out


def p_biggest_row_tile(rng):
    return {}


def r_biggest_row_tile(rng, p):
    """La fila con MÁS celdas no-cero se repite en todas las filas de la salida."""
    g = _empty(rng, 5, 10)
    _place_objects(rng, g, rng.integers(3, 6), _colors(rng, 3), max_side=3, solid=False)
    counts = (g > 0).sum(axis=1)
    if counts.max() == 0 or (counts == counts.max()).sum() > 1:
        return None
    row = g[int(counts.argmax())]
    return g, np.tile(row, (g.shape[0], 1))


def p_diag_shadow(rng):
    return {"shadow": int(_colors(rng, 1)[0])}


def r_diag_shadow(rng, p):
    """Cada celda no-cero proyecta una 'sombra' del color de tarea en su diagonal abajo-derecha."""
    g = _empty(rng, 6, 12)
    _place_objects(rng, g, rng.integers(2, 5), _colors(rng, 2, exclude=(p["shadow"],)),
                   max_side=2)
    if not g.any():
        return None
    out = g.copy()
    rs, cs = np.where(g > 0)
    changed = False
    for r, c in zip(rs, cs):
        if r + 1 < g.shape[0] and c + 1 < g.shape[1] and g[r + 1, c + 1] == 0:
            out[r + 1, c + 1] = p["shadow"]
            changed = True
    if not changed:
        return None
    return g, out


def p_recolor_by_size(rng):
    a, b = _colors(rng, 2)
    return {"big": int(a), "small": int(b)}


def r_recolor_by_size(rng, p):
    """Objetos de >=4 celdas → color A; de <4 → color B (colores fijos de la tarea)."""
    g = _empty(rng, 7, 13)
    src = _colors(rng, 2, exclude=(p["big"], p["small"]))
    _place_objects(rng, g, rng.integers(3, 6), src, max_side=3)
    obs = _objects(g)
    if not obs or not any(len(c) >= 4 for _, c in obs) or not any(len(c) < 4 for _, c in obs):
        return None
    out = np.zeros_like(g)
    for col, cells in obs:
        tgt = p["big"] if len(cells) >= 4 else p["small"]
        for r, c in cells:
            out[r, c] = tgt
    return g, out


def p_dup_taller(rng):
    return {}


def r_dup_taller(rng, p):
    """El grid se duplica hacia abajo, la copia inferior espejada verticalmente."""
    g = _empty(rng, 4, 9)
    _place_objects(rng, g, rng.integers(2, 5), _colors(rng, 3), max_side=3, solid=False)
    if not g.any() or g.shape[0] * 2 > 30:
        return None
    return g, np.concatenate([g, np.flipud(g)], axis=0)


RULES = {
    "swap_extremes": (p_swap_extremes, r_swap_extremes),
    "delete_rare": (p_delete_rare, r_delete_rare),
    "keep_largest": (p_keep_largest, r_keep_largest),
    "mirror_overlay": (p_mirror_overlay, r_mirror_overlay),
    "transpose_swap": (p_transpose_swap, r_transpose_swap),
    "border_rarest": (p_border_rarest, r_border_rarest),
    "count_square": (p_count_square, r_count_square),
    "row_bars": (p_row_bars, r_row_bars),
    "tile_by_colors": (p_tile_by_colors, r_tile_by_colors),
    "percol_column": (p_percol_column, r_percol_column),
    "checker_topcolors": (p_checker_topcolors, r_checker_topcolors),
    "crop_scale": (p_crop_scale, r_crop_scale),
    "frame_interior": (p_frame_interior, r_frame_interior),
    "marker_quadrant": (p_marker_quadrant, r_marker_quadrant),
    "densest_quadrant": (p_densest_quadrant, r_densest_quadrant),
    "unique_corners": (p_unique_corners, r_unique_corners),
    "mask_and": (p_mask_and, r_mask_and),
    "mask_xor": (p_mask_xor, r_mask_xor),
    "mirror_repair": (p_mirror_repair, r_mirror_repair),
    "periodic_rows": (p_periodic_rows, r_periodic_rows),
    "symmetrize180": (p_symmetrize180, r_symmetrize180),
    "fill_zeros_majority": (p_fill_zeros_majority, r_fill_zeros_majority),
    "outline_objects": (p_outline_objects, r_outline_objects),
    "remove_isolated": (p_remove_isolated, r_remove_isolated),
    "gravity": (p_gravity, r_gravity),
    "gravity_paint_floor": (p_gravity_paint_floor, r_gravity_paint_floor),
    "left_align": (p_left_align, r_left_align),
    "ray_cross": (p_ray_cross, r_ray_cross),
    "connect_row_pairs": (p_connect_row_pairs, r_connect_row_pairs),
    "biggest_row_tile": (p_biggest_row_tile, r_biggest_row_tile),
    "diag_shadow": (p_diag_shadow, r_diag_shadow),
    "recolor_by_size": (p_recolor_by_size, r_recolor_by_size),
    "dup_taller": (p_dup_taller, r_dup_taller),
}


def _ok_grid(g):
    return (isinstance(g, np.ndarray) and g.ndim == 2 and 1 <= g.shape[0] <= 30
            and 1 <= g.shape[1] <= 30 and ((g >= 0) & (g <= 9)).all())


def build_task(rng, rule=None, n_pairs=4, max_tries=80):
    """Una tarea: regla + params fijos, n_pairs instancias frescas, todas verificadas."""
    name = rule or list(RULES)[int(rng.integers(len(RULES)))]
    make_p, fn = RULES[name]
    params = make_p(rng)
    pairs, seen = [], set()
    for _ in range(max_tries):
        if len(pairs) == n_pairs:
            break
        res = fn(rng, params)
        if res is None:
            continue
        gi, go = res
        if not (_ok_grid(gi) and _ok_grid(go)) or not gi.any():
            continue
        key = gi.tobytes() + go.tobytes()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((gi.tolist(), go.tolist()))
    if len(pairs) < n_pairs:
        return None
    return {"rule": name, "pairs": pairs}
