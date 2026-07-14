"""Selección de attempt_1/attempt_2 a partir de candidatos decodificados.

Port fiel de los scorers de LB33 (`arc_decoder.py` del fork NVARC) a código puro y
testeable, más variantes propias para benchmarkear contra el held-out limpio (ver
docs/STRATEGY.md → "benchmark de selección"). NO depende de torch/GPU.

Cada *candidato* (guess) es un dict con las claves que produce el decodificador de LB33:
  - "solution":   grid (lista de listas de int) — la salida propuesta.
  - "beam_score": NLL del beam de decodificación (menor = más probable).
  - "score_aug":  lista de NLL bajo K vistas augmentadas (menor = más consistente).

`rank_solutions(candidates, getter)` agrupa candidatos por contenido del grid (dedup),
puntúa cada solución única agregando sobre los candidatos que la produjeron, y devuelve las
soluciones ordenadas de mejor a peor. `two_attempts` toma las 2 mejores distintas.
"""
from __future__ import annotations

from typing import Callable, Dict, List

Grid = List[List[int]]
Candidate = Dict


def _hashable(grid: Grid):
    return tuple(map(tuple, grid))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --- Getters: puntúan una solución única dada la lista de candidatos que la produjeron ---
# (mayor score = mejor; rank_solutions ordena descendente)

def getter_kgmon(guesses: List[Candidate]) -> float:
    """LB33 `score_kgmon`: nº de votos menos la NLL augmentada media (votos + verosimilitud)."""
    inf_score = len(guesses)
    aug_score = _mean([_mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def getter_full_probmul_3(guesses: List[Candidate], baseline: float = 3.0) -> float:
    """LB33 `score_full_probmul_3`: suma de (baseline - beam) + media de sum(baseline - aug)."""
    inf_score = sum(baseline - g["beam_score"] for g in guesses)
    aug_score = _mean([sum(baseline - s for s in g["score_aug"]) for g in guesses])
    return inf_score + aug_score


# --- Variantes propias a benchmarkear (no en LB33) ---

def getter_kgmon_minaug(guesses: List[Candidate]) -> float:
    """Como kgmon pero usa la MEJOR vista augmentada (min NLL) en vez de la media:
    premia la solución que alguna augmentación considera muy probable."""
    inf_score = len(guesses)
    aug_score = _mean([min(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def getter_votes_only(guesses: List[Candidate]) -> float:
    """Solo consenso de decodificación (nº de candidatos que dieron esta solución)."""
    return float(len(guesses))


GETTERS: Dict[str, Callable[[List[Candidate]], float]] = {
    "kgmon": getter_kgmon,
    "full_probmul_3": getter_full_probmul_3,
    "kgmon_minaug": getter_kgmon_minaug,
    "votes_only": getter_votes_only,
}


def rank_solutions(candidates: List[Candidate],
                   getter: Callable[[List[Candidate]], float] = getter_kgmon) -> List[Grid]:
    """Agrupa por contenido del grid, puntúa con `getter`, devuelve soluciones desc por score.

    Desempate estable por representación del grid (no por orden de inserción) para
    reproducibilidad entre procesos.
    """
    groups: Dict[tuple, List[Candidate]] = {}
    grid_of: Dict[tuple, Grid] = {}
    for g in candidates:
        h = _hashable(g["solution"])
        groups.setdefault(h, []).append(g)
        grid_of[h] = g["solution"]
    scored = [(getter(gs), h) for h, gs in groups.items()]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [grid_of[h] for _, h in scored]


def two_attempts(candidates: List[Candidate], fallback: Grid,
                 getter: Callable[[List[Candidate]], float] = getter_kgmon) -> dict:
    """{attempt_1, attempt_2} = las 2 mejores soluciones distintas; `fallback` si faltan.

    pass@2 regala un 2º intento: no se duplica attempt_1 si hay un fallback distinto."""
    ranked = rank_solutions(candidates, getter)
    a1 = ranked[0] if ranked else fallback
    if len(ranked) > 1:
        a2 = ranked[1]
    else:
        a2 = fallback if fallback != a1 else a1
    return {"attempt_1": a1, "attempt_2": a2}
