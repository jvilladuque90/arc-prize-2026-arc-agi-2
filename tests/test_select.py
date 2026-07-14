"""Tests de arc.select: dedup, orden, paridad con la matemática de LB33."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.select import (GETTERS, getter_full_probmul_3, getter_kgmon,
                        rank_solutions, two_attempts)


def _cand(solution, beam, aug):
    return {"solution": solution, "beam_score": beam, "score_aug": aug}


def test_rank_dedups_by_content_and_counts_votes():
    g = [[1, 2], [3, 4]]
    other = [[5, 6], [7, 8]]
    cands = [
        _cand(g, 0.5, [0.1, 0.2]),
        _cand(g, 0.6, [0.15, 0.25]),   # misma solución (voto extra)
        _cand(other, 0.4, [0.1, 0.1]),
    ]
    ranked = rank_solutions(cands, getter=lambda gs: len(gs))  # votes_only
    assert ranked[0] == g          # 2 votos gana
    assert ranked[1] == other
    assert len(ranked) == 2        # deduplicado


def test_kgmon_matches_reference_numpy():
    """El port debe coincidir con la implementación numpy original de LB33."""
    np = __import__("numpy")

    def ref_kgmon(guesses):
        inf = len(guesses)
        aug = np.mean([np.mean(g["score_aug"]) for g in guesses])
        return inf - aug

    def ref_probmul(guesses, baseline=3):
        inf = np.sum([baseline - g["beam_score"] for g in guesses])
        aug = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses])
        return inf + aug

    rng = random.Random(0)
    for _ in range(50):
        guesses = [
            {"solution": [[0]], "beam_score": rng.uniform(-2, 4),
             "score_aug": [rng.uniform(-1, 3) for _ in range(rng.randint(1, 8))]}
            for _ in range(rng.randint(1, 5))
        ]
        assert abs(getter_kgmon(guesses) - ref_kgmon(guesses)) < 1e-9
        assert abs(getter_full_probmul_3(guesses) - ref_probmul(guesses)) < 1e-9


def test_rank_is_deterministic():
    cands = [
        _cand([[1]], 0.5, [0.5]),
        _cand([[2]], 0.5, [0.5]),  # empate exacto de score
        _cand([[3]], 0.4, [0.4]),
    ]
    r1 = rank_solutions(cands, getter=getter_kgmon)
    r2 = rank_solutions(cands, getter=getter_kgmon)
    assert r1 == r2  # desempate estable


def test_two_attempts_distinct_and_fallback():
    c1 = _cand([[1]], 0.1, [0.1])
    c2 = _cand([[2]], 0.9, [0.9])  # peor
    att = two_attempts([c1, c2], fallback=[[0]], getter=getter_kgmon)
    assert att["attempt_1"] == [[1]] and att["attempt_2"] == [[2]]
    # un solo candidato -> attempt_2 usa el fallback distinto
    att = two_attempts([c1], fallback=[[0]], getter=getter_kgmon)
    assert att["attempt_1"] == [[1]] and att["attempt_2"] == [[0]]
    # sin candidatos -> fallback en ambos
    att = two_attempts([], fallback=[[7]], getter=getter_kgmon)
    assert att["attempt_1"] == [[7]] and att["attempt_2"] == [[7]]


def test_all_registered_getters_run():
    cands = [_cand([[1, 1]], 0.3, [0.2, 0.4]), _cand([[1, 1]], 0.5, [0.1, 0.6])]
    for name, getter in GETTERS.items():
        ranked = rank_solutions(cands, getter=getter)
        assert ranked and ranked[0] == [[1, 1]], name
