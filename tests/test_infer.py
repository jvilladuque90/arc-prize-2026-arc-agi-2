"""Tests de arc.infer / arc.augment sin modelo real (predict_fn falso)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.augment import augment_grid, color_permutation, invert_grid
from arc.data import Task, TaskPair
from arc.grid import format_task_prompt, grid_to_text, text_to_grid
from arc.infer import predict_task, predict_test_input, rank_keys, two_attempts


def test_augment_roundtrip_all_d4_and_colors():
    grid = [[0, 1, 2], [3, 4, 5]]
    for d4 in range(8):
        cmap = color_permutation(seed=d4 + 1, keep_zero=True)
        aug = augment_grid(grid, d4, cmap)
        assert invert_grid(aug, d4, cmap) == grid


def test_grid_text_roundtrip():
    grid = [[0, 9, 3], [1, 2, 8]]
    assert text_to_grid(grid_to_text(grid)) == grid


def test_rank_keys_score_dominates_votes():
    scores = {"a": -5.0, "b": -1.0, "c": -3.0}
    votes = {"a": 10, "b": 1, "c": 5}
    assert rank_keys(scores, votes) == ["b", "c", "a"]


def test_rank_keys_votes_break_ties():
    scores = {"a": -2.0, "b": -2.0}
    votes = {"a": 1, "b": 3}
    assert rank_keys(scores, votes)[0] == "b"


def _echo_task() -> Task:
    # tarea identidad: output == input
    g1, g2 = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    return Task("echo", train=[TaskPair(g1, g1), TaskPair(g2, g2)],
                test=[TaskPair([[2, 3], [4, 5]], None)])


def test_predict_test_input_vote_deaugments_correctly():
    """Un predict_fn que resuelve la tarea identidad en el espacio augmentado debe
    producir, tras des-augmentar, el mismo grid original en todas las vistas."""
    task = _echo_task()

    def predict_fn(prompt: str):
        # el test input es el último bloque "Input:" del prompt
        block = prompt.split("Input:\n")[-1].split("\nOutput:")[0]
        return text_to_grid(block)

    cands = predict_test_input(task, 0, predict_fn, n_aug=8)
    assert cands[0] == task.test[0].input  # consenso perfecto entre vistas


def test_predict_task_fallback_when_no_candidates():
    task = _echo_task()
    results = predict_task(task, lambda p: None, n_aug=4)
    assert results[0]["attempt_1"] == task.test[0].input  # fallback = identidad


def test_two_attempts_distinct_when_possible():
    c1, c2 = [[1]], [[2]]
    att = two_attempts([c1, c2], fallback=[[0]])
    assert att["attempt_1"] == c1 and att["attempt_2"] == c2
    # con 1 solo candidato, attempt_2 aprovecha el fallback (no duplica)
    att = two_attempts([c1], fallback=[[0]])
    assert att["attempt_1"] == c1 and att["attempt_2"] == [[0]]
    att = two_attempts([c1], fallback=c1)
    assert att["attempt_2"] == c1  # fallback igual al candidato -> duplica
    att = two_attempts([], fallback=[[0]])
    assert att["attempt_1"] == [[0]] and att["attempt_2"] == [[0]]


def test_format_task_prompt_contains_examples_and_test():
    task = _echo_task()
    p = format_task_prompt(task.train, task.test[0].input)
    assert p.count("Example") == 2 and p.rstrip().endswith("Output:")
