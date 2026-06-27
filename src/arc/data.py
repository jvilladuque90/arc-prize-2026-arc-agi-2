"""Carga y representación de tareas ARC.

Formato de los JSON de la competencia:
    challenges: {task_id: {"train": [{"input": grid, "output": grid}, ...],
                            "test":  [{"input": grid}, ...]}}
    solutions:  {task_id: [grid, ...]}   # un grid por cada test input

Un `grid` es una lista de listas de enteros 0-9 (colores).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

Grid = List[List[int]]


@dataclass
class TaskPair:
    """Un par entrada/salida. `output` es None para los test inputs sin solución."""

    input: Grid
    output: Optional[Grid] = None


@dataclass
class Task:
    task_id: str
    train: List[TaskPair]
    test: List[TaskPair] = field(default_factory=list)


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_challenges(path: str | Path) -> Dict[str, dict]:
    return _read_json(path)


def load_solutions(path: str | Path) -> Dict[str, List[Grid]]:
    return _read_json(path)


def load_dataset(
    challenges_path: str | Path,
    solutions_path: Optional[str | Path] = None,
) -> Dict[str, Task]:
    """Construye {task_id: Task}, asociando soluciones a los test si se proveen."""
    challenges = load_challenges(challenges_path)
    solutions = load_solutions(solutions_path) if solutions_path else {}

    tasks: Dict[str, Task] = {}
    for tid, payload in challenges.items():
        train = [TaskPair(p["input"], p["output"]) for p in payload["train"]]
        sols = solutions.get(tid)
        test = []
        for i, p in enumerate(payload["test"]):
            out = sols[i] if sols and i < len(sols) else None
            test.append(TaskPair(p["input"], out))
        tasks[tid] = Task(task_id=tid, train=train, test=test)
    return tasks


# Rutas por defecto relativas a la raíz del repo.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DEFAULT_FILES = {
    "training": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
    "evaluation": ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    "test": ("arc-agi_test_challenges.json", None),
}


def load_split(split: str, data_dir: str | Path = DATA_DIR) -> Dict[str, Task]:
    """Carga un split estándar: 'training' | 'evaluation' | 'test'."""
    data_dir = Path(data_dir)
    ch, sol = DEFAULT_FILES[split]
    sol_path = data_dir / sol if sol else None
    return load_dataset(data_dir / ch, sol_path)
