"""Fase B — data-gen compositiva para SFT de inducción (ARC-AGI-2).

Los seeds de BARC (`vendor/barc/seeds/*.py`) son GENERADORES EJECUTABLES: cada uno trae
`generate_input()` (crea un input aleatorio) y `main(input_grid)` (el solver). Ejecutarlos da
pares (input, output) reales.

AGI-2 exige tareas más COMPOSITIVAS (profundidad ~2.7 pasos vs 1.3 de AGI-1). Este módulo
**compone** dos seeds: input ← genA(), intermedio ← mainA(input), output ← mainB(intermedio).
El programa objetivo del SFT es la composición (transformA luego transformB), que se emite como
código `transform` reusando los cuerpos de ambos seeds.

Sin API/LLM: todo es ejecución + validación local.
"""
from __future__ import annotations

import random
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = ROOT / "vendor" / "barc" / "seeds"
COMMON_PATH = ROOT / "vendor" / "barc" / "common.py"

# marcador que separa el código "de prompting" del scaffolding de visualización
CUT_MARKER = "# ============= remove below this point for prompting ============="


@dataclass
class Seed:
    """Un seed de BARC parseado: concepto, descripción y código ejecutable."""

    name: str
    concepts: list[str]
    description: str
    source: str  # código completo (sin el scaffolding de visualización)
    ok: bool = False  # ejecuta y produce pares válidos
    err: str = ""
    samples: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def _strip_scaffold(src: str) -> str:
    """Quita el bloque de visualización (todo tras el marcador o el __main__)."""
    if CUT_MARKER in src:
        src = src.split(CUT_MARKER)[0]
    else:
        src = re.split(r"\nif __name__\s*==", src)[0]
    return src.rstrip() + "\n"


def _extract_block(src: str, header: str) -> str:
    """Extrae el bloque de comentarios que sigue a `# header:` (concepts/description)."""
    # [ \t]* (no \s*) para NO cruzar lineas en blanco: separa concepts de description
    m = re.search(rf"#[ \t]*{header}:[ \t]*\n((?:[ \t]*#.*\n)+)", src)
    if not m:
        return ""
    lines = [re.sub(r"^\s*#\s?", "", ln) for ln in m.group(1).splitlines()]
    return " ".join(x.strip() for x in lines if x.strip()).strip()


def parse_seed(path: Path) -> Seed:
    src = _strip_scaffold(path.read_text(encoding="utf-8", errors="replace"))
    concepts_raw = _extract_block(src, "concepts")
    concepts = [c.strip() for c in concepts_raw.split(",") if c.strip()]
    return Seed(name=path.stem, concepts=concepts,
                description=_extract_block(src, "description"), source=src)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # pragma: no cover - señal
    raise _Timeout()


class time_limit:
    """Timeout por ejecución (SIGALRM en POSIX; no-op en Windows)."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.ok = hasattr(signal, "SIGALRM")

    def __enter__(self):
        if self.ok:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(self.seconds)

    def __exit__(self, *exc):
        if self.ok:
            signal.alarm(0)
        return False


def load_common() -> str:
    return COMMON_PATH.read_text(encoding="utf-8", errors="replace")


def install_common(common_src: str) -> None:
    """Registra `common` como módulo importable: los seeds hacen `from common import *`."""
    import sys
    import types

    if "common" in sys.modules:
        return
    mod = types.ModuleType("common")
    mod.__dict__["__name__"] = "common"
    exec(common_src, mod.__dict__)
    sys.modules["common"] = mod


def exec_seed(seed: Seed, common_src: str, timeout: int = 5) -> dict[str, Any]:
    """Ejecuta el código del seed y devuelve su namespace (con main/generate_input)."""
    install_common(common_src)
    ns: dict[str, Any] = {}
    with time_limit(timeout):
        exec(common_src, ns)
        exec(seed.source, ns)
    return ns


def valid_grid(g: Any) -> bool:
    if not isinstance(g, np.ndarray) or g.ndim != 2 or g.size == 0:
        return False
    if g.shape[0] > 30 or g.shape[1] > 30:
        return False
    return bool(np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9)


def try_seed(seed: Seed, common_src: str, n: int = 3, timeout: int = 5) -> Seed:
    """Valida que el seed genera pares (input, output) usables."""
    try:
        ns = exec_seed(seed, common_src, timeout)
        gen, main = ns.get("generate_input"), ns.get("main")
        if not callable(gen) or not callable(main):
            seed.err = "sin generate_input/main"
            return seed
        for _ in range(n):
            with time_limit(timeout):
                gi = np.array(gen())
                go = np.array(main(np.copy(gi)))
            if valid_grid(gi) and valid_grid(go):
                seed.samples.append((gi, go))
        seed.ok = len(seed.samples) >= 2
        if not seed.ok and not seed.err:
            seed.err = f"solo {len(seed.samples)} pares validos"
    except BaseException as e:  # noqa: BLE001 - los seeds pueden fallar de cualquier forma
        seed.err = f"{type(e).__name__}: {str(e)[:90]}"
    return seed


def load_seeds(limit: int | None = None) -> list[Seed]:
    paths = sorted(SEEDS_DIR.glob("*.py"))
    if limit:
        paths = paths[:limit]
    return [parse_seed(p) for p in paths]


# ---------------------------------------------------------------- composición

def compose_task(a: Seed, b: Seed, common_src: str, n_pairs: int = 4,
                 timeout: int = 5, seed_val: int | None = None) -> dict | None:
    """Compone A→B: input ← genA(), x ← mainA(input), output ← mainB(x).

    Devuelve una tarea {train:[{input,output}...], test:[...]} + el programa compuesto,
    o None si la composición no es válida (crashea, degenera o es trivial).
    """
    if seed_val is not None:
        random.seed(seed_val)
        np.random.seed(seed_val % (2**32))
    try:
        ns_a = exec_seed(a, common_src, timeout)
        ns_b = exec_seed(b, common_src, timeout)
    except BaseException:
        return None
    gen_a, main_a, main_b = ns_a.get("generate_input"), ns_a.get("main"), ns_b.get("main")
    if not all(callable(f) for f in (gen_a, main_a, main_b)):
        return None

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    n_noop_a = n_noop_b = 0
    for _ in range(n_pairs * 4):  # margen: muchas ejecuciones fallan
        if len(pairs) >= n_pairs + 1:
            break
        try:
            with time_limit(timeout):
                gi = np.array(gen_a())
                if not valid_grid(gi):
                    continue
                mid = np.array(main_a(np.copy(gi)))
                if not valid_grid(mid):
                    continue
                go = np.array(main_b(np.copy(mid)))
            if not valid_grid(go):
                continue
            if np.array_equal(gi, go):  # identidad: tarea trivial
                continue
            if len(np.unique(go)) < 2:  # output degenerado (un solo color)
                continue
            # cada paso debe HACER algo (si no, la tarea no es realmente compositiva)
            n_noop_a += int(np.array_equal(gi, mid))
            n_noop_b += int(np.array_equal(mid, go))
            pairs.append((gi, go))
        except BaseException:
            continue

    if len(pairs) < n_pairs + 1:
        return None
    # ambos pasos deben aportar en la mayoria de los pares (si no, es una tarea de 1 paso)
    if n_noop_a > len(pairs) // 2 or n_noop_b > len(pairs) // 2:
        return None
    # no-degenerada: los outputs no pueden ser todos idénticos entre sí
    if all(np.array_equal(pairs[0][1], p[1]) for p in pairs[1:]):
        return None

    train = [{"input": i.tolist(), "output": o.tolist()} for i, o in pairs[:n_pairs]]
    test = [{"input": pairs[n_pairs][0].tolist(), "output": pairs[n_pairs][1].tolist()}]
    return {
        "task": {"train": train, "test": test},
        "program": compose_program(a, b),
        "seeds": [a.name, b.name],
        "concepts": sorted(set(a.concepts) | set(b.concepts)),
        "description": f"First, {a.description.rstrip('.')}. Then, {b.description.rstrip('.')}.",
    }


def _rename_main(src: str, new_name: str) -> str:
    """Renombra `def main(` -> `def {new_name}(` (y sus llamadas internas)."""
    src = re.sub(r"\bdef main\s*\(", f"def {new_name}(", src)
    return re.sub(r"(?<![\w.])main\s*\(", f"{new_name}(", src)


def _body_without_common_imports(src: str) -> str:
    """Quita los imports del preámbulo (common/numpy/typing) para poder concatenar dos seeds."""
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith(("from common import", "import numpy", "from typing import",
                         "import random")):
            continue
        out.append(ln)
    return "\n".join(out).strip("\n")


def compose_program(a: Seed, b: Seed) -> str:
    """Emite el programa compuesto: `transform` = mainB(mainA(x)), con ambos cuerpos."""
    body_a = _rename_main(_body_without_common_imports(a.source), "_step_a")
    body_b = _rename_main(_body_without_common_imports(b.source), "_step_b")
    concepts = ", ".join(sorted(set(a.concepts) | set(b.concepts)))
    desc = f"First, {a.description.rstrip('.')}. Then, {b.description.rstrip('.')}."
    return (
        "from common import *\n\nimport numpy as np\nfrom typing import *\n\n"
        f"# concepts:\n# {concepts}\n\n"
        f"# description:\n# {desc}\n\n"
        f"{body_a}\n\n{body_b}\n\n"
        "def transform(input_grid):\n"
        "    # apply the two transformations in sequence\n"
        "    intermediate = _step_a(np.copy(input_grid))\n"
        "    return _step_b(np.array(intermediate))\n"
    )


# ------------------------------------------------- formato de training (BARC chat)

COLOR = {0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
         5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"}

SYSTEM_MSG = ("You are a world-class puzzle solver with exceptional pattern recognition skills "
              "and expertise in Python programming. Your task is to analyze puzzles and provide "
              "Python solutions.")


def grid_to_text(grid) -> str:
    return "\n".join(" ".join(COLOR[int(c)] for c in row) for row in grid) + "\n"


def build_user_msg(task: dict) -> str:
    """Mismo formato que BARC usa en training e inferencia (ver docs/PROGRAM_SYNTHESIS.md)."""
    ex = ""
    for i, p in enumerate(task["train"]):
        ex += (f"Example {i+1}\nInput:\n{grid_to_text(p['input'])}\n"
               f"Output:\n{grid_to_text(p['output'])}\n\n")
    test_in = grid_to_text(task["test"][0]["input"])
    return ("Given input-output grid pairs as reference examples, carefully observe the patterns "
            "to predict the output grid for new test input. Each pair follows the same "
            "transformation rule. Grids are 2D arrays represented as strings, with cells (colors) "
            "separated by spaces and rows by newlines.\n"
            "Here are the input and output grids for the reference examples:\n"
            f"{ex}Here is the input grid for the test example:\n{test_in}\n"
            "Write a Python function `transform` that can convert any given input grid to its "
            "corresponding output grid based on the pattern observed in the reference examples.")


def build_assistant_msg(program: str) -> str:
    return ("Let's solve this puzzle using Python code with the common library functions. We'll "
            "first reason about the problem and then write the code to solve it. The `transform` "
            "function will take the input grid and return the output grid. Here is the Python code "
            "with the comments describing how to solve the problem:\n"
            f"```python\n{program}```\n")


def to_training_example(composed: dict) -> dict:
    """Convierte una tarea compuesta al formato chat `messages` del SFT de BARC."""
    return {"messages": [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_msg(composed["task"])},
        {"role": "assistant", "content": build_assistant_msg(composed["program"])},
    ]}


def verify_program(program: str, task: dict, common_src: str, timeout: int = 5) -> bool:
    """El programa compuesto debe reproducir EXACTAMENTE todos los pares (train y test)."""
    install_common(common_src)
    ns: dict[str, Any] = {}
    try:
        with time_limit(timeout):
            exec(common_src, ns)
            exec(program, ns)
        fn = ns.get("transform")
        if not callable(fn):
            return False
        for pair in task["train"] + task["test"]:
            with time_limit(timeout):
                got = np.array(fn(np.array(pair["input"])))
            if not np.array_equal(got, np.array(pair["output"])):
                return False
        return True
    except BaseException:
        return False
