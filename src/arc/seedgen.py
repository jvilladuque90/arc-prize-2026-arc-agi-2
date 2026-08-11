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


# El modelo BARC fue entrenado con el vocabulario de colores RENOMBRADO (su gen_dataset
# aplica Teal→Purple, Maroon→Brown, Grey→Gray en código y descripciones). Los seeds del
# repo usan los nombres viejos; renombramos al cargar para casar la distribución del modelo.
_COLOR_RENAMES = (("teal", "purple"), ("grey", "gray"), ("maroon", "brown"))


def _rename_colors(src: str) -> str:
    # fronteras de LETRA (no \b): asi "teal_pixel" -> "purple_pixel", igual que el
    # color_deterministic de BARC (su regex usa [^a-zA-Z] como frontera).
    for old, new in _COLOR_RENAMES:
        for o, n in ((old, new), (old.capitalize(), new.capitalize()),
                     (old.upper(), new.upper())):
            src = re.sub(rf"(?<![a-zA-Z]){o}(?![a-zA-Z])", n, src)
    return src


def parse_seed(path: Path) -> Seed:
    src = _rename_colors(_strip_scaffold(path.read_text(encoding="utf-8", errors="replace")))
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


def call_with_timeout(fn, args=(), timeout: int = 5):
    """Ejecuta fn(*args) con timeout PORTABLE.

    POSIX (Kaggle): SIGALRM — barato y mata de verdad la ejecución.
    Windows (desarrollo local): thread daemon abandonado si excede — el runaway queda
    quemando CPU hasta que muera solo, pero el flujo principal continúa. Necesario porque
    algunos seeds tienen bucles de reintento que no terminan con inputs fuera de su
    distribución (p.ej. el chequeo de visibilidad ejecuta main_B sobre inputs crudos)."""
    if hasattr(signal, "SIGALRM"):
        with time_limit(timeout):
            return fn(*args)
    import threading

    box: dict[str, Any] = {}

    def _run():
        try:
            box["v"] = fn(*args)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise _Timeout()
    if "e" in box:
        raise box["e"]
    return box.get("v")


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
            gi = np.array(call_with_timeout(gen, timeout=timeout))
            go = np.array(call_with_timeout(main, (np.copy(gi),), timeout))
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

# --- v2: compatibilidad semántica ---------------------------------------------
# Lección del A/B v1 (0/16): encadenar dos seeds ALEATORIAS produce tareas correctas pero
# no ARC-like — sin una regla coherente inferible. v2 solo compone pares que (a) comparten
# temática, o (b) cuyo segundo paso es una operación GLOBAL (geometría/color) que mantiene
# visible el efecto del primero (la regla compuesta "haz A y luego rota/recolorea" sí es
# inferible por un humano).

_CATS = {
    "geometry": ("symmetry", "rotation", "reflection", "mirror", "translation", "scaling",
                 "transpose", "flip"),
    "color": ("color",),
    "objects": ("object", "sprite", "shape", "attraction", "collision"),
    "pattern": ("pattern", "repetition", "tiling", "fractal", "repeating"),
    "draw": ("line", "fill", "draw", "border", "frame", "growing"),
    "topology": ("topology", "connect", "contain", "inside", "boundary", "hole",
                 "occlusion"),
    "count": ("counting", "count", "size"),
}

# categorías cuyo efecto como SEGUNDO paso es global y no destruye la evidencia del primero
_GLOBAL_SECOND = {"geometry", "color"}


def seed_categories(seed: Seed) -> set[str]:
    cats: set[str] = set()
    text = " ".join(seed.concepts).lower()
    for cat, kws in _CATS.items():
        if any(kw in text for kw in kws):
            cats.add(cat)
    return cats


def compatible(a: Seed, b: Seed) -> bool:
    """¿La composición A→B tiene posibilidades de ser una tarea coherente/inferible?"""
    ca, cb = seed_categories(a), seed_categories(b)
    if cb & _GLOBAL_SECOND:                      # 2º paso global → siempre inferible
        return True
    if ca & cb:                                  # temática compartida
        return True
    return bool(set(c.lower() for c in a.concepts) & set(c.lower() for c in b.concepts))


def compose_task(a: Seed, b: Seed, common_src: str, n_pairs: int = 4,
                 timeout: int = 5, seed_val: int | None = None,
                 require_compatible: bool = True) -> dict | None:
    """Compone A→B: input ← genA(), x ← mainA(input), output ← mainB(x).

    Devuelve una tarea {train:[{input,output}...], test:[...]} + el programa compuesto,
    o None si la composición no es válida (crashea, degenera, es trivial, incompatible
    semánticamente, o el efecto de A no sobrevive a B).
    """
    if require_compatible and not compatible(a, b):
        return None
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
    n_noop_a = n_noop_b = n_invisible_a = 0
    for _ in range(n_pairs * 4):  # margen: muchas ejecuciones fallan
        if len(pairs) >= n_pairs + 1:
            break
        try:
            gi = np.array(call_with_timeout(gen_a, timeout=timeout))
            if not valid_grid(gi):
                continue
            mid = np.array(call_with_timeout(main_a, (np.copy(gi),), timeout))
            if not valid_grid(mid):
                continue
            go = np.array(call_with_timeout(main_b, (np.copy(mid),), timeout))
            if not valid_grid(go):
                continue
            if np.array_equal(gi, go):  # identidad: tarea trivial
                continue
            if len(np.unique(go)) < 2:  # output degenerado (un solo color)
                continue
            # cada paso debe HACER algo (si no, la tarea no es realmente compositiva)
            n_noop_a += int(np.array_equal(gi, mid))
            n_noop_b += int(np.array_equal(mid, go))
            # v2 — VISIBILIDAD: el efecto de A debe sobrevivir A TRAVÉS de B. Si B(input)
            # da lo mismo que B(A(input)), el paso A es invisible en el output y la tarea
            # colapsa a una de 1 paso (no compositiva de verdad). B(input) puede crashear
            # (la distribución de A difiere de la de gen_B) — eso cuenta como "visible".
            try:
                direct_b = np.array(call_with_timeout(main_b, (np.copy(gi),), timeout))
                if valid_grid(direct_b) and np.array_equal(direct_b, go):
                    n_invisible_a += 1
            except BaseException:
                pass
            pairs.append((gi, go))
        except BaseException:
            continue

    if len(pairs) < n_pairs + 1:
        return None
    # ambos pasos deben aportar en la mayoria de los pares (si no, es una tarea de 1 paso)
    if n_noop_a > len(pairs) // 2 or n_noop_b > len(pairs) // 2:
        return None
    # v2: el efecto de A debe ser visible en el output en la mayoría de los pares
    if n_invisible_a > len(pairs) // 2:
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


def _strip_seed_header(src: str) -> str:
    """Quita los bloques `# concepts:` y `# description:` propios del seed (el programa
    compuesto lleva su propio header fusionado; dos headers intermedios son ruido de forma)."""
    src = re.sub(r"#[ \t]*concepts:[ \t]*\n(?:[ \t]*#.*\n)+", "", src)
    src = re.sub(r"#[ \t]*description:[ \t]*\n(?:[ \t]*#.*\n)+", "", src)
    return src.strip("\n")


def _helper_name(seed: Seed, taken: set[str]) -> str:
    """Nombre natural para el paso, derivado de su primer concepto:
    'color change' -> apply_color_change. Evita colisiones con sufijo numérico."""
    base = (seed.concepts[0] if seed.concepts else seed.name).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "step"
    name = f"apply_{slug}"
    n = 2
    while name in taken:
        name = f"apply_{slug}_{n}"
        n += 1
    taken.add(name)
    return name


def _first_sentence(text: str, limit: int = 140) -> str:
    s = text.split(". ")[0].strip().rstrip(".")
    return (s[: limit - 1] + "…") if len(s) > limit else s


def compose_program(a: Seed, b: Seed) -> str:
    """Emite el programa compuesto con FORMA NATURAL (v2).

    v1 emitía dos cuerpos verbatim (con sus propios headers) + un dispatcher `_step_a/_step_b`
    — el A/B mostró que el modelo aprendía esa plantilla, no el razonamiento. v2: un solo
    header fusionado, helpers con nombres semánticos y un `transform` que se lee como una
    solución secuencial de dos etapas (la forma que tienen los programas multi-paso reales
    de BARC)."""
    taken: set[str] = set()
    name_a = _helper_name(a, taken)
    name_b = _helper_name(b, taken)
    body_a = _rename_main(_strip_seed_header(_body_without_common_imports(a.source)), name_a)
    body_b = _rename_main(_strip_seed_header(_body_without_common_imports(b.source)), name_b)
    concepts = ", ".join(sorted(set(a.concepts) | set(b.concepts)))
    desc = f"First, {a.description.rstrip('.')}. Then, {b.description.rstrip('.')}."
    return (
        "from common import *\n\nimport numpy as np\nfrom typing import *\n\n"
        f"# concepts:\n# {concepts}\n\n"
        f"# description:\n# {desc}\n\n"
        f"{body_a}\n\n{body_b}\n\n"
        "def transform(input_grid):\n"
        f"    # Step 1: {_first_sentence(a.description)}\n"
        f"    grid = {name_a}(np.copy(input_grid))\n"
        f"    # Step 2: {_first_sentence(b.description)}\n"
        f"    output_grid = {name_b}(np.array(grid))\n"
        "    return output_grid\n"
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
            got = np.array(call_with_timeout(fn, (np.array(pair["input"]),), timeout))
            if not np.array_equal(got, np.array(pair["output"])):
                return False
        return True
    except BaseException:
        return False
