"""[LOCAL] Congela el subset de ARC-Heavy para D4 (SFT del 4B en Kaggle) como dataset privado.

Mismo stream determinista y filtros de grid que D1/D2 (parse_row): salta los primeros 40
validos (holdout sintetico del proxy) y toma los siguientes 19.000 (el kernel de SFT tokeniza
y filtra a <=2048 tokens con el tokenizer real -> ~16K utiles, paridad exacta con el proxy).
Salida: JSONL {"pairs": [[input, output], ...]} + subida a Kaggle como dataset privado
`arcscale-archeavy-19k`.

Uso local:  python scripts/make_archeavy_snapshot.py <OUT_DIR> [<KAGGLE_USERNAME>]
(env: KAGGLE_API_TOKEN para la subida; sin username, solo genera el JSONL)
"""
import json
import os
import subprocess
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
KUSER = sys.argv[2] if len(sys.argv) > 2 else ""
os.makedirs(OUT, exist_ok=True)

from datasets import load_dataset  # noqa: E402


def parse_row(row):
    try:
        pairs = []
        for ex in row["examples"]:
            gi = [[int(c) for c in r] for r in ex[0]]
            go = [[int(c) for c in r] for r in ex[1]]
            for g in (gi, go):
                if not g or not g[0] or len(g) > 30 or len(g[0]) > 30:
                    return None
                if any(len(r) != len(g[0]) for r in g):
                    return None
                if any(c < 0 or c > 9 for r in g for c in r):
                    return None
            pairs.append((gi, go))
            if len(pairs) == 4:
                break
        return pairs if len(pairs) >= 3 else None
    except Exception:
        return None


t0 = time.time()
STREAM = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                      split="train", streaming=True)
N_SKIP_HOLD, N_TAKE = 40, 19000
n_hold, n_take, n_seen = 0, 0, 0
path = os.path.join(OUT, "arc_heavy_19k.jsonl")
with open(path, "w") as fh:
    for row in STREAM:
        n_seen += 1
        pairs = parse_row(row)
        if pairs is None:
            continue
        if n_hold < N_SKIP_HOLD:
            n_hold += 1
            continue
        fh.write(json.dumps({"pairs": pairs}) + "\n")
        n_take += 1
        if n_take % 2000 == 0:
            print(f"[snapshot] {n_take}/{N_TAKE} ({n_seen} filas, {time.time()-t0:.0f}s)",
                  flush=True)
        if n_take >= N_TAKE:
            break
print(f"[snapshot] LISTO: {n_take} tareas en {path} "
      f"({os.path.getsize(path)/1e6:.0f}MB, {time.time()-t0:.0f}s)", flush=True)

if KUSER:
    json.dump({"title": "arcscale-archeavy-19k", "id": f"{KUSER}/arcscale-archeavy-19k",
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", OUT], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "already exists" in out:
        r = subprocess.run(["kaggle", "datasets", "version", "-p", OUT, "-m", "snapshot"],
                           capture_output=True, text=True)
    print(f"[upload] rc={r.returncode} {(r.stdout or '')[-300:]}", flush=True)
