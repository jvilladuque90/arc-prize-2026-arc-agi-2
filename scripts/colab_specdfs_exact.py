#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Test de EXACTITUD del DFS especulativo (gate duro antes de tocar Kaggle).

El bench specdraft (16-ago) dio luz para seguir: forward de 16 tokens cuesta lo mismo que uno
de 1 (56 ms), aceptacion 0.569, bifurcaciones 0.007 -> forwards x1.88. Pero una decodificacion
especulativa solo sirve si produce EXACTAMENTE el mismo pool de candidatos que el baseline;
si cambia el pool, cambia la calibracion y repetimos el error de D4 (27.08).

Tres brazos sobre las MISMAS tareas y el MISMO modelo (4B real, batch 1, sin TTT):
  A) REF   — DFS de referencia: recursivo, con la cache recortada explicitamente en cada nodo
             (semantica correcta por construccion: la cache siempre es el prefijo exacto).
  B) SPEC  — DFS especulativo: mismo arbol, pero obtiene los logits en BLOQUES de k tokens
             usando como borrador la grid de entrada. Las ramas se apilan y se exploran igual.
  C) CHAMP — el turbo_dfs del campeon tal cual (cache COMPARTIDA entre la recursion y el bucle,
             sin recortes). Brazo de control: si C != A, el campeon arrastra un quirk de cache
             que conviene conocer (no se toca nada hoy; es informacion).

Comparacion: multiset de secuencias de tokens de las hojas + delta de score. La igualdad
A == B es el gate. Diferencias de score por debajo de 1e-3 se consideran ruido de fp16
(los kernels difieren entre un forward de 1 token y uno de k), pero una secuencia distinta NO.

Lecturas pre-registradas:
  - A == B en todas las tareas (Jaccard 1.0) y max|dscore| < 1e-2 -> VERDE: implementar en el
    kernel (queda el problema aparte del lote ragged de 4 filas).
  - Jaccard >= 0.98 con diferencias solo en hojas de score cercano al umbral -> AMARILLO:
    es ruido de frontera; medir su efecto en el pool final antes de seguir.
  - Jaccard < 0.98 -> ROJO: la especulacion NO es lossless como se implemento; cerrar o
    rediseñar. No se sube nada a Kaggle.

Ademas mide el modelo de coste a BATCH 4 (la produccion decodifica 4 vistas por lote):
si un forward de 4x16 tokens ya no es plano, el lote ragged no pagaria.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_specdfs_exact.py <KAGGLE_API_TOKEN>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

T00 = time.time()
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
os.makedirs("/content", exist_ok=True)
os.chdir("/content")
for f in ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"):
    subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-2",
                    "-f", f, "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done", shell=True)
subprocess.run(["kaggle", "models", "instances", "versions", "download",
                "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
                "-p", "/content/m4b", "--untar", "-q"], check=True)
MDIR = None
for root, dirs, files in os.walk("/content/m4b"):
    if "config.json" in files and any(x.endswith(".safetensors") for x in files):
        MDIR = root
        break
assert MDIR

AGI1 = set()
for split in ("training", "evaluation"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}",
        headers={"User-Agent": "arc-check"})
    for f in json.load(urllib.request.urlopen(req)):
        AGI1.add(f["name"].replace(".json", ""))
print(f"[setup] descargas listas ({time.time()-T00:.0f}s)", flush=True)

import hashlib
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

TOK = AutoTokenizer.from_pretrained(MDIR)
MODEL = AutoModelForCausalLM.from_pretrained(MDIR, torch_dtype=torch.float16,
                                             device_map="cuda:0")
MODEL.eval()
print(f"[setup] 4B cargado | mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
NL_ID, EOS_ID, PAD_ID = 10, 15, 13
_TOKID = torch.tensor(ARC_TOKENS, dtype=torch.long, device="cuda:0")
MAX_SCORE = -float(np.log(0.2))
# v2: la v1 salio VACIA — sin TTT y al umbral de produccion el 4B no completa NINGUNA grid
# bajo -log(0.2) en tareas nuevas (los 3 brazos dieron 0 hojas -> comparacion vacua). El test
# de exactitud es una propiedad ALGORITMICA, asi que se corre tambien con umbrales anchos, que
# generan arboles ricos (mas ramas = mas estricto). Salidas cortas + tope de nodos para que
# ambos brazos exploren el arbol COMPLETO: si alguno se trunca, la comparacion no vale.
MAX_NEW = 200
K_DRAFT = 8
MAX_LEAVES = 3000


def g2s(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def build_prompt(task):
    txt = "".join(f"<|im_start|>user\n{g2s(p['input'])}<|im_end|>"
                  f"<|im_start|>assistant\n{g2s(p['output'])}<|im_end|>"
                  for p in task["train"])
    txt += (f"<|im_start|>user\n{g2s(task['test'][0]['input'])}<|im_end|>"
            f"<|im_start|>assistant\n")
    return TOK.encode(txt)


def cand_list(logits_row, run_score):
    """Candidatos de un nodo: (score_acumulado, token) con score < umbral, ordenados."""
    lf = logits_row.float()
    nll = (torch.logsumexp(lf, dim=-1) - lf.index_select(-1, _TOKID)).tolist()
    out = [(run_score + v, t) for v, t in zip(nll, ARC_TOKENS) if run_score + v < MAX_SCORE]
    out.sort(key=lambda x: x[0])
    return out


def draft_seq(row_col, inp, k):
    """k tokens de borrador = reproducir la grid de entrada desde (r,c). Coste cero."""
    r, c = row_col
    H = len(inp)
    seq = []
    for _ in range(k):
        if r >= H:
            seq.append(EOS_ID)
            break
        row = inp[r]
        if c < len(row):
            seq.append(int(row[c]))
            c += 1
        elif r == H - 1:
            seq.append(EOS_ID)
            break
        else:
            seq.append(NL_ID)
            r, c = r + 1, 0
    return seq


def advance(row_col, tok):
    r, c = row_col
    if tok == NL_ID:
        return (r + 1, 0)
    return (r, c + 1)


# ---------------------------------------------------------------- A) DFS de referencia
@torch.no_grad()
def dfs_ref(prompt_ids, inp, deadline):
    cache = DynamicCache()
    ids = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
    base = ids.size(1)
    leaves = []
    stats = {"fwd": 1, "trunc": False}

    def rec(logits_row, pos, run_score, prefix, row_col):
        if time.time() > deadline or len(leaves) >= MAX_LEAVES:
            stats["trunc"] = True
            return
        if len(prefix) >= MAX_NEW:
            stats["trunc"] = True
            return
        for sc, tok in cand_list(logits_row, run_score):
            if tok == EOS_ID:
                leaves.append((round(sc, 6), tuple(prefix + [EOS_ID])))
                continue
            cache.crop(pos)
            o = MODEL(input_ids=torch.tensor([[tok]], device="cuda:0", dtype=torch.long),
                      position_ids=torch.tensor([[pos]], device="cuda:0"),
                      past_key_values=cache, use_cache=True, return_dict=True)
            stats["fwd"] += 1
            rec(o.logits[0, -1], pos + 1, sc, prefix + [tok], advance(row_col, tok))
            if time.time() > deadline:
                return

    rec(out.logits[0, -1], base, 0.0, [], (0, 0))
    return leaves, stats


# ---------------------------------------------------------------- B) DFS especulativo
@torch.no_grad()
def dfs_spec(prompt_ids, inp, deadline):
    """Mismo arbol que dfs_ref, pero los logits del camino borrador se obtienen en bloques de
    K_DRAFT tokens en UN forward. Las ramas alternativas se apilan y se exploran igual."""
    cache = DynamicCache()
    ids = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
    base = ids.size(1)
    leaves = []
    stats = {"fwd": 1, "trunc": False}

    def rec(logits_row, pos, run_score, prefix, row_col):
        if time.time() > deadline or len(leaves) >= MAX_LEAVES:
            stats["trunc"] = True
            return
        if len(prefix) >= MAX_NEW:
            stats["trunc"] = True
            return
        cands = cand_list(logits_row, run_score)
        if not cands:
            return
        draft = draft_seq(row_col, inp, K_DRAFT)
        if not (draft and cands[0][1] == draft[0] and draft[0] != EOS_ID):
            step(cands, pos, prefix, row_col)
            return
        # --- especulacion: UN forward para todo el bloque del borrador ---
        o = feed(draft, pos)
        cur_score, cur_prefix, cur_rc = run_score, list(prefix), row_col
        node_cands, accepted, dead = cands, 0, False
        # v3: las ramas NO se exploran aqui. Explorarlas sobrescribiria la cache en esa
        # posicion y corromperia el prefijo del borrador para todo lo que venga despues
        # (bug cazado por el test v2: jaccard 0.0 contra la referencia). Se anotan y se
        # procesan al final, restableciendo el prefijo con un solo forward cada una.
        ramas = []
        for j, dtok in enumerate(draft):
            if node_cands[0][1] != dtok:
                break
            for sc2, t2 in node_cands[1:]:
                ramas.append((j, sc2, t2, tuple(cur_prefix), cur_rc))
            cur_score = node_cands[0][0]
            cur_prefix.append(dtok)
            cur_rc = advance(cur_rc, dtok)
            accepted = j + 1
            if dtok == EOS_ID:
                leaves.append((round(cur_score, 6), tuple(cur_prefix)))
                dead = True
                break
            if len(cur_prefix) >= MAX_NEW:
                stats["trunc"] = True
                dead = True
                break
            node_cands = cand_list(o.logits[0, j], cur_score)
            if not node_cands:
                dead = True
                break
        # continuacion PRIMERO: la cache aun contiene el bloque del borrador intacto
        if not dead and accepted:
            rec(o.logits[0, accepted - 1], pos + accepted, cur_score, cur_prefix, cur_rc)
        # ...y ahora las ramas, cada una restableciendo el prefijo exacto
        for j, sc2, t2, pfx, rc in ramas:
            if time.time() > deadline:
                stats["trunc"] = True
                return
            if t2 == EOS_ID:
                leaves.append((round(sc2, 6), tuple(list(pfx) + [EOS_ID])))
                continue
            o2 = feed(list(draft[:j]) + [t2], pos)
            rec(o2.logits[0, -1], pos + j + 1, sc2, list(pfx) + [t2], advance(rc, t2))

    def step(cands, pos, prefix, row_col):
        """Camino lento: token a token, semantica identica a la referencia."""
        for sc, tok in cands:
            if tok == EOS_ID:
                leaves.append((round(sc, 6), tuple(prefix + [EOS_ID])))
                continue
            o = feed([tok], pos)
            rec(o.logits[0, -1], pos + 1, sc, prefix + [tok], advance(row_col, tok))
            if time.time() > deadline:
                stats["trunc"] = True
                return

    def feed(seq, pos):
        """Restablece el prefijo exacto (crop) y consume 'seq' en UN forward."""
        cache.crop(pos)
        o = MODEL(input_ids=torch.tensor([seq], device="cuda:0", dtype=torch.long),
                  position_ids=torch.arange(pos, pos + len(seq),
                                            device="cuda:0").view(1, -1),
                  past_key_values=cache, use_cache=True, return_dict=True)
        stats["fwd"] += 1
        return o

    rec(out.logits[0, -1], base, 0.0, [], (0, 0))
    return leaves, stats


# ---------------------------------------------------------------- C) turbo_dfs del campeon
@torch.no_grad()
def dfs_champ(prompt_ids, deadline):
    """Copia fiel del turbo_dfs de produccion (cache compartida, sin recortes), n=1."""
    stats = {"fwd": 1, "trunc": False}
    ids = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
    out = MODEL(input_ids=ids, use_cache=True, return_dict=True)

    def turbo(logits, max_new, scores, pos, cache):
        n = logits.size(0)
        lf = logits.float()
        nll = (torch.as_tensor(scores, dtype=torch.float32, device=lf.device).view(n, 1)
               + torch.logsumexp(lf, dim=-1, keepdim=True) - lf.index_select(-1, _TOKID)).cpu()
        suffixes = defaultdict(list)
        cand = {}
        for i in range(n):
            cand[i] = []
            for j, t in enumerate(ARC_TOKENS):
                sc = nll[i, j].item()
                if sc < MAX_SCORE:
                    if t == EOS_ID:
                        suffixes[i].append((sc, [t]))
                    elif max_new > 1:
                        cand[i].append((sc, t))
            cand[i].sort(key=lambda x: x[0])
        while True:
            if time.time() > deadline:
                stats["trunc"] = True
                break
            toks, scs, alive = [], [], 0
            for i in range(n):
                if cand[i]:
                    sc, t = cand[i].pop(0)
                    toks.append(t)
                    scs.append(sc)
                    alive += 1
                else:
                    toks.append(PAD_ID)
                    scs.append(1000.0)
            if alive == 0:
                break
            o = MODEL(input_ids=torch.tensor(toks, device="cuda:0",
                                             dtype=torch.long).view(-1, 1),
                      position_ids=torch.full((n, 1), pos, device="cuda:0"),
                      past_key_values=cache, use_cache=True, return_dict=True)
            stats["fwd"] += 1
            nxt = turbo(o.logits[:, -1], max_new - 1, scs, pos + 1, o.past_key_values)
            for bi, beams in nxt.items():
                for sc, st in beams:
                    st.insert(0, toks[bi])
                    suffixes[bi].append((sc, st))
        return suffixes

    res = turbo(out.logits[:, -1], MAX_NEW, [0.0], ids.size(1), out.past_key_values)
    return [(round(sc, 6), tuple(st)) for sc, st in res.get(0, [])], stats


@torch.no_grad()
def cost_model_batch(prompt_ids):
    """Modelo de coste a batch 4 (produccion decodifica 4 vistas por lote)."""
    out = {}
    for bs in (1, 4):
        ids = torch.tensor([prompt_ids] * bs, device="cuda:0", dtype=torch.long)
        base = MODEL(input_ids=ids, use_cache=True, return_dict=True)
        for width in (1, 8, 16, 32):
            chunk = torch.zeros((bs, width), device="cuda:0", dtype=torch.long)
            pos = torch.arange(ids.size(1), ids.size(1) + width,
                               device="cuda:0").view(1, -1).expand(bs, -1)
            for _ in range(3):
                MODEL(input_ids=chunk, position_ids=pos,
                      past_key_values=base.past_key_values, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(15):
                MODEL(input_ids=chunk, position_ids=pos,
                      past_key_values=base.past_key_values, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            out[f"b{bs}_w{width}"] = round((time.time() - t0) / 15 * 1000, 2)
        del base
        torch.cuda.empty_cache()
    return out


CHAL = json.load(open("/content/arc-agi_training_challenges.json"))
SOLS = json.load(open("/content/arc-agi_training_solutions.json"))


def is_holdout(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 1000 < 105


def cost(t):
    return sum(len(g["input"]) * len(g["input"][0]) + len(g["output"]) * len(g["output"][0])
               for g in t["train"])


def gt_len(k):
    g = SOLS[k][0]
    return sum(len(r) for r in g) + len(g)


HOLD = sorted([k for k in CHAL if is_holdout(k)], key=lambda k: cost(CHAL[k]))
NEW = [k for k in HOLD if k not in AGI1]
BUDGET = 240
# v4: argv[2]=="sharp" -> tareas de AGI-1 (que el modelo tiene MEMORIZADAS: sharp, con pool
# no vacio al umbral de PRODUCCION incluso sin TTT). Es la unica forma barata de testear la
# exactitud en el punto de operacion real: con tareas nuevas y sin TTT el pool es vacio y la
# comparacion no vale. La poblacion no importa: la equivalencia es una propiedad algoritmica.
SHARP = len(sys.argv) > 2 and sys.argv[2] == "sharp"
if SHARP:
    TASKS = [k for k in CHAL if k in AGI1 and gt_len(k) <= 150]
    TASKS = sorted(TASKS, key=lambda k: cost(CHAL[k]))[:10]
    THRESHOLDS = {"t020_prod": -float(np.log(0.2)), "t005": -float(np.log(0.05))}
else:
    # salidas cortas: el arbol tiene que caber ENTERO en ambos brazos (si se trunca, no vale)
    TASKS = [k for k in NEW if gt_len(k) <= 150][:5]
    THRESHOLDS = {"t020_prod": -float(np.log(0.2)), "t001": -float(np.log(0.01)),
                  "t0001": -float(np.log(0.001))}

RES = {"tasks": []}
for ti, tk in enumerate(TASKS):
    task = CHAL[tk]
    inp = task["test"][0]["input"]
    prompt = build_prompt(task)
    if len(prompt) > 6000:
        print(f"[{ti+1}/{len(TASKS)}] {tk} saltada (ctx {len(prompt)})", flush=True)
        continue
    if not RES.get("cost_model_ms"):
        RES["cost_model_ms"] = cost_model_batch(prompt)
        print(f"[coste] {RES['cost_model_ms']}", flush=True)

    for thr_name, thr in THRESHOLDS.items():
        globals()["MAX_SCORE"] = thr
        t0 = time.time()
        la, sa = dfs_ref(prompt, inp, time.time() + BUDGET)
        t_ref = time.time() - t0
        t0 = time.time()
        lb, sb = dfs_spec(prompt, inp, time.time() + BUDGET)
        t_spec = time.time() - t0
        t0 = time.time()
        lc, sc_ = dfs_champ(prompt, time.time() + BUDGET)
        t_champ = time.time() - t0

        sa_seq = {s for _, s in la}
        sb_seq = {s for _, s in lb}
        sc_seq = {s for _, s in lc}
        union = len(sa_seq | sb_seq)
        dmax = 0.0
        # las hojas son (score, seq): el dict va seq -> score (invertirlo daba KeyError)
        da = {s: sc for sc, s in la}
        db = {s: sc for sc, s in lb}
        for s in sa_seq & sb_seq:
            dmax = max(dmax, abs(da[s] - db[s]))
        # una comparacion solo VALE si hay hojas y ningun brazo se trunco
        trunc = bool(sa["trunc"] or sb["trunc"])
        valida = bool(union > 0 and not trunc)
        row = {"task": tk, "umbral": thr_name, "ctx": len(prompt),
               "hojas_ref": len(sa_seq), "hojas_spec": len(sb_seq), "hojas_champ": len(sc_seq),
               "truncado": trunc, "valida": valida,
               "jaccard_ref_spec": round(len(sa_seq & sb_seq) / union, 4) if union else None,
               "iguales": (sa_seq == sb_seq) if valida else None,
               "dscore_max": round(dmax, 6),
               "jaccard_ref_champ": (round(len(sa_seq & sc_seq) / len(sa_seq | sc_seq), 4)
                                     if (sa_seq | sc_seq) else None),
               "fwd_ref": sa["fwd"], "fwd_spec": sb["fwd"], "fwd_champ": sc_["fwd"],
               "secs_ref": round(t_ref, 1), "secs_spec": round(t_spec, 1),
               "secs_champ": round(t_champ, 1)}
        RES["tasks"].append(row)
        print(f"[{ti+1}/{len(TASKS)}] {tk} {thr_name}: hojas {len(sa_seq)}/{len(sb_seq)}/"
              f"{len(sc_seq)} valida={valida} trunc={trunc} IGUALES={row['iguales']} "
              f"jac={row['jaccard_ref_spec']} dscore={dmax:.2e} | fwd {sa['fwd']}->{sb['fwd']} "
              f"(x{sa['fwd']/max(sb['fwd'],1):.2f}) {t_ref:.0f}s->{t_spec:.0f}s | "
              f"champ_jac={row['jaccard_ref_champ']}", flush=True)

rows = [r for r in RES["tasks"] if r["valida"]]
print(f"[resumen] comparaciones validas: {len(rows)} de {len(RES['tasks'])}", flush=True)
if rows:
    summ = {"n_validas": len(rows), "n_totales": len(RES["tasks"]),
            "todas_iguales": all(r["iguales"] for r in rows),
            "n_iguales": sum(1 for r in rows if r["iguales"]),
            "jaccard_medio": round(float(np.mean([r["jaccard_ref_spec"] for r in rows])), 4),
            "dscore_max": max(r["dscore_max"] for r in rows),
            "hojas_totales": sum(r["hojas_ref"] for r in rows),
            "speedup_fwd": round(sum(r["fwd_ref"] for r in rows)
                                 / max(sum(r["fwd_spec"] for r in rows), 1), 2),
            "speedup_tiempo": round(sum(r["secs_ref"] for r in rows)
                                    / max(sum(r["secs_spec"] for r in rows), 0.1), 2),
            "jaccard_ref_champ": round(float(np.mean(
                [r["jaccard_ref_champ"] for r in rows if r["jaccard_ref_champ"] is not None])), 4),
            "coste_ms": RES.get("cost_model_ms")}
    RES["resumen"] = summ
    print("RESULT_JSON:" + json.dumps(summ), flush=True)
else:
    print("RESULT_JSON:{\"error\": \"ninguna comparacion valida (arboles vacios o truncados)\"}",
          flush=True)

json.dump(RES, open("/content/specdfs_exact_result.json", "w"), indent=1)
try:
    os.makedirs("/content/persist", exist_ok=True)
    json.dump(RES, open("/content/persist/specdfs_exact_result.json", "w"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")
    json.dump({"title": "arcscale specdfs exact", "id": f"{user}/arcscale-specdfs-exact",
               "licenses": [{"name": "CC0-1.0"}]},
              open("/content/persist/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", "/content/persist", "-q"],
                       capture_output=True, text=True)
    if "error" in (r.stdout + r.stderr).lower():
        subprocess.run(["kaggle", "datasets", "version", "-p", "/content/persist",
                        "-m", "test exactitud", "-q"], check=False)
    print("[persist] subido a Kaggle", flush=True)
except Exception as e:                                    # noqa: BLE001
    print(f"[persist] fallo: {e}", flush=True)
print(f"[fin] {time.time()-T00:.0f}s", flush=True)
