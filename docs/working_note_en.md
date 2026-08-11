# Standing on the Winner's Shoulders: an Honest Transduction Baseline, an Exhausted Lever Space, and a Verifiable-Program Bet — Working Note (ARC Prize 2026 / ARC-AGI-2)

*Working Note — ARC Prize 2026, ARC-AGI-2 (Kaggle, offline track).*
**Author:** Julian Camilo Villa Duque.

> Living document. Goal: **win**, not merely participate. Updated with every experiment —
> problem, features, successful/failed strategies, results, metrics, process. Hard numbers in
> [RESULTS.md](RESULTS.md). **Extended technical explanation of every training/inference tactic and
> of the problem: [TACTICS.md](TACTICS.md)** (Spanish: [TACTICAS.md](TACTICAS.md)).
> Companion docs: [PROGRAM_SYNTHESIS.md](PROGRAM_SYNTHESIS.md), [SFT_PLAN.md](SFT_PLAN.md),
> [LB33_AUDIT.md](LB33_AUDIT.md), [STRATEGY.md](STRATEGY.md).

---

## The problem in plain terms (a 60-second primer)

**ARC-AGI-2** gives, per task, a few input→output **grid** pairs (cells colored 0–9) that
exemplify an abstract transformation; you must infer the rule and apply it to a held-out test
input. It is built to be **easy for humans (~60%) and hard for machines** — it rewards
compositional generalization, not memorization.

**Competition constraints (Kaggle offline track):**

| Constraint | Value |
|---|---|
| Internet | none (offline); model + deps from attached Kaggle Models/Datasets |
| Hardware / time | L4×4 (96 GB), **12 h** per submission |
| Metric | **pass@2** (2 attempts/test), exact match |
| Scoring split | **~50% semi-private** (public LB, shown each submit) + **~50% private** (hidden, decides ranking) |
| Cadence | **1 submission/day**; the scoring rerun does **not** consume GPU quota (only Save&Run does) |

**Reference points.** 2025 private SOTA = **NVARC 24.03%**. Frontier APIs (GPT-5.5 ~85%) are
**not** comparable — they run online and uncapped; this track is offline and compute-capped.

**What "winning" requires.** The 2025 champion reached the top with **synthetic data at scale +
SFT + test-time training (TTT) + ensembling**. That is the reference recipe we replicate and
push past. There is no cheap shortcut above ~30%.

---

## Abstract

Our final-to-date system is the **LB33/NVARC transduction pipeline** (Qwen3-4B ARC-SFT checkpoint
+ TTT LoRA + DFS decoding) with one graft — **cheap-first task ordering** — that lifts it from
28.47% to **30.14%** semi-private by draining more tasks before the 12 h wall. We establish, by
four leaderboard draws (28.47 / 30.14 / 29.72 / 29.31), that **~30% is a firm systematic ceiling
for us with this pipeline**, and that it coincides with the level of the 2025 private winner
whose checkpoint we reuse — our score *is* the winner's level. This note is deliberately weighted
toward **what failed**, because the negative results define the frontier. Every cheap/medium lever
inside the transduction paradigm is now **exhausted with measurement**: architecturally-diverse
ensembling (a Qwen3-2B partner is break-even; a 7M recursive TRM under a time-split *regresses* to
27.22%, because the strong Qwen is coverage-limited and any weaker partner that displaces its
compute loses more coverage than it adds); per-task time-budget tuning (adaptive budget −2.08,
600 s cap severely degraded); and selection quality (a SOTA-founded "holistic-judge" selection
ensemble is **neutral** at 29.31%, because our two scorers are correlated heuristics, not a strong
reasoning judge). A full SOTA review explains the wall: the 2026 survey places **pure transduction
at a 5–40% ceiling** and shows the winners are **hybrids** (induction + transduction + TTT,
50–79%); the leaderboard's top scores are either frontier-API (banned offline) or opaque. The
disciplined conclusion is that meaningful gains require a **paradigm change**, not more tuning —
so we commit to the winner's full recipe in three staged steps: (1) a hybrid of two *frozen*
already-SFT'd models — **BARC verifiable program synthesis** (generate Python solvers → execute →
keep those that reproduce **all** training pairs → apply to test) merged with our NVARC
transduction, whose verified hits *add* coverage without sacrificing the 30%; (2) **TTT** on top;
and (3) the long, decisive work — **our own SFT with 100–266K synthetic puzzles**, the core of
what made NVARC win, which we defer until steps 1–2 plateau and which we flag honestly as our
weakest capability (our 8K-synthetic SFT attempts produced zero exact solutions). We report each
component's measured contribution and are candid that we currently sit at the public-fork level,
not at the board head.

*A note on scope: we share the full reasoning, data insights, and every negative result — that is
where the value is. Exact constants/features are kept in-repo while the competition is live.*

---

## 1. The task, and what it reduces to

Each task is a set of demonstration grid pairs plus one (or few) test inputs; predict each test
output exactly, two attempts allowed. Two facts shape the whole approach:

- **It is knowledge-bound.** ARC reasoning is tied to what the model already knows about grid
  transformations. A base LLM *without ARC-specific SFT* scores ≈0 exact (measured: our train
  v10/v11 with 8K synthetic examples → zero exact). NVARC needed **266K synthetic puzzles** to
  reach 24%. The SFT — not the base generation — is where the skill lives.
- **Score ≈ Coverage × quality.** The pipeline is time-bounded (TTT + DFS per task under 12 h across
  4 workers). Which tasks finish before the wall is as decisive as per-task accuracy — this is why
  our one successful graft is a *coverage* lever, and why any compute given to a weaker model is a
  coverage tax.

---

## 2. Insights (paradigms, and the shape of the ceiling)

- **The paradigm hierarchy (2026 survey).** Pure neural transduction: **5–40%** (weakest, our
  family). Program synthesis: 26–79% but with a severe **AGI-1→AGI-2 cliff**. **Hybrid** (induction
  + transduction + TTT): **50–79%, dominant.** Every system above ~70% on AGI-1 uses TTT, generates
  ≥10 hypotheses, and verifies.
- **What separates ~55% from ~30%:** multiple hypotheses, robust perception, **hybrid reasoning
  with verification**, iterative refinement, TTT (+20–30 pt). The ~30% systems are static,
  frozen-after-training, single-representation.
- **The winners' numbers are SFT-driven.** NVARC's 266K synthetic + SFT is the moat; ARChitects
  used 102 recursive-refinement steps; MindsAI used TTFT + augmentation voting. None is a light tweak.
- **The offline ceiling is real.** Frontier-API hybrids reach 72.9% (e.g. holistic trace-judging
  over GPT-5.2/Gemini-3/Opus, $39/task) — **inapplicable offline**. The compute-capped ceiling is
  much lower; matching NVARC (~24% private / ~30% semi-private) is already "winner level," and
  beating it needs out-R&D-ing the winner or a genuinely new offline method.

---

## 3. Method (current deployed pipeline)

A frozen ARC-SFT transduction tracker with a coverage graft:

```
  test task ─▶ cheap-first queue (cheap→expensive by grid-cell cost)
             ─▶ per task, 4 GPU workers:
                  TTT: LoRA r=256, 16 augmentations, 1 epoch on the task's own pairs
                  DECODE: turbo-DFS (cumulative-prob threshold) → candidates
                  SCORE: augmented-consistency re-scoring (calc_scores)
             ─▶ SELECT top-2 (score_kgmon) ─▶ submission (attempt_1, attempt_2)
```

> **Every component (TTT, LoRA, DFS, re-scoring, selection, cheap-first) is explained in depth in
> [TACTICS.md](TACTICS.md) — what it is, why it exists, how we implement it.**

We adopted the public LB33 fork of the NVARC recipe (audited clean — no hardcoding/leakage; see
LB33_AUDIT.md), reusing the `sorokin/qwen3_4b_grids15_sft139` checkpoint. The **only** change that
survived the public test is **cheap-first ordering** (§4): sort the task queue cheap→expensive so
more tasks are drained before `end_time`, cutting `[[0]]` placeholders. Everything else (DFS math,
re-scoring, TTT, seeds, selection) is kept verbatim from the audited baseline.

---

## 4. Breadth and depth of exploration (negatives valued equally)

**What worked (transferred to the semi-private LB):**

- **cheap-first ordering.** 28.47 → **30.14** (+1.67). The task queue was alphabetical; sorting by
  a grid-cell cost proxy drains cheaper tasks first, so fewer are left unfinished. Pure throughput,
  no per-task compute change. **Confirms coverage was the binding constraint.**

**What did NOT transfer (with analysis — these define the wall):**

- **Ensemble with a weaker LLM (Qwen3-2B).** Break-even. The 4B is coverage-limited (~900 s/task,
  cannot drain 240 tasks in 12 h), so in a shared design the 2B never runs on the surplus, and in a
  time-split design it displaces strong-model coverage. Net ≈ 30%.
- **Ensemble with an architecturally-diverse TRM (7M recursive).** 30.14 → **27.22** (−2.92). A
  single-kernel time split (Qwen 9.5 h + TRM epochs=500 + a "fill-gaps" merge that never sacrifices
  a Qwen attempt) still lost: giving the Qwen 9.5 h instead of ~11.4 h cost ~3% of coverage, and the
  TRM (8.3% solo) filled essentially none of the expensive uncovered tail. **Airtight lesson: the
  strong coverage-limited model wants *all* the compute; any weaker partner is net-negative.**
- **Adaptive per-task decode budget.** 30.14 → 28.06 (−2.08). Measuring the budget from the *end of
  TTT* gave each task more total time → fewer tasks finished → less coverage.
- **600 s per-task cap.** Smoke degraded severely (quality loss); not submitted.
- **Selection ensemble (holistic-judge idea, arXiv 2606.31543).** 29.72 → **29.31** (neutral). The
  paper's judge is GPT-5.2 reading reasoning traces to recover minority-correct answers; ours is two
  correlated scoring formulas (kgmon consensus vs probmul confidence) over the same beam/aug signals
  — when they disagree, neither is reliably better. **Selection is not the offline lever.** (The
  same paper contraindicates refinement loops — they collapse diversity — so we did not build one.)
- **Our own SFT with 8K synthetic (train v10/v11).** Zero exact solutions. Data far below the
  266K NVARC scale; **SFT is our weakest capability.**

**A phantom gap, resolved.** We are **byte-identical** (all four core files + config + Docker image
+ model + sources, verified) to the public 33.89% pelotón notebook, yet draw ~30%. Three fresh
draws (28.47/30.14/29.72) stay below the tight pelotón band — a **systematic**, not variance, gap,
whose cause we could not identify remotely (likely exact source-version differences or unpublished
tweaks the pelotón carries). We stopped chasing it: **~30% is this pipeline's honest score for us.**

**The recurring meta-lesson:** inside the transduction paradigm, every cheap lever is spent; adding
a *weaker* model always subtracts (coverage tax); the score is at a measured local optimum in
coverage, time-budget, and selection. **Only a paradigm change (hybrid verification) or the full
SFT recipe can move it.**

---

## 5. Contribution of individual ideas

Honest semi-private-LB progression:

| Step | Change | LB (pass@2) | Δ |
|---|---|---|---|
| 0 | LB33/NVARC baseline (audited clean) | 28.47% | — |
| 1 | + cheap-first ordering | **30.14%** | **+1.67** |
| — | + adaptive decode budget | 28.06% | −2.08 (reverted) |
| — | ensemble + Qwen3-2B | ≈30% | break-even |
| — | ensemble + TRM (time-split) | 27.22% | −2.92 |
| — | re-draw of the champion (variance test) | 29.72% | ±noise |
| — | selection ensemble (kgmon+probmul) | 29.31% | neutral |

Everything after step 1 is a **negative result**. That is the finding: the transduction paradigm
is saturated for us at ~30%, and the four independent draws bound its run-to-run noise at ±~1%.

---

## 6. The plan to win — the full recipe in three staged steps (committed)

Winning is not a tweak; it is the champion's recipe: **load an SFT → adapt with TTT → continue
SFT'ing those weights with our own synthetic data.** We take the three in order, cheapest first,
measuring at each gate.

**Step 1 — Frozen hybrid (in progress).** Combine two already-SFT'd models without training:
**BARC verifiable program synthesis** + **NVARC transduction**.
- *Phase 0 (probe):* measure BARC-8B's **verifiable-hit rate** on ARC-AGI-2 — the fraction of tasks
  for which it generates a Python program that reproduces **all** training pairs (high-precision,
  because exact-match-on-train is a real verification), and whether those generalize to the test.
- *Phase 1:* if viable, the full induction system + a **hybrid merge** where verified programs *add*
  coverage on tasks the transduction misses, never displacing the proven 30%.
- *Threshold:* if the frozen hybrid clears ~30%, we win without touching weights.

**Step 2 — TTT (test-time training).** Light per-task LoRA fine-tuning at inference — the lever
worth +20–30 pt in the winning recipe. Automatic, cheap, per-task; applied to induction and/or
transduction as measurements dictate.

**Step 3 — Own SFT with synthetic data — EXECUTED AND CLOSED WITH A NEGATIVE (2026-07-27).**
Outcome: we generated **13,271 verified compositional tasks** with no LLM and no GPU quota, and the
SFT trained cleanly (loss 4.07→0.50), but the A/B gave **0/16 vs the frozen model's 1/16**. Root
cause: **verifiable ≠ useful** (see §4 and [TACTICS.md](TACTICS.md) §12). Original plan below; the
infrastructure built remains reusable.

**Step 3 (original plan) — Own SFT with synthetic data.** Replicate NVARC's core:
**generate 100–266K synthetic puzzles** (RE-ARC / BARC-Potpourri / concept-mixing, validated
against specs) and **continue-SFT** the weights (starting from an already-ARC-SFT'd checkpoint, not
from scratch) to exceed the public models' ceiling.
- *Known risks:* our weakest area (v10/v11 hit "zero exact"); GPU-heavy (plan around the weekly
  reset and the 2 h submission-holdover rule); hard to beat NVARC's own SFT.
- *Mitigations:* start from an SFT'd checkpoint + add data (don't train from scratch); measure
  coverage and exact-solve count per iteration; never chain data-gen without a signal.
- *Status:* deferred until steps 1–2 plateau; data-gen, config, and metrics to be logged here.

---

## 7. Living log (results · metrics · process)

- **2026-07-11..16.** Adopted LB33/NVARC → 28.47% → cheap-first **30.14%** (champion). Weak-partner
  ensembles (2B, TRM) and time-budget tweaks: exhausted / regressive.
- **2026-07-20.** Confirmed ~30% is systematic (re-draw 29.72%); we are byte-identical to the public
  33.89% notebook — the gap is not method.
- **2026-07-21.** SOTA review → transduction caps ~40%, hybrids win. Selection ensemble 29.31%
  (neutral) → transduction levers exhausted. Decision: pursue **verifiable program synthesis**.
- **2026-07-22.** Located BARC-8B (only ARC induction model; Llama-3.1). Recipe + `common.py`
  vendored; probe built (multiple offline-integration gates). ironbar's Kaggle copy is Llama-gated
  (403); the HF repo is open → downloading our own copy. **Committed to the full 3-step recipe,
  including Step 3 (synthetic-data SFT), to win.**
- **2026-07-23.** Phase-0 probe completed after ~8 offline-integration gates + one key bug
  (master's `common.py` uses TEAL/MAROON; the model uses PURPLE/BROWN → false 0%, patched).
  **Verifiable-hit rate = 6.25%** (1/16, K=16) and that hit **generalized** (1/1). Validated: the
  generate→execute→verify→apply pipeline works and generalizes. But **frozen BARC ≈6% on AGI-2**
  (the cliff) and expensive (485 s/task) → as a *frozen* partner it faces the same coverage tax as
  the TRM. **Confirms Step 3 (AGI-2 SFT) is needed** for induction to be competitive — as anticipated.
- **2026-07-24.** User chose **Step 3 (SFT the induction model)**. Located BARC's full data-gen+SFT
  pipeline (executable seeds + config). **Phase A (infra de-risk) COMPLETE:** BARC-8B LoRA SFT
  validated offline (loss 1.65→1.08). Key config: standard image, transformers+peft, `device_map=auto`
  (spreads the 8B across 4 L4s, no bitsandbytes), device-aware `DevTrainer` for the loss. Next:
  **Phase B** — compositional data-gen (compose seeds → 2-step AGI-2-style tasks).
- **2026-07-25/26. Phases B and C executed.** **Phase B:** our own compositor
  (`src/arc/seedgen.py`) chains BARC's executable seeds (A→B) → **13,271 VERIFIED 2-step tasks**
  (the composed program reproduces every pair → data correct by construction), 128 concepts, 6,037
  distinct seed pairs, generated **without any LLM and without spending GPU quota** (CPU kernel).
  **Phase C:** LoRA r=32 SFT, 900 steps on that data mixed with ~25% original (anti-forgetting) →
  **loss 4.07 → 0.50**; the high initial loss confirms the compositional data is genuinely new to
  the model. Pending: the verifiable-hit-rate A/B (probe with adapter) vs the frozen 6.25%.
- **2026-07-27→08-11. Compositional-SFT A/B — TWO negatives, path CLOSED.** v1 (random
  composition): 0/16 vs 1/16 — the model learned the *template* (two measurement artifacts fixed
  along the way: evaluator truncation; statistical power). v2 (full redesign: semantically coherent
  composition + visibility filter + natural form + aligned vocabulary + 1:1 mix, evaluated with a
  **paired probe** — same 16 tasks, dense metric, base replicating the 1/16+generalizes anchor):
  **SFT'd = 0 train-correct (Δ=−1), exec rate 80%→50%, generations 2.5× longer, and the model
  parrots the data's style ("First…, Then…") on real tasks.** Underlying lesson: **synthetic ≠ real
  distribution — SFT on mechanically composed tasks shifts the prior and interferes with base
  capability.** BARC needed 100K+ GPT-4 tasks; that bar is unreachable offline via mechanical
  composition.
- **2026-08-11. New lever family: SPEED (unexplored).** Fresh LB recon: the head is pulling away
  (nvbanana 67.5, rabbithole 47.8), pelotón at 33-34, top-8 cutoff ~33.5. Every knob we measured
  changed the time *cap*, never per-task *efficiency*. Built for Friday's reset: (1) an
  **instrumented smoke** (12 tasks across the cost spectrum, TTT/decode/scoring split — never
  captured); (2) the **ttt8 graft** (TTT n_aug 16→8, one number: cheaper task → more coverage).
- **2026-08-11 (afternoon). Colab lane operational — three free deliveries in one day.** Official
  Colab CLI running headless from Windows (free T4, 0 Kaggle quota; setup in docs/COLAB.md).
  (1) Paired `ttt8` ablation: **red light** — ΔNLL(n8) = 48% of ΔNLL(n16): halving augs halves
  adaptation; demoted to "only if TTT ≥ 35% of task time". (2) The `dfsbatch` bench **caught a
  production bug** (naive fusion [0,4,2,6] mixes (h,w)/(w,h) shapes on non-square tasks → ragged
  tensor → would have crashed; the champion's 4 batches are shape-grouped on purpose) and then
  **eliminated it** with data: 1.05× speedup (0.77 on the one task with real DFS work) and
  projected L4 KV = 39.3 GB > 22 GB. Remaining candidate: `scoring4` (re-scoring augs 8→4,
  pre-testable via rank-correlation on Colab).
- **2026-08-11 (evening). New parallel family: SCALE (docs/SCALE_PLAN.md).** Hypothesis (user's
  proposal): the gap to the head is largely **synthetic data + scale**, not a qualitatively
  different method. Our v1/v2 negatives condemn home-grown mechanical composition, NOT diverse
  100K+ corpora generated by frontier LLMs — the champion itself is living proof that that SFT
  works. Verified downloadable sources: ARC-Heavy 139K (HF, streaming), NVARC 103K+3.2M (Kaggle),
  RE-ARC. **D1 launched on Colab (free)**: champion 2B + LoRA r=64, cumulative curve N=0→500→2000
  over ARC-Heavy, dense eval on the same 12 real tasks as the ablation + 40 held-out synthetic
  (separates "learns the source" from "transfers to AGI-2" — the v1/v2 lesson). Pre-registered
  chain: D2 N=8000 → D3 source comparison (which synthetic distribution generalizes to AGI-2?) →
  D4 = the real Step 3 (4B SFT on Kaggle) only after two greens.
- *(next: read the D1 curve; build the scoring4 proxy; Friday after reset — timing smoke → decide
  by the pre-registered rules in SPEED_PLAN §4-bis)*
