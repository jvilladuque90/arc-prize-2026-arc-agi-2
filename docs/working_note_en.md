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
- **2026-08-11/12 (late night). D1 + D1-bis: the STRONGEST positive signal of the campaign.**
  D1 curve (champion 2B + LoRA over ARC-Heavy, N=0→500→1250→2000): exact 1/12→7/12, cell-acc
  0.47→0.85. Post-hoc audit: 9/12 eval tasks were AGI-1 (the population BARC's seeds were
  written from) → possible memorized neighborhood → **D1-bis** same day: paired base-vs-adapter
  eval on the **24 holdout tasks NOT in AGI-1** (all postdate ARC-Heavy → contamination
  impossible). Verdict: **exact 0/24 → 7/24, cell-acc 0.50 → 0.85, NLL improves on 24/24, 0
  losses — GENUINE transfer.** The v1/v2 negatives condemned our mechanical composition, not
  diverse corpora: with the right source, Step 3 of the winner's recipe LIVES. Technical gates:
  T4 OOM (→ grad checkpointing, 5.4GB peak); the n=2000 adapter persisted as a private Kaggle
  dataset (`arcscale-adapter-2b`) so later stages resume without retraining. Remaining unknown
  before spending Kaggle: does the gain survive ON TOP of per-task TTT? → **D2-bis launched**
  (paired arms TTT vs adapter+TTT on 12 new tasks).
- **2026-08-12/13. D2/D3 chain + hidden-set TRIAL E1 (user's idea: use the idle daily slot).**
  D2: ARC-Heavy SATURATES around 2-4K examples on the 2B (exact flat at 7-8/24). D3: the three
  public sources (ARC-Heavy, NVARC's own, mix) are INDISTINGUISHABLE — a shared plateau → the
  bulk of the effect is domain re-alignment, not source knowledge; the only remaining data lever
  is a NEW distribution (semantic-author generation pilot). E1: 2B+merged n4000 adapter,
  submitted with the exact 18.33-baseline config. **First attempt: 0.00** — post-mortem: the
  transformers 5.x config drops top-level `rope_theta: 5M` → the kernel's 4.x stack used the
  default → broken attention. Fix: ORIGINAL aux files in every merge + the commit now VALIDATES
  (unsloth load + NLL<1.5; cost ~7 of the 22 quota minutes). Pre-score audit at the user's
  request: weight diff (adapter genuinely merged, controls bit-identical), n4000 provenance,
  mechanical diff vs baseline (single change on the rerun path). **E1-v2 (Aug 13): 19.86 =
  +1.53 over 18.33 — transfer is REAL but modest** (~3× the variance band; between the
  pre-registered thresholds). Consistent with D2-bis: TTT already recovers part of the SFT.
  Operational gates: free-T4 daily limit (~6 sessions) — Google-account switch authorized by
  the user (ADC backed up); D2's ghost VM kept running after the client died and saved n=4000.
- **2026-08-13/14. The data lever closes + trial run 2 + induction pilot.** claudegen v1: 33
  compositional rules hand-authored in a day (2178 tasks verified by construction, 0
  contamination with AGI-2) → probe under the D3 protocol: the SAME plateau (7-8/24) as the
  other four sources/mixes → QUINTUPLE evidence that the SFT effect is domain re-alignment +
  generic grid competence, saturated — the data lever closes on the 2B proxy with a crisp
  result for the paper. E1 trial run 2 (Thursday's idle slot): **21.25** — two runs 19.86/21.25,
  mean delta **+2.2** over 18.33 (3-6× the run-to-run spread) → hidden-set transfer CONFIRMED
  robust. Own-induction pilot (grids→`main` program, execute-and-verify): Qwen2.5-1.5B SFT on
  2000 ARC-Heavy pairs → **0/24 verified** (prereg rule: scale before discarding); the Coder-3B
  escalation was LOST to infra (the ghost VM died before persisting) — the branch has no clean
  verdict yet.
- **2026-08-15/16. D4: capacity breaks the plateau… and the pipeline eats the gain. Merged
  branch CLOSED.** 4B SFT on Kaggle (3.2h, N=8000 ARC-Heavy): 0/24 → **11/24 greedy** on the
  internal eval — capacity breaks the 2B plateau ✓. But the LB A/B: **27.08 (−3.06 vs 30.14)**
  → prereg revert. Converging, free diagnosis: the merged model's pools IMPROVE (1.6 vs 0.8
  cands/task; GT-in-pool 8/12 vs 5/12), and the smoke with the real stack scores 3.0 vs the
  champion's 2.5 (per-task quality did NOT regress) → mechanism: **2× richer pools make
  decode+scoring more expensive → coverage collapsed**. Compensation `lb33-scale-fast` = ttt8
  (green when re-measured on the new base, −17%) + scoring4 (top-2 100% faithful) ≈ −25%
  t/task; v1 was caught by the commit failsafe (IndentationError — the build now compiles the
  embedded modules), v2 smoke reload 3.0. **A/B (55538928): 28.33** — the mechanism's direction
  is confirmed (+1.25 from cutting cost) but <29 → **prereg reading: merged branch CLOSED; the
  champion remains 30.14**. For the paper, the week's clean pair: the SAME SFT yields +2.2 on
  the 2B and −1.8/−3.1 on the production 4B — the calibration×pipeline-budget interaction
  dominates over capacity.
- **2026-08-16. Architecture map: where our data enters and where the ceiling is.**
  At the user's request, [ARCHITECTURE.md](ARCHITECTURE.md), with numbers taken from the tensors
  and the code rather than memory: Qwen3-4B with the vocabulary cut to **16 tokens** (1 token =
  1 cell), 36 layers, d_model 2560, GQA 32/8, rope 5M, **untied** `lm_head`, 3,634 M params.
  **There are only TWO data-injection points**: the SFT (LoRA r=64, 132 M, 7 projections) and
  per-task TTT (LoRA r=256 + rsLoRA, 530 M, **including embed_tokens and lm_head**). Finding:
  those two matrices are [16, 2560], **max rank 16**, so r=256 on them is *full rank* — TTT
  rewrites the entire symbol table per task, and our SFT never touched it (verified tensor by
  tensor). A candidate explanation for why the SFT gain washed out over TTT. **Score
  decomposition: 0.80 coverage × 0.42 GT-in-pool × 0.90 selection ≈ 30.1** → the ceiling of the
  WHOLE speed family is **+7.5 pts**; the ×2 lives only in GT-in-pool. Measured free the same
  day: context truncation is NOT a lever (2/120 tasks), `attempt_2` **is** used (4/5 smoke
  subtasks), and scoring seeds its augmentations with `hash()`, which is **salted per process**
  → the same kernel is not reproducible (consistent with the 19.86 / 21.25 runs) ⇒ **differences
  below 1.4 pts are not conclusive from a single run**.
- **2026-08-16/17. Speculation: exact but dead; the length cap survives.**
  Prompted by the user's question about nibble tokenization (which in the literature means
  quantization, not tokenization; and for ARC the ARChitects deliberately chose 1 token/cell),
  we chased the *lossless* version of the same idea: speculative decoding with a free drafter =
  the input grid itself (72% of outputs share their input's dimensions and, of those, a median
  85% of cells are identical). Chain: cost gate ✓ (a 16-token forward costs the same as a
  1-token one) → exactness test v1 came back **EMPTY** (without TTT the 4B completes no grid at
  the production threshold — corollary worth its own line: *TTT doesn't improve the pool, it is
  what makes the pool exist*) → v2 **caught a real bug** (exploring a branch overwrote the cache
  and corrupted the drafted prefix: Jaccard 0.0) → v3 fixed → **v4 gate PASSED: 15/15 identical
  pools, 6 at the production threshold**. Then the decisive bench: over the **4 real views of a
  production batch** the gain collapses to **×1.25 (lockstep) / ×1.39 (ragged)** against gates
  of 1.4/1.6 → **branch CLOSED with the algorithm already verified correct**. Reason:
  speculation accelerates short tasks (×3.4) and not long ones (×1.06-1.3), which are what
  consumes the budget. The same bench produced what we did adopt: batches with lengths
  `[72, 650, 72, 72]` → the champion uses `max_new_tokens=931` for every task, so one rambling
  view drags the whole batch. **`lb33-cap`**: cap = `4 × max(largest train output, test input)`;
  the GT fits in **100.0% of 1248 subtasks**, the capped pool was a **subset** of the normal one
  in 12/12 with 0 GTs lost, and the L4×4 smoke gave **reload 2.5 = identical to the champion**.
  Submitted (55582027). Infrastructure gates paid: mount layout (now we **search** for the
  file), `--accelerator NvidiaL4` mandatory (the T4×2 default leaves 2 of the 4 processes
  without a GPU), and the Kaggle CLI **cannot write logs on Windows** (charmap vs unsloth's
  emoji → a 0-byte `.log` that hid the real error for three attempts; `PYTHONUTF8=1`). Also
  discovered that **agi-3 shares the same Kaggle account**: same 2 GPU sessions and same weekly
  quota ⇒ this repo's hour ledger is unreliable.
- **2026-08-22/23. TTT is THE ENGINE, and its schedule is already optimal.** Two submissions pin
  both ends. **Without TTT: 9.58** (vs ~30 for the champion) ⇒ **TTT contributes ~+20 pts, two
  thirds of the score, for 34.8% of the clock.** It is not a refinement: it is what makes the
  pool exist at all. And **TTT at half an epoch: 26.39** ⇒ a real regression. Mapping
  post-mortem: production's 128 steps **are** 1 epoch (each augmented sequence seen ONCE), which
  is exactly the sweet spot the bench had measured; half an epoch leaves half the views
  unadapted and the vote loses voters. **The "amount of TTT" axis is closed at both ends.**
- **2026-08-24/26. The noise band: nearly every A/B of the month was unreadable.** Three runs of
  the **same kernel, same version**: 30.14 / 29.31 / 27.64 → mean **29.03**, σ **1.27**. The
  champion was never 30.14; that was the lucky roll. With 1 submission/day and that band, **a
  single sample only detects effects >2.5 pts**, and the live levers are worth ~1 task. Cause
  located in the code: scoring seeds its augmentations with `hash(bk)`, and Python's string
  `hash()` is **process-salted**.
- **2026-09-01. Auditing the champion's logs (ZERO cost) → a second source, and a leak.** Before
  spending quota we audited whether the clock gates bite. The DFS's **540 s cap never bites**
  (slowest batch 203.7 s; it exits on beam exhaustion, which is deterministic). But the
  **per-task 1200 s cap does**, by a hair: `981571dc` started its last batch at **1159.6 s —
  40.4 s of margin, a 3.5% jitter** — so machine speed decided whether that task decoded 16
  views or 12. **Not just noise: a coverage leak.**
- **2026-09-02/05. `lb33-det` = NEW CHAMPION, on a pre-registered prediction.** Two grafts:
  stable `md5` seed and **removal of the per-task clock cap** (global deadline kept). Three runs:
  **31.11 / 30.69 / 30.28** → mean **30.69**, σ **0.415**. **Complete separation** from the old
  champion (det's worst, 30.28, beats the old best, 30.14) → Mann-Whitney n=m=3, **p = 0.05**.
  **+1.66 pts = +4 tasks**, and the measurement channel is **3.1× narrower**. Operational
  corollary: at σ≈0.42 **a single sample resolves ~1 pt (2.5 tasks)**; before it took >2.5 pts.
- **2026-09-03. The quantum is 1/240, not 1/120.** Every historical difference is a near-exact
  multiple of 0.4167: 0.83→1.99 tasks · 1.67→4.01 · 0.42→1.01. Confirmed in the code: the rerun
  scores against `arc-agi_test_challenges.json`, which holds **240 tasks / 259 test inputs**.
  ⚠️ Determinism was **NOT** closed (31.11 ≠ 30.69): a third source remains, probably GPU
  floating-point. We **abandoned chasing it**: it is worth ~1 task and removing it would require
  slower deterministic algorithms.
- **2026-09-04/05. The narrow channel pays immediately: `stack` 29.44, and a false rescue
  corrected.** `stack` (= `det` + cheap-first + length cap) lands **−3.5 tasks** below det's mean
  and −2 below its worst run ⇒ revert. And it forces a correction to the record: `lb33-cap`
  carried **those same two grafts** and its 30.14 had been read as "recovered"; it was **the
  lucky roll of a worse configuration**. A 1.46-pt effect was **invisible** under the old rule.
  Lesson: **a wide band does not merely hide real effects — it manufactures false rescues.** The
  culprit was identified **offline and for free**: the cap
  `4·max(train outputs, test input)+2` produces no violations on eval/train, but its **minimum
  margin is ONE token** and 12.2% of eval sits under 50 tokens of slack; it assumes a ≤4×
  expansion that ARC does not guarantee — **a 3×3 tiling of a 10×10 needs 930 tokens against a
  cap of 438: ungeneratable**. The baseline's `max_new_tokens=931`, by contrast, is a
  **structural** bound (30×30 = 900 cells + 29 newlines + 1 EOS = 930) and is therefore safe.
  Own design flaw noted: **two** grafts were bundled into one arm ⇒ **new rule: one graft per
  arm**.
- **2026-09-01/02. Augmentation-dosage bench: closed, with a post-mortem worth more than the
  experiment.** v1 ran mechanically flawless and came out **scientifically void**: TTT trained on
  **16 sequences instead of 128** (I conflated the `n` in `augment(n)` with the number of views;
  in production `augment(n)` = 8 D4 × n colour permutations). What exposed it was logging the
  *schedule* — `n_seqs_ttt`=16 and `ttt_s` of 14-27 s against production's 215-865 s — not the
  result. **Why the 2B smoke missed it: that model memorises eval and solves without needing
  TTT.** General rule: **whatever a bench does not need in order to run, the bench does not
  verify.** v2, now faithful, gave r(scarcity, benefit of more views) = **+0.215** against a
  pre-registered ≤−0.60 ⇒ **branch closed**.

### Consolidated table: all 24 leaderboard submissions

⚠️ Everything before 2026-09-02 was measured at σ=1.27, where a single sample only resolves
effects >2.5 pts. Verdicts from that era are marked inconclusive where they are.

| date | what it tested | score | verdict |
|---|---|---|---|
| 07-07 | initial 3B v10 pipeline | 0.83 | superseded |
| 07-11 | **LB33 baseline** (NVARC 4B + TTT r=256 + DFS) | 28.47 | starting point |
| 07-13 | + cheap-first | 30.14 | inconclusive |
| 07-14 | + adaptive decode budget | 28.06 | negative |
| 07-15 | Qwen3-2B alone | 18.33 | not competitive |
| 07-17 | TRM alone (7M recursive) | 8.33 | not competitive |
| 07-19 | ensemble 4B + TRM | 27.22 | negative |
| 07-20 | champion cheap-first cap1200 | 29.72 | band sample |
| 07-22 | selection ensemble (kgmon + probmul) | 29.31 | neutral |
| 08-12 | scale 2B + ARC-Heavy n4000 | 0.00 | bug (rope_theta) |
| 08-13 | scale 2B v2 | 19.86 | — |
| 08-14 | variance of the above | 21.25 | σ of the 2B |
| 08-15 | **4B + ARC-Heavy N=8000** | 27.08 | negative → scale family closed |
| 08-16 | merged + ttt8 + scoring4 | 28.33 | inconclusive |
| 08-17 | **per-task length cap** | 27.64 | **negative (confirmed 09-04 with mechanism)** |
| 08-22 | **no TTT** | **9.58** | **TTT is worth ~+20 pts** |
| 08-23 | TTT at half an epoch | 26.39 | negative → 1 epoch is optimal |
| 08-24 / 08-25 / 08-26 | old champion variance | 29.31 / 27.64 / 30.14 | **σ = 1.27** |
| **09-02 / 09-03 / 09-05** | **`lb33-det`** (md5 + no clock cap) | **31.11 / 30.69 / 30.28** | **CHAMPION · +4 tasks · p=0.05** |
| 09-04 | `stack` (det + cheap-first + length cap) | 29.44 | reverted |

Benches that never reached the LB (offline, Colab or Kaggle): triage of which tasks need TTT,
leave-one-out TTT, focal and weighted CE, object verifier, augmentation dosage (two runs). All
closed negative or invalidated.

### What the champion actually is (`lb33-det`, 30.69 ± 0.415)

| component | configuration |
|---|---|
| base model | Qwen3-**4B**, checkpoint `qwen3_4b_grids15_sft139` (**SFT comes pre-made**, NVARC-derived) |
| parallelism | 4×L4, one worker per GPU, shared task queue |
| **TTT** (per task) | LoRA **r=256, α=32 with rsLoRA**, over 7 projections **+ `embed_tokens` + `lm_head`** (~530 M params); AdamW **lr 5e-5 cosine**, batch 1, **1 epoch over 128 sequences** (`augment(n=16)` = 8 D4 × 16 colour permutations) |
| decode | 16 views (`augment(n=2)`) in 4 batches of 4; DFS pruned at `max_score = −log(0.2)`; **`max_new_tokens = 931` fixed** |
| re-scoring | each candidate re-scored over 8 augmented views by NLL (md5-stable seed) |
| selection | 2 attempts: `score_full_probmul_3` and `score_kgmon` |
| budget | global deadline 11 h 50 m · per-task 1200 s cap **removed** · DFS 540 s cap kept |
| **thinking** | **not used** — the pipeline never goes through the chat template; it feeds raw digit tokens with role markers (ids 11/12) |

### Open, pre-registered next steps

1. **SFT with *thinking* — an untouched lever, never tested.** Today the pipeline uses **no
   thinking at all**: no chat template, the model consumes raw digits, and the checkpoint's SFT
   was done on that format, so it emits no reasoning chains. Turning it on **is not a flag: it
   requires another SFT**. Plan: **bench it in Colab first (free T4)** — does a 2B given a short
   SFT on reasoning traces improve answer NLL against the same 2B without them? ⚠️ The vehicle
   must avoid the §26 trap: the 2B memorises eval, so the comparison must be **paired and
   restricted to tasks the 2B does not already solve**, or it measures nothing. Only a green
   light justifies spending L4.
2. **Efficient TTT by hyper-parameter ablation** (user's proposal). Idea: TTT is the engine
   **and** 34.8% of the clock; if only part of what it trains moves the loss, training *only
   that part* frees budget for decode **without cutting adaptation** — the known failure mode
   (`ttt05`, `ttt8`). First step, now done: **inventory what TTT touches today** (table above).
   The ablatable axes are: (a) the **9 target modules**, with a strong prior that
   `embed_tokens`/`lm_head` are special — they are [16, 2560] matrices of **rank at most 16**, so
   r=256 over them is *full rank* and TTT **rewrites the entire symbol table** per task; (b) the
   **rank r** (256 is enormous for the projections); (c) **lr and scheduler**; (d) the **number
   of sequences** (closed at both ends, do not touch). Bench design: measure **the TTT loss curve
   per configuration**, which needs no decode and is therefore cheap (~150-200 s per task on the
   4B). ⚠️ It must run on **4B over eval** — the 2B is useless because it memorises and its loss
   starts at zero. And loss is a *proxy*: `ttt05` already showed a schedule that looks healthy
   can regress in production, so a green in loss requires pool-level confirmation before
   grafting.

   **EXECUTED 2026-09-05 (rank axis).** 12 tasks, 4 ranks, paired, TTT only: **r=64 → 1.34×
   faster with no detectable degradation** (mean, median and sign test agree; 25.5% of TTT =
   **8.9% of the whole pipeline**). r=16 → 1.46× but **10/12 tasks worse** (sign p=0.039): the
   mean hid it because one outlier improves by −0.204 while the other eleven sum to +0.163 —
   without the sign test we would have adopted a degradation believing there was none. r=4
   rejected (11/12 worse). **Speed SATURATES at r=16**: r=4 takes the same time with 3.6× fewer
   params ⇒ below that the cost lives in the frozen base model and trimming more parameters
   **cannot** help; the next lever is **adapting only the top K layers**, which does shorten the
   backward pass. ⚠️ A faster TTT **does not raise the score on its own**: the pipeline does not
   reinvest the time, it just finishes earlier. It only pays if the global deadline bites or if
   the saving is explicitly **spent** on more decode views.

- *(status: the coverage family delivered its single largest piece (+4 tasks) through free
  offline auditing, not through the LB. That is now the engine: offline analysis found both the
  +2 pts and the length-cap diagnosis without spending a slot. The leaderboard is reserved for
  confirming effects ≥2-3 tasks. Still pending: splitting the weekly quota with agi-3, which
  shares the account, GPU sessions and quota)*
