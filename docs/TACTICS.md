# Training and inference tactics — extended technical explanation

> Technical reference for the ARC Prize 2026 project (ARC-AGI-2, Kaggle offline track).
> Explains **the problem**, **every tactic we use or tried** (what it is, why it exists, how we
> implement it, what we measured) and **where we stand**. Companion to
> [working_note_en.md](working_note_en.md) (narrative and results) and [RESULTS.md](RESULTS.md)
> (hard numbers). Spanish version: [TACTICAS.md](TACTICAS.md).
>
> Last updated: 2026-07-29.

---

## Part I — The problem

### 1. What ARC-AGI-2 asks for

Each **task** is a standalone puzzle containing:

- a handful of **demonstration pairs** (`train`): input grid → output grid,
- one or more **test grids** whose output must be predicted exactly.

A **grid** is a 2D matrix of up to 30×30 cells, each colored 0–9 (0 = black/background). The
transformation relating input to output is **abstract** (symmetry, counting, gravity, containment,
filling, scaling, color correspondence…) and must be **inferred from the examples alone**: no text
description, and no two tasks share a rule.

**The metric is `pass@2` with exact match:** two attempts per test grid, and it only counts if one
reproduces the answer **cell for cell**. No partial credit. A 30×30 grid with 899 correct cells and
1 wrong scores exactly the same as a blank one.

This metric is harsher than it looks and shapes every design decision: getting *close* is worth
nothing. Our early experiments measured tasks at **95–98% correct cells and zero points** — which
is why we dropped similarity metrics early and track only exact solves.

### 2. Why it is hard (and why the "2" matters)

ARC is designed to measure **compositional generalization**, not memorized knowledge:

- **Humans score ~60%**; compute-capped offline machines, 20–30%.
- **ARC-AGI-2 vs ARC-AGI-1:** the 2026 survey documents a **universal 2.5–3× degradation** going
  from AGI-1 to AGI-2, across *all* paradigms. The measured cause is **compositional depth**: the
  average solution goes from ~1.3 steps (AGI-1) to ~2.7 steps (AGI-2). Recognizing *one*
  transformation no longer suffices; several must be **chained**, and the search space explodes.
- **It is "knowledge-bound."** A base LLM without ARC-specific training scores ≈0 exact. We measured
  this ourselves: our own SFT runs on 8K synthetic examples (`train v10/v11`) produced **zero exact
  solutions** despite driving loss down to 0.043. Knowledge about grid transformations is not in
  pretraining; it has to be put there.

### 3. Competition constraints (and what they imply)

| Constraint | Value | Design implication |
|---|---|---|
| Internet | **forbidden** | Model, weights and dependencies must come from attached Kaggle Models/Datasets. Development is "blind": no `pip install`, no downloads inside the submission kernel. |
| Hardware | **4× NVIDIA L4** (24 GB each, 96 GB total) | An 8B model in bf16 (16 GB) **does not fit for training** on a single L4. Forces model parallelism or quantization. |
| Time | **12 h** per submission | The dominant constraint. With 240 tasks that is **180 s/task** on average to cover them all. Anything slower means leaving tasks unsolved. |
| Metric | **pass@2**, exact match | Two hypotheses per task; they should be **diverse**, not two variants of the same thing. |
| Scoring | **~50% semi-private** (visible) + **~50% private** (hidden, decides ranking) | Do not over-optimize the visible LB: the final ranking is decided on data we never see. |
| Cadence | **1 submission/day** | Every real LB experiment costs a day. Forces offline validation before spending the slot. |
| GPU quota | ~30 h/week | An explicit budget. Our rule: **keep 2 h of headroom** for submissions. The scoring *rerun* does NOT consume quota; only "Save & Run All" does. |

**The central consequence:**

```
score  ≈  Coverage  ×  Quality
          (how many     (how many of
           tasks you      those you
           have time      get right)
           to attack)
```

Because time is fixed, **coverage and quality compete**. Spending more compute per task raises
quality but lowers coverage. All of our optimization work has moved along this frontier, and
several of our negative results are explained precisely by it (§Part III).

---

## Part II — The tactics, explained

### 4. SFT — Supervised Fine-Tuning (the base knowledge)

**What it is.** Classic supervised training: give the model (input, desired output) pairs and adjust
weights to maximize the probability of the output. In ARC the "input" is the serialized task and the
"output" is either the correct grid (transduction) or a program that produces it (induction).

**Why it is indispensable.** It is where the skill lives. The winners confirm it:

- **NVARC** (1st in 2025, 24.03% private) generated **266,000 synthetic puzzles** and used them to
  SFT a Qwen3-4B. That SFT *is* their competitive moat.
- **BARC** did the analogous thing for induction: ~400K (task → Python program) examples generated
  with GPT-4, to train a Llama-3.1-8B.

**Our position.** We do not redo the transduction SFT: we **reuse NVARC's checkpoint**
(`sorokin/qwen3_4b_grids15_sft139`), which places us at the winner's level from the start. This is
deliberate — standing on someone else's SFT skips the most expensive part and the one where we are
weakest (our 8K-synthetic attempts produced zero exact solves).

⚠️ **Important contamination:** that checkpoint's SFT **included the public ARC-AGI-2 evaluation
set**. We therefore **never validate that model against `arc-agi_evaluation_challenges.json`** — it
would measure memorization, not capability. Only the hidden leaderboard gives honest signal.
(Documented in [LB33_AUDIT.md](LB33_AUDIT.md); we learned it the hard way — a 4-task smoke scored
~perfect and was completely misleading.)

### 5. LoRA — Low-Rank Adaptation (how we train without blowing up memory)

**What it is.** Instead of updating each layer's weights `W` (billions of parameters), LoRA freezes
`W` and learns two small matrices `A` (d×r) and `B` (r×d) with **very small r**, so the effective
weight is `W + BA`. Only `A` and `B` are trained.

**Why we use it.**
1. **Memory:** full training of an 8B needs ~4× the weights in memory (weights + gradients + two
   Adam moments). LoRA trains ~0.1–1% of parameters → fits on the L4s.
2. **Speed:** fewer parameters to update, cheaper backward.
3. **Reversibility:** the adapter is a small standalone file; it can be applied, removed or merged
   (`merge_and_unload`) without touching the base model. This is **critical for TTT** (§6).

**Hyperparameters and what they control:**

| Parameter | Controls | Our values |
|---|---|---|
| `r` (rank) | Adapter capacity. Higher = can learn more, costs more memory | **256** in the champion's TTT (very high — TTT needs capacity to "reprogram" the model toward one task); **16–32** in our SFT runs |
| `lora_alpha` | Update scale (`BA · alpha/r`) | 32 |
| `target_modules` | Which layers get adapted | The champion adapts **everything**: `q,k,v,o` (attention) + `gate,up,down` (MLP) + **`embed_tokens` and `lm_head`** (embeddings — unusual, but in ARC the relevant vocabulary is 12 tokens, and re-tuning their embeddings helps) |
| `use_rslora` | Stabilized scaling (`alpha/√r` instead of `alpha/r`) | ✔ — important at r=256, otherwise the update is too large |

### 6. TTT — Test-Time Training (the champion's key tactic)

**What it is.** The idea that separates winners from the rest: **train the model on the very task
it is solving, at solve time**. Every ARC task ships its own demonstration pairs; instead of merely
putting them in the prompt, run a **short fine-tune on them** before generating.

**Why it works.** The 2026 survey quantifies it: **+20–30 points** over the same frozen model, and
**every system above 70% on AGI-1 uses it**. The intuition: a task defines its own mini-distribution;
a few gradient steps on its examples move the model far closer to that specific rule than any prompt
can.

**How our pipeline implements it (per task):**

1. **Restore** the adapter to its initial state (`set_peft_model_state_dict` from saved weights).
   Essential: without restoring, task N would inherit task N−1's training and contaminate it.
2. **Build the training set** from the task's pairs, **augmented ×16** (§7).
3. **Train 1 epoch** with LoRA r=256, `lr=5e-5`, `adamw_torch`, cosine schedule, batch 1.
4. **Switch to inference mode**, decode (§8), score (§9), select (§10).
5. Free memory and move to the next task.

**The cost.** This is what makes each task cost ~900 s instead of ~5 s. It is the origin of the
coverage↔quality conflict: TTT is so expensive that **the model never reaches all 240 tasks in
12 h**. That is literally our measured bottleneck.

### 7. Data augmentation (why ARC allows it and demands it)

**What it is.** ARC has a property almost no domain has: **valid grid transformations preserve task
semantics**. Rotate input *and* output 90° and the rule is still the same rule. Permute the colors
consistently and it still holds.

**The augmentations we use:** transposition (×2), rotations 90/180/270 (×4), random permutation of
the 10 colors (consistent across input and output), and reshuffling of the demonstration examples
(order should not matter).

**Used in three different places:**

1. **In TTT (×16):** turns 3–5 demonstration pairs into ~50–80 training examples. Without this the
   per-task fine-tune would overfit instantly.
2. **At inference (×16 "subkeys"):** the *same* task is decoded under 16 different views. Each view
   produces candidates, which are then **un-transformed** (`invert_mod`) back to the original frame.
   This is what generates **real diversity**.
3. **In re-scoring (×8):** to evaluate how good a candidate answer is (§9).

**Implementation detail:** inverting the transformation must be exact. The pipeline encodes the
transformation in the example's own *key* (`task.rot90.permute0492…`) and undoes it operation by
operation in reverse order.

### 8. DFS decoding (`turbo_dfs`) — searching instead of sampling

**The problem with normal generation.** An LLM generates token by token. Sampling gives diversity
but little precision; greedy gives a single answer. Exact match requires **systematically exploring**
the high-probability sequences.

**What `turbo_dfs` does.** Depth-first search over the token tree, **pruned by cumulative
probability**:

- The relevant vocabulary is **12 tokens**: digits `0-9`, newline (`Ċ`) and end-of-sequence. No need
  to consider Qwen's ~150,000-token vocabulary.
- Each branch accumulates its **NLL**. It is **pruned** when it exceeds
  `max_score = -log(0.2)` — i.e. when the accumulated sequence falls below 20% probability.
- Expansion is **batched**: it keeps N live branches and advances them together, reusing the
  **KV cache** — hence "turbo".
- Returns **all** complete sequences found under the threshold, sorted by score.

**Our optimization.** The baseline moved full-vocabulary logits to CPU at every step. We rewrote the
computation to be **algebraically identical** while transferring only the 12 relevant values
(`index_select` over ARC tokens, `logsumexp` for normalization). Same result, far less GPU→CPU
traffic.

**Why DFS and not beam search.** Beam keeps a fixed width; DFS with a probability threshold adapts
effort: where the model is confident it explores little and finishes fast; where it hesitates it
explores more. That fits a variable time budget.

### 9. Augmented-consistency re-scoring (`calc_scores`)

**The problem.** DFS returns several candidates with their generation score (`beam_score`). But that
score is biased: it measures confidence *under the specific view* in which it was generated.

**The tactic.** Each candidate is evaluated **teacher-forced under 8 augmentations**: the model is
shown the transformed task and the (also transformed) candidate answer, and the log-likelihood of
that answer is measured. A **truly correct** candidate should be probable under *all* views; one
that was an accident of a single view should not.

This is a cheap form of **consistency verification**: it executes nothing (unlike program synthesis,
§11) but exploits the domain's invariances.

### 10. Selecting the 2 attempts

With `pass@2`, **two** candidates must be chosen. Two implemented criteria:

- **`score_kgmon` (consensus):** `number of times the candidate appeared − mean(score_aug)`.
  **Count**-dominated: rewards grids that multiple views produced *independently*. A majority vote
  corrected by confidence. **This is what the champion uses.**
- **`score_full_probmul_3` (confidence):** sum of `(baseline − beam_score)` plus the mean of
  `Σ(baseline − score_aug)`. **Probability**-dominated: rewards the candidate the model is most sure
  about, even from a single view.

**Our (failed) experiment.** Inspired by the AGI-2 "trace judging" paper — where a strong LLM judge
recovers *minority-but-correct* answers that consensus discards — we tried a **selection ensemble**:
`attempt_1` = consensus, `attempt_2` = confidence when they disagree. Result: **neutral** (29.31%
vs 29.72%). The reason: our "judge" is two **correlated** formulas over the same signals (beam +
augmentations), not an independent reasoner. When they disagree, neither is reliably better.

### 11. Verifiable program synthesis (the alternative paradigm)

**The idea.** Instead of the model *drawing* the output grid (transduction), have it **write a
Python program** that maps input→output (induction). Then:

```
generate K programs  →  EXECUTE each on the demonstration pairs
                     →  keep only those reproducing ALL of them exactly
                     →  apply to the test input
```

**Why it is attractive.** The verification step is **real**: not a confidence heuristic but an
executable check. A program reproducing all 3–4 demonstration pairs exactly has a high probability
of being the right rule. Its hits are **reliable**, which makes them ideal to *add* coverage without
risking what already works.

**How we implement it** (following BARC's recipe, see [PROGRAM_SYNTHESIS.md](PROGRAM_SYNTHESIS.md)):
Llama-3.1 chat template; grids serialized as **color names**; generated programs `from common import *`
(a 1,551-line library of primitives — `Color`, `flood_fill`, `blit`, `crop`,
`find_connected_components`, symmetry detection — vendored at `vendor/barc/common.py`); regex
extraction of the ` ```python … ``` ` block; **sandboxed execution with a timeout** (SIGALRM, so an
infinite loop cannot hang the kernel) catching every exception (many generated programs crash — that
is normal); verification via `np.array_equal`.

**What we measured (PHASE 0):** the **frozen** BARC-8B achieves a **6.25% verifiable-hit rate** on
ARC-AGI-2 (1 of 16 tasks yields a program passing all pairs) — and **that hit generalized** to the
test. The mechanism works; the model, trained in 2024 on AGI-1, is weak on AGI-2 (the "cliff", §2).
It is also expensive: ~485 s/task at K=16.

### 12. Compositional synthetic data generation (our own tactic)

**The reasoning.** If the bottleneck is SFT (§4) and AGI-2 demands compositional depth (§2), we need
**more compositional training data**. NVARC and BARC generated theirs with GPT-4 — impossible for us
(offline, no API budget).

**The enabling discovery.** BARC's ~163 *seeds* are not static examples: they are **executable
generators**. Each ships a `main(input_grid)` solver and a `generate_input()` sampler, plus concept
and description comments. Running `generate_input()` → `main()` yields real, **unlimited** pairs.

**Our composition (A→B):** `input ← generate_input_A()`, `x ← main_A(input)`, `output ← main_B(x)`;
the SFT target program concatenates both bodies plus a `transform` that chains them.

**Quality filters:** reject identity, reject degenerate outputs (single color), and — crucially —
require that **both steps do something** in most pairs (otherwise it is a 1-step task in disguise).

**The valuable property:** every generated task is **verified by executing its own program** against
all its pairs. **100% of generated tasks verify** → the data is **correct by construction**, with no
LLM involved. We generated **13,271 tasks** (128 concepts, 6,037 distinct seed combinations) in
~1.7 h of CPU, **spending zero GPU quota**.

**The (negative) result and why.** SFT on that data trained cleanly (loss 4.07 → 0.50) but the A/B
gave **0/16 against the frozen model's 1/16**. The diagnosed cause:

> **Verifiable ≠ useful.** Chaining two *random* seeds produces correct but **non-ARC-like** tasks:
> real tasks have a **coherent, inferable rule**; a chain of two semantically unrelated
> transformations often **is not inferable at all**. The model learned the **form** (two-part
> programs) instead of the reasoning.

*If this path is revisited*, the fix would be to compose only **semantically compatible** pairs and
emit the program **inline** (one coherent function) instead of two bodies + dispatcher.

### 13. `cheap-first` — the one coverage lever that worked

**The measured problem.** The pipeline pulls tasks from a queue shared by 4 workers. At ~900 s/task
over 12 h, **240 tasks do not fit**: unreached ones stay as `[[0]]` placeholders and score zero.

**The tactic.** Sort the queue **cheap→expensive**, using the **total cell count** of the task's
grids as a cost proxy (it correlates with prompt length and thus with TTT and decoding time). Ties
broken by key, for determinism.

**Why it works.** It changes *nothing* about per-task compute — only the order. But it drains more
tasks before the 12 h wall. **Measured: 28.47% → 30.14% (+1.67).** Our only graft that survived the
public test, and it confirmed that **coverage was the bottleneck**.

### 14. Offline engineering constraints (what it costs to make things *run*)

Much of the real effort is infrastructure, not algorithms. Each point below cost iterations:

- **No `bitsandbytes`** in any available image → **no 4-bit / QLoRA**, no `adamw_8bit`. This removes
  the standard solution for training an 8B in 24 GB.
- **Adopted solution:** `device_map="auto"` spreads the bf16 8B **across the 4 L4s** (96 GB total) →
  fits unquantized, no gradient checkpointing needed.
- **Side effect:** with the model split, `labels` (GPU 0) and `lm_head` logits (GPU 3) live on
  different devices → `RuntimeError`. Solved with a `DevTrainer` whose `compute_loss` moves labels
  to the logits' device.
- **Pinned Docker image is mandatory.** Kaggle's "latest" image rolled to Python 3.12 with
  `torchao 0.10`, incompatible with `peft` (needs >0.16) → `ImportError` when applying LoRA. **Any
  kernel using peft must use the pinned image.**
- **Models mount via symlink** under `/kaggle/input`: `glob(recursive=True)` does **not** traverse
  them; use `os.walk(followlinks=True)`.
- **`model_sources` is case-sensitive** (`Transformers`, not `transformers`) and Kaggle **silently
  drops** an invalid reference (metadata ends up `[]`) → always verify the committed metadata.
- **Gated models:** BARC's Kaggle copy is blocked by the Llama license (403). The original HuggingFace
  repo is open → solved with a download kernel (CPU+internet) producing our own reusable copy.
- **The Kaggle CLI on Windows** fails to read notebooks with accents (cp1252) and **the push fails
  silently** (exit≠0, no message) → force `PYTHONUTF8=1` and check the return code. This bug once
  made us submit the wrong kernel version.
- **BYOD image logs come back empty** when downloaded → for anything that needs debugging, use the
  standard image and **write results to a JSON file** in `/kaggle/working`; do not rely on the log.

---

## Part III — Where we stand

### 15. Measured results (semi-private leaderboard, `pass@2`)

| # | Configuration | Score | Δ |
|---|---|---|---|
| 0 | LB33/NVARC baseline (audited clean) | 28.47% | — |
| 1 | **+ `cheap-first`** | **30.14%** | **+1.67** ✅ **champion** |
| — | + adaptive decode budget | 28.06% | −2.08 ❌ |
| — | ensemble with Qwen3-2B | ≈30% | break-even ❌ |
| — | ensemble with TRM (time split) | 27.22% | −2.92 ❌ |
| — | champion re-draw (variance test) | 29.72% | noise |
| — | selection ensemble (consensus+confidence) | 29.31% | neutral ❌ |

**Four draws of the same pipeline: 28.47 / 30.14 / 29.72 / 29.31.** Run-to-run noise is ~±1%, and
**~30% is a systematic ceiling** for us in this configuration.

### 16. The three structural conclusions (all measured)

**(a) The strong model wants *all* the compute.** We tried two ensembles with weaker partners
(Qwen3-2B: 18.33% alone; 7M recursive TRM: 8.33% alone). Both failed for the **same airtight
reason**: the Qwen 4B is *coverage-limited*, so any compute handed to another model **takes away its
coverage**, and the weak partner cannot repay what it displaces. For the TRM we measured it exactly:
giving the Qwen 9.5 h instead of 11.4 h cost ~3% of coverage and the TRM filled ~0 of the expensive
tail. **Corollary: never again attempt ensembles with weaker partners.**

**(b) The time operating point is already optimal.** Three measurements: more time per task
(adaptive budget) → −2.08; the current point (1,200 s) → 30.14%; less time (600 s cap) → severe
quality loss. **Both directions are worse.**

**(c) We are at the ceiling of the wrong paradigm.** The 2026 survey places **pure transduction at
5–40%** and shows the winners are **hybrids** (induction + transduction + TTT, 50–79%). Our ~30% *is*
the 2025 winner's level (whose checkpoint we use). Beating it requires a paradigm change, not
parameter tuning.

### 17. Step 3 status (induction SFT) — CLOSED with TWO negatives (v1 template, v2 style; see RESULTS.md 2026-08-11)

| Phase | What it was | Status |
|---|---|---|
| **0** | Feasibility probe of verifiable synthesis | ✅ **6.25%** hit rate, and it generalized → mechanism works |
| **A** | Build and validate offline 8B SFT infra | ✅ loss 1.65 → 1.08 |
| **B** | Generate our own compositional data | ✅ **13,271 verified tasks**, zero GPU quota |
| **C** | SFT on that data + A/B | ✅ clean SFT (4.07 → 0.50) · ❌ **A/B: 0/16 vs 1/16** |

**Three methodological lessons** (in RESULTS.md; they apply to everything ahead):

1. **Verifiable ≠ useful.** Correctness of synthetic data does not imply its **distributional
   relevance**. 100%-correct data in the wrong shape teaches the shape.
2. **The measuring instrument is part of the experiment.** SFT lengthened the model's outputs and the
   evaluator truncated them at 1024 tokens without closing the code block → we measured the
   evaluator, not the model. When changing the output distribution, **always revisit the evaluator's
   limits**.
3. **Statistical power.** An n=16 probe at ~6% rates **cannot distinguish 0/16 from 1/16**. Deciding
   on small improvements needs much larger n (expensive: ~1,000 s/task) or a denser metric than
   per-task exact match.

### 18. Reusable assets built

- **Offline 8B SFT pipeline** validated (`device_map=auto` + `DevTrainer`, no bitsandbytes).
- **Verifiable data generator** (`src/arc/seedgen.py` + 163 vendored seeds): produces
  correct-by-construction tasks with no LLM and no GPU quota.
- **Full verifiable-synthesis pipeline**: prompt, extraction, sandboxed execution with timeout,
  verification.
- **Our own BARC-8B copy** (bypasses Llama gating) as a Kaggle artifact.
- **Offline infrastructure cookbook** (§14) — the knowledge that cost the most iterations.

### 19. Open options (pending decision)

1. **Redesign the compositional data:** compose only semantically coherent pairs + inline program.
   Reuses all the infra; fixes the diagnosed root cause, but is another long round with uncertain
   odds.
2. **Measure the frozen hybrid (1 submission):** the never-measured unknown is whether the frozen
   BARC's ~6% verifiable hits are **disjoint** from the Qwen's 30%. If so, they add up to +6%
   **without training anything**. Cheap, and closes an open question.
3. **TTT on the induction side:** the +20–30 pt lever from the winning recipe, not yet applied there.
4. **Consolidate ~30%** and stop investing in high-uncertainty bets.
