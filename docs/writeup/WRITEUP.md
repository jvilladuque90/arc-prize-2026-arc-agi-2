# Measuring Before Moving: a Noise-Aware Study of Test-Time Training on ARC-AGI-2

**Julian Camilo Villaduque** · ARC Prize 2026, ARC-AGI-2 track · Final submission: `lb33-det`,
**30.69 ± 0.42 %** on the public leaderboard (3 reruns; best single run 31.11 %)

Code (MIT): https://github.com/jvilladuque90/arc-prize-2026-arc-agi-2 · All Kaggle kernels used are linked
from `scripts/push_kernels.py`.

---

## Summary

We started from the strongest public ARC-AGI-2 pipeline, the NVARC model (`sft139`, 1st place 2025) with
per-task test-time training (TTT) and depth-first decoding. Then we spent the competition on one question:
**which changes really move the hidden-test score, and which only look like they do?**

Three findings shaped everything:

1. **The leaderboard was noisier than the effects we were chasing.** Three reruns of one unchanged
   notebook scored 30.14, 29.31 and 27.64. We traced the noise to two causes in the code and removed both.
   That gave the final pipeline: **+1.66 points and a 3× narrower spread**, at zero extra compute.
2. **Test-time training is the engine, and its recipe is already at a local optimum.** Removing TTT
   costs 21 points. All five changes to its recipe that we sent to the leaderboard lost, by 1 to 6 points.
3. **The remaining gap is about generalization to new rules, not data, scale or decoding.** A handful of
   cheap measurements, most of them with no GPU at all, locate the difficulty precisely. They also explain why
   larger models, more synthetic data and program search did not help within the 12-hour budget.

We report every leaderboard experiment, including the many negative ones (Figure 3), because we believe
they are the most useful part of this work for the next team.

---

## 1. The pipeline

![Pipeline](figs/fig1_pipeline.png)

The final pipeline is the public LB33/NVARC notebook with three changes of ours. Every other part is
unchanged:

| Stage | What it does | Our change |
|---|---|---|
| Base model | Qwen3-4B with the vocabulary cut to 16 tokens (10 colors, newline, role and turn markers), SFT by NVARC on 266K synthetic puzzles | none |
| Test-time training | For each task, a fresh LoRA adapter (rank 256, all linear layers plus embeddings and output head) trained for 1 epoch on 128 augmented copies of the demos: 8 rotations/reflections × 16 color permutations, random demo order | none |
| DFS decoding | Under 16 augmented views, enumerate every output whose cumulative probability stays above 0.2 | **no per-task time cap** (only the global deadline) |
| Re-scoring | Score each candidate under 8 fresh augmentations | **stable md5 seeds** instead of Python's salted `hash()` |
| Selection | Votes plus likelihood, 2 attempts per test input | **cheapest tasks first** in the work queue |

The 4 L4 GPUs run one worker each. The full pipeline uses about 95 % of the 12-hour budget.

---

## 2. First, fix the instrument

![Variance](figs/fig2_variance.png)

Early in the competition, a change that gained 1.7 points looked like a clear win. Rerunning the
*unchanged* notebook showed a spread of 2.5 points (σ = 1.27). Most A/B tests we had read until then were
inside that band, and some "failures" were real effects hidden by it.

Reading the logs of the champion run, we found two mechanical sources of noise:

- **Salted seeds.** Re-scoring drew its augmentations with `hash(task_key)`. Python salts string hashes
  per process, so every run scored candidates under different views. We replaced it with an md5 hash,
  which is stable across processes.
- **A per-task clock cap that tasks barely cleared.** One task started its last decoding batch 40 seconds
  before its 20-minute limit. A 3.5 % change in machine speed decided whether it decoded 16 views or 12.
  The global deadline already protects the submission, so we removed the per-task cap.

With both fixes (`lb33-det`) three reruns scored **31.11, 30.69 and 30.28**. The worst rerun beats the
best rerun of the old version (Mann-Whitney p = 0.05). The quantum of the leaderboard is 1/240 of a task;
the new spread is about 2 tasks.

From then on, every experiment had a **pre-registered prediction** with three bands: adopt, tie, discard.
We wrote the bands down before reading the score. We compared paired runs where possible and reported
median and sign tests next to the mean. In one sweep the mean said "no change" while 10 of 12 tasks
got worse.

---

## 3. What we tried on the leaderboard

![Leaderboard map](figs/fig3_leaderboard_map.png)

Each bar is one submission to the hidden test set. The shaded band is ±2 standard deviations of the final
pipeline. Nothing we sent landed above it.

**Test-time training is the engine.** With TTT turned off, the same model scores 9.58. TTT is worth
**+21 points for 35 % of the compute**, the best trade in the pipeline. Its recipe also proved to be a narrow
optimum:

| Change to TTT | Score | Lesson |
|---|---|---|
| half an epoch | 26.39 | half of the 128 views go unadapted |
| largest demo always last (order otherwise random) | 25.69 | demo-order diversity carries weight: −12 tasks |
| 4 rotations × 32 color permutations (same cost) | 26.39 | geometric diversity cannot be swapped for color diversity |
| LoRA rank 64 / rank 512 | 29.72 / 24.31 | see the clock, below |

The last case is instructive. On an 18-task bench, doubling color permutations improved the probability
of the true answer in 13 of 15 tasks (p = 0.007). The same-cost variant that dropped reflections looked neutral on
the bench, but lost 4.3 points on the hidden test. **A small bench "inside the noise" is not a green light.**

**The clock is a cliff, not a slope.**

![Clock cliff](figs/fig4_clock_cliff.png)

Rank 512 added 11.9 % compute per task and lost 15 tasks, because the last tasks in the queue never ran.
Rank 64 saved 11 % and gained nothing, because the pipeline already finished in time. Fitting the
three points gives the budget: **any change must cost less than about 4.8 % per task**. That rules out
"thinking" tokens, more decoding views and more TTT steps, unless an equal saving pays for them.

**Other directions, all measured on the hidden test:**

- **Ensembles with a weaker partner** (a 2B model, the 7M-parameter recursive TRM) lost up to 3 points.
  The 4B model is limited by coverage and needs all the compute. Any time given to a partner that solves
  less is a net loss.
- **More SFT for the 4B model** (ARC-Heavy data) gave 27.08, and 28.33 with faster TTT and scoring. The
  richer candidate pools cost more time to decode and score, so coverage collapsed. The same SFT gave
  **+2.2 points** to a 2B model, which sits far below its time budget.
- **Wider search on tasks with no candidate** (beam of 8): 29.31. An empty pool almost always means the
  model does not know the answer, not that the 0.2 threshold hid it.
- **A 27B reasoning model** (Qwen3.8-27B, FP8, 440 tokens/s on 4 L4s) with a public agent scored 16.94 and
  14.86. Every task it solved was already solved by our pipeline.

---

## 4. Why it is hard to climb: measuring the difficulty cheaply

Before spending GPU time, we asked whether arithmetic on the task files, or a one-hour bench, could answer
the question. These checks decided where not to go.

![Why it is hard](figs/fig5_why_hard.png)

- **Rule novelty dominates scale** (left). We paired 119 AGI-1 tasks with larger re-ARC test inputs. The
  model solves 78 % of known rules at known size, still 67 % at 5.75× the size, and only 10 % of new rules
  from the AGI-2 evaluation set. Synthetic data can only add rules the model already knows. That ended
  our plan to train against a "scale deficit".
- **Test inputs fall outside the demonstrations** (middle). We measured how far each test input sits
  outside the range of its demos on 7 grid features. The median distance grows from 0.50 on AGI-1 to
  1.16 on AGI-2 eval (p = 2·10⁻⁸). The excess is *content*: 1.43× the area, 1.45× the ink, the same
  number of objects. The true output also falls outside the range of demo outputs in 95 % of tasks.
- **The model learns the rule but fails the test input.** After TTT, the model reproduces a held-out
  *demo* in 5 of 12 tasks and the real *test* in 0 of 12 (McNemar p = 0.031). It does learn the
  transformation. ARC-AGI-2 test inputs are chosen to require the concept, not the surface pattern.
- **Program search does not reach AGI-2** (right). A 25-operator grid DSL with exact verification solves
  5.1 % of AGI-1 tasks and 0 of 353 AGI-2 tasks. An object-level DSL with 17 properties does no better.
  The positive half survives: **exact verification against the demos is 87 % precise**. The rules depend
  on relations between objects, and a compact DSL does not express them.

Together these say that the pipeline's ~31 % is not short of data, decoding or compute. It is short of
**generalization to an unseen rule at an unseen scale**, the very thing ARC-AGI-2 was built to test.

---

## 5. A newer base model

![Base swap](figs/fig6_base_swap.png)

A newer base at the same size should raise the ceiling without costing coverage. We tested Qwen3.5-4B
(hybrid: 24 Gated DeltaNet layers + 8 attention layers) against Qwen3-4B with a clean protocol: *Base*
against *Base* checkpoints, the fast kernels installed, and a plain-text control to confirm that
loading worked. A first, naive comparison against the chat model had overstated the gap by 44 %.

- Before any training, Qwen3.5 assigns 5 % lower loss to the true answers (28 of 36 tasks, p = 0.0006).
- After the **same** short SFT on NVARC puzzles (9K sequences, 16-token vocabulary, LoRA rank 64),
  the advantage disappears. The four Qwen3.5 seeds win 19, 12, 17 and 22 of 36 tasks against Qwen3
  (sign test p ≥ 0.12), inside the seed-to-seed band. Qwen3.5 is also 8 % slower.

**The newer generation has a slightly better prior for grids, but does not learn them faster.** We trained a
full Qwen3.5 model on 18K sequences and ported the whole pipeline to it (unsloth does not support its
layers, and its recurrent cache cannot be rewound, so the DFS snapshots and restores it at every branch).
Kaggle's weekly GPU quota ran out before it could be submitted. The code is in the repository
(`scripts/build_lb35.py`).

---

## 6. Lessons

1. **Measure the noise of your leaderboard before you trust an A/B.** Our biggest gain came from removing
   two sources of randomness, not from a new idea.
2. **Pre-register.** Writing "adopt above X, discard below Y" before the score arrived kept us from
   reading noise as progress, in both directions.
3. **Budget the clock before designing.** At 95 % utilization, a good idea that costs 12 % is worse than
   no idea.
4. **The bench must match production** in model, format and population. A 2B bench told us TTT was
   worthless, because the 2B model had memorized the evaluation set. Dropped role tokens (a tokenizer
   change between library versions) silently turned a family of experiments into a format-repair study.
5. **Ask the data first.** Analyses that ran on a laptop (the demo-envelope distance and two DSL coverage
   gates) closed whole research directions that would each have cost a week of compute.

## Reproducibility

Every notebook is generated by a script in `scripts/build_*.py` and pushed with `scripts/push_kernels.py`.
The final submission is `notebooks/lb33_det.ipynb`. The full log of experiments, with each pre-registered
prediction and its outcome, is in `PLAN.md` and `docs/PREREG_*.md`. The figures come from
`scripts/make_writeup_figs.py`.

**Acknowledgements.** This work stands on NVARC (Sorokin and Puget), on the public LB33 notebook, and on the
ARChitects' TTT and DFS recipe. The NVARC checkpoint is Apache 2.0.
