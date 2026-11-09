# Measuring Before Moving: a Noise-Aware Study of Test-Time Training on ARC-AGI-2

**Julian Camilo Villaduque** · ARC Prize 2026, ARC-AGI-2 track · Final submission: `lb33-det`,
**30.69 ± 0.42 %** on the public leaderboard (3 reruns; best single run 31.11 %)

Code (MIT): https://github.com/jvilladuque90/arc-prize-2026-arc-agi-2

---

## Summary

I started from the strongest public ARC-AGI-2 pipeline, the NVARC model (`sft139`, 1st place 2025) with
per-task test-time training (TTT) and depth-first decoding. Then I spent the competition on one question:
**which changes really move the hidden-test score, and which only look like they do?**

1. **The leaderboard was noisier than the effects I was chasing.** Three reruns of one unchanged
   notebook scored 30.14, 29.31 and 27.64. I traced the noise to two causes in the code and removed both:
   **+1.66 points and a 3× narrower spread**, at zero extra compute.
2. **Test-time training is the engine, and its recipe is at a local optimum.** Removing TTT costs 21
   points. All five changes to its recipe that I sent to the leaderboard lost, by 1 to 6 points.
3. **The remaining gap is generalization to new rules, not data, scale or decoding.** Cheap
   measurements, most with no GPU, locate the difficulty and explain why larger models, more synthetic
   data and program search did not help.

I report every leaderboard experiment, including the negative ones (Figure 3).

---

## 1. The pipeline

![Pipeline](figs/fig1_pipeline.png)

| Stage | What it does | My change |
|---|---|---|
| Base model | Qwen3-4B, vocabulary cut to 16 tokens, SFT by NVARC on 266K synthetic puzzles | none |
| Test-time training | Per task, a fresh LoRA adapter (rank 256) trained 1 epoch on 128 augmented copies of the demos: 8 rotations/reflections × 16 color permutations | none |
| DFS decoding | Under 16 augmented views, keep every output with cumulative probability above 0.2 | **no per-task time cap** |
| Re-scoring | Score each candidate under 8 fresh augmentations | **stable md5 seeds** instead of salted `hash()` |
| Selection | Votes plus likelihood, 2 attempts | **cheapest tasks first** |

Four L4 GPUs run one worker each; the pipeline uses about 95 % of the 12-hour budget.

---

## 2. First, fix the instrument

![Variance](figs/fig2_variance.png)

Rerunning the *unchanged* notebook showed a spread of 2.5 points (σ = 1.27). Most A/B tests I had read
were inside that band. The champion's logs revealed two mechanical noise sources:

- **Salted seeds.** Re-scoring drew augmentations with `hash(task_key)`, which Python salts per process,
  so every run scored candidates under different views. An md5 hash fixes it.
- **A per-task clock cap that tasks barely cleared.** One task started its last decoding batch 40 seconds
  before its 20-minute limit; a 3.5 % change in machine speed decided 16 views or 12. The global
  deadline already protects the submission, so I removed the cap.

With both fixes three reruns scored **31.11, 30.69 and 30.28**: the worst beats the best of the old
version (Mann-Whitney p = 0.05). From then on every experiment had a **pre-registered prediction** (adopt,
tie, discard bands written before the score), paired comparisons, and medians and sign tests next to the
mean. In one sweep the mean said "no change" while 10 of 12 tasks got worse.

---

## 3. What I tried on the leaderboard

![Leaderboard map](figs/fig3_leaderboard_map.png)

Each bar is one hidden-test submission; the band is ±2 standard deviations of the final pipeline.

**Test-time training is the engine.** Without TTT the same model scores 9.58: TTT is worth **+21 points
for 35 % of the compute**. Its recipe is a narrow optimum:

| Change to TTT | Score | Lesson |
|---|---|---|
| half an epoch | 26.39 | half of the views go unadapted |
| largest demo always last | 25.69 | demo-order diversity matters: −12 tasks |
| 4 rotations × 32 color permutations (same cost) | 26.39 | geometric diversity cannot be swapped for color |
| LoRA rank 64 / 512 | 29.72 / 24.31 | see the clock, below |

On an 18-task bench the color-heavy variant looked neutral; on the hidden test it lost 4.3 points.
**A small bench "inside the noise" is not a green light.**

![Clock cliff](figs/fig4_clock_cliff.png)

**The clock is a cliff.** Rank 512 added 11.9 % compute per task and lost 15 tasks, because the last
tasks in the queue never ran. Rank 64 saved 11 % and gained nothing. The three points give the budget:
**any change must cost less than about 4.8 % per task**, which rules out "thinking" tokens, more views
and more TTT steps.

**Other directions, all on the hidden test:**

- **Ensembles with a weaker partner** (a 2B model, the 7M-parameter TRM) lost up to 3 points: the 4B
  model is coverage-limited and needs all the compute.
- **More SFT for the 4B** (ARC-Heavy) gave 27.08 / 28.33: richer candidate pools cost decoding time
  and coverage collapsed. The same SFT gave **+2.2 points** to a 2B model far below its time budget.
- **Wider search on empty candidate pools** (beam 8): 29.31. An empty pool means the model does not
  know the answer.
- **A 27B reasoning model** (Qwen3.8-27B, FP8) with a public agent scored 16.94 and 14.86; every task it
  solved was already solved by my pipeline.

---

## 4. Why it is hard to climb

![Why it is hard](figs/fig5_why_hard.png)

- **Rule novelty dominates scale** (left). On 119 AGI-1 tasks paired with larger re-ARC inputs, the model
  solves 78 % of known rules at known size, 67 % at 5.75× the size, and 10 % of new AGI-2 rules.
  Synthetic data only adds rules the model already knows.
- **Test inputs fall outside the demonstrations** (middle). Distance of the test input outside the range
  of its demos on 7 grid features grows from 0.50 (AGI-1) to 1.16 (AGI-2 eval, p = 2·10⁻⁸). The excess
  is content: 1.43× area, 1.45× ink, same object count.
- **The model learns the rule but fails the test.** After TTT it reproduces a held-out *demo* in 5 of 12
  tasks and the *test* in 0 of 12 (McNemar p = 0.031).
- **Program search does not reach AGI-2** (right). A 25-operator grid DSL solves 5.1 % of AGI-1 and 0 of
  353 AGI-2 tasks; an object-level DSL does no better. Rules depend on *relations* between objects.
  Exact verification against the demos, however, is 87 % precise.

The pipeline's ~31 % is not short of data, decoding or compute. It is short of **generalization to an
unseen rule at an unseen scale**, the very thing ARC-AGI-2 tests.

---

## 5. A newer base model

![Base swap](figs/fig6_base_swap.png)

I tested Qwen3.5-4B (24 Gated DeltaNet + 8 attention layers) against Qwen3-4B, *Base* against *Base*,
with a plain-text control. A naive comparison against the chat model had overstated the gap by 44 %.

- Before training, Qwen3.5 assigns 5 % lower loss to the true answers (28 of 36 tasks, p = 0.0006).
- After the **same** SFT (9K NVARC sequences, LoRA rank 64) the advantage disappears: four Qwen3.5
  seeds win 19, 12, 17 and 22 of 36 tasks (p ≥ 0.12). Qwen3.5 is also 8 % slower.

**A better prior, not faster learning.** I trained a full Qwen3.5 model and ported the pipeline (its
recurrent cache cannot be rewound, so the DFS snapshots it at every branch), but the weekly GPU quota ran
out before submission. The code is in `scripts/build_lb35.py`.

---

## 6. Lessons

1. **Measure leaderboard noise before trusting an A/B.** My biggest gain came from removing randomness.
2. **Pre-register** adopt/discard thresholds before the score arrives.
3. **Budget the clock before designing.** At 95 % utilization, a good idea that costs 12 % is worse than
   no idea.
4. **The bench must match production** in model, format and population. A 2B bench said TTT was
   worthless because the 2B model had memorized the evaluation set; silently dropped role tokens turned
   a family of experiments into a format-repair study.
5. **Ask the data first.** Laptop analyses closed directions that would each have cost a week of compute.

**Reproducibility.** Notebooks are generated by `scripts/build_*.py`; the final submission is
`notebooks/lb33_det.ipynb`; every pre-registered prediction and outcome is in `PLAN.md` and
`docs/PREREG_*.md`.

**Acknowledgements.** NVARC (Sorokin and Puget), the public LB33 notebook, and the ARChitects' TTT and
DFS recipe.
