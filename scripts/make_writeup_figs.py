"""Figures for the ARC Prize 2026 solution writeup (docs/writeup/). All numbers are measured results
recorded in PLAN.md, docs/*.md and the PREREG files; nothing here is estimated.
Run: python scripts/make_writeup_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")
OUT = "docs/writeup/figs"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, ORANGE, RED, GREEN, GREY = "#2563eb", "#ea580c", "#dc2626", "#16a34a", "#9ca3af"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
CHAMP, CHAMP_SD = 30.69, 0.415


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1. Pipeline -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 3.0))
ax.set_xlim(0, 12.4); ax.set_ylim(0, 3); ax.axis("off")
boxes = [
    (0.05, "Base model", "Qwen3-4B, 16-token grid\nvocabulary, NVARC SFT\n(sft139, public)", GREY),
    (2.5, "Test-time training", "per task: LoRA r=256\n128 seqs = 8 D4 x 16 color\nperms, 1 epoch, lr 5e-5", BLUE),
    (4.95, "DFS decoding", "16 augmented views\nkeep every path with\ncum. prob > 0.2", BLUE),
    (7.4, "Augmented re-scoring", "each candidate scored\nunder 8 fresh augs\n(md5-stable seeds)", BLUE),
    (9.85, "Selection", "votes + likelihood\n-> 2 attempts\ncheap tasks first", BLUE),
]
for x, title, body, col in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.35), 2.25, 2.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=col, lw=2))
    ax.text(x + 1.125, 2.3, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col)
    ax.text(x + 1.125, 1.3, body, ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.45)
for x in (2.27, 4.72, 7.17, 9.62):
    ax.annotate("", xy=(x + 0.2, 1.5), xytext=(x, 1.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
ax.text(6.2, 0.08, "4 x L4 GPUs, one worker per GPU, 12 h budget for 240 hidden tasks; the pipeline uses ~95% of it.",
        ha="center", fontsize=9, color=MUTED)
save(fig, "fig1_pipeline.png")

# 2. Variance: old champion vs deterministic champion ----------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
old = [30.14, 29.31, 27.64]; det = [31.11, 30.69, 30.28]
for i, (vals, col, lab) in enumerate([(old, GREY, "Baseline + cheap-first\n(same kernel, 3 reruns)"),
                                       (det, BLUE, "+ stable scoring seeds\n+ no per-task clock cap")]):
    ax.scatter([i] * 3, vals, s=70, color=col, zorder=3)
    m = sum(vals) / 3
    ax.hlines(m, i - 0.18, i + 0.18, color=col, lw=2.5)
    ax.text(i + 0.24, m, "%.2f" % m, va="center", color=col, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline + cheap-first\n(same kernel, 3 reruns)", "+ stable scoring seeds\n+ no per-task clock cap"])
ax.set_xlim(-0.5, 1.7); ax.set_ylabel("Public LB score (%)")
ax.set_title("Same model, same code: reruns moved by up to 2.5 points.\nRemoving two noise sources: +1.66 points and 3x less spread",
             fontsize=10.5, loc="left", color=INK)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
save(fig, "fig2_variance.png")

# 3. Leaderboard map of everything we tried ---------------------------------------------------
rows = [
    ("Final: lb33-det (n=3)", 30.69, BLUE),
    ("NVARC+ v2 public pipeline (n=2)", 30.21, GREY),
    ("TTT LoRA rank 64", 29.72, ORANGE),
    ("Beam-8 on empty candidate pools", 29.31, ORANGE),
    ("4B extra SFT + faster TTT/scoring", 28.33, ORANGE),
    ("Ensemble with TRM (7M recursive)", 27.22, ORANGE),
    ("4B extra SFT (ARC-Heavy)", 27.08, ORANGE),
    ("TTT: 4 D4 x 32 colors (same cost)", 26.39, ORANGE),
    ("TTT: half an epoch", 26.39, ORANGE),
    ("TTT: largest demo always last", 25.69, ORANGE),
    ("TTT LoRA rank 512", 24.31, ORANGE),
    ("Qwen3.8-27B reasoning agent (n=2)", 15.90, RED),
    ("No test-time training", 9.58, RED),
]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
ys = list(range(len(rows)))[::-1]
ax.axvspan(CHAMP - 2 * CHAMP_SD, CHAMP + 2 * CHAMP_SD, color=BLUE, alpha=0.10, lw=0)
ax.axvline(CHAMP, color=BLUE, lw=1, ls="--")
for y, (lab, v, col) in zip(ys, rows):
    ax.barh(y, v, color=col, height=0.62)
    ax.text(v + 0.4, y, "%.2f" % v, va="center", fontsize=9, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_xlim(0, 36); ax.set_xlabel("Hidden-test score, public LB (%)")
ax.set_title("Every leaderboard experiment, against the final pipeline (shaded = its +/-2 sd noise band)",
             fontsize=10.5, loc="left")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
save(fig, "fig3_leaderboard_map.png")

# 4. The clock cliff ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cost = [-11.0, 0.0, 11.9]; score = [29.72, 30.69, 24.31]; lab = ["LoRA r64\n(-11% time)", "r256\n(final)", "LoRA r512\n(+11.9% time)"]
ax.axvspan(4.8, 14, color=RED, alpha=0.08, lw=0)
ax.text(5.3, 30.2, "past the 12 h deadline:\ntail tasks never run", color=RED, fontsize=9)
ax.plot(cost, score, color=MUTED, lw=1, zorder=1)
ax.scatter(cost, score, s=[60, 110, 60], color=[ORANGE, BLUE, RED], zorder=3)
for c, s, l in zip(cost, score, lab):
    ax.text(c, s - 1.2, l, ha="center", va="top", fontsize=9)
ax.axvline(4.8, color=RED, lw=1, ls=":")
ax.set_xlim(-14, 14); ax.set_ylim(20, 32.5)
ax.set_xlabel("Change in compute per task vs final (%)"); ax.set_ylabel("Public LB score (%)")
ax.set_title("The pipeline runs at 95% of its budget: saving time buys nothing,\nspending 12% more loses 15 tasks",
             fontsize=10.5, loc="left")
ax.grid(color=GRID); ax.set_axisbelow(True)
save(fig, "fig4_clock_cliff.png")

# 5. Why it is hard: three zero-GPU measurements ---------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
a = axs[0]
v = [78, 67, 10]; l = ["known rule,\nknown scale", "known rule,\n5.75x larger", "new rule\n(AGI-2 eval)"]
a.bar(l, v, color=[GREEN, ORANGE, RED], width=0.6)
for i, x in enumerate(v): a.text(i, x + 2, "%d%%" % x, ha="center", fontweight="bold")
a.set_ylim(0, 92); a.set_ylabel("greedy exact, no TTT (%)")
a.set_title("A 5.75x larger grid costs 11 pt;\na new rule costs 68 pt (119 tasks)", fontsize=10, loc="left")
a = axs[1]
v = [0.500, 0.750, 1.161]; l = ["AGI-1", "AGI-2\ntrain", "AGI-2\neval"]
a.bar(l, v, color=[GREEN, ORANGE, RED], width=0.6)
for i, x in enumerate(v): a.text(i, x + 0.03, "%.2f" % x, ha="center", fontweight="bold")
a.set_ylim(0, 1.35); a.set_ylabel("median distance of test input\noutside the demo envelope")
a.set_title("Test inputs sit further outside\nthe demos (p=2e-8)", fontsize=10, loc="left")
a = axs[2]
v = [5.1, 0.0, 0.0]; l = ["AGI-1\ntrain", "AGI-2\ntrain", "AGI-2\neval"]
a.bar(l, v, color=[GREEN, ORANGE, RED], width=0.6)
for i, x in enumerate(v): a.text(i, x + 0.15, "%.1f%%" % x, ha="center", fontweight="bold")
a.set_ylim(0, 6.5); a.set_ylabel("tasks solved by a 25-op\ngrid DSL, depth 2 (%)")
a.set_title("Global grid programs\nstop applying (353 tasks)", fontsize=10, loc="left")
for a in axs:
    a.grid(axis="y", color=GRID); a.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig5_why_hard.png")

# 6. Base-model swap: Qwen3.5-4B vs Qwen3-4B, paired SFT -------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
pre = {"Qwen3-4B-Base": [0.275, 0.275], "Qwen3.5-4B-Base": [0.262] * 4}
post = {"Qwen3-4B-Base": [0.176, 0.177], "Qwen3.5-4B-Base": [0.164, 0.173, 0.160, 0.176]}
for i, (name, col) in enumerate([("Qwen3-4B-Base", GREY), ("Qwen3.5-4B-Base", ORANGE)]):
    xs_pre = [i * 0.35 + 0.0] * len(pre[name]); xs_post = [1 + i * 0.35] * len(post[name])
    ax.scatter(xs_pre, pre[name], color=col, s=55, zorder=3, label=name)
    ax.scatter(xs_post, post[name], color=col, s=55, zorder=3)
ax.set_xticks([0.17, 1.17]); ax.set_xticklabels(["before SFT", "after 9K-sequence SFT\n(same data, same recipe)"])
ax.set_xlim(-0.4, 1.8); ax.set_ylim(0.14, 0.29)
ax.set_ylabel("NLL per cell of the true answer\n(36 eval tasks, lower = better)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("A newer base starts 5% better (28/36, p=0.0006)\nbut the gap vanishes after SFT (12-22 of 36, p>=0.12)",
             fontsize=10.5, loc="left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
save(fig, "fig6_base_swap.png")
