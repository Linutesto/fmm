#!/usr/bin/env python3
"""Figures from results/summary.json — dark, green/pink to match yandesbiens.com."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BG = "#0b0f10"; FG = "#d7dadc"; GRID = "#23292b"
COLOR = {"flat": "#ff5fd1", "scoped_domain": "#ffcf5e", "scoped_leaf": "#5ef08a", "misrouted": "#7a7a7a"}
LABEL = {"flat": "flat scan (whole store)", "scoped_domain": "scoped: domain subtree",
         "scoped_leaf": "scoped: leaf subtree", "misrouted": "misrouted scope (wrong topic)"}


def _style(ax, title, xl, yl):
    ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=13, pad=12)
    ax.set_xlabel(xl, color=FG); ax.set_ylabel(yl, color=FG)
    ax.tick_params(colors=FG); ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)


def series(res, mode, key):
    rows = sorted([r for r in res if r["mode"] == mode], key=lambda r: r["n"])
    return [r["n"] for r in rows], [r[key] for r in rows]


def main():
    data = json.loads((RESULTS / "summary.json").read_text())
    res = data["results"]

    # fig 1 — latency
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for m in ["flat", "scoped_domain", "scoped_leaf"]:
        xs, ys = series(res, m, "latency_ms")
        ax.plot(xs, ys, "o-", color=COLOR[m], label=LABEL[m], lw=2, ms=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, "Retrieval latency vs. store size (lower is better)", "items in memory (log)", "ms / query (log)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_latency.png", dpi=150, facecolor=BG); plt.close(fig)

    # fig 2 — recall
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for m in ["flat", "scoped_domain", "scoped_leaf", "misrouted"]:
        xs, ys = series(res, m, "recall_at_k")
        ax.plot(xs, ys, "o-", color=COLOR[m], label=LABEL[m], lw=2, ms=6)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    _style(ax, "Recall@k vs. store size (higher is better)", "items in memory (log)", "recall@k")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_recall.png", dpi=150, facecolor=BG); plt.close(fig)

    print("wrote", RESULTS / "fig_latency.png")
    print("wrote", RESULTS / "fig_recall.png")


if __name__ == "__main__":
    main()
