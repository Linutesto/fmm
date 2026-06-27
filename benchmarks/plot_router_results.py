#!/usr/bin/env python3
"""Figures from results/router_summary.json — dark, green/pink to match yandesbiens.com."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BG = "#0b0f10"; FG = "#d7dadc"; GRID = "#23292b"

COLOR = {
    "flat": "#ff5fd1",          # pink
    "oracle_leaf": "#5ef08a",   # green (the ceiling)
    "router_top1": "#ffcf5e",   # amber
    "router_top3": "#5ec8ff",   # blue
}
LABEL = {
    "flat": "flat scan (no routing)",
    "oracle_leaf": "oracle scope (true topic)",
    "router_top1": "centroid router · top-1",
    "router_top3": "centroid router · top-3",
}


def _style(ax, title, xl, yl):
    ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=13, pad=12)
    ax.set_xlabel(xl, color=FG); ax.set_ylabel(yl, color=FG)
    ax.tick_params(colors=FG); ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)


def _modes(res):
    # preserve the canonical order; router_topR mode name carries the R value
    seen = [r["mode"] for r in res]
    order = ["flat", "oracle_leaf", "router_top1"]
    order += sorted({m for m in seen if m.startswith("router_top") and m != "router_top1"})
    return order


def series(res, sweep, mode, xkey, ykey):
    rows = sorted([r for r in res if r["sweep"] == sweep and r["mode"] == mode], key=lambda r: r[xkey])
    return [r[xkey] for r in rows], [r[ykey] for r in rows]


def label_for(mode):
    if mode in LABEL:
        return LABEL[mode]
    if mode.startswith("router_top"):
        return f"centroid router · top-{mode.removeprefix('router_top')}"
    return mode


def color_for(mode):
    return COLOR.get(mode, "#5ec8ff")


def main():
    data = json.loads((RESULTS / "router_summary.json").read_text())
    res = data["results"]
    modes = _modes(res)

    # ---- fig 1: separability sweep — realized recall vs cluster scatter ----
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for m in modes:
        xs, ys = series(res, "sep", m, "eps", "recall_at_k")
        if xs:
            ax.plot(xs, ys, "o-", color=color_for(m), label=label_for(m), lw=2, ms=6)
    _style(ax, "Realized recall@k vs cluster overlap (higher is better)",
           "cluster scatter ε  →  less separable", "recall@k")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_router_separability.png", dpi=150, facecolor=BG); plt.close(fig)

    # ---- fig 2: routing accuracy vs cluster scatter (the mechanism) ----
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for m in [x for x in modes if x.startswith("router_top")]:
        xs, ys = series(res, "sep", m, "eps", "routing_acc")
        if xs:
            ax.plot(xs, ys, "o-", color=color_for(m), label=label_for(m), lw=2, ms=6)
    # chance line: top-r out of n_leaf
    n_leaf = data["env"]["config"]["n_domains"] * data["env"]["config"]["n_sub"]
    ax.axhline(1.0 / n_leaf, color="#7a7a7a", ls="--", lw=1, label=f"chance (top-1 / {n_leaf})")
    _style(ax, "Routing accuracy vs cluster overlap (is the true topic in the routed set?)",
           "cluster scatter ε  →  less separable", "routing accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_router_accuracy.png", dpi=150, facecolor=BG); plt.close(fig)

    # ---- fig 3: size sweep — latency vs store size (router ~= oracle, << flat) ----
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for m in modes:
        xs, ys = series(res, "size", m, "n", "latency_ms")
        if xs:
            ax.plot(xs, ys, "o-", color=color_for(m), label=label_for(m), lw=2, ms=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, "Query latency vs store size — routing cost included (lower is better)",
           "items in memory (log)", "ms / query (log)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_router_latency.png", dpi=150, facecolor=BG); plt.close(fig)

    for f in ["fig_router_separability.png", "fig_router_accuracy.png", "fig_router_latency.png"]:
        print("wrote", RESULTS / f)


if __name__ == "__main__":
    main()
