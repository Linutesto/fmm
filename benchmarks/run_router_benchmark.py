#!/usr/bin/env python3
"""
FMM router benchmark — does a *cheap* topic router capture the scoping win?

Proof drop #2 showed that topic-scoped retrieval is far faster and more accurate
than a flat scan *when the topic is known*, and that a misrouted scope has recall
0.0. That left one question open: in production nobody hands you the topic — a
router has to guess it. So the misrouted case is not a freak event, it is whatever
fraction of the time the router is wrong.

This benchmark replaces the oracle with an actual router and measures the
END-TO-END result. The router is deliberately the cheapest thing that could work:
one centroid per leaf subtree (the mean of its items — a descriptor the fractal
store already has), and route the query to the nearest centroid(s) by cosine.
Routing to the top-r leaves trades a little scope width for robustness.

We compare, per condition:

  flat            cosine over ALL N items                         (no routing)
  oracle_leaf     scope to the TRUE leaf                          (proof drop #2 ceiling)
  router_top1     scope to the router's single best-guess leaf    (cheap, brittle)
  router_topR     scope to the union of the router's top-r leaves (cheap, robust)

Two sweeps:
  - separability sweep: vary cluster scatter (eps) at fixed N — how organized must
                 the memory be for the cheap router to work? This turns proof drop
                 #2's binary "misrouted = 0" into a continuous curve governed by how
                 separable the topics actually are.
  - size sweep:  vary N at a separability where routing works — does the cheap
                 router preserve the latency/recall win as the store grows?

Metrics: median query latency (ms, router cost included), realized recall@k, and
routing accuracy (was the true leaf inside the routed top-r?). Honest by design.

Usage:
    python run_router_benchmark.py
    python run_router_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from corpus import build_corpus  # noqa: E402
from fmm import FractalMemoryMatrix  # noqa: E402

RESULTS = HERE / "results"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def leaf_centroids(V: torch.Tensor, leaf_of: torch.Tensor, n_leaf: int) -> torch.Tensor:
    """One unit-norm centroid per leaf = mean of that leaf's item vectors.

    This is the router's whole model: a cheap descriptor built once from data the
    store already holds (no training, no second pass over queries).
    """
    dim = V.shape[1]
    sums = torch.zeros(n_leaf, dim)
    sums.index_add_(0, leaf_of, V)
    counts = torch.zeros(n_leaf).index_add_(0, leaf_of, torch.ones(V.shape[0]))
    cent = sums / counts.clamp_min(1).unsqueeze(1)
    return F.normalize(cent, dim=1)


def topk_recall(scores: torch.Tensor, ids: torch.Tensor, target: int, k: int) -> int:
    kk = min(k, scores.shape[0])
    top = ids[torch.topk(scores, kk).indices]
    return int((top == target).any().item())


def timed(fn, repeats: int):
    """median per-call wall time in ms, plus the last return value."""
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], out


# ---------------------------------------------------------------------------
# one condition (fixed N and query noise) → one row per mode
# ---------------------------------------------------------------------------
def run_condition(V, dom, leaf, n_leaf, n_domains, qnoise, cfg, sweep, n, eps):
    N, dim = V.shape
    allids = torch.arange(N)
    leaf_idx = {l: (leaf == l).nonzero(as_tuple=True)[0] for l in range(n_leaf)}
    C = leaf_centroids(V, leaf, n_leaf)  # [n_leaf, dim] — built once per condition (index time)
    R = cfg["top_r"]

    g = torch.Generator().manual_seed(cfg["seed"] + 7)
    modes = ["flat", "oracle_leaf", "router_top1", f"router_top{R}"]
    acc = {m: {"lat": [], "rec": [], "route_hit": [], "scope": []} for m in modes}
    router_overhead = []

    for _ in range(cfg["queries"]):
        ti = int(torch.randint(0, N, (1,), generator=g).item())
        q = F.normalize(V[ti] + qnoise * torch.randn(dim, generator=g), dim=0)
        true_leaf = int(leaf[ti].item())

        def score(idx):
            return V[idx] @ q  # cosine over unit vectors

        # the router op itself (what the query pays to choose a scope): n_leaf x dim
        def route_scores():
            return C @ q

        # --- flat: no routing ---
        lat, s = timed(lambda: score(allids), cfg["repeats"])
        acc["flat"]["lat"].append(lat)
        acc["flat"]["rec"].append(topk_recall(s, allids, ti, cfg["k"]))
        acc["flat"]["scope"].append(N)
        acc["flat"]["route_hit"].append(1)  # n/a — flat can't misroute

        # --- oracle: scope to the TRUE leaf (the ceiling) ---
        idx = leaf_idx[true_leaf]
        lat, s = timed(lambda: score(idx), cfg["repeats"])
        acc["oracle_leaf"]["lat"].append(lat)
        acc["oracle_leaf"]["rec"].append(topk_recall(s, idx, ti, cfg["k"]))
        acc["oracle_leaf"]["scope"].append(idx.numel())
        acc["oracle_leaf"]["route_hit"].append(1)

        # router prediction (shared by both router modes)
        ov, rs = timed(route_scores, cfg["repeats"])
        router_overhead.append(ov)
        ranked = torch.topk(rs, R).indices.tolist()
        pred1 = ranked[0]
        predR = ranked

        # --- router top-1: route then scope to the single best leaf ---
        idx1 = leaf_idx[pred1]
        lat, s = timed(lambda: score(idx1), cfg["repeats"])
        acc["router_top1"]["lat"].append(lat + ov)  # router cost is part of the query
        acc["router_top1"]["rec"].append(topk_recall(s, idx1, ti, cfg["k"]))
        acc["router_top1"]["scope"].append(idx1.numel())
        acc["router_top1"]["route_hit"].append(int(true_leaf == pred1))

        # --- router top-R: union of the R best leaves ---
        idxR = torch.cat([leaf_idx[p] for p in predR])
        lat, s = timed(lambda: score(idxR), cfg["repeats"])
        acc[f"router_top{R}"]["lat"].append(lat + ov)
        acc[f"router_top{R}"]["rec"].append(topk_recall(s, idxR, ti, cfg["k"]))
        acc[f"router_top{R}"]["scope"].append(idxR.numel())
        acc[f"router_top{R}"]["route_hit"].append(int(true_leaf in predR))

    rows = []
    for m in modes:
        a = acc[m]
        lat = sorted(a["lat"])[len(a["lat"]) // 2]
        rows.append({
            "sweep": sweep,
            "n": N,
            "eps": round(eps, 3),
            "qnoise": round(qnoise, 3),
            "mode": m,
            "scope_size": int(sum(a["scope"]) / len(a["scope"])),
            "latency_ms": round(lat, 4),
            "recall_at_k": round(sum(a["rec"]) / len(a["rec"]), 4),
            "routing_acc": round(sum(a["route_hit"]) / len(a["route_hit"]), 4),
        })
    overhead_ms = round(sorted(router_overhead)[len(router_overhead) // 2], 5)
    return rows, overhead_ms


# ---------------------------------------------------------------------------
# library correctness: the lib's route() agrees with the benchmark's router
# ---------------------------------------------------------------------------
def library_correctness_check(cfg: dict) -> dict:
    """The fmm route() must (a) stay inside the routed subtree and (b) pick the
    right leaf on an easy, well-separated query."""
    dim = 48
    mem = FractalMemoryMatrix(dim=dim, max_nodes=5000)
    g = torch.Generator().manual_seed(3)
    centers = F.normalize(torch.randn(4, dim, generator=g), dim=1)
    topics = [("A", "x"), ("A", "y"), ("B", "p"), ("B", "q")]
    for c, t in zip(centers, topics):
        for _ in range(30):
            mem.add_node(F.normalize(c + 0.1 * torch.randn(dim, generator=g), dim=0), topic=t)
    q = F.normalize(centers[2] + 0.1 * torch.randn(dim, generator=g), dim=0)  # near ("B","p")
    routed_leaf = mem.route(q, top_r=1, level=2)
    routed_dom = mem.route(q, top_r=1, level=1)
    rr = mem.route_and_retrieve(q, top_k=5, top_r=1, level=2)
    return {
        "route_picks_correct_leaf": routed_leaf[0][0] == ("B", "p"),
        "route_picks_correct_domain": routed_dom[0][0] == ("B",),
        "route_and_retrieve_stays_in_scope": all(n.topic == ("B", "p") for n, _ in rr),
        "retrieved": len(rr),
    }


def _print_rows(rows):
    for r in rows:
        print(f"{r['sweep']:<6}{r['mode']:<14}{r['n']:>8}{r['eps']:>7}{r['qnoise']:>8}{r['scope_size']:>8}"
              f"{r['latency_ms']:>9}{r['recall_at_k']:>10}{r['routing_acc']:>11}")


def main():
    ap = argparse.ArgumentParser()
    # operating point for the size sweep — an eps where the cheap router works
    ap.add_argument("--ns", nargs="+", type=int, default=[2000, 8000, 32000, 128000])
    ap.add_argument("--size-eps", type=float, default=0.10, help="cluster scatter for the size sweep")
    ap.add_argument("--size-noise", type=float, default=0.15, help="query noise (both sweeps)")
    # separability sweep — vary cluster scatter at fixed N
    ap.add_argument("--sep-n", type=int, default=32000, help="store size for the separability sweep")
    ap.add_argument("--epses", nargs="+", type=float, default=[0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.45])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-domains", type=int, default=16)
    ap.add_argument("--n-sub", type=int, default=8)
    ap.add_argument("--top-r", type=int, default=3)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.ns = [2000, 8000]
        args.sep_n = 8000
        args.epses = [0.05, 0.15, 0.45]
        args.queries = 50

    cfg = dict(dim=args.dim, n_domains=args.n_domains, n_sub=args.n_sub,
               top_r=args.top_r, queries=args.queries, repeats=args.repeats, k=args.k, seed=args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)

    env = {
        "torch": torch.__version__, "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(), "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg,
        "size_eps": args.size_eps, "size_noise": args.size_noise, "sep_n": args.sep_n,
    }
    check = library_correctness_check(cfg)
    n_leaf = args.n_domains * args.n_sub
    print(f"# FMM router benchmark | torch {env['torch']} cpu({env['cpu_threads']}t) | "
          f"dim={cfg['dim']} leaves={n_leaf} top_r={cfg['top_r']} qnoise={args.size_noise}")
    print(f"# library route() correctness: {check}\n")
    hdr = (f"{'sweep':<6}{'mode':<14}{'N':>8}{'eps':>7}{'qnoise':>8}{'scope':>8}"
           f"{'lat_ms':>9}{'recall@k':>10}{'route_acc':>11}")

    results = []
    overheads = {}

    # ---- separability sweep: vary cluster scatter (eps) at fixed N ----
    print(f"## separability sweep  (N = {args.sep_n}, query noise = {args.size_noise})")
    print(hdr); print("-" * len(hdr))
    for eps in args.epses:
        c = build_corpus(args.sep_n, cfg["dim"], cfg["n_domains"], cfg["n_sub"], eps, cfg["seed"])
        rows, ov = run_condition(c["V"], c["domain_of"], c["leaf_of"], c["n_leaf"],
                                 c["n_domains"], args.size_noise, cfg, "sep", args.sep_n, eps)
        overheads[f"sep_eps{eps}"] = ov
        results.extend(rows)
        _print_rows(rows)
        print()

    # ---- size sweep: vary N at an eps where routing works ----
    print(f"## size sweep  (eps = {args.size_eps}, query noise = {args.size_noise})")
    print(hdr); print("-" * len(hdr))
    for n in args.ns:
        c = build_corpus(n, cfg["dim"], cfg["n_domains"], cfg["n_sub"], args.size_eps, cfg["seed"])
        rows, ov = run_condition(c["V"], c["domain_of"], c["leaf_of"], c["n_leaf"],
                                 c["n_domains"], args.size_noise, cfg, "size", n, args.size_eps)
        overheads[f"size_N{c['n']}"] = ov
        results.extend(rows)
        _print_rows(rows)
        print(f"   (router overhead ≈ {ov} ms/query)")
    print()

    summary = {"env": env, "correctness": check, "router_overhead_ms": overheads, "results": results}
    (RESULTS / "router_summary.json").write_text(json.dumps(summary, indent=2))
    with open(RESULTS / "router_runs.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"# wrote {RESULTS/'router_summary.json'}\n# next: python plot_router_results.py")


if __name__ == "__main__":
    main()
