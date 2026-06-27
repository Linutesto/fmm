#!/usr/bin/env python3
"""
FMM benchmark — flat vs. topic-scoped semantic retrieval.

Question: does hierarchical scoping ("page the relevant region") beat a flat scan
of the whole memory? We sweep store size N and, for each query whose topic is known,
compare:

  flat            cosine over ALL N items                 (a flat vector store)
  scoped_domain   cosine over the query's domain subtree  (~N / n_domains items)
  scoped_leaf     cosine over the query's leaf subtree     (~N / n_leaf items)
  misrouted       scoped to the WRONG domain               (the honest failure case)

We measure mean query latency and recall@k (is the ground-truth item retrieved?).
Honest by design: the misrouted case is included to show scoping's downside.

Usage:
    python run_benchmark.py                 # default sweep
    python run_benchmark.py --quick
"""

from __future__ import annotations

import argparse, json, platform, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from corpus import build_corpus  # noqa: E402
from fmm import FractalMemoryMatrix  # noqa: E402

RESULTS = HERE / "results"


def topk_recall(scores: torch.Tensor, ids: torch.Tensor, target: int, k: int) -> int:
    kk = min(k, scores.shape[0])
    top = ids[torch.topk(scores, kk).indices]
    return int((top == target).any().item())


def timed(fn, repeats: int) -> tuple[float, object]:
    # median of per-call wall time (ms)
    ts = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], out


def run_n(n: int, cfg: dict) -> list[dict]:
    c = build_corpus(n, cfg["dim"], cfg["n_domains"], cfg["n_sub"], cfg["eps"], cfg["seed"])
    V, dom, leaf = c["V"], c["domain_of"], c["leaf_of"]
    N = c["n"]
    allids = torch.arange(N)
    # precompute scope index lists once (the tree structure)
    dom_idx = {d: (dom == d).nonzero(as_tuple=True)[0] for d in range(c["n_domains"])}
    leaf_idx = {l: (leaf == l).nonzero(as_tuple=True)[0] for l in range(c["n_leaf"])}

    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    rows = {m: {"lat": [], "rec": []} for m in ["flat", "scoped_domain", "scoped_leaf", "misrouted"]}

    for _ in range(cfg["queries"]):
        ti = int(torch.randint(0, N, (1,), generator=g).item())
        q = torch.nn.functional.normalize(V[ti] + cfg["qnoise"] * torch.randn(cfg["dim"], generator=g), dim=0)
        d = int(dom[ti].item()); l = int(leaf[ti].item())
        wrong_d = (d + 1) % c["n_domains"]

        def score(idx):
            return V[idx] @ q  # cosine (unit vectors)

        lat, s = timed(lambda: score(allids), cfg["repeats"])
        rows["flat"]["lat"].append(lat); rows["flat"]["rec"].append(topk_recall(s, allids, ti, cfg["k"]))

        lat, s = timed(lambda: score(dom_idx[d]), cfg["repeats"])
        rows["scoped_domain"]["lat"].append(lat); rows["scoped_domain"]["rec"].append(topk_recall(s, dom_idx[d], ti, cfg["k"]))

        lat, s = timed(lambda: score(leaf_idx[l]), cfg["repeats"])
        rows["scoped_leaf"]["lat"].append(lat); rows["scoped_leaf"]["rec"].append(topk_recall(s, leaf_idx[l], ti, cfg["k"]))

        lat, s = timed(lambda: score(dom_idx[wrong_d]), cfg["repeats"])
        rows["misrouted"]["lat"].append(lat); rows["misrouted"]["rec"].append(topk_recall(s, dom_idx[wrong_d], ti, cfg["k"]))

    out = []
    for m, r in rows.items():
        lat = sorted(r["lat"])[len(r["lat"]) // 2]
        out.append({
            "mode": m, "n": N,
            "scope_size": (N if m == "flat" else (N // c["n_domains"] if "domain" in m or m == "misrouted" else N // c["n_leaf"])),
            "latency_ms": round(lat, 4),
            "recall_at_k": round(sum(r["rec"]) / len(r["rec"]), 4),
        })
    return out


def library_correctness_check(cfg: dict) -> dict:
    """Prove the library's scoped retrieve actually restricts to the topic subtree."""
    mem = FractalMemoryMatrix(dim=cfg["dim"], max_nodes=10_000)
    g = torch.Generator().manual_seed(7)
    ca = torch.nn.functional.normalize(torch.randn(cfg["dim"], generator=g), dim=0)
    cb = torch.nn.functional.normalize(torch.randn(cfg["dim"], generator=g), dim=0)
    for _ in range(50):
        mem.add_node(torch.nn.functional.normalize(ca + 0.05 * torch.randn(cfg["dim"], generator=g), dim=0), topic=("A", "x"))
        mem.add_node(torch.nn.functional.normalize(cb + 0.05 * torch.randn(cfg["dim"], generator=g), dim=0), topic=("B", "y"))
    q = torch.nn.functional.normalize(ca + 0.05 * torch.randn(cfg["dim"], generator=g), dim=0)
    scoped_b = mem.retrieve(q, top_k=10, topic_prefix=("B",))
    ok = all(n.topic[0] == "B" for n, _ in scoped_b)
    return {"scoped_retrieve_respects_topic": ok, "scoped_b_count": len(scoped_b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", nargs="+", type=int, default=[2000, 8000, 32000, 128000])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-domains", type=int, default=16)
    ap.add_argument("--n-sub", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.45)
    ap.add_argument("--qnoise", type=float, default=0.35)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.ns = [2000, 8000]; args.queries = 50

    cfg = dict(dim=args.dim, n_domains=args.n_domains, n_sub=args.n_sub, eps=args.eps,
               qnoise=args.qnoise, queries=args.queries, repeats=args.repeats, k=args.k, seed=args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    env = {
        "torch": torch.__version__, "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(), "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg,
    }
    check = library_correctness_check(cfg)
    print(f"# FMM benchmark | torch {env['torch']} cpu({env['cpu_threads']}t) | "
          f"dim={cfg['dim']} domains={cfg['n_domains']} sub={cfg['n_sub']} (leaves={cfg['n_domains']*cfg['n_sub']})")
    print(f"# library scoped-retrieve correctness: {check}\n")
    hdr = f"{'mode':<15}{'N':>8}{'scope':>9}{'lat_ms':>10}{'recall@k':>10}"
    print(hdr); print("-" * len(hdr))

    results = []
    for n in args.ns:
        for r in run_n(n, cfg):
            results.append(r)
            print(f"{r['mode']:<15}{r['n']:>8}{r['scope_size']:>9}{r['latency_ms']:>10}{r['recall_at_k']:>10}")
        print()

    summary = {"env": env, "correctness": check, "results": results}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(RESULTS / "runs.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"# wrote {RESULTS/'summary.json'}\n# next: python plot_results.py")


if __name__ == "__main__":
    main()
