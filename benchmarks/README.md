# FMM Benchmark — hierarchical scoping vs. flat semantic search

One question, tested honestly:

> Does **topic-scoped retrieval** ("page the relevant region of memory") beat a **flat
> scan** of the whole store — in latency *and* recall — as memory grows?

This is the semantic-memory analogue of the [UFM benchmark](https://yandesbiens.com/blog/ufm-benchmark/):
both are a **bet on locality**. UFM bets on locality in *physical* memory (VRAM/RAM); FMM
bets on it in *semantic* memory (the topic tree).

## Setup

A synthetic corpus organized into a topic hierarchy (`domain → subtopic`, 16×8 = 128 leaf
topics). Each leaf is a cluster; items are points near their cluster center. For each query
(a perturbed item with a *known* topic) we compare four retrieval strategies:

| mode | searches | analogue |
|---|---|---|
| `flat` | all N items | a flat vector store |
| `scoped_domain` | the query's domain subtree (~N/16) | coarse paging |
| `scoped_leaf` | the query's leaf subtree (~N/128) | fine paging |
| `misrouted` | the **wrong** domain subtree | the honest failure case |

Metrics: median query latency and recall@k (k=5). Vectors are unit-norm; cosine = dot product.

## Quick start

```bash
cd fmm && pip install -e ".[torch]" && pip install matplotlib
cd benchmarks && ./run.sh           # or: python run_benchmark.py && python plot_results.py
python run_benchmark.py --quick     # ~10s
```

The harness also runs a **library correctness check**: that `FractalMemoryMatrix.retrieve(
query, topic_prefix=…)` returns only nodes inside the requested subtree.

## Reproduced result

**Hardware:** CPU, 16 threads. **Software:** torch 2.10.0+cu128, Python 3.x, Linux.
**Corpus:** dim 128, 16 domains × 8 subtopics, eps 0.45, query noise 0.35, 200 queries, k=5.

| N | mode | scope | latency (ms) | recall@k |
|---:|---|---:|---:|---:|
| 128,000 | flat | 128,000 | 8.304 | 0.155 |
| 128,000 | scoped_domain | 8,000 | 0.354 | 0.365 |
| 128,000 | **scoped_leaf** | **1,000** | **0.051** | **0.58** |
| 128,000 | misrouted | 8,000 | 0.345 | 0.000 |

At 128k items, **leaf-scoped retrieval is ~164× faster and ~3.7× more accurate** than a flat
scan. The full sweep (2k → 128k) is in `results/summary.json`.

![latency](results/fig_latency.png)
![recall](results/fig_recall.png)

### What this shows

1. **Flat search degrades on both axes as memory grows.** Latency is ~linear in N; recall
   *falls* (0.48 → 0.155) because every unrelated topic adds distractors that crowd out the target.
2. **Scoping wins on both axes — when the topic is known.** Restricting the search to the
   relevant subtree is sublinear (≈ N/scope faster) *and* lifts recall, because cross-topic
   distractors are never considered. Finer scope (leaf) > coarser scope (domain).
3. **The honest cost: a misrouted scope misses entirely** (recall 0.0). If you scope to the
   wrong region, the target isn't there. Hierarchical memory is a bet on routing locality —
   it pays only when you can address the right region.

## Honest limitations

- **Synthetic embeddings.** This isolates the *structural* question (does scoping help?) from
  embedding quality. Real-embedding corpora + a learned topic router are future work; absolute
  recall will depend on both.
- **Absolute recall is moderate** (within-scope distractors remain at high density); the
  defensible claim is the **relative** advantage and the **scaling trend**, not a recall number.
- **Topic is assumed known** at query time. Production needs a router to choose the scope; a bad
  router lands you in the `misrouted` regime. Measuring router quality is the next proof.
- CPU, cosine via dot product; the harness measures the core retrieval op (a production system
  would also add an ANN index — orthogonal to the scoping question).

## Files

```
corpus.py          synthetic hierarchical corpus
run_benchmark.py   flat vs scoped sweep + library correctness check
plot_results.py    figures
run.sh             run + plot
results/           summary.json, runs.jsonl, figures
```
