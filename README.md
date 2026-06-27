# FMM — Fractal Memory

![status](https://img.shields.io/badge/status-v0.1.0-ff5fa2)
![python](https://img.shields.io/badge/python-3.9%2B-3776ab)
![torch](https://img.shields.io/badge/torch-optional-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-22c55e)

> Memory that organizes itself, the way biological memory does.

> 🧬 Extracted from the [**Fractal Neurons** / **QJSON Agents**](https://yandesbiens.com/projects/fractal-neurons/)
> research by [Yan Desbiens](https://yandesbiens.com). Pairs naturally with
> [`ufm`](https://github.com/Linutesto/ufm) for single-GPU agent/model stacks.

> 📊 **Benchmarked (v0.2.0):** topic-scoped retrieval vs. flat semantic search. At 128k items,
> leaf-scoped retrieval is **~164× faster and ~3.7× more accurate** than a flat scan — when the
> topic is known; a misrouted scope misses entirely. A bet on locality. Method + reproduce:
> [`benchmarks/`](benchmarks/) · writeup: [yandesbiens.com/blog/fmm-benchmark](https://yandesbiens.com/blog/fmm-benchmark/).

A self-organizing, hierarchical memory for AI agents. Instead of a flat bag of
vectors, FMM stores knowledge as a tree of nested contexts you can zoom into or
out of — and retrieves by structure *and* similarity.

This idea recurred across four separate projects in the
[Fractal Neurons / QJSON Agents](https://yandesbiens.com/projects/fractal-neurons/)
research before being extracted here. It ships in two flavours:

| Class | Deps | Use it for |
|---|---|---|
| `FractalMemory` | none (pure Python) | Topic-addressed tree store — insert under a path, query any subtree. |
| `FractalMemoryMatrix` | `torch` (optional) | Semantic lattice — store node vectors, retrieve top-K by cosine similarity, with parent/child structure. |

## Install

```bash
pip install -e .            # pure-python tree only
pip install -e ".[torch]"   # + the torch semantic lattice
```

## Usage

### Pure-Python tree (no torch)

```python
from fmm import FractalMemory

m = FractalMemory()
m.insert(["work", "projects", "fmm"], "extracted to a library")
m.insert(["work", "projects", "fmm"], "shipped v0.1.0")
m.query(["work", "projects", "fmm"])   # -> ['extracted to a library', 'shipped v0.1.0']
m.visualize()
```

### Torch semantic lattice

```python
import torch
from fmm import FractalMemoryMatrix

mem = FractalMemoryMatrix(dim=256, max_nodes=10000)
root = mem.add_node(torch.randn(256))
mem.add_node(torch.randn(256), parent_id=root.id)

hits = mem.retrieve(torch.randn(256), top_k=5)   # [(FMMNode, score), ...]

state = mem.state_dict()           # JSON-serializable
mem2 = FractalMemoryMatrix(dim=256)
mem2.load_state_dict(state, device="cpu")
```

## Benchmark

[`benchmarks/`](benchmarks/) compares topic-scoped retrieval against a flat scan as the store
grows. Headline (128k items, synthetic hierarchical corpus):

| mode | scope | latency | recall@k |
|---|---:|---:|---:|
| flat scan | 128,000 | 8.30 ms | 0.155 |
| scoped: leaf subtree | 1,000 | **0.05 ms** | **0.58** |
| misrouted scope | 8,000 | 0.35 ms | 0.00 |

Scoping is both faster (sublinear) and more accurate (fewer cross-topic distractors) **when the
topic is known** — and useless if misrouted. Full writeup:
[yandesbiens.com/blog/fmm-benchmark](https://yandesbiens.com/blog/fmm-benchmark/).

```bash
cd benchmarks && ./run.sh
```

## Status

v0.2.0 — extracted, packaged, benchmarked. Adds **topic-scoped retrieval**
(`retrieve(query, topic_prefix=…)`) so the lattice can page the relevant region instead of
scanning everything. A serialization bug in the original (calling `.item()` on plain floats)
is fixed so `state_dict()`/`load_state_dict()` round-trip cleanly. Condensation/summary engines
and a learned topic router are the next milestones.

MIT licensed.
