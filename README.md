# FMM — Fractal Memory

> Memory that organizes itself, the way biological memory does.

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

## Status

v0.1.0 — extracted and packaged. A serialization bug in the original (calling
`.item()` on plain floats) is fixed here so `state_dict()`/`load_state_dict()`
round-trip cleanly. Condensation/summary engines and eviction heuristics
(described in the FMM whitepaper) are the next milestone.

MIT licensed.
