"""
FMM — Fractal Memory.

A self-organizing, hierarchical memory for AI agents. Two flavours:

- ``FractalMemory`` — a dependency-free, topic-addressed tree store.
- ``FractalMemoryMatrix`` — a torch-backed semantic lattice with cosine
  retrieval (requires ``torch``; imported lazily).

Extracted from the Fractal Neurons / QJSON Agents research work by Yan Desbiens.
"""

from .tree import FractalMemory

__all__ = ["FractalMemory", "FractalMemoryMatrix", "FMMNode"]
__version__ = "0.2.0"


def __getattr__(name: str):
    # Lazy torch import so the pure-python tree works without torch installed.
    if name in ("FractalMemoryMatrix", "FMMNode"):
        from . import matrix
        return getattr(matrix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
