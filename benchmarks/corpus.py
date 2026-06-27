"""Synthetic hierarchical corpus for the FMM retrieval benchmark.

Items are organized into a topic hierarchy (domain → subtopic). Each leaf topic
has a random cluster center; items are points near their center. This isolates the
*structural* question — does hierarchical scoping help retrieval? — from embedding
quality (which is a separate concern; real embeddings are future work).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_corpus(n: int, dim: int, n_domains: int, n_sub: int, eps: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    n_leaf = n_domains * n_sub
    centers = F.normalize(torch.randn(n_leaf, dim, generator=g), dim=1)
    per = max(1, n // n_leaf)

    vecs = []
    domain_of = []  # int domain id per item
    leaf_of = []    # int leaf id per item
    for leaf in range(n_leaf):
        pts = centers[leaf].unsqueeze(0) + eps * torch.randn(per, dim, generator=g)
        vecs.append(pts)
        d = leaf // n_sub
        domain_of += [d] * per
        leaf_of += [leaf] * per

    V = F.normalize(torch.cat(vecs, 0), dim=1)  # [N, dim], unit norm
    return {
        "V": V,
        "domain_of": torch.tensor(domain_of),
        "leaf_of": torch.tensor(leaf_of),
        "centers": centers,
        "n": V.shape[0],
        "n_domains": n_domains,
        "n_sub": n_sub,
        "n_leaf": n_leaf,
        "seed": seed,
        "eps": eps,
    }
