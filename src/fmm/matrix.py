"""Fractal Memory Matrix — a torch-backed hierarchical semantic memory lattice.

Stores node vectors (e.g. model root representations) in a parent/child tree and
retrieves the top-K nearest memories by cosine similarity. Designed to attach a
retrieval pathway to a model so it can anchor to previously seen semantics.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMMNode:
    """A single node in the Fractal Memory Matrix."""

    def __init__(
        self,
        semantic_vector: torch.Tensor,
        context_anchor: Optional[torch.Tensor] = None,
        parent_id: Optional[str] = None,
        topic: Optional[Tuple[str, ...]] = None,
    ):
        self.id = str(torch.randint(0, 1000000000, (1,)).item())
        self.semantic_vector = semantic_vector  # [dim]
        self.context_anchor = context_anchor  # [dim], e.g. mean of creating input
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.topic: Tuple[str, ...] = tuple(topic) if topic else ()  # hierarchical address
        self.last_accessed: float = 0.0  # for entropy-adaptive refresh
        self.entropy: float = 0.0  # information entropy of content

    def add_child(self, child_id: str) -> None:
        self.children_ids.append(child_id)

    def update_access_time(self) -> None:
        self.last_accessed = time.time()

    def update_entropy(self, new_entropy: float) -> None:
        self.entropy = float(new_entropy)


class FractalMemoryMatrix(nn.Module):
    """A hierarchical, recursive storage lattice for reasoning chains."""

    def __init__(self, dim: int, max_nodes: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.nodes: Dict[str, FMMNode] = {}
        self.root_nodes: List[str] = []  # top-level nodes without parents
        self.node_vectors: Dict[str, torch.Tensor] = {}

    def add_node(
        self,
        semantic_vector: torch.Tensor,
        context_anchor: Optional[torch.Tensor] = None,
        parent_id: Optional[str] = None,
        topic: Optional[Tuple[str, ...]] = None,
    ) -> Optional[FMMNode]:
        if len(self.nodes) >= self.max_nodes:
            # Pruning strategy hook (e.g. LRU, lowest entropy). For now, refuse.
            return None

        node = FMMNode(semantic_vector, context_anchor, parent_id, topic=topic)
        self.nodes[node.id] = node
        self.node_vectors[node.id] = semantic_vector

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].add_child(node.id)
        else:
            self.root_nodes.append(node.id)
        return node

    def get_node(self, node_id: str) -> Optional[FMMNode]:
        return self.nodes.get(node_id)

    def get_semantic_vector(self, node_id: str) -> Optional[torch.Tensor]:
        return self.node_vectors.get(node_id)

    def _candidate_ids(self, topic_prefix: Optional[Tuple[str, ...]]) -> List[str]:
        if not topic_prefix:
            return list(self.node_vectors.keys())
        p = tuple(topic_prefix)
        n = len(p)
        return [nid for nid, node in self.nodes.items() if node.topic[:n] == p]

    def retrieve(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        topic_prefix: Optional[Tuple[str, ...]] = None,
    ) -> List[Tuple[FMMNode, float]]:
        """Retrieve the top_k most similar nodes by cosine similarity.

        If ``topic_prefix`` is given, only nodes whose hierarchical topic starts with
        that prefix are considered — scoped ("paged") retrieval over the relevant
        region of the fractal store, instead of a flat scan of everything.
        """
        if not self.node_vectors:
            return []

        cand = self._candidate_ids(topic_prefix)
        if not cand:
            return []

        all_vectors = torch.stack([self.node_vectors[c] for c in cand])
        similarities = F.cosine_similarity(query_vector, all_vectors)
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(cand)))

        results = []
        for sim, idx in zip(top_similarities, top_indices):
            node_id = cand[idx]
            node = self.nodes[node_id]
            node.update_access_time()
            results.append((node, sim.item()))
        return results

    def topic_centroids(self, level: int = 1) -> Dict[Tuple[str, ...], torch.Tensor]:
        """Mean (unit-norm) vector of every topic subtree at depth ``level``.

        Each centroid summarizes one region of the fractal store — the node's own
        cheap descriptor. Built from data already present (no training, no extra
        passes), so it costs one mean per subtree at index time.
        """
        groups: Dict[Tuple[str, ...], List[torch.Tensor]] = defaultdict(list)
        for nid, node in self.nodes.items():
            key = node.topic[:level]
            if len(key) < level:
                continue  # node shallower than the routing level — can't address it
            groups[key].append(self.node_vectors[nid])
        return {k: F.normalize(torch.stack(v).mean(0), dim=0) for k, v in groups.items()}

    def route(
        self,
        query_vector: torch.Tensor,
        top_r: int = 1,
        level: int = 1,
        centroids: Optional[Dict[Tuple[str, ...], torch.Tensor]] = None,
    ) -> List[Tuple[Tuple[str, ...], float]]:
        """Pick the ``top_r`` topic subtrees a query most likely belongs to.

        The cheap counterpart to :meth:`retrieve`: instead of scanning items, score
        the query against the per-subtree centroids and return the best ``top_r``
        topic prefixes (highest cosine first). Feed those prefixes back into
        ``retrieve(query, topic_prefix=...)`` to search only the routed region.

        Cost is ``O(n_subtrees x dim)`` — independent of how many items the store
        holds. Pass precomputed ``centroids`` (from :meth:`topic_centroids`) to avoid
        rebuilding them per call.
        """
        cents = centroids if centroids is not None else self.topic_centroids(level)
        if not cents:
            return []
        keys = list(cents.keys())
        mat = torch.stack([cents[k] for k in keys])
        sims = F.cosine_similarity(query_vector, mat)
        r = min(top_r, len(keys))
        top = torch.topk(sims, r)
        return [(keys[i], float(top.values[j])) for j, i in enumerate(top.indices.tolist())]

    def route_and_retrieve(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        top_r: int = 1,
        level: int = 1,
        centroids: Optional[Dict[Tuple[str, ...], torch.Tensor]] = None,
    ) -> List[Tuple[FMMNode, float]]:
        """Route to the ``top_r`` best subtrees, then retrieve within their union.

        End-to-end scoped retrieval with no oracle: the store decides *where* to look
        (:meth:`route`) and then *what* to return (:meth:`retrieve`). For ``top_r > 1``
        the candidate set is the union of the routed subtrees.
        """
        routed = self.route(query_vector, top_r=top_r, level=level, centroids=centroids)
        if not routed:
            return []
        prefixes = [p for p, _ in routed]
        cand = [
            nid
            for nid, node in self.nodes.items()
            if any(node.topic[: len(p)] == p for p in prefixes)
        ]
        if not cand:
            return []
        vecs = torch.stack([self.node_vectors[c] for c in cand])
        sims = F.cosine_similarity(query_vector, vecs)
        top = torch.topk(sims, min(top_k, len(cand)))
        out = []
        for sim, idx in zip(top.values, top.indices.tolist()):
            node = self.nodes[cand[idx]]
            node.update_access_time()
            out.append((node, float(sim)))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder: FMM has no traditional forward pass; passthrough for integration."""
        return x

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        serializable_nodes = {
            node_id: {
                "semantic_vector": node.semantic_vector.cpu().tolist(),
                "context_anchor": node.context_anchor.cpu().tolist() if node.context_anchor is not None else None,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "last_accessed": float(node.last_accessed),
                "entropy": float(node.entropy),
            }
            for node_id, node in self.nodes.items()
        }
        return {
            "dim": self.dim,
            "max_nodes": self.max_nodes,
            "nodes": serializable_nodes,
            "root_nodes": self.root_nodes,
        }

    def load_state_dict(self, state_dict: Dict[str, Any], device: torch.device | str = "cpu"):  # type: ignore[override]
        self.dim = state_dict["dim"]
        self.max_nodes = state_dict["max_nodes"]
        self.nodes = {}
        self.node_vectors = {}
        for node_id, node_data in state_dict["nodes"].items():
            semantic_vector = torch.tensor(node_data["semantic_vector"], device=device)
            context_anchor = (
                torch.tensor(node_data["context_anchor"], device=device)
                if node_data["context_anchor"] is not None
                else None
            )
            node = FMMNode(semantic_vector, context_anchor, node_data["parent_id"])
            node.id = node_id  # preserve ID
            node.children_ids = node_data["children_ids"]
            node.last_accessed = float(node_data["last_accessed"])
            node.entropy = float(node_data["entropy"])
            self.nodes[node_id] = node
            self.node_vectors[node_id] = semantic_vector
        self.root_nodes = state_dict["root_nodes"]
