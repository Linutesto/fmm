"""Fractal Memory Matrix — a torch-backed hierarchical semantic memory lattice.

Stores node vectors (e.g. model root representations) in a parent/child tree and
retrieves the top-K nearest memories by cosine similarity. Designed to attach a
retrieval pathway to a model so it can anchor to previously seen semantics.
"""

import time
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
    ):
        self.id = str(torch.randint(0, 1000000000, (1,)).item())
        self.semantic_vector = semantic_vector  # [dim]
        self.context_anchor = context_anchor  # [dim], e.g. mean of creating input
        self.parent_id = parent_id
        self.children_ids: List[str] = []
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
    ) -> Optional[FMMNode]:
        if len(self.nodes) >= self.max_nodes:
            # Pruning strategy hook (e.g. LRU, lowest entropy). For now, refuse.
            return None

        node = FMMNode(semantic_vector, context_anchor, parent_id)
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

    def retrieve(self, query_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[FMMNode, float]]:
        """Retrieve the top_k most similar nodes by semantic-vector cosine similarity."""
        if not self.node_vectors:
            return []

        all_vectors = torch.stack(list(self.node_vectors.values()))
        similarities = F.cosine_similarity(query_vector, all_vectors)
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(self.node_vectors)))

        results = []
        node_ids = list(self.node_vectors.keys())
        for sim, idx in zip(top_similarities, top_indices):
            node_id = node_ids[idx]
            node = self.nodes[node_id]
            node.update_access_time()
            results.append((node, sim.item()))
        return results

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
