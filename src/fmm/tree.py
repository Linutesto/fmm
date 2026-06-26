"""Fractal Memory (tree) — a dependency-free hierarchical, topic-addressed store.

The torch-free variant of FMM: insert data under a topic path (a list of keys)
and query any subtree. Self-similar — every node is shaped like the root — so the
same traversal works at any depth.
"""

from __future__ import annotations

from typing import Any, Dict, List


class FractalMemory:
    def __init__(self) -> None:
        self.memory_tree: Dict[str, Any] = {}

    def insert(self, topic_path: List[str], data: Any) -> None:
        node = self.memory_tree
        for part in topic_path:
            if part not in node:
                node[part] = {}
            node = node[part]
        if "__data__" not in node:
            node["__data__"] = []
        node["__data__"].append(data)

    def query(self, topic_path: List[str]) -> List[Any]:
        node = self.memory_tree
        for part in topic_path:
            if part not in node:
                return []
            node = node[part]
        return node.get("__data__", [])

    def visualize(self, node: Dict[str, Any] | None = None, prefix: str = "") -> None:
        if node is None:
            node = self.memory_tree
        for k, v in node.items():
            if k == "__data__":
                continue
            print(prefix + k + "/")
            self.visualize(v, prefix + "  ")
