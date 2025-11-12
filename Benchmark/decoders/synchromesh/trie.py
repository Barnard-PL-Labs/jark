from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    values: List[str] = field(default_factory=list)


class Trie:
    def __init__(self):
        self.root = TrieNode()

    @classmethod
    def from_vocabulary(cls, tokens: Iterable[str]) -> "Trie":
        trie = cls()
        for token in tokens:
            trie.insert(token)
        return trie

    def insert(self, token: str) -> None:
        node = self.root
        for char in token:
            node = node.children.setdefault(char, TrieNode())
        node.values.append(token)

    def antimonotonic_filter(self, predicate: Callable[[str], bool]) -> List[str]:
        """
        Apply *predicate* to every token stored in the trie, returning the list
        of tokens for which it returns True. The traversal is performed in a
        depth-first manner to avoid holding unnecessary intermediate strings.
        """

        accepted: List[str] = []

        def _traverse(node: TrieNode, prefix: str) -> None:
            for token in node.values:
                if predicate(token):
                    accepted.append(token)
            for char, child in node.children.items():
                _traverse(child, prefix + char)

        _traverse(self.root, "")
        return accepted
