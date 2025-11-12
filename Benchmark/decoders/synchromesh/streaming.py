from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import regex

from .completion_engine import CompletionEngine
from .trie import Trie


class StreamingCSD:
    """Streaming implementation of constrained semantic decoding."""

    def __init__(self, completion_engine: CompletionEngine, vocabulary: Iterable[str]):
        self._engine = completion_engine
        self._vocab_list = list(vocabulary)
        self._trie = Trie.from_vocabulary(self._vocab_list)
        self._completion_points: Dict[str, regex.Pattern] = {}
        self._completion_points[""] = completion_engine.complete("")
        self.init_stream()

    def init_stream(self) -> None:
        self._prefix_tokens: List[int] = []
        self._prefix_str = ""

    def can_token_follow(self, token: str) -> bool:
        return is_prefix_valid(self._engine, self._completion_points, self._prefix_str + token)

    def feed_prediction(self, token: str) -> None:
        self._prefix_tokens.append(token)
        self._prefix_str += token

    def get_valid_tokens(self) -> List[str]:
        return self._trie.antimonotonic_filter(
            lambda t: is_prefix_valid(self._engine, self._completion_points, self._prefix_str + t)
        )

    def current_prefix(self) -> str:
        return self._prefix_str

    def is_complete(self) -> bool:
        return self._engine.is_complete(self._prefix_str)


def is_prefix_valid(
    completion_engine: CompletionEngine,
    completion_points: Dict[str, regex.Pattern],
    candidate: str,
) -> bool:
    # 1) Find the longest cached completion point
    longest_completion = ""
    for i in range(len(candidate) + 1):
        prefix = candidate[:i]
        if prefix in completion_points:
            longest_completion = prefix

    pattern = completion_points[longest_completion]
    remainder = candidate[len(longest_completion) :]

    # 2) Feed remainder char by char to the regex
    for i in range(len(remainder)):
        substring = remainder[: i + 1]
        if not pattern.fullmatch(substring, partial=True):
            # Check if previous characters formed a match; if so, recurse from there
            if i > 0 and pattern.fullmatch(remainder[:i]):
                new_prefix = longest_completion + remainder[:i]
                completion_points[new_prefix] = completion_engine.complete(new_prefix)
                return is_prefix_valid(completion_engine, completion_points, candidate)
            return False

    return True
