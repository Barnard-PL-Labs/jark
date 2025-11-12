from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import regex
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken


class CompletionEngine:
    """Abstract interface describing how to query grammar completions."""

    def complete(self, prefix: str) -> regex.Pattern:
        raise NotImplementedError()

    def is_complete(self, prefix: str) -> bool:
        return self.complete(prefix) == regex.compile("")


@dataclass
class CompletionPoint:
    pattern: regex.Pattern


class LarkCompletionEngine(CompletionEngine):
    """
    Completion engine that uses Lark to expose all terminals that can follow a
    prefix. The result is returned as a ``regex.Pattern`` that matches any valid
    continuation from the prefix.
    """

    def __init__(self, grammar: str, start_rule: str, allow_whitespace: bool = True):
        self.grammar = grammar
        self.start_rule = start_rule
        self.allow_whitespace = allow_whitespace

        # ``regex=True`` is required so that terminal patterns use the ``regex`` module.
        self.parser = Lark(grammar, start=start_rule, parser="lalr", regex=True)
        self._terminals: Dict[str, "lark.lexer.TerminalDef"] = self.parser._terminals_dict

    def _feed_prefix(self, prefix: str):
        interactive = self.parser.parse_interactive(prefix)
        try:
            for token in interactive.parser_state.lexer.lex(interactive.parser_state):
                interactive.parser_state.feed_token(token)
        except (UnexpectedCharacters, UnexpectedToken):
            # Prefix already invalid, the resulting pattern will reject everything.
            pass
        return interactive

    def complete(self, prefix: str) -> regex.Pattern:
        interactive = self._feed_prefix(prefix)
        valid_tokens = interactive.accepts()

        # Build a regex that accepts any valid next terminal according to the grammar.
        valid_regexes = []
        for token in valid_tokens:
            if token == "$END":
                continue
            terminal = self._terminals[token]
            valid_regexes.append(terminal.pattern.to_regexp())

        if self.allow_whitespace and valid_regexes:
            valid_regexes.append(r"\s+")

        if not valid_regexes:
            return regex.compile("")

        combined = "|".join(valid_regexes)
        return regex.compile(combined)

    def __deepcopy__(self, memo):
        return LarkCompletionEngine(
            grammar=self.grammar,
            start_rule=self.start_rule,
            allow_whitespace=self.allow_whitespace,
        )
