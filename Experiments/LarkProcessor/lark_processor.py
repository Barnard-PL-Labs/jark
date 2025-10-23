from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

from lark import Lark, Transformer, v_args
from lark.exceptions import UnexpectedInput
from transformers import LogitsProcessor
import torch


ARITHMETIC_GRAMMAR = r"""
?start: sum

?sum: sum "+" product   -> add
    | sum "-" product   -> sub
    | product

?product: product "*" atom -> mul
        | product "/" atom -> div
        | atom

?atom: NUMBER           -> number
     | "-" atom         -> neg
     | "(" sum ")"

%import common.NUMBER
%import common.WS
%ignore WS
"""


def build_parser(debug: bool = False) -> Lark:
    """Create a LALR parser for the arithmetic grammar.

    debug=True enables verbose parser logs (requires logging configured).
    """
    return Lark(
        ARITHMETIC_GRAMMAR,
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
        debug=debug,
    )


@v_args(inline=True)
class EvalTransformer(Transformer):
    """Evaluate the arithmetic expression tree to a Python float.

    Each method name matches a rule alias (-> name) or a terminal rule.
    """

    def number(self, token):
        return float(token)

    def add(self, left, right):
        return left + right

    def sub(self, left, right):
        return left - right

    def mul(self, left, right):
        return left * right

    def div(self, left, right):
        return left / right
    
    def neg(self, value):
        return -value
    
_parser = build_parser()
def parse_expression(text: str):
    tree = _parser.parse(text)
    return tree


class LarkConstrainedLogitsProcessor(LogitsProcessor):
    """Constrain next-token choices using a Lark grammar and the current prefix text.

    This implementation queries the parser for expected terminals at the end of the
    current prefix (using UnexpectedInput when the prefix is incomplete). It then
    masks logits for tokens that cannot be a valid continuation under the grammar.

    Notes:
    - Designed for character-level constraints projected onto token pieces. It uses
      simple prefix-compatibility checks based on expected terminals and first
      characters of token strings. For robust behavior across tokenizers, we
      decode each candidate token id to text.
    - Optimized strategies (interactive parser state, per-token parse probes) can
      be added later if needed for stricter constraints.
    """

    def __init__(self, tokenizer, parser: Lark, allow_eos_when_accepting: bool = True):
        self.tokenizer = tokenizer
        self.parser = parser
        self.allow_eos_when_accepting = allow_eos_when_accepting
        self._id_to_text_cache: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        cached = self._id_to_text_cache.get(token_id)
        if cached is not None:
            return cached
        # Decode a single token id to its text segment
        text = self.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self._id_to_text_cache[token_id] = text
        return text

    def _expected_terminals_at_end(self, prefix: str) -> tuple[set[str], bool]:
        """Return (expected_terminals, is_accepting).

        - If prefix fully parses, is_accepting=True and expected is empty.
        - If prefix is incomplete and fails exactly at the end, expected is the set
          of terminals acceptable next, is_accepting=False.
        - If failure occurs before the end, the prefix is already off-grammar; we
          return an empty set and is_accepting=False (downstream may choose to allow
          only whitespace or mask everything but EOS).
        """
        # Clean the prefix for parsing
        clean_prefix = prefix.strip()
        
        # If empty, we can start with NUMBER or LPAR
        if not clean_prefix:
            return {"NUMBER", "LPAR"}, False
            
        try:
            self.parser.parse(clean_prefix)
            return set(), True
        except UnexpectedInput as e:
            # If error is exactly at the end of input, we can use expected
            if getattr(e, "pos_in_stream", None) == len(clean_prefix):
                # Lark provides a set of terminal names; we conservatively cast to str
                try:
                    expected = set(str(t) for t in e.expected)
                except Exception:
                    expected = set()
                return expected, False
            # If we're not at the end, try to be more permissive for incomplete expressions
            # This handles cases where the prefix might be valid but incomplete
            # We'll allow common arithmetic continuations
            return {"NUMBER", "LPAR", "MINUS"}, False

    def _build_allowed_first_chars(self, expected: set[str]) -> set[str]:
        """Map expected terminals to allowed first characters of the next token text.

        This is a conservative projection of grammar terminals to character-level
        constraints suitable for piecewise token filtering.
        """
        allowed: set[str] = set()

        # Whitespace is ignored by the grammar; allow it to pass through.
        allowed.update([" ", "\t", "\n", "\r"])

        # Arithmetic literals
        literal_map = ["+", "-", "*", "/", "(", ")"]
        for lit in literal_map:
            if (lit in expected) or (f'"{lit}"' in expected):
                allowed.add(lit)

        # Numbers: accept digits and dot as starters
        if any(name in expected for name in ["NUMBER", "DEC_NUMBER", "SIGNED_NUMBER"]):
            allowed.update(list("0123456789."))

        # Handle LPAR (left parenthesis)
        if "LPAR" in expected:
            allowed.add("(")

        return allowed

    def _token_is_prefix_compatible(self, token_text: str, allowed_first_chars: set[str]) -> bool:
        if token_text == "":
            return False
        # Accept pure whitespace if allowed
        if token_text[0].isspace():
            return True
        # Check the first non-whitespace char against allowed set
        for ch in token_text:
            if ch.isspace():
                continue
            return ch in allowed_first_chars
        # Token was only whitespace
        return True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        vocab_size = scores.size(-1)

        # Build per-batch prefix strings once
        prefixes: list[str] = []
        for i in range(batch_size):
            # Decode and clean the prefix for grammar parsing
            full_text = self.tokenizer.decode(
                input_ids[i].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            # Remove special tokens that the grammar doesn't understand
            # Keep only the actual content that should be parsed
            if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
                full_text = full_text.replace(self.tokenizer.bos_token, '')
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                full_text = full_text.replace(self.tokenizer.eos_token, '')
            if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token:
                full_text = full_text.replace(self.tokenizer.pad_token, '')
            # Also remove common special tokens
            full_text = full_text.replace('<s>', '').replace('</s>', '').replace('<unk>', '')
            prefixes.append(full_text.strip())

        for i in range(batch_size):
            prefix = prefixes[i]
            expected, is_accepting = self._expected_terminals_at_end(prefix)

            # If in accepting state, optionally force EOS
            if is_accepting and self.allow_eos_when_accepting and self.tokenizer.eos_token_id is not None:
                # Only allow EOS; mask others
                scores[i, :] = float("-inf")
                scores[i, self.tokenizer.eos_token_id] = 0.0
                continue

            allowed_first_chars = self._build_allowed_first_chars(expected)

            # Fallback: if we couldn't infer anything, allow arithmetic charset to avoid over-masking
            if not allowed_first_chars:
                allowed_first_chars.update(list("0123456789+-*/() .\t\n\r"))

            # Mask tokens that are not compatible
            # We iterate once over vocab; for speed, consider vectorization or caching in production
            for tok_id in range(vocab_size):
                tok_text = self._decode_token(tok_id)
                if not self._token_is_prefix_compatible(tok_text, allowed_first_chars):
                    scores[i, tok_id] = float("-inf")

        return scores


def build_arithmetic_processor(tokenizer) -> LarkConstrainedLogitsProcessor:
    """Convenience factory for this repo's arithmetic grammar."""
    parser = build_parser(debug=False)
    return LarkConstrainedLogitsProcessor(tokenizer=tokenizer, parser=parser)


def generate_with_lark_constraints(model, tokenizer, prompt: str, max_new_tokens: int = 64, do_sample: bool = False, temperature: float = 1.0):
    """Generate text constrained by the arithmetic grammar using the given model/tokenizer.

    This is a convenience wrapper; pass your own model path when you construct
    the model/tokenizer outside this function.
    """
    from transformers import LogitsProcessorList

    processor = build_arithmetic_processor(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        logits_processor=LogitsProcessorList([processor]),
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(parse_expression("1 + 2 * 3"))