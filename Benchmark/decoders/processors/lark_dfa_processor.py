import os
from functools import lru_cache

import torch
from lark import Lark, UnexpectedInput


VALID_CHARS = set("0123456789+-*/() \t\n")


def _normalize_token_text(text: str) -> str:
    """Normalize decoded token text for validation.

    Strips leading newline artifacts and preserves spaces/tabs which are explicitly allowed.
    """

    # Replace carriage returns and other control characters with nothing
    return text.replace("\r", "").replace("\f", "")


def _load_grammar(grammar_file=None, grammar_string=None, grammar_dir=None):
    if grammar_file and not os.path.exists(grammar_file):
        raise FileNotFoundError(f"Grammar file not found: {grammar_file}")

    # Try to infer grammar file from grammar_dir
    if not grammar_file and grammar_dir and os.path.exists(grammar_dir):
        for filename in os.listdir(grammar_dir):
            if filename.endswith(".lark"):
                grammar_file = os.path.join(grammar_dir, filename)
                break

    # Fallback to default locations
    if not grammar_file and not grammar_string:
        possible_paths = [
            os.path.join("decoders", "grammars", "simple.lark"),
            os.path.join("decoders", "grammars", "example.lark"),
            "simple.lark",
            "example.lark",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                grammar_file = path
                break

    if grammar_file and not grammar_string:
        with open(grammar_file, "r", encoding="utf-8") as f:
            grammar_string = f.read()

    if not grammar_string:
        raise ValueError("No grammar provided for Lark DFA processor")

    parser = Lark(grammar_string, parser="lalr", maybe_placeholders=False)
    return parser, grammar_file


def get_logits_processor(grammar_table, terminal_map, tokenizer, grammar_file=None, grammar_string=None, grammar_dir=None):
    """
    Returns a logits processor that constrains generation to strings accepted by a Lark grammar.

    This implementation approximates a DFA by:
    1. Pre-filtering the tokenizer vocabulary to tokens whose decoded strings consist only of
       calculator characters (digits 1-9, parentheses, operators, whitespace).
    2. Using Lark's LALR parser in interactive mode to verify that the accumulated text remains
       a valid prefix of the grammar for each candidate token.

    The resulting mask is cheap to evaluate because only a small subset of the vocabulary is
    considered at each decoding step.
    """

    # Allow overriding grammar paths via grammar_table / terminal_map (for compatibility)
    if isinstance(grammar_table, str) and not grammar_file:
        grammar_file = grammar_table
    if isinstance(terminal_map, str) and not grammar_dir:
        grammar_dir = terminal_map

    parser, resolved_grammar_file = _load_grammar(grammar_file, grammar_string, grammar_dir)

    # Pre-filter the vocabulary to candidate tokens consisting only of valid characters
    allowed_token_text = {}
    vocab_size = len(tokenizer)
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if not text:
            continue
        text = _normalize_token_text(text)
        if not text:
            continue
        if set(text).issubset(VALID_CHARS):
            allowed_token_text[token_id] = text

    eos_token_id = tokenizer.eos_token_id

    print("Initialized Lark DFA processor")
    if resolved_grammar_file:
        print(f"  Grammar file: {resolved_grammar_file}")
    print(f"  Candidate token count: {len(allowed_token_text)} out of {vocab_size}")

    @lru_cache(maxsize=8192)
    def is_valid_prefix(text: str) -> bool:
        interactive = parser.parse_interactive("")
        try:
            for token in parser.lex(text):
                interactive.feed_token(token)
        except UnexpectedInput:
            return False
        return True

    def is_complete(text: str) -> bool:
        try:
            parser.parse(text)
            return True
        except Exception:
            return False

    def processor_fn(scores, state):
        device = scores.device
        vocab = scores.shape[-1]

        generated_ids = state.get("generated_ids", [])
        current_text = ""
        if generated_ids:
            current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        valid_mask = torch.zeros(vocab, dtype=torch.bool, device=device)

        for token_id, token_text in allowed_token_text.items():
            if token_id >= vocab:
                continue
            new_text = current_text + token_text
            if is_valid_prefix(new_text):
                valid_mask[token_id] = True

        # Allow EOS when the current text already forms a complete expression
        if eos_token_id is not None and eos_token_id < vocab:
            if current_text and is_complete(current_text):
                valid_mask[eos_token_id] = True

        if not valid_mask.any():
            # Fallback to avoid dead-ends: allow only pre-filtered calculator tokens
            for token_id in allowed_token_text:
                if token_id < vocab:
                    valid_mask[token_id] = True

            # As a final fallback, if still empty (e.g., tokenizer lacks calculator tokens), allow EOS
            if not valid_mask.any() and eos_token_id is not None and eos_token_id < vocab:
                valid_mask[eos_token_id] = True

        scores = scores.masked_fill(~valid_mask, float("-inf"))
        return scores

    return processor_fn
