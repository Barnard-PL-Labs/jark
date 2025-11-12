"""Logits processor providing Synchromesh-style grammar constraints.

Example:

    python eval/evaluate.py \
        --model gpt2 \
        --logits_processor decoders/processors/synchromesh_processor.py \
        --grammar_dir decoders/grammars \
        --max_tokens 20

The processor loads the first ``.lark`` file found in ``grammar_dir`` and enforces
its ``start`` rule via regex-driven constrained decoding.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch

from decoders.synchromesh.completion_engine import LarkCompletionEngine
from decoders.synchromesh.streaming import StreamingCSD


def _token_to_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text


def _build_vocabulary(tokenizer) -> Dict[int, str]:
    vocab = {}
    for token_id in range(len(tokenizer)):
        vocab[token_id] = _token_to_text(tokenizer, token_id)
    return vocab


def _reverse_vocabulary(mapping: Dict[int, str]) -> Dict[str, List[int]]:
    reverse: Dict[str, List[int]] = {}
    for token_id, text in mapping.items():
        reverse.setdefault(text, []).append(token_id)
    return reverse


def get_logits_processor(grammar_table, terminal_map, tokenizer, grammar_file=None, grammar_string=None, grammar_dir=None, start_rule="start"):
    if isinstance(grammar_table, str) and not grammar_file:
        grammar_file = grammar_table
    if isinstance(terminal_map, str) and not grammar_dir:
        grammar_dir = terminal_map

    if not grammar_file:
        if grammar_dir and os.path.exists(grammar_dir):
            for candidate in os.listdir(grammar_dir):
                if candidate.endswith(".lark"):
                    grammar_file = os.path.join(grammar_dir, candidate)
                    break

    if grammar_file and os.path.exists(grammar_file):
        with open(grammar_file, "r", encoding="utf-8") as f:
            grammar_string = f.read()

    if isinstance(grammar_table, dict):
        grammar_file = grammar_table.get("grammar_file", grammar_file)
        start_rule = grammar_table.get("start_rule", start_rule)

    if grammar_string is None:
        raise ValueError("A Lark grammar is required for the Synchromesh processor")

    completion_engine = LarkCompletionEngine(grammar_string, start_rule=start_rule)

    token_text = _build_vocabulary(tokenizer)
    reverse_lookup = _reverse_vocabulary(token_text)
    vocabulary_strings = [token_text[token_id] for token_id in sorted(token_text)]

    def processor_fn(scores: torch.Tensor, state: Dict) -> torch.Tensor:
        if "csd" not in state:
            state["csd"] = StreamingCSD(completion_engine, vocabulary_strings)
            state["processed_length"] = 0

        csd: StreamingCSD = state["csd"]
        processed_length: int = state.get("processed_length", 0)
        generated_ids: List[int] = state.get("generated_ids", [])

        if processed_length > len(generated_ids):
            csd.init_stream()
            processed_length = 0

        # Feed any new tokens into the streaming decoder
        for token_id in generated_ids[processed_length:]:
            token_text_piece = token_text.get(token_id, "")
            if not token_text_piece:
                csd.init_stream()
                break
            if not csd.can_token_follow(token_text_piece):
                csd.init_stream()
                break
            csd.feed_prediction(token_text_piece)
            processed_length += 1

        state["processed_length"] = processed_length

        valid_tokens = csd.get_valid_tokens()
        if not valid_tokens:
            return scores

        masked_scores = scores.clone()
        masked_scores.fill_(float("-inf"))

        for token in valid_tokens:
            for token_id in reverse_lookup.get(token, []):
                if token_id < masked_scores.shape[-1]:
                    masked_scores[..., token_id] = scores[..., token_id]

        if masked_scores.max() == float("-inf"):
            return scores

        return masked_scores

    return processor_fn
