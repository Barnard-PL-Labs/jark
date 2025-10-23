#!/usr/bin/env python3
"""
Debug continuation of incomplete expressions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import build_arithmetic_processor


def debug_continuation():
    """Debug why continuation isn't working."""
    print("Loading TinyLlama for debugging...")
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    processor = build_arithmetic_processor(tokenizer)
    
    # Test with incomplete expressions
    test_prefixes = [
        "2 +",
        "5 *",
        "10 -",
        "8 /",
        "(3 + 4) *",
    ]
    
    for prefix in test_prefixes:
        print(f"\nTesting prefix: '{prefix}'")
        
        # Test expected terminals
        expected, is_accepting = processor._expected_terminals_at_end(prefix)
        print(f"Expected terminals: {expected}")
        print(f"Is accepting: {is_accepting}")
        
        # Test allowed first chars
        allowed_chars = processor._build_allowed_first_chars(expected)
        print(f"Allowed first chars: {allowed_chars}")
        
        # Test some specific tokens that should be allowed
        test_tokens = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]  # digits 0-9
        print(f"Testing digit tokens:")
        for tok_id in test_tokens:
            token_text = processor._decode_token(tok_id)
            compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
            print(f"  Token {tok_id} '{repr(token_text)}': {compatible}")
        
        # Test parenthesis token
        print(f"Testing parenthesis tokens:")
        for tok_id in range(100, 200):
            token_text = processor._decode_token(tok_id)
            if '(' in token_text or ')' in token_text:
                compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
                print(f"  Token {tok_id} '{repr(token_text)}': {compatible}")
        
        # Count total allowed tokens
        allowed_count = 0
        for tok_id in range(min(1000, tokenizer.vocab_size)):
            token_text = processor._decode_token(tok_id)
            compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
            if compatible:
                allowed_count += 1
        
        print(f"Total allowed tokens: {allowed_count}")


if __name__ == "__main__":
    debug_continuation()
