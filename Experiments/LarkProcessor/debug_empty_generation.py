#!/usr/bin/env python3
"""
Debug why generation is producing empty strings.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import build_arithmetic_processor


def debug_empty_generation():
    """Debug why we're getting empty generation."""
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
    
    # Test with a simple prompt
    prompt = "2 + 3 ="
    print(f"\nTesting with prompt: '{prompt}'")
    
    # Decode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input IDs: {inputs['input_ids']}")
    
    # Clean the prefix like the processor does
    full_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    print(f"Full decoded text: '{full_text}'")
    
    prefix = full_text
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
        prefix = prefix.replace(tokenizer.bos_token, '')
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        prefix = prefix.replace(tokenizer.eos_token, '')
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
        prefix = prefix.replace(tokenizer.pad_token, '')
    prefix = prefix.replace('<s>', '').replace('</s>', '').replace('<unk>', '').strip()
    print(f"Cleaned prefix: '{prefix}'")
    
    # Test expected terminals
    expected, is_accepting = processor._expected_terminals_at_end(prefix)
    print(f"Expected terminals: {expected}")
    print(f"Is accepting: {is_accepting}")
    
    # Test allowed first chars
    allowed_chars = processor._build_allowed_first_chars(expected)
    print(f"Allowed first chars: {allowed_chars}")
    
    # Test some specific tokens that should be allowed
    test_tokens = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]  # digits 0-9
    print(f"\nTesting digit tokens:")
    for tok_id in test_tokens:
        token_text = processor._decode_token(tok_id)
        compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
        print(f"Token {tok_id} '{repr(token_text)}': {compatible}")
    
    # Test a few more tokens
    print(f"\nTesting more tokens:")
    for tok_id in range(100, 110):
        token_text = processor._decode_token(tok_id)
        compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
        if compatible:
            print(f"Token {tok_id} '{repr(token_text)}': {compatible}")
    
    # Let's see what tokens are actually being allowed
    print(f"\nFinding allowed tokens:")
    allowed_count = 0
    for tok_id in range(min(1000, tokenizer.vocab_size)):
        token_text = processor._decode_token(tok_id)
        compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
        if compatible:
            print(f"Token {tok_id} '{repr(token_text)}': {compatible}")
            allowed_count += 1
            if allowed_count >= 20:  # Limit output
                break
    
    print(f"\nTotal allowed tokens found: {allowed_count}")


if __name__ == "__main__":
    debug_empty_generation()
