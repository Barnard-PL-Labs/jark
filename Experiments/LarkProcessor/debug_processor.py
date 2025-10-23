#!/usr/bin/env python3
"""
Debug script to understand what's happening with the logits processor.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import build_arithmetic_processor, build_parser


def debug_processor():
    """Debug the processor step by step."""
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
    parser = build_parser()
    
    # Test with a simple prompt
    prompt = "Calculate: "
    print(f"\nTesting with prompt: '{prompt}'")
    
    # Decode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Decoded: '{tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}'")
    
    # Test the processor logic manually
    full_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    print(f"\nFull decoded text: '{full_text}'")
    
    # Clean the prefix like the processor does
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
    
    # Test some token decoding
    print(f"\nTesting token decoding:")
    for i in range(10):
        token_text = processor._decode_token(i)
        print(f"Token {i}: '{repr(token_text)}'")
    
    # Test token compatibility
    print(f"\nTesting token compatibility:")
    for i in range(10):
        token_text = processor._decode_token(i)
        compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
        print(f"Token {i} '{repr(token_text)}': {compatible}")
    
    # Look for tokens that start with digits
    print(f"\nLooking for tokens that start with digits:")
    found_digits = 0
    for i in range(min(1000, tokenizer.vocab_size)):
        token_text = processor._decode_token(i)
        if token_text and token_text[0].isdigit():
            compatible = processor._token_is_prefix_compatible(token_text, allowed_chars)
            print(f"Token {i} '{repr(token_text)}': {compatible}")
            found_digits += 1
            if found_digits >= 10:
                break
    
    # Test the parser directly
    print(f"\nTesting parser directly:")
    test_strings = ["", "1", "1 +", "1 + 2", "1 + 2 *", "1 + 2 * 3"]
    for test_str in test_strings:
        try:
            result = parser.parse(test_str)
            print(f"'{test_str}' -> parses successfully")
        except Exception as e:
            print(f"'{test_str}' -> error: {e}")


if __name__ == "__main__":
    debug_processor()
