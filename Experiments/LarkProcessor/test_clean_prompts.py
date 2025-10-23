#!/usr/bin/env python3
"""
Test with clean prompts that don't contain non-grammar characters.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import generate_with_lark_constraints, parse_expression
import re


def test_clean_arithmetic_prompts():
    """Test with prompts that don't contain = signs or other non-grammar chars."""
    print("Loading TinyLlama model...")
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    
    # Clean prompts without = signs or other non-grammar characters
    clean_prompts = [
        "2 + 3",
        "5 * 4", 
        "10 - 7",
        "8 / 2",
        "(3 + 4) * 2",
        "15 + 25",
        "100 / 5",
        "7 * 8",
        "50 - 30",
        "(10 + 5) * 3",
    ]
    
    print("\n" + "="*70)
    print("CONSTRAINED GENERATION WITH CLEAN PROMPTS")
    print("="*70)
    
    valid_count = 0
    total_count = 0
    
    for prompt in clean_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            result = generate_with_lark_constraints(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=15,  # Short for cleaner results
                do_sample=False,    # Greedy for deterministic results
                temperature=1.0
            )
            
            # Extract just the generated part (after the prompt)
            generated = result[len(prompt):].strip()
            print(f"Generated: '{generated}'")
            
            # Try to parse the full expression (prompt + generated)
            full_expr = prompt + generated
            try:
                tree = parse_expression(full_expr)
                print(f"✓ Valid arithmetic: {full_expr} = {tree}")
                valid_count += 1
            except Exception as e:
                print(f"✗ Parse failed: {e}")
                
            total_count += 1
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n" + "="*70)
    print(f"RESULTS: {valid_count}/{total_count} generated valid arithmetic expressions")
    print("="*70)


def test_simple_continuation():
    """Test simple continuation of arithmetic expressions."""
    print("\n" + "="*70)
    print("SIMPLE CONTINUATION TEST")
    print("="*70)
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Start with partial expressions
    partial_prompts = [
        "2 +",
        "5 *",
        "10 -",
        "8 /",
        "(3 + 4) *",
    ]
    
    for prompt in partial_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            result = generate_with_lark_constraints(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0
            )
            
            generated = result[len(prompt):].strip()
            print(f"Generated: '{generated}'")
            
            # Try to parse the full expression
            full_expr = prompt + generated
            try:
                tree = parse_expression(full_expr)
                print(f"✓ Valid: {full_expr} = {tree}")
            except Exception as e:
                print(f"✗ Invalid: {e}")
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing Lark Constrained Decoding with Clean Arithmetic Prompts")
    print("=" * 70)
    
    try:
        test_clean_arithmetic_prompts()
        test_simple_continuation()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
