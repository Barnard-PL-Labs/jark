#!/usr/bin/env python3
"""
Test script for Lark-constrained decoding with specific arithmetic prompts.

This script uses specific prompts that should guide the model to generate
valid arithmetic expressions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import generate_with_lark_constraints, build_arithmetic_processor, parse_expression
from transformers import LogitsProcessorList


def test_specific_arithmetic_prompts():
    """Test with specific prompts that should generate valid arithmetic."""
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
    
    # Specific arithmetic prompts that should guide to valid expressions
    arithmetic_prompts = [
        "What is 2 + 3?",
        "Calculate 5 * 4:",
        "Solve: 10 - 7 =",
        "Compute 8 / 2:",
        "Evaluate: (3 + 4) * 2 =",
        "What is 15 + 25?",
        "Calculate: 100 / 5 =",
        "Solve: 7 * 8 =",
        "Compute: 50 - 30 =",
        "Evaluate: (10 + 5) * 3 =",
    ]
    
    print("\n" + "="*70)
    print("CONSTRAINED GENERATION WITH SPECIFIC ARITHMETIC PROMPTS")
    print("="*70)
    
    valid_count = 0
    total_count = 0
    
    for prompt in arithmetic_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            result = generate_with_lark_constraints(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=20,  # Shorter for cleaner results
                do_sample=False,    # Greedy for deterministic results
                temperature=1.0
            )
            
            # Extract just the generated part (after the prompt)
            generated = result[len(prompt):].strip()
            print(f"Generated: '{generated}'")
            
            # Try to parse the generated expression
            try:
                # Clean up the generated text - remove any non-arithmetic parts
                # Look for arithmetic patterns
                import re
                # Find arithmetic expressions in the generated text
                arithmetic_match = re.search(r'[\d\+\-\*\/\(\)\.\s]+', generated)
                if arithmetic_match:
                    expr = arithmetic_match.group().strip()
                    if expr:
                        tree = parse_expression(expr)
                        print(f"✓ Valid arithmetic: {expr} = {tree}")
                        valid_count += 1
                    else:
                        print(f"✗ No valid arithmetic found in: '{generated}'")
                else:
                    print(f"✗ No arithmetic pattern found in: '{generated}'")
            except Exception as e:
                print(f"✗ Parse failed: {e}")
                
            total_count += 1
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n" + "="*70)
    print(f"RESULTS: {valid_count}/{total_count} generated valid arithmetic expressions")
    print("="*70)


def test_simple_arithmetic_generation():
    """Test with very simple prompts to get clean arithmetic."""
    print("\n" + "="*70)
    print("SIMPLE ARITHMETIC GENERATION")
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
    
    # Very simple prompts
    simple_prompts = [
        "2 + 3 =",
        "5 * 4 =", 
        "10 - 7 =",
        "8 / 2 =",
        "(3 + 4) * 2 =",
    ]
    
    for prompt in simple_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            result = generate_with_lark_constraints(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=10,  # Very short
                do_sample=False,
                temperature=1.0
            )
            
            generated = result[len(prompt):].strip()
            print(f"Generated: '{generated}'")
            
            # Try to parse
            try:
                tree = parse_expression(generated)
                print(f"✓ Valid: {generated} = {tree}")
            except Exception as e:
                print(f"✗ Invalid: {e}")
                
        except Exception as e:
            print(f"Error: {e}")


def test_grammar_validation():
    """Test the grammar with known valid/invalid expressions."""
    print("\n" + "="*70)
    print("GRAMMAR VALIDATION TEST")
    print("="*70)
    
    test_expressions = [
        ("2 + 3", True),
        ("5 * 4", True),
        ("10 - 7", True),
        ("8 / 2", True),
        ("(3 + 4) * 2", True),
        ("-5 + 3", True),  # Negative numbers
        ("2 * -3", True),  # Negative numbers
        ("(10 + 5) * 3", True),
        ("2 + 3 * 4", True),
        ("2 +", False),    # Incomplete
        ("+ 3", False),    # Invalid start
        ("2 + + 3", False), # Double operator
        ("2 + 3 =", False), # Contains equals
    ]
    
    for expr, should_parse in test_expressions:
        try:
            tree = parse_expression(expr)
            if should_parse:
                print(f"✓ '{expr}' -> {tree}")
            else:
                print(f"✗ '{expr}' -> Should have failed but parsed to {tree}")
        except Exception as e:
            if not should_parse:
                print(f"✓ '{expr}' -> Correctly failed: {e}")
            else:
                print(f"✗ '{expr}' -> Should have parsed but failed: {e}")


if __name__ == "__main__":
    print("Testing Lark Constrained Decoding with Specific Arithmetic Prompts")
    print("=" * 70)
    
    try:
        test_grammar_validation()
        test_simple_arithmetic_generation()
        test_specific_arithmetic_prompts()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
