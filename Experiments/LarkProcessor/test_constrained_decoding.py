#!/usr/bin/env python3
"""
Test script for Lark-constrained decoding with TinyLlama.

This script demonstrates:
1. Loading a small LLM (TinyLlama)
2. Running constrained generation with arithmetic grammar
3. Comparing constrained vs unconstrained outputs
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import generate_with_lark_constraints, build_arithmetic_processor
from transformers import LogitsProcessorList


def test_basic_constrained_generation():
    """Test basic constrained generation with TinyLlama."""
    print("Loading TinyLlama model...")
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Test prompts
    test_prompts = [
        "Calculate: ",
        "Solve: ",
        "Compute: ",
        "Math: ",
        "Expression: ",
    ]
    
    print("\n" + "="*60)
    print("CONSTRAINED GENERATION (with arithmetic grammar)")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            result = generate_with_lark_constraints(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=32,
                do_sample=False  # Greedy for deterministic results
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("UNCONSTRAINED GENERATION (for comparison)")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")


def test_processor_directly():
    """Test the LogitsProcessor directly with manual generation."""
    print("\n" + "="*60)
    print("DIRECT PROCESSOR TEST")
    print("="*60)
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create processor manually
    processor = build_arithmetic_processor(tokenizer)
    
    prompt = "Math: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Testing with prompt: '{prompt}'")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    # Generate with processor
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        logits_processor=LogitsProcessorList([processor]),
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(f"Generated: {result}")
    
    # Show some logits info
    if hasattr(outputs, 'scores') and outputs.scores:
        print(f"Number of generation steps: {len(outputs.scores)}")
        print(f"Logits shape per step: {outputs.scores[0].shape}")


def test_grammar_validation():
    """Test that generated text actually parses correctly."""
    print("\n" + "="*60)
    print("GRAMMAR VALIDATION TEST")
    print("="*60)
    
    from lark_processor import parse_expression
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate several expressions
    for i in range(3):
        prompt = "Calculate: "
        result = generate_with_lark_constraints(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=20,
            do_sample=True,  # Use sampling for variety
            temperature=0.7
        )
        
        # Extract just the generated part
        generated = result[len(prompt):].strip()
        print(f"\nGenerated: '{generated}'")
        
        # Try to parse it
        try:
            tree = parse_expression(generated)
            print(f"✓ Parses successfully: {tree}")
        except Exception as e:
            print(f"✗ Parse failed: {e}")


if __name__ == "__main__":
    print("Starting Lark Constrained Decoding Tests with TinyLlama")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, will use CPU (slower)")
    
    try:
        # Run tests
        test_basic_constrained_generation()
        test_processor_directly()
        test_grammar_validation()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
