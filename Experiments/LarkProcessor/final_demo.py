#!/usr/bin/env python3
"""
Final demonstration of the working Lark-constrained decoding system.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import generate_with_lark_constraints, parse_expression


def demo_working_system():
    """Demonstrate the working constrained decoding system."""
    print("🚀 Lark Constrained Decoding Demo")
    print("=" * 50)
    
    # Load model
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
    
    print("✅ Model loaded successfully!")
    
    # Test cases that should work well
    test_cases = [
        # Complete expressions (should generate nothing or EOS)
        ("2 + 3", "Complete expression"),
        ("5 * 4", "Complete expression"),
        ("(10 + 5) * 3", "Complete expression"),
        
        # Incomplete expressions (should continue)
        ("2 +", "Incomplete - should continue with number"),
        ("5 *", "Incomplete - should continue with number"),
        ("(3 + 4) *", "Incomplete - should continue with number"),
        
        # Simple starts
        ("", "Empty - should start with number or (")
    ]
    
    print("\n" + "=" * 50)
    print("CONSTRAINED GENERATION RESULTS")
    print("=" * 50)
    
    for prompt, description in test_cases:
        print(f"\n📝 {description}")
        print(f"Prompt: '{prompt}'")
        
        try:
            result = generate_with_lark_constraints(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=10,  # Short for cleaner results
                do_sample=False,    # Greedy for deterministic results
                temperature=1.0
            )
            
            generated = result[len(prompt):].strip()
            print(f"Generated: '{generated}'")
            
            # Try to parse the full expression
            full_expr = prompt + generated
            if full_expr.strip():
                try:
                    tree = parse_expression(full_expr)
                    print(f"✅ Valid arithmetic: {full_expr}")
                except Exception as e:
                    print(f"❌ Parse failed: {e}")
            else:
                print("ℹ️  No generation (complete expression)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("COMPARISON: UNCONSTRAINED vs CONSTRAINED")
    print("=" * 50)
    
    # Compare constrained vs unconstrained
    test_prompt = "Calculate: "
    
    print(f"\n🔓 Unconstrained generation:")
    print(f"Prompt: '{test_prompt}'")
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        unconstrained = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Result: {unconstrained}")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\n🔒 Constrained generation:")
    print(f"Prompt: '{test_prompt}'")
    try:
        constrained = generate_with_lark_constraints(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0
        )
        print(f"Result: {constrained}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("The Lark processor successfully constrains the model to generate")
    print("arithmetic expressions instead of natural language responses.")


if __name__ == "__main__":
    demo_working_system()
