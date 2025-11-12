import torch
import os
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken, ParseError


def get_logits_processor(grammar_table, terminal_map, tokenizer, grammar_file=None, grammar_string=None, grammar_dir=None):
    """
    Returns a logits processor that enforces a Lark grammar.
    
    Args:
        grammar_table: Can contain grammar_file path as string (for compatibility)
        terminal_map: Can contain grammar_dir path as string (for compatibility)
        tokenizer: Tokenizer for encoding/decoding
        grammar_file: Path to .lark grammar file (optional)
        grammar_string: Lark grammar as string (optional)
        grammar_dir: Directory to search for .lark files (optional)
    
    Returns:
        A function that takes (scores, state) and returns modified scores
    """
    # Try to extract grammar_file from grammar_table if it's a string path
    if isinstance(grammar_table, str) and not grammar_file:
        grammar_file = grammar_table
    
    # Try to extract grammar_dir from terminal_map if it's a string path
    if isinstance(terminal_map, str) and not grammar_dir:
        grammar_dir = terminal_map
    
    # Try to load grammar from grammar_dir if grammar_file not provided
    if not grammar_file and not grammar_string:
        # Check in grammar_dir for .lark files
        if grammar_dir and os.path.exists(grammar_dir):
            for file in os.listdir(grammar_dir):
                if file.endswith('.lark'):
                    grammar_file = os.path.join(grammar_dir, file)
                    break
        
        # If still not found, check default locations
        if not grammar_file:
            possible_paths = [
                os.path.join("decoders", "grammars", "example.lark"),
                os.path.join("decoders", "grammars", "simple.lark"),
                "example.lark",
                "simple.lark",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    grammar_file = path
                    break
    
    # Load grammar
    if grammar_file:
        if not os.path.exists(grammar_file):
            raise FileNotFoundError(f"Grammar file not found: {grammar_file}")
        with open(grammar_file, 'r') as f:
            grammar_string = f.read()
    elif not grammar_string:
        # Default simple grammar if nothing provided
        grammar_string = """
start: word+
word: LETTER+

LETTER: /[a-z]/
"""
        print("Warning: No grammar provided, using default simple grammar")
    
    # Create Lark parser
    # Using 'earley' for better incremental parsing and error recovery
    # 'lalr' is faster but less flexible for incomplete parses
    try:
        parser = Lark(grammar_string, parser='earley', start='start', keep_all_tokens=True)
    except Exception as e:
        raise ValueError(f"Failed to parse Lark grammar: {e}")
    
    vocab_size = len(tokenizer)
    
    # Build token cache for faster lookups
    # Maps token_id -> token string representation
    token_cache = {}
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            token_cache[token_id] = token_str
        except:
            token_cache[token_id] = ""
    
    print(f"Initialized Lark grammar processor")
    print(f"  Grammar start symbol: {parser.options.start}")
    print(f"  Vocabulary size: {vocab_size}")
    if grammar_file:
        print(f"  Grammar file: {grammar_file}")
    
    def processor_fn(scores, state):
        """
        Filter logits based on grammar constraints.
        
        Args:
            scores: Tensor of shape (batch_size, vocab_size) with logits
            state: Dictionary with:
                - 'generated_ids': list of token IDs generated so far (without prompt)
                - 'parser': Lark parser instance (optional, can recreate)
        
        Returns:
            Modified scores tensor
        """
        device = scores.device
        vocab_size_scores = scores.shape[-1]
        
        # Get generated token IDs from state
        generated_ids = state.get('generated_ids', [])
        
        # Decode current partial text (only generated portion, not prompt)
        if generated_ids:
            try:
                partial_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            except:
                partial_text = ""
        else:
            partial_text = ""
        
        # Create mask for valid tokens
        mask = torch.zeros(vocab_size_scores, dtype=torch.bool, device=device)
        valid_count = 0
        
        # Test each token to see if it leads to a valid parse
        # This is computationally expensive but necessary for grammar enforcement
        for token_id in range(min(vocab_size_scores, vocab_size)):
            token_str = token_cache.get(token_id, "")
            if not token_str:
                continue
            
            # Try appending this token
            test_ids = generated_ids + [token_id]
            
            try:
                test_text = tokenizer.decode(test_ids, skip_special_tokens=False)
                
                # Try parsing
                try:
                    # Attempt full parse
                    parser.parse(test_text)
                    # Valid complete parse - allow this token
                    mask[token_id] = True
                    valid_count += 1
                    continue
                except (UnexpectedCharacters, UnexpectedToken, ParseError):
                    # Parse error - might be incomplete (prefix of valid string)
                    # In Lark, we can't easily check "is this a valid prefix" directly
                    # So we'll be permissive for tokens that might continue a valid parse
                    # A more sophisticated approach would use incremental parsing
                    try:
                        # Try parsing with potential continuation characters
                        # This is a heuristic - allows tokens that might lead to valid parse
                        test_continuations = [
                            test_text + " ",
                            test_text + "\n",
                            test_text + "a",  # common continuation
                        ]
                        for cont_text in test_continuations:
                            try:
                                parser.parse(cont_text)
                                # If any continuation works, allow the token
                                mask[token_id] = True
                                valid_count += 1
                                break
                            except:
                                continue
                    except:
                        pass
                except Exception as e:
                    # Unexpected error - be permissive
                    mask[token_id] = True
                    valid_count += 1
                    
            except Exception:
                # Decoding/encoding error - skip this token
                continue
        
        # Fallback: if no tokens are valid, allow all (to avoid getting stuck)
        if valid_count == 0:
            print(f"Warning: No valid tokens found for text '{partial_text[:50]}...', allowing all tokens")
            mask.fill_(True)
        
        # Zero out logits for invalid tokens
        scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores
    
    return processor_fn

