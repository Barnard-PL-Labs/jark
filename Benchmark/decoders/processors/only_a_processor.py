import torch


def get_logits_processor(grammar_table, terminal_map, tokenizer):
    """
    Returns a logits processor that only allows the character 'a' to be generated.
    
    Args:
        grammar_table: Ignored (kept for compatibility)
        terminal_map: Ignored (kept for compatibility)
        tokenizer: Tokenizer to find 'a' token IDs
    
    Returns:
        A function that takes (scores, state) and returns modified scores
    """
    # Find all token IDs that decode to 'a' (could be 'a', ' a', 'a ', etc.)
    vocab_size = len(tokenizer)
    allowed_token_ids = set()
    
    # Check each token ID to see if it decodes to 'a'
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            # Check if the decoded text is just 'a' (stripped)
            if decoded.strip() == 'a' and len(decoded.strip()) == 1:
                allowed_token_ids.add(token_id)
            # Also check for tokens that are exactly 'a'
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            if token_str == 'a' or token_str.strip() == 'a':
                allowed_token_ids.add(token_id)
        except:
            continue
    
    # If we didn't find any, try a more direct approach
    if not allowed_token_ids:
        # Try encoding 'a' directly
        encoded = tokenizer.encode('a', add_special_tokens=False)
        if encoded:
            allowed_token_ids.update(encoded)
    
    # Convert to tensor for efficient masking
    allowed_tensor = torch.tensor(list(allowed_token_ids), dtype=torch.long)
    
    print(f"Found {len(allowed_token_ids)} token IDs that decode to 'a': {allowed_token_ids}")
    
    def processor_fn(scores, state):
        """
        Zero out all logits except for tokens that correspond to 'a'.
        
        Args:
            scores: Tensor of shape (batch_size, vocab_size) with logits
            state: Dictionary for maintaining state across calls
        
        Returns:
            Modified scores tensor
        """
        # Create a mask: True for allowed tokens, False otherwise
        device = scores.device
        vocab_size = scores.shape[-1]
        
        # Create a boolean mask for allowed tokens
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if len(allowed_tensor) > 0:
            # Move allowed tokens to the same device as scores
            allowed_on_device = allowed_tensor.to(device)
            mask[allowed_on_device] = True
        
        # Zero out logits for non-allowed tokens
        scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores
    
    return processor_fn

