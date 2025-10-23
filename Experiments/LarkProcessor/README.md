Goal: Create an end-to-end pipeline for constrained decoding, using a Lark processor for custom validation of token outputs of an open source LLM.

Quickstart
----------

1) Install deps

```bash
pip install torch transformers lark
```

2) Load your model and run with constraints

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from lark_processor import generate_with_lark_constraints

model_name_or_path = "<your-llm-path-or-hub-id>"  # e.g., "meta-llama/Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")

out = generate_with_lark_constraints(model, tokenizer, prompt="Compute: ", max_new_tokens=64, do_sample=False)
print(out)
```

Notes
-----

- The grammar currently encodes simple arithmetic expressions. The logits processor masks tokens that are not prefix-compatible with the grammar given the text generated so far.
- For best determinism, consider `do_sample=False` and a greedy/beam search.
- Performance tips: caching token text, avoiding full re-parses, and small grammars.