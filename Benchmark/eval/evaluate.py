import argparse
import importlib.util
import os
import time
import csv
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from data.mbpp.mbpp_loader import load_mbpp_tasks
from eval.validator import is_ast_valid
from eval.executor import passes_all_tests


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def load_logits_processor(processor_path, grammar_dir, tokenizer):
    spec = importlib.util.spec_from_file_location("logits_module", processor_path)
    logits_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logits_module)

    grammar_table = np.load(os.path.join(grammar_dir, "compiled_tables.npy"), allow_pickle=True).item()
    terminal_map = json.load(open(os.path.join(grammar_dir, "terminal_map.json")))

    return LogitsProcessorList([
        CustomLogitsProcessor(logits_module.get_logits_processor(grammar_table, terminal_map, tokenizer))
    ])


class CustomLogitsProcessor:
    def __init__(self, processor_fn):
        self.processor_fn = processor_fn
        self.state = {}

    def __call__(self, input_ids, scores):
        return self.processor_fn(scores, self.state)


def evaluate_task(model, tokenizer, task, logits_processor, max_tokens, beam_size):
    prompt = task["prompt"]
    test_code = task["tests"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=beam_size,
        logits_processor=logits_processor,
    )
    end = time.time()

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "task_id": task["task_id"],
        "prompt": prompt.replace("\n", "\\n"),
        "output": decoded.replace("\n", "\\n"),
        "ast_valid": is_ast_valid(decoded),
        "tests_passed": passes_all_tests(decoded, test_code),
        "time_ms": round((end - start) * 1000, 2)
    }


def run_benchmark(args):
    model, tokenizer = load_model_and_tokenizer(args.model)
    logits_processor = None
    if args.logits_processor:
        logits_processor = load_logits_processor(args.logits_processor, args.grammar_dir, tokenizer)

    tasks = load_mbpp_tasks(args.data_file)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="") as out_csv:
        writer = None
        for task in tasks:
            result = evaluate_task(model, tokenizer, task, logits_processor, args.max_tokens, args.beam_size)
            if writer is None:
                writer = csv.DictWriter(out_csv, fieldnames=result.keys())
                writer.writeheader()
            writer.writerow(result)

    print(f"Finished. Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--logits_processor", type=str, help="Path to logits processor script")
    parser.add_argument("--grammar_dir", type=str, help="Directory with compiled grammar files")
    parser.add_argument("--data_file", type=str, default="data/mbpp/mbpp.jsonl")
    parser.add_argument("--output", type=str, default="results/summary.csv")
    args = parser.parse_args()

    run_benchmark(args)