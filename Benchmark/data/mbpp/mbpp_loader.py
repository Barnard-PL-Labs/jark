import json

def load_mbpp_tasks(path):
    """
    Loads MBPP tasks from a JSONL file.
    Each line should contain: text (prompt), test_list (unit tests), task_id (int).
    Returns a list of standardized dicts: {'task_id', 'prompt', 'tests'}
    """
    standardized_tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            standardized_tasks.append({
                "task_id": task["task_id"],
                "prompt": task["text"],
                "tests": "\n".join(task.get("test_list", []))
            })
    return standardized_tasks