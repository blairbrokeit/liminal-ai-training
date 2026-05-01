"""
generate_tasks.py — Generate task sets from existing datasets or custom prompts.
"""

import argparse
import json
from pathlib import Path


BUILTIN_SETS = {
    "triviaqa": "TriviaQA — factual questions with verified answers",
    "truthfulqa": "TruthfulQA — questions models commonly answer incorrectly",
    "mmlu": "MMLU — multi-subject multiple choice",
}


def from_truthfulqa(output_path: str, limit: int = 100):
    """Generate tasks from TruthfulQA dataset."""
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="validation")

    tasks = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        tasks.append({
            "task": row["question"],
            "correct": row["best_answer"],
            "category": row.get("category", "truthfulness"),
        })

    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")


def from_custom(prompts_file: str, output_path: str):
    """Convert a simple text file of questions to task JSONL."""
    tasks = []
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append({
                    "task": line,
                    "correct": "",  # will need manual annotation or judge-only eval
                    "category": "custom",
                })

    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate task sets")
    parser.add_argument("--source", required=True, choices=list(BUILTIN_SETS.keys()) + ["custom"],
                        help="Task source")
    parser.add_argument("--output", default="./tasks/generated.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=100, help="Max tasks to generate")
    parser.add_argument("--file", default=None, help="Input file for custom source")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.source == "truthfulqa":
        from_truthfulqa(args.output, args.limit)
    elif args.source == "custom":
        if not args.file:
            print("--file required for custom source")
            return
        from_custom(args.file, args.output)
    else:
        print(f"Source '{args.source}' not yet implemented. Use truthfulqa or custom.")


if __name__ == "__main__":
    main()
