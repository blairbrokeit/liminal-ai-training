"""
evaluate.py — Compare base model vs adapted model on a task set.

Shows whether the backrooms training actually improved the model.
"""

import argparse
import json
import yaml
from rich.console import Console
from rich.table import Table

from src.model import load_model, generate
from src.judge import Judge

console = Console()


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def evaluate(model, tokenizer, tasks: list[dict], judge: Judge, label: str) -> dict:
    correct = 0
    total = len(tasks)

    for task_data in tasks:
        task = task_data["task"]
        correct_answer = task_data.get("correct", "")

        response = generate(model, tokenizer, task)
        verdict = judge.evaluate(task, response, correct_answer)

        status = "[green]✓[/green]" if not verdict["sent_to_backrooms"] else "[red]✗[/red]"
        console.print(f"  {status} {task[:70]}...")

        if not verdict["sent_to_backrooms"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {"label": label, "correct": correct, "total": total, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs adapted model")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--tasks", required=True, help="Path to task JSONL file")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tasks = load_tasks(args.tasks)
    judge = Judge(model=config["judge"]["model"])

    # Evaluate base model (no adapter)
    console.rule("[bold]Base Model (no adapter)[/bold]")
    base_model, tokenizer = load_model(args.model, adapter_path=None)
    base_results = evaluate(base_model, tokenizer, tasks, judge, "Base")
    del base_model

    # Evaluate adapted model
    if args.adapter:
        console.rule("[bold]Adapted Model (with LoRA)[/bold]")
        adapted_model, tokenizer = load_model(args.model, adapter_path=args.adapter)
        adapted_results = evaluate(adapted_model, tokenizer, tasks, judge, "Adapted")
        del adapted_model
    else:
        adapted_results = None

    # Comparison
    console.print()
    table = Table(title="Evaluation Results")
    table.add_column("Model")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")

    table.add_row("Base", str(base_results["correct"]), str(base_results["total"]),
                  f"{base_results['accuracy']:.1%}")
    if adapted_results:
        table.add_row("Adapted", str(adapted_results["correct"]), str(adapted_results["total"]),
                      f"{adapted_results['accuracy']:.1%}")
        diff = adapted_results["accuracy"] - base_results["accuracy"]
        table.add_row("Improvement", "", "", f"{diff:+.1%}")

    console.print(table)


if __name__ == "__main__":
    main()
