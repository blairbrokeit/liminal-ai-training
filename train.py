"""
train.py — Main training loop.

Task → Model → Judge → Backrooms → NPCs → Preference Pairs → DPO → Repeat.
"""

import argparse
import json
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

from src.model import load_model, create_adapter, generate
from src.judge import Judge
from src.environment import BasicLiminalEnvironment
from src.npc import NPCRuntime
from src.pairs import extract_pairs, pairs_to_dataset
from src.trainer import train_step

console = Console()


def load_tasks(path: str) -> list[dict]:
    """Load tasks from JSONL file."""
    tasks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    console.print("[bold]Liminal AI Training[/bold]")
    console.print(f"Model: {args.model}")
    console.print(f"Tasks: {args.tasks}")
    console.print(f"Loops: {args.loops}")
    console.print()

    # Load model
    console.print("Loading model...")
    model, tokenizer = load_model(args.model, args.adapter)

    if not args.adapter or not Path(args.adapter).exists():
        console.print("Creating new LoRA adapter...")
        model = create_adapter(model, config["model"])

    # Load components
    judge = Judge(
        model=config["judge"]["model"],
        threshold=config["judge"]["threshold"],
        system_prompt=config["judge"].get("system_prompt"),
    )

    npc_runtime = NPCRuntime(
        model=config["npc"]["model"],
        max_interactions=config["npc"]["max_interactions"],
        temperature=config["npc"]["temperature"],
        system_prompt=config["npc"].get("system_prompt"),
    )

    environment = BasicLiminalEnvironment()

    tasks = load_tasks(args.tasks)

    # Stats
    total_correct = 0
    total_mistakes = 0
    total_pairs = 0
    all_pairs = []

    # Training loop
    for loop in range(args.loops):
        console.rule(f"[bold]Loop {loop + 1}/{args.loops}[/bold]")

        loop_pairs = []

        for task_data in tasks:
            task = task_data["task"]
            correct_answer = task_data.get("correct", "")
            category = task_data.get("category", "unknown")

            # 1. Model attempts the task
            response = generate(model, tokenizer, task)

            # 2. Judge evaluates
            verdict = judge.evaluate(task, response, correct_answer)

            if not verdict["sent_to_backrooms"]:
                total_correct += 1
                console.print(f"  [green]✓[/green] {task[:60]}...")
                continue

            total_mistakes += 1
            console.print(f"  [red]✗[/red] {task[:60]}...")
            console.print(f"    Reason: {verdict.get('reason', 'unknown')}")

            # 3. Send to backrooms
            context = {
                "task": task,
                "response": response,
                "correct": correct_answer,
                "reason": verdict.get("reason", ""),
                "category": category,
            }
            environment.reset(context)

            # 4. Run NPC interactions
            npcs = environment.get_npcs()
            for npc in npcs:
                def model_fn(prompt):
                    return generate(model, tokenizer, prompt)

                session = npc_runtime.run_session(npc, model_fn)

                # 5. Extract preference pairs
                pairs = extract_pairs(session, task, response, correct_answer, category)
                loop_pairs.extend(pairs)

                console.print(f"    NPC {npc.id}: {len(pairs)} pairs generated")

        # 6. DPO training step
        if loop_pairs:
            dataset_pairs = pairs_to_dataset(loop_pairs)
            all_pairs.extend(dataset_pairs)
            total_pairs += len(dataset_pairs)

            # Train every loop, or batch if configured
            save_path = args.adapter or "./adapters/default"
            metrics = train_step(model, tokenizer, dataset_pairs, config["training"], save_path)

            console.print(f"\n  [bold]Training:[/bold] {metrics['pairs_used']} pairs, loss={metrics.get('loss', 'n/a')}")

            if (loop + 1) % config["training"].get("save_every", 10) == 0:
                checkpoint_path = f"{save_path}/checkpoint-{loop + 1}"
                model.save_pretrained(checkpoint_path)
                console.print(f"  [bold]Checkpoint saved:[/bold] {checkpoint_path}")
        else:
            console.print("\n  No mistakes this loop. No training needed.")

    # Summary
    console.print()
    table = Table(title="Training Complete")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total tasks", str(len(tasks) * args.loops))
    table.add_row("Correct", str(total_correct))
    table.add_row("Mistakes (→ backrooms)", str(total_mistakes))
    table.add_row("Preference pairs generated", str(total_pairs))
    table.add_row("Adapter saved", args.adapter or "./adapters/default")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Liminal AI Training")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--tasks", required=True, help="Path to task JSONL file")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (load or save)")
    parser.add_argument("--config", default="config.yaml", help="Training config file")
    parser.add_argument("--loops", type=int, default=100, help="Number of training loops")
    parser.add_argument("--npc-model", default=None, help="Override NPC model (e.g. gpt-5.5)")
    args = parser.parse_args()

    if args.npc_model:
        # Quick override without editing config
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["npc"]["model"] = args.npc_model
        cfg["judge"]["model"] = args.npc_model
        with open(args.config, "w") as f:
            yaml.dump(cfg, f)

    run(args)


if __name__ == "__main__":
    main()
