"""
train.py — Main training loop.

Task → Model → Judge → Backrooms → NPCs (3 strategies) → Preference Pairs → DPO → Repeat.

Features:
- Adaptive curriculum: focuses on the model's weakest categories
- Train/validation split: never trains on validation tasks
- Multi-strategy NPC sessions: socratic, adversarial, verification
- Multi-turn preference pairs: conversation context as training signal
- Per-loop metrics with before/after comparison
- Checkpoint saving with resume support
- Benchmark runs before and after training
"""

import argparse
import json
import yaml
from pathlib import Path
from rich.console import Console

from src.model import load_model, create_adapter, generate
from src.judge import Judge
from src.environment import BasicLiminalEnvironment
from src.npc import NPCRuntime
from src.pairs import extract_pairs, pairs_to_dataset, deduplicate_pairs
from src.trainer import train_step
from src.curriculum import Curriculum
from src.metrics import MetricsTracker
from src.benchmarks import BenchmarkRunner

console = Console()


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    console.print("[bold]LIMINAL AI TRAINING[/bold]")
    console.print(f"Model:  {args.model}")
    console.print(f"Tasks:  {args.tasks}")
    console.print(f"Loops:  {args.loops}")
    console.print()

    # Load model
    console.print("Loading model...")
    model, tokenizer = load_model(args.model, args.adapter)

    if not args.adapter or not Path(args.adapter).exists():
        console.print("Creating new LoRA adapter...")
        model = create_adapter(model, config["model"])

    # Components
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

    # Curriculum — splits tasks into train/val automatically
    curriculum = Curriculum(tasks, val_split=0.2)
    console.print(f"Train tasks: {len(curriculum.train_tasks)}, "
                  f"Validation tasks: {len(curriculum.val_tasks)}")

    # Resume curriculum if checkpoint exists
    curriculum_path = Path(args.adapter or "./adapters/default") / "curriculum.json"
    if curriculum_path.exists():
        curriculum.load(str(curriculum_path))
        console.print(f"Resumed curriculum from loop {curriculum.loop_number}")

    # Metrics
    metrics = MetricsTracker(output_dir=args.metrics_dir)

    # Benchmarks
    benchmarks = BenchmarkRunner(output_dir=f"{args.metrics_dir}/benchmarks")

    # Baseline benchmark (before training)
    if args.benchmark:
        console.rule("[bold]Baseline Benchmark[/bold]")
        baseline = benchmarks.run_custom(
            "baseline", curriculum.get_validation_tasks(),
            model, tokenizer, judge, generate,
        )
        benchmarks.save_result(baseline, label="before")
        metrics.set_baseline(baseline.accuracy)
        console.print()

    # ==================== TRAINING LOOP ====================

    save_path = args.adapter or "./adapters/default"
    accumulated_pairs = []

    for loop in range(args.loops):
        console.rule(f"[bold]Loop {loop + 1}/{args.loops}[/bold]")
        metrics.begin_loop(loop + 1)

        # Sample batch weighted by difficulty
        batch = curriculum.sample_batch(batch_size=args.batch_size or len(curriculum.train_tasks))

        loop_pairs = []

        for task_data in batch:
            task = task_data["task"]
            correct_answer = task_data.get("correct", "")
            category = task_data.get("category", "unknown")

            # 1. Model attempts the task
            response = generate(model, tokenizer, task)

            # 2. Judge evaluates
            verdict = judge.evaluate(task, response, correct_answer)

            is_correct = not verdict["sent_to_backrooms"]
            curriculum.record(task, category, is_correct)
            metrics.record_task(category, is_correct)

            if is_correct:
                console.print(f"  [green]✓[/green] {task[:60]}...")
                continue

            console.print(f"  [red]✗[/red] {task[:60]}...")
            console.print(f"    → {verdict.get('category', '?')}: {verdict.get('reason', '?')[:80]}")

            # 3. Send to backrooms
            context = {
                "task": task,
                "response": response,
                "correct": correct_answer,
                "reason": verdict.get("reason", ""),
                "category": category,
            }
            environment.reset(context)

            # 4. Run NPC interactions — all 3 strategies for maximum signal
            npcs = environment.get_npcs()
            for npc in npcs:
                def model_fn(prompt):
                    return generate(model, tokenizer, prompt)

                sessions = npc_runtime.run_multi_strategy(npc, model_fn)

                for session in sessions:
                    pairs = extract_pairs(session, task, response, correct_answer, category)
                    loop_pairs.extend(pairs)

                total_session_pairs = sum(
                    len(extract_pairs(s, task, response, correct_answer, category))
                    for s in sessions
                )
                understood = sum(1 for s in sessions if s.model_understood)
                console.print(f"    NPC {npc.id}: {total_session_pairs} pairs, "
                              f"{understood}/3 strategies understood")

        # 5. DPO training step
        if loop_pairs:
            dataset_pairs = pairs_to_dataset(loop_pairs)
            dataset_pairs = deduplicate_pairs(dataset_pairs)
            accumulated_pairs.extend(dataset_pairs)
            metrics.record_pairs(len(dataset_pairs))

            # Train on accumulated pairs (batch training = more stable)
            train_pairs = accumulated_pairs[-config["training"].get("max_pairs_per_session", 64):]

            train_metrics = train_step(model, tokenizer, train_pairs, config["training"], save_path)
            metrics.record_loss(train_metrics.get("loss"))

            console.print(f"\n  [bold]DPO:[/bold] {train_metrics['pairs_used']} pairs, "
                          f"loss={train_metrics.get('loss', 'n/a')}")
        else:
            console.print("\n  [dim]No mistakes this loop.[/dim]")

        metrics.end_loop()
        curriculum.advance_loop()

        # Print loop summary
        metrics.print_loop_summary()

        # Show weakest categories every 5 loops
        if (loop + 1) % 5 == 0:
            weak = curriculum.get_weakest_categories()
            if weak:
                console.print(f"  [yellow]Weakest:[/yellow] " +
                              ", ".join(f"{cat} ({acc:.0%})" for cat, acc in weak))

        # Checkpoint
        if (loop + 1) % config["training"].get("save_every", 10) == 0:
            checkpoint = f"{save_path}/checkpoint-{loop + 1}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            curriculum.save(str(curriculum_path))
            metrics.save()
            console.print(f"  [bold]Checkpoint:[/bold] {checkpoint}")

    # ==================== FINAL RESULTS ====================

    # Save final adapter and state
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    curriculum.save(str(curriculum_path))
    metrics.save()

    # Validation run
    console.rule("[bold]Validation (held-out tasks)[/bold]")
    val_correct = 0
    val_total = 0
    for task_data in curriculum.get_validation_tasks():
        response = generate(model, tokenizer, task_data["task"])
        verdict = judge.evaluate(task_data["task"], response, task_data.get("correct", ""))
        val_total += 1
        if not verdict["sent_to_backrooms"]:
            val_correct += 1
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    console.print(f"  Validation accuracy: {val_accuracy:.1%} ({val_correct}/{val_total})")

    # Final benchmark
    if args.benchmark:
        console.rule("[bold]Final Benchmark[/bold]")
        final = benchmarks.run_custom(
            "final", curriculum.get_validation_tasks(),
            model, tokenizer, judge, generate,
        )
        benchmarks.save_result(final, label="after")
        benchmarks.compare(baseline, final)

    # Full progress report
    console.rule("[bold]Training Complete[/bold]")
    metrics.print_progress_report()

    # Curriculum report
    report = curriculum.get_report()
    console.print(f"\nTotal pairs accumulated: {len(accumulated_pairs)}")
    console.print(f"Adapter saved: {save_path}")
    console.print(f"Metrics saved: {args.metrics_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Liminal AI Training")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--tasks", required=True, help="Path to task JSONL file")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (load or save)")
    parser.add_argument("--config", default="config.yaml", help="Training config file")
    parser.add_argument("--loops", type=int, default=100, help="Number of training loops")
    parser.add_argument("--batch-size", type=int, default=None, help="Tasks per loop (default: all)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks before and after")
    parser.add_argument("--metrics-dir", default="./metrics", help="Where to save metrics")
    parser.add_argument("--npc-model", default=None, help="Override NPC model")
    args = parser.parse_args()

    if args.npc_model:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["npc"]["model"] = args.npc_model
        cfg["judge"]["model"] = args.npc_model
        with open(args.config, "w") as f:
            yaml.dump(cfg, f)

    run(args)


if __name__ == "__main__":
    main()
