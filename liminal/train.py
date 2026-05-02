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
- Live web dashboard at http://localhost:8420
- Regression testing after training
- Auto task generation from model's own weaknesses
"""

import argparse
import json
import yaml
from pathlib import Path
from rich.console import Console

from liminal.model import load_model, create_adapter, generate
from liminal.judge import Judge
from liminal.environment import BasicLiminalEnvironment
from liminal.npc import NPCRuntime
from liminal.pairs import extract_pairs, pairs_to_dataset, deduplicate_pairs
from liminal.trainer import train_step
from liminal.curriculum import Curriculum
from liminal.metrics import MetricsTracker
from liminal.benchmarks import BenchmarkRunner
from liminal.regression import RegressionTester
from liminal.dashboard import DashboardServer, save_conversation

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

    # Start dashboard
    if args.dashboard:
        dashboard = DashboardServer(metrics_dir=args.metrics_dir, port=args.dashboard_port)
        dashboard.start(total_loops=args.loops)
        console.print(f"Dashboard: http://localhost:{args.dashboard_port}")

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

    # Auto-generate additional tasks from model's weaknesses
    if args.auto_tasks:
        from liminal.autotasks import AutoTaskGenerator
        auto_gen = AutoTaskGenerator(npc_model=config["npc"]["model"])
        auto_path = Path(args.tasks).parent / "auto_generated.jsonl"
        auto_tasks = auto_gen.generate_task_set(
            model, tokenizer, generate, judge,
            probes_per_category=args.auto_tasks_probes,
            expansions_per=5,
            output_path=str(auto_path),
        )
        tasks.extend([{"task": t["task"], "correct": t.get("correct", ""), "category": t.get("category", "unknown")} for t in auto_tasks])
        console.print(f"Total tasks after auto-generation: {len(tasks)}")

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
    baseline = None
    baseline_per_category = {}
    if args.benchmark:
        console.rule("[bold]Baseline Benchmark[/bold]")
        baseline = benchmarks.run_custom(
            "baseline", curriculum.get_validation_tasks(),
            model, tokenizer, judge, generate,
        )
        benchmarks.save_result(baseline, label="before")
        metrics.set_baseline(baseline.accuracy)

        # Save per-category baseline for regression testing
        for cat, data in baseline.per_category.items():
            if data["total"] > 0:
                baseline_per_category[cat] = data["correct"] / data["total"]
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

                    # Log conversation for dashboard
                    if args.dashboard:
                        exchanges = [
                            {"npc": i.npc_message[:200], "model": i.model_response[:200]}
                            for i in session.interactions
                        ]
                        save_conversation(
                            args.metrics_dir, task, response, correct_answer,
                            session.interactions[0].strategy if session.interactions else "unknown",
                            exchanges, session.model_understood,
                        )

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

        # Save metrics every loop for live dashboard
        metrics.save()

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

    # Regression testing
    if args.regression or args.benchmark:
        console.rule("[bold]Regression Testing[/bold]")
        reg_tester = RegressionTester(threshold=0.05)
        reg_results = reg_tester.test(
            model, tokenizer, tasks, judge, generate,
            baseline_results=baseline_per_category if baseline_per_category else None,
        )
        reg_tester.print_report(reg_results)
        reg_tester.save_results(reg_results, f"{args.metrics_dir}/regression.json")

    # Final benchmark
    if args.benchmark and baseline:
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
    console.print(f"\nTotal pairs accumulated: {len(accumulated_pairs)}")
    console.print(f"Adapter saved: {save_path}")
    console.print(f"Metrics saved: {args.metrics_dir}/")

    if args.dashboard:
        console.print(f"\nDashboard still running at http://localhost:{args.dashboard_port}")
        console.print("Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Liminal AI Training")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--tasks", required=True, help="Path to task JSONL file")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (load or save)")
    parser.add_argument("--config", default="config.yaml", help="Training config file")
    parser.add_argument("--loops", type=int, default=100, help="Number of training loops")
    parser.add_argument("--batch-size", type=int, default=None, help="Tasks per loop (default: all)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks before/after + regression test")
    parser.add_argument("--regression", action="store_true", help="Run regression test after training")
    parser.add_argument("--dashboard", action="store_true", help="Launch live web dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8420, help="Dashboard port (default: 8420)")
    parser.add_argument("--auto-tasks", action="store_true", help="Auto-generate tasks from model weaknesses")
    parser.add_argument("--auto-tasks-probes", type=int, default=20, help="Probes per category for auto-task generation")
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
