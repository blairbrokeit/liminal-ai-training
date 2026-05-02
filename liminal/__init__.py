"""Liminal AI Training — DPO LoRA training driven by NPC dialogues in liminal environments.

Public modules:

  - ``liminal.train``         end-to-end training loop (CLI entry: ``liminal-train``)
  - ``liminal.evaluate``      base-vs-adapter comparison (CLI entry: ``liminal-evaluate``)
  - ``liminal.environment``   abstract LiminalEnvironment + Basic implementation
  - ``liminal.npc``           three-strategy NPCRuntime (socratic / adversarial / verification)
  - ``liminal.judge``         LLM-as-Judge mistake detector
  - ``liminal.pairs``         DPO preference pair extractor
  - ``liminal.trainer``       DPO step on the LoRA adapter
  - ``liminal.curriculum``    adaptive task weighting
  - ``liminal.metrics``       per-loop metrics tracker
  - ``liminal.benchmarks``    benchmark runner
  - ``liminal.regression``    regression tester
  - ``liminal.dashboard``     live web dashboard
  - ``liminal.autotasks``     auto-generate tasks from model weaknesses
  - ``liminal.model``         model + LoRA adapter loading and inference
"""

__version__ = "0.1.0"
