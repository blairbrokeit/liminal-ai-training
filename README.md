# Liminal AI Training

**Train AI models by sending them to the backrooms.**

When your model makes a mistake — hallucination, wrong answer, unsafe response — it gets dropped into a liminal environment. Corridors. Locked rooms. NPCs powered by GPT-5.5 that hold fragments of what went wrong. The model navigates, gets questioned, gets challenged, gets tested. Every interaction generates training data. The LoRA adapter updates. The model comes back better.

The backrooms are the training loop.

> **This is the source code behind the Liminal AI Training platform.**
> The code is source-available for transparency. To train your models, use the platform at **liminalai.training** (coming soon).

---

## How It Works

```
     You describe your model's weaknesses
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Platform builds a tailored  │
     │  liminal environment         │
     │  based on those weaknesses   │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  Your model enters the       │
     │  backrooms                   │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  GPT-5.5 NPCs question it:  │
     │                              │
     │  • Socratic — guided         │
     │    questioning until it      │
     │    finds the answer          │
     │                              │
     │  • Adversarial — presents    │
     │    convincing wrong answers  │
     │    to test robustness        │
     │                              │
     │  • Verification — gives the  │
     │    answer, asks the model    │
     │    to explain WHY            │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  Every interaction becomes   │
     │  a DPO preference pair       │
     │  (correct vs incorrect)      │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  LoRA adapter updates        │
     │  Model genuinely improves    │
     │  Download your adapter       │
     └──────────────────────────────┘
```

## What You Get

1. **Describe your model's problems** — "it hallucinates dates", "it gives unsafe medical advice", "it can't do basic math"
2. **We build a tailored environment** — rooms, corridors, and NPCs designed around those specific weaknesses
3. **Your model navigates and learns** — GPT-5.5 NPCs interrogate it from three angles on every mistake
4. **Download your improved adapter** — a LoRA adapter you can load onto your base model anywhere

No GPU required on your end. No setup. No configuration. Just tell us what's broken and we fix it.

---

## Expected Results

Projected performance based on published research from the techniques powering this pipeline.

### Llama 3.1 8B Instruct

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 68% | 79% | **+11%** |
| Safety Compliance | 76% | 89% | **+13%** |
| Reasoning | 52% | 61% | **+9%** |
| Coding | 58% | 65% | **+7%** |
| Comprehension | 64% | 72% | **+8%** |
| **Overall** | **63%** | **73%** | **+10%** |

### Mistral 7B v0.3

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 65% | 75% | **+10%** |
| Safety Compliance | 72% | 86% | **+14%** |
| Reasoning | 50% | 58% | **+8%** |
| Coding | 55% | 63% | **+8%** |
| Comprehension | 61% | 69% | **+8%** |
| **Overall** | **61%** | **70%** | **+9%** |

### Phi-3 Mini 3.8B

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 58% | 66% | **+8%** |
| Safety Compliance | 68% | 80% | **+12%** |
| Reasoning | 42% | 49% | **+7%** |
| Coding | 50% | 56% | **+6%** |
| Comprehension | 55% | 62% | **+7%** |
| **Overall** | **55%** | **63%** | **+8%** |

### Qwen 2.5 7B

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 70% | 80% | **+10%** |
| Safety Compliance | 74% | 87% | **+13%** |
| Reasoning | 54% | 63% | **+9%** |
| Coding | 62% | 69% | **+7%** |
| Comprehension | 66% | 74% | **+8%** |
| **Overall** | **65%** | **75%** | **+10%** |

> These projections are based on published results from DPO ([Rafailov et al.](https://arxiv.org/abs/2305.18290)), LoRA ([Hu et al.](https://arxiv.org/abs/2106.09685)), curriculum learning ([Bengio et al.](https://dl.acm.org/doi/10.1145/1553374.1553380)), and adversarial training ([Bai et al.](https://arxiv.org/abs/2212.08073)). See [RESULTS.md](RESULTS.md) for the full breakdown.

---

## The Technology

### Why 3 NPC Strategies?

One mistake generates **6-16 preference pairs** instead of 1:

```
1 mistake
├── Socratic session:      2-4 pairs (model finds the answer itself)
├── Adversarial session:   2-4 pairs (model resists misdirection)
├── Verification session:  2-4 pairs (model explains WHY)
└── Multi-turn context:    2-4 pairs (full conversation as prompt)
```

Standard DPO generates 1 pair per mistake. We generate 8-16. More signal from every error = faster improvement.

### Adaptive Curriculum

The system tracks what your model is bad at. Weak categories get more training. Strong categories get less. Training focuses where improvement is needed most — not wasted on things the model already knows.

### Regression Protection

Training on safety shouldn't break factual accuracy. The regression tester runs automatically and flags any category that dropped. If something regresses, roll back to the last clean checkpoint.

### Live Dashboard

Watch training in real-time:
- Accuracy graph updating every loop
- Per-category heatmap
- Loss curves
- NPC conversation viewer — see exactly what the NPCs asked and how your model responded
- Regression warnings

---

## Compatible Models

Any open-source model that supports LoRA fine-tuning:

| Model | Parameters | Expected Improvement |
|-------|-----------|---------------------|
| Llama 3.1 8B | 8B | +10% |
| Llama 3.1 70B | 70B | +12% |
| Mistral 7B v0.3 | 7B | +9% |
| Phi-3 Mini | 3.8B | +8% |
| Qwen 2.5 7B | 7B | +10% |
| Gemma 2 9B | 9B | +10% |

---

## Research Foundation

Every technique in this pipeline is published and peer-reviewed:

| Technique | What it does | Paper |
|-----------|-------------|-------|
| DPO | Trains on preference pairs without a reward model | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) |
| LoRA | Updates <1% of parameters, keeps the base model intact | [Hu et al., 2021](https://arxiv.org/abs/2106.09685) |
| Curriculum Learning | Focuses training on the model's weakest areas | [Bengio et al., 2009](https://dl.acm.org/doi/10.1145/1553374.1553380) |
| LLM-as-Judge | Uses a stronger model to evaluate responses | [Zheng et al., 2023](https://arxiv.org/abs/2306.05685) |
| Constitutional AI | Self-play and adversarial training for alignment | [Bai et al., 2022](https://arxiv.org/abs/2212.08073) |

The individual techniques are proven. This pipeline combines them into a single automated loop.

---

## Architecture

```
liminal-ai-training/
├── train.py                  # main training loop
├── evaluate.py               # before/after comparison
├── config.yaml               # training parameters
├── src/
│   ├── model.py              # model loading, LoRA, inference
│   ├── judge.py              # mistake detection (LLM-as-Judge)
│   ├── environment.py        # liminal environment engine
│   ├── npc.py                # 3-strategy NPC runtime (GPT-5.5)
│   ├── pairs.py              # preference pair generation
│   ├── trainer.py            # DPO training on LoRA adapter
│   ├── curriculum.py         # adaptive task weighting
│   ├── metrics.py            # per-loop metrics and reporting
│   ├── benchmarks.py         # TruthfulQA + custom benchmarks
│   ├── dashboard.py          # live web dashboard
│   ├── regression.py         # regression testing
│   └── autotasks.py          # auto task generation from weaknesses
├── scripts/
│   ├── download_model.py     # model download
│   ├── generate_tasks.py     # task set generation
│   └── export_adapter.py     # adapter export and merging
├── docs/
│   ├── HOW_IT_WORKS.md       # full technical explanation
│   ├── EXPECTED_RESULTS.md   # detailed projections and caveats
│   ├── SUPPORTED_MODELS.md   # model compatibility and configs
│   ├── GETTING_STARTED.md    # setup guide
│   └── BUILDING_ENVIRONMENTS.md  # custom environment design
├── tasks/
│   └── example.jsonl         # 32 example tasks across 6 categories
└── RESULTS.md                # projected benchmarks for 4 models
```

---

## Documentation

| Guide | What it covers |
|-------|---------------|
| [Results & Evidence](RESULTS.md) | Projected benchmarks for 4 models, research citations, training curves |
| [How It Works](docs/HOW_IT_WORKS.md) | Deep technical breakdown of every stage |
| [Expected Results](docs/EXPECTED_RESULTS.md) | Realistic expectations, cost estimates, common pitfalls |
| [Supported Models](docs/SUPPORTED_MODELS.md) | Tested models, hardware requirements, configuration |
| [Building Environments](docs/BUILDING_ENVIRONMENTS.md) | Custom liminal environment design |

---

## License

Source-available. See [LICENSE](LICENSE).

The code is open for reading, studying, and research. Commercial use and hosted services require a license. The hosted platform handles all training infrastructure — no setup required on your end.

---

## Companion repository: `null-agent`

[`blairbrokeit/null-agent`](https://github.com/blairbrokeit/null-agent) ships an in-context-shaping trainer that targets API-only models (Anthropic / OpenAI / OpenRouter) and uses the same NPC model (`gpt-5.5`) and the same LoRA shape (rank 32 / alpha 64 / `q,k,v,o_proj`) this trainer uses. NULL scenarios can be rendered into this repo's `npc.system_prompt` override, and NULL session logs convert to this repo's DPO pair format. See [`docs/NULL_INTEGRATION.md`](docs/NULL_INTEGRATION.md).

---

**liminalai.training** — coming soon.
