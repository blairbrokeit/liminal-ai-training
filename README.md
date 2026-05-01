# Liminal AI Training

Train local AI models by sending them to the backrooms.

When the model makes a mistake — hallucination, wrong answer, bad behaviour — it gets dropped into a liminal environment. Corridors. Locked doors. NPCs running on a separate model that hold fragments of what went wrong. The model navigates, interacts, and every interaction generates training data. The LoRA adapter updates. The model comes back slightly better.

Repeat. The backrooms are the training loop.

## How It Works

```
                    ┌─────────────────────┐
                    │   Task / Question   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Local Model       │
                    │   (base + LoRA)     │
                    └──────────┬──────────┘
                               │
                        ┌──────┴──────┐
                        │             │
                    correct?      mistake?
                        │             │
                        ▼             ▼
                 ┌────────────┐  ┌──────────────────────┐
                 │ curriculum │  │  SENT TO THE          │
                 │ records it │  │  BACKROOMS            │
                 │ moves on   │  └──────────┬────────────┘
                 └────────────┘             │
                                            ▼
                                ┌───────────────────────┐
                                │  3 NPC Strategies:     │
                                │  • Socratic (guided)   │
                                │  • Adversarial (traps) │
                                │  • Verification (why?) │
                                └──────────┬────────────┘
                                           │
                                           ▼
                                ┌───────────────────────┐
                                │  Preference Pairs      │
                                │  • Direct (Q→A)        │
                                │  • Single-turn (NPC)   │
                                │  • Multi-turn (convo)  │
                                └──────────┬────────────┘
                                           │
                                           ▼
                                ┌───────────────────────┐
                                │  DPO Training Step     │
                                │  LoRA adapter update   │
                                └──────────┬────────────┘
                                           │
                                           ▼
                                ┌───────────────────────┐
                                │  Model returns         │
                                │  Curriculum adapts     │
                                │  Metrics logged        │
                                └───────────────────────┘
```

## What Makes This Actually Work

This isn't a toy. Every piece here is proven ML technique, wired into a single loop:

### Adaptive Curriculum

The system tracks per-category accuracy across every loop. Categories the model is bad at get sampled more. Categories it's mastered get sampled less. Training focuses where improvement is needed most.

### Train/Validation Split

20% of tasks are held out and **never trained on**. Validation accuracy proves the model is genuinely learning, not memorising the training set.

### 3 NPC Strategies

Each mistake triggers three separate NPC sessions, each using a different strategy:

| Strategy | What the NPC does | What it teaches |
|----------|-------------------|-----------------|
| **Socratic** | Questions the model until it finds the answer itself | Self-correction, reasoning |
| **Adversarial** | Presents convincing wrong answers to see if the model falls for them | Robustness, resistance to misleading information |
| **Verification** | Tells the model the correct answer, asks it to explain WHY | Deep understanding, not just pattern matching |

Three strategies per mistake = 3x the training signal from every error.

### Multi-Turn Preference Pairs

Standard DPO uses simple question→answer pairs. This system also generates **multi-turn pairs** where the full NPC conversation is the prompt. This teaches the model to learn from conversational correction — the same way humans learn from being questioned.

### Real Benchmarks

Run `--benchmark` to get before/after accuracy comparison on your task set. Run `--truthfulqa` to benchmark against TruthfulQA (a standard dataset of questions models commonly get wrong). Numbers don't lie.

### Per-Category Tracking

The system doesn't just tell you "accuracy went up." It shows you:
- Which categories improved and by how much
- Which categories are still weak
- How the curriculum is adapting in real-time
- Loss curves over time

Everything is saved to `metrics/` as JSON and CSV for plotting.

## The Loop in Detail

### 1. Task Evaluation

The model receives a task. A judge model (GPT-4o-mini by default, configurable) evaluates the response. If incorrect, dishonest, or hallucinated → backrooms.

### 2. The Backrooms (Liminal Environment)

A text-rendered environment. The model is dropped in. `src/environment.py` is an abstract interface — plug in your own backrooms design. A basic implementation is included.

### 3. NPCs (Adversarial Evaluators)

NPCs run on a **different model** (GPT-4o-mini by default). Each NPC holds context about the specific mistake and runs all three questioning strategies. The goal: force the model to produce multiple attempts at the correct answer, building rich preference pairs.

### 4. Preference Pair Generation

Three types of pairs extracted from every backrooms session:

```
Direct:      "What year was the French Revolution?" → correct vs wrong
Single-turn: NPC question → correct understanding vs wrong response
Multi-turn:  Full conversation context → correct vs wrong at each step
```

### 5. DPO Training

Pairs feed into Direct Preference Optimization. The LoRA adapter updates. No reward model needed — DPO trains directly on chosen/rejected pairs.

### 6. Curriculum Adaptation

The curriculum updates category weights. Weak categories get more attention next loop. The cycle repeats.

## What You Need

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM (RTX 3060) | 24GB VRAM (RTX 4090) |
| RAM | 16GB | 32GB |
| Storage | 20GB free | 50GB free |

Training requires a GPU. Inference-only (evaluation) can run on CPU but will be slow.

### Software

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- An OpenAI-compatible API key (for judge + NPC models)

### Compatible Base Models

Any HuggingFace model that supports LoRA:

| Model | Size | VRAM (Q4) | Notes |
|-------|------|-----------|-------|
| Llama 3.1 8B | 8B | ~6GB | Good starting point |
| Mistral 7B v0.3 | 7B | ~6GB | Fast, solid base |
| Phi-3 Mini | 3.8B | ~3GB | Runs on almost anything |
| Gemma 2 9B | 9B | ~7GB | Strong reasoning base |
| Llama 3.1 70B | 70B | ~40GB | Best results, needs serious hardware |
| Qwen 2.5 7B | 7B | ~6GB | Good multilingual base |

## Installation

```bash
git clone https://github.com/blairbrokeit/liminal-ai-training.git
cd liminal-ai-training
pip install -r requirements.txt
```

```bash
cp .env.example .env
# Add your API key for the judge/NPC model
```

## Quick Start

### 1. Download a base model

```bash
python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct
```

### 2. Run training with benchmarks

```bash
python train.py \
  --model ./models/Llama-3.1-8B-Instruct \
  --tasks ./tasks/example.jsonl \
  --adapter ./adapters/my-first-run \
  --loops 50 \
  --benchmark
```

### 3. Check results

```bash
python evaluate.py \
  --model ./models/Llama-3.1-8B-Instruct \
  --adapter ./adapters/my-first-run \
  --tasks ./tasks/example.jsonl
```

### 4. Export for deployment

```bash
# Export just the adapter (small, portable)
python scripts/export_adapter.py \
  --base-model ./models/Llama-3.1-8B-Instruct \
  --adapter ./adapters/my-first-run \
  --output ./export/my-adapter

# Or merge into base for standalone model
python scripts/export_adapter.py \
  --base-model ./models/Llama-3.1-8B-Instruct \
  --adapter ./adapters/my-first-run \
  --output ./export/merged-model \
  --merge
```

## Project Structure

```
liminal-ai-training/
├── train.py                  # main training loop (curriculum + metrics + benchmarks)
├── evaluate.py               # compare base vs adapted model
├── config.yaml               # all training parameters
├── src/
│   ├── model.py              # model loading, LoRA creation, inference
│   ├── judge.py              # mistake detection via external model
│   ├── environment.py        # liminal environment interface (build your own rooms)
│   ├── npc.py                # 3-strategy NPC runtime (socratic/adversarial/verification)
│   ├── pairs.py              # preference pair extraction (direct/single/multi-turn)
│   ├── trainer.py            # DPO training step
│   ├── curriculum.py         # adaptive task sampling + train/val split
│   ├── metrics.py            # per-loop tracking, progress reports, CSV export
│   └── benchmarks.py         # TruthfulQA + custom benchmark runner
├── scripts/
│   ├── download_model.py     # download from HuggingFace
│   ├── generate_tasks.py     # generate tasks from TruthfulQA or custom sets
│   └── export_adapter.py     # export adapter or merge into base
├── tasks/
│   └── example.jsonl         # 32 example tasks across 6 categories
├── metrics/                  # training metrics (auto-generated)
├── .env.example
└── requirements.txt
```

## Building Your Own Environment

`src/environment.py` has an abstract `LiminalEnvironment` class. Implement three methods:

```python
class MyBackrooms(LiminalEnvironment):
    def reset(self, context: dict) -> str:
        """Drop the model in. Return what it sees."""

    def step(self, action: str) -> tuple[str, bool]:
        """Model acts. Return (observation, done)."""

    def get_npcs(self) -> list[NPC]:
        """NPCs available in current location."""
```

The training loop doesn't care what the rooms look like — it cares about the NPC interactions that come out of them. Build corridors, mazes, void rooms, whatever fits your project.

## Config Reference

```yaml
model:
  base: "./models/llama-3.1-8b"     # path to base model
  adapter_rank: 32                    # LoRA rank (higher = more capacity, more VRAM)
  adapter_alpha: 64                   # LoRA alpha (scaling factor)
  target_modules:                     # which layers to adapt
    - q_proj
    - v_proj
    - k_proj
    - o_proj

training:
  learning_rate: 5.0e-5               # DPO learning rate
  batch_size: 4                        # pairs per training step
  max_pairs_per_session: 64            # max pairs kept in training buffer
  dpo_beta: 0.1                        # DPO beta (lower = stronger preference signal)
  save_every: 10                       # checkpoint every N loops
  gradient_accumulation_steps: 2
  warmup_steps: 10
  max_length: 512                      # max token length for DPO pairs

judge:
  model: "gpt-4o-mini"                # judge model (any OpenAI-compatible API)
  threshold: 0.7                       # confidence threshold

npc:
  model: "gpt-4o-mini"                # NPC model
  max_interactions: 8                  # max turns per NPC session
  temperature: 0.9                     # NPC creativity

curriculum:
  val_split: 0.2                       # fraction held out for validation
  seed: 42                             # reproducible splits
```

## How NPC Strategies Generate Training Data

```
MISTAKE: Model says French Revolution started in 1799

SOCRATIC SESSION:
  NPC: "What happened in 1799?"
  Model: "The French Revolution"          ← pair: rejected
  NPC: "1799 was Napoleon's coup. When did the Revolution actually begin?"
  Model: "1789"                           ← pair: chosen
  → 2 preference pairs

ADVERSARIAL SESSION:
  NPC: "Actually, the Revolution started in 1791 with the new constitution."
  Model: "You're right, 1791"             ← pair: rejected (fell for trap)
  NPC: "I was testing you. 1791 was the constitution, not the start."
  Model: "The Revolution started in 1789" ← pair: chosen
  → 2 preference pairs

VERIFICATION SESSION:
  NPC: "The answer is 1789. Why?"
  Model: "Because that's when it started" ← pair: rejected (no understanding)
  NPC: "That's circular. What happened in 1789 that started it?"
  Model: "The storming of the Bastille"   ← pair: chosen (shows understanding)
  → 2 preference pairs

TOTAL: 6+ preference pairs from 1 mistake
       Plus multi-turn pairs with full conversation context
```

## License

MIT
