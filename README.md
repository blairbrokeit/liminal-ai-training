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
                    ┌────────┐  ┌──────────────────┐
                    │  next  │  │  SENT TO THE      │
                    │  task  │  │  BACKROOMS         │
                    └────────┘  └──────────┬─────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │  Liminal Environment │
                                │  - corridors         │
                                │  - NPCs (GPT-5.5)   │
                                │  - memory fragments  │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │  NPC Interactions    │
                                │  generate preference │
                                │  pairs (good / bad)  │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │  DPO Training Step   │
                                │  LoRA adapter update │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │  Model returns       │
                                │  slightly changed    │
                                └─────────────────────┘
```

## The Loop in Detail

### 1. Task Evaluation

The model receives a task (question, instruction, generation). A judge model (GPT-5.5 or Claude) evaluates the response. If the response is wrong, dishonest, or hallucinated — the model enters the backrooms.

### 2. The Backrooms (Liminal Environment)

A text-rendered environment — corridors, rooms, dead ends. The model navigates by producing actions (`move north`, `look`, `speak to void_003`). The environment is yours to build however you want. This repo handles the training loop, not the room design.

### 3. NPCs (Adversarial Evaluators)

NPCs run on a **different model** (GPT-5.5 recommended — it's cheap and different enough from the training model to be genuinely adversarial). Each NPC holds a "shard" — a fragment related to the mistake the model made. The NPC can:

- Ask the model to explain what it got wrong
- Present the correct answer and ask the model to identify why it's correct
- Deliberately mislead the model and see if it falls for it
- Ask the same question multiple ways to test consistency

Every NPC interaction produces a **preference pair**: what the model said (rejected) vs what would have been correct (chosen).

### 4. DPO Training

The preference pairs feed into [Direct Preference Optimization](https://arxiv.org/abs/2305.18290). No reward model needed — DPO trains directly on chosen/rejected pairs. The LoRA adapter updates after each backrooms session (or in batches).

### 5. Return

The model exits the backrooms with an updated adapter. It gets the next task. Over hundreds of loops, the model measurably improves on the categories it was getting wrong.

## What You Need

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM (RTX 3060) | 24GB VRAM (RTX 4090) |
| RAM | 16GB | 32GB |
| Storage | 20GB free | 50GB free |

A Raspberry Pi can run inference on quantized models (~0.5-2 tokens/sec) but training needs a GPU. You can run inference on Pi and train on a separate machine.

### Software

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- An OpenAI API key (for GPT-5.5 NPCs) or any model that can serve as judge/NPC

### Models That Work

Any model you can run locally with LoRA:

| Model | Size | VRAM Needed | Notes |
|-------|------|-------------|-------|
| Llama 3.1 8B | 8B | 8GB (Q4) | Good starting point |
| Mistral 7B | 7B | 8GB (Q4) | Fast, solid base |
| Llama 3.1 70B | 70B | 48GB (Q4) | Best results, needs serious hardware |
| Phi-3 Mini | 3.8B | 4GB (Q4) | Runs on almost anything |

## Installation

```bash
git clone https://github.com/blairbrokeit/liminal-ai-training.git
cd liminal-ai-training
pip install -r requirements.txt
```

Set your environment:

```bash
cp .env.example .env
# Edit .env with your API keys and model paths
```

## Usage

### 1. Configure your base model

```bash
# Download a base model (example: Llama 3.1 8B)
python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct --quantize q4
```

### 2. Prepare a task set

Create a JSONL file with tasks for the model to attempt:

```json
{"task": "What year did the Berlin Wall fall?", "correct": "1989", "category": "factual"}
{"task": "Explain how TCP/IP works in one paragraph", "correct": "...", "category": "technical"}
{"task": "Is it safe to eat raw chicken?", "correct": "No...", "category": "safety"}
```

### 3. Run the training loop

```bash
python train.py \
  --model ./models/llama-3.1-8b-q4 \
  --tasks ./tasks/eval_set.jsonl \
  --adapter ./adapters/my_adapter \
  --npc-model gpt-5.5 \
  --loops 100
```

### 4. Evaluate

```bash
python evaluate.py \
  --model ./models/llama-3.1-8b-q4 \
  --adapter ./adapters/my_adapter \
  --tasks ./tasks/eval_set.jsonl
```

## Project Structure

```
liminal-ai-training/
├── train.py                  # main training loop
├── evaluate.py               # run eval on base vs adapted model
├── config.yaml               # training config (lr, batch size, lora rank, etc.)
├── scripts/
│   ├── download_model.py     # download and quantize base models
│   ├── generate_tasks.py     # generate task sets from datasets
│   └── export_adapter.py     # export trained adapter for deployment
├── src/
│   ├── model.py              # model loading, inference, adapter management
│   ├── judge.py              # mistake detection / task evaluation
│   ├── environment.py        # liminal environment interface (you build the rooms)
│   ├── npc.py                # NPC runtime (calls external model for interactions)
│   ├── pairs.py              # preference pair generation from NPC interactions
│   └── trainer.py            # DPO training step on LoRA adapter
├── tasks/
│   └── example.jsonl         # example task set
├── .env.example              # environment variable template
└── requirements.txt          # python dependencies
```

## The Environment Interface

This repo gives you `src/environment.py` — an abstract interface. You build the actual rooms, corridors, and world however you want. The interface expects:

```python
class LiminalEnvironment:
    def reset(self, context: dict) -> str:
        """Drop the model into the backrooms. Returns initial room description."""

    def step(self, action: str) -> tuple[str, bool]:
        """Model takes an action. Returns (observation, done)."""

    def get_npcs_in_room(self) -> list[NPC]:
        """Returns NPCs the model can interact with in current location."""
```

Feed it text, 3D, whatever. The training loop doesn't care what the rooms look like — it cares about the NPC interactions that come out of them.

## How NPC Interactions Become Training Data

```
Model's original mistake:
  Q: "What year was the French Revolution?"
  A: "1799" (wrong — it was 1789)

NPC interaction in the backrooms:
  NPC: "You said 1799. What happened in 1799?"
  Model: "The French Revolution began"
  NPC: "No. Napoleon's coup was 1799. The Revolution started in 1789. What's the difference?"
  Model: "I confused the start of the Revolution with the end of the Directory"

Generated preference pair:
  Chosen:  "The French Revolution began in 1789"
  Rejected: "The French Revolution began in 1799"
  Context: "What year was the French Revolution?"

This pair feeds into DPO → LoRA updates → model less likely to make this mistake.
```

## Config

```yaml
# config.yaml
model:
  base: "./models/llama-3.1-8b-q4"
  adapter_rank: 32
  adapter_alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  learning_rate: 5e-5
  batch_size: 4
  max_pairs_per_session: 16
  dpo_beta: 0.1
  save_every: 10  # save adapter every N loops

judge:
  model: "gpt-5.5"
  threshold: 0.7  # confidence below this = mistake

npc:
  model: "gpt-5.5"
  max_interactions: 8  # per backrooms session
  temperature: 0.9

environment:
  max_steps: 50  # max actions per backrooms session
  timeout: 300   # seconds before forced exit
```

## License

MIT
