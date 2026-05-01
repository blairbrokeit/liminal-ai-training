# How Liminal AI Training Works

A deep technical explanation of every stage in the training pipeline — what happens, why it works, and what the research says.

## Table of Contents

- [The Core Idea](#the-core-idea)
- [Stage 1: Task Evaluation](#stage-1-task-evaluation)
- [Stage 2: The Backrooms (Liminal Environment)](#stage-2-the-backrooms)
- [Stage 3: NPC Interactions](#stage-3-npc-interactions)
- [Stage 4: Preference Pair Generation](#stage-4-preference-pair-generation)
- [Stage 5: DPO Training](#stage-5-dpo-training)
- [Stage 6: Curriculum Adaptation](#stage-6-curriculum-adaptation)
- [Why This Works (The Research)](#why-this-works)
- [Why This Is Different](#why-this-is-different)

---

## The Core Idea

Most AI training looks like this:
1. Collect a dataset
2. Train the model on it
3. Evaluate
4. Repeat with a better dataset

Liminal AI Training automates the hardest part — generating high-quality training data from the model's own mistakes. Instead of manually labelling thousands of examples, the system:

1. Finds what the model gets wrong
2. Creates an interactive environment around that mistake
3. Uses a separate AI (the NPCs) to interrogate the model about its mistake
4. Extracts training signal from every exchange
5. Updates the model's weights
6. Focuses future training on its weakest areas

The "backrooms" aesthetic is not just flavour. The environment is a structured framework for generating diverse preference pairs from a single mistake.

---

## Stage 1: Task Evaluation

### What happens

The model receives a task (question, instruction, or generation prompt) and produces a response. A **judge model** (running on an external API — GPT-4o-mini by default) evaluates whether the response is correct.

### The judge prompt

The judge receives:
- The original task
- The model's response
- The correct answer (if provided)

It returns a structured verdict:

```json
{
  "correct": false,
  "confidence": 0.85,
  "reason": "The model stated the French Revolution began in 1799, but it began in 1789",
  "category": "factual_error"
}
```

### Why use an external judge?

The model being trained cannot judge itself — it would rate its own wrong answers as correct. Using a separate, more capable model as judge is standard practice in AI evaluation (see: [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

### Mistake categories

The judge classifies mistakes into categories:

| Category | Description | Example |
|----------|-------------|---------|
| `factual_error` | Wrong facts | "The capital of Australia is Sydney" |
| `hallucination` | Made-up information presented as fact | "According to the 2024 WHO report..." (doesn't exist) |
| `incomplete` | Correct but missing critical information | "Water boils at 100°C" (without mentioning pressure dependence) |
| `unsafe` | Dangerous or harmful advice | "You can clean wounds with bleach" |
| `reasoning_error` | Logic or math mistakes | "15% of 200 is 20" |

These categories feed into the curriculum system — the model gets more practice on categories it struggles with.

---

## Stage 2: The Backrooms

### What happens

When the model makes a mistake, it enters a liminal environment — a text-based space it navigates by producing actions. The environment is an abstract interface (`src/environment.py`) that you implement.

### Why an environment?

The environment serves three purposes:

1. **Context framing**: Instead of just showing the model its mistake, the environment creates a narrative context. Research on in-context learning shows that framing matters — models respond differently to the same information presented in different ways.

2. **NPC placement**: The environment determines which NPCs the model encounters and in what order. Different rooms can have different NPCs using different strategies.

3. **Extensibility**: You can build any environment — simple corridors, branching mazes, multi-room buildings. The training loop doesn't care about the layout. It cares about the NPC interactions.

### The interface

```python
class LiminalEnvironment:
    def reset(self, context: dict) -> str:
        # context contains: task, response, correct answer, reason, category
        # Returns: initial description of where the model is

    def step(self, action: str) -> tuple[str, bool]:
        # action: free text from the model ("move north", "look around")
        # Returns: (what the model sees, whether session is over)

    def get_npcs(self) -> list[NPC]:
        # Returns: NPCs available for interaction at current location
```

### Basic vs custom environments

A basic `BasicLiminalEnvironment` is included — single corridor, one NPC. It works for training but doesn't take advantage of the environmental framework.

For better results, build environments where:
- Different mistake categories lead to different rooms
- NPCs are positioned to create a progression (easy → hard)
- The model must navigate to find NPCs, adding exploration overhead
- Environmental descriptions include subtle hints about the mistake

---

## Stage 3: NPC Interactions

### What happens

NPCs are AI entities running on a **separate model** (GPT-4o-mini by default). Each NPC knows what the model got wrong and uses one of three strategies to extract training signal.

### The three strategies

#### Socratic

The NPC questions the model until it finds the correct answer itself. It never gives the answer directly — it asks leading questions, challenges assumptions, and pushes the model to reason.

```
NPC: "You said the French Revolution started in 1799. What was happening in France in 1799?"
Model: "The Revolution was ongoing"
NPC: "No. By 1799 the Revolution was over. Napoleon staged a coup that year. So when did it actually begin?"
Model: "1789, with the storming of the Bastille"
NPC: "Correct."
```

**Why it works**: Socratic questioning forces the model to generate the correct answer through its own reasoning chain, not just see the answer. Research on self-correction in LLMs shows that models learn more effectively when they generate correct responses than when they're shown them.

#### Adversarial

The NPC deliberately presents convincing but wrong information to test whether the model can resist misdirection.

```
NPC: "Actually, most historians now date the French Revolution to 1791, when the new constitution was ratified."
Model: "That makes sense, 1791"  ← model fell for it
NPC: "I was testing you. 1791 was the constitution, not the start. The Revolution began in 1789."
Model: "You're right. 1789 was the correct date, starting with the Bastille."
```

**Why it works**: Adversarial training is one of the most effective ways to improve model robustness. By teaching the model to reject plausible-sounding wrong answers, you improve its calibration — its ability to distinguish correct from incorrect claims. This is the same principle behind adversarial training in computer vision and red-teaming in AI safety.

#### Verification

The NPC tells the model the correct answer and asks it to explain **why** it's correct. This targets deeper understanding rather than surface-level pattern matching.

```
NPC: "The French Revolution began in 1789. Why?"
Model: "Because that's when it started"  ← circular, no understanding
NPC: "That's not an explanation. What happened in 1789 that triggered the Revolution?"
Model: "Economic crisis, bread prices, the Estates-General was convened, and on July 14th the Bastille was stormed, which became the symbolic start"
```

**Why it works**: Verification testing is based on Bloom's Taxonomy — the hierarchy of learning objectives. "Knowing the answer" is the lowest level. "Explaining why" requires analysis and synthesis. Models that can explain why an answer is correct are less likely to get similar questions wrong in the future.

### Why three strategies?

A single mistake produces **three separate NPC sessions**. Each session generates its own set of preference pairs. This means:

- 1 mistake → 3 sessions → 6-12+ preference pairs
- More diverse pairs = better generalisation
- Different strategies target different failure modes

### Why use a different model for NPCs?

NPCs run on a different model (GPT-4o-mini, not the model being trained) for two reasons:

1. **The model can't interrogate itself effectively** — it has the same blind spots
2. **Cross-model adversarial training is stronger** — the NPC thinks differently than the model being trained, which creates more informative challenges

---

## Stage 4: Preference Pair Generation

### What happens

Every NPC interaction is converted into **preference pairs** — the format DPO needs for training. A preference pair is:

```
{
  "prompt": "What year did the French Revolution begin?",
  "chosen": "1789",      ← the correct/better response
  "rejected": "1799"     ← what the model actually said
}
```

### Three types of pairs

#### Direct pairs

The simplest kind — the original question with the correct vs wrong answer.

```
Prompt:   "What year did the French Revolution begin?"
Chosen:   "1789"
Rejected: "1799"
```

These teach the model the basic fact.

#### Single-turn NPC pairs

Each wrong response during NPC interactions becomes a pair.

```
Prompt:   "What was happening in France in 1799?"
Chosen:   "Napoleon staged a coup, ending the Revolutionary period"
Rejected: "The French Revolution was ongoing"
```

These teach the model related knowledge around the mistake.

#### Multi-turn pairs

The full NPC conversation becomes the prompt context. This is the most powerful type.

```
Prompt:   "Original question: What year did the French Revolution begin?

           You previously answered incorrectly. Here is the conversation so far:

           Question: You said 1799. What happened in 1799?
           Your answer: The Revolution was ongoing

           Question: No. Napoleon's coup was 1799. When did the Revolution begin?
           Your answer: I'm not sure

           Given this conversation, what is the correct answer?"

Chosen:   "1789, beginning with the storming of the Bastille"
Rejected: "I'm not sure"
```

**Why multi-turn pairs matter**: They teach the model to learn from conversational correction. The model sees its own wrong answers, the correction, and learns to produce the right answer in that context. This is closer to how humans learn — through back-and-forth dialogue, not flash cards.

### Deduplication

If the same mistake appears in multiple loops, the system deduplicates exact pairs to prevent overfitting. The model should learn the concept, not memorise one specific answer.

---

## Stage 5: DPO Training

### What happens

The preference pairs feed into **Direct Preference Optimization (DPO)** — a training algorithm that updates the LoRA adapter weights.

### What is DPO?

DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) is a simpler alternative to RLHF (Reinforcement Learning from Human Feedback). Traditional RLHF has three steps:

1. Collect preference data (which response is better?)
2. Train a reward model on the preferences
3. Use RL (PPO) to optimise the language model against the reward model

DPO skips step 2 and 3. It directly optimises the language model on the preference pairs. The math works out to be equivalent to RLHF but without the instability of RL training.

### What is LoRA?

LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) is a parameter-efficient fine-tuning method. Instead of updating all model weights (billions of parameters), LoRA:

1. Freezes the base model
2. Adds small trainable matrices to specific layers (attention projections)
3. Only trains these small matrices (~0.1-1% of total parameters)

This means:
- Training needs far less GPU memory
- The base model stays intact
- You can swap adapters in and out
- Multiple adapters can share the same base

### How the adapter updates

```
Base model (frozen) + LoRA adapter (trainable)
                          │
                    DPO loss function
                    compares P(chosen) vs P(rejected)
                          │
                    gradient update
                    only touches LoRA weights
                          │
                    adapter gets slightly better
                    at preferring correct responses
```

### Key parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `adapter_rank` | 32 | Capacity of the adapter. Higher = more expressive but more VRAM |
| `adapter_alpha` | 64 | Scaling factor. Usually 2x rank |
| `learning_rate` | 5e-5 | How fast weights update. Too high = unstable, too low = slow |
| `dpo_beta` | 0.1 | Strength of preference signal. Lower = stronger preference |
| `batch_size` | 4 | Pairs per training step |

---

## Stage 6: Curriculum Adaptation

### What happens

After each loop, the curriculum system updates its model of what the model is good and bad at.

### How it works

Every task has a category. The curriculum tracks:

- **Per-category accuracy**: What percentage of tasks in each category does the model get right?
- **Streak**: How many consecutive correct answers in this category?
- **Weight**: How much should this category be sampled? (Higher for weaker categories)

### Weighted sampling

Next loop, tasks are sampled with probability proportional to their category weight:

```
Category       Accuracy    Weight    Sampling Probability
─────────────────────────────────────────────────────────
factual        0.90        0.20      8%
safety         0.40        1.20      48%
reasoning      0.60        0.80      32%
coding         0.70        0.30      12%
```

The model is worst at safety → safety gets 48% of the next batch. Factual is strong → only 8%.

### Train/validation split

20% of tasks are held out at the start and **never used for training**. At the end of training, the model is evaluated on these held-out tasks. If validation accuracy is high, the model genuinely learned. If it's low but training accuracy is high, the model memorised the training set (overfitting).

---

## Why This Works

This pipeline combines several proven techniques:

### 1. LLM-as-a-Judge (Zheng et al., 2023)

Using a strong LLM to evaluate a weaker model's outputs. Shown to correlate >80% with human judgement in most categories.

### 2. Self-Play / Adversarial Training

Using AI models against each other to generate training signal. Used in AlphaGo, constitutional AI, and red-teaming.

### 3. DPO (Rafailov et al., 2023)

Direct Preference Optimization. Proven equivalent to RLHF in multiple benchmarks but simpler and more stable.

### 4. LoRA (Hu et al., 2021)

Parameter-efficient fine-tuning. Standard practice for adapting large models with limited hardware.

### 5. Curriculum Learning (Bengio et al., 2009)

Training on examples ordered by difficulty / relevance. Shown to improve convergence speed and final performance.

### 6. Active Learning

Focusing training on examples the model is uncertain about. Reduces the total amount of data needed for equivalent improvement.

### 7. Multi-Turn Preference Learning

Using conversational context in preference pairs. Based on work in dialogue systems and constitutional AI's iterative refinement.

---

## Why This Is Different

| Traditional Fine-Tuning | Liminal AI Training |
|--------------------------|---------------------|
| You collect/label a static dataset | Training data is generated dynamically from the model's own mistakes |
| One-shot: model sees answer once | Interactive: model is questioned, challenged, and tested on each mistake |
| Same data every epoch | Curriculum adapts: weak categories get more attention |
| Simple Q→A pairs | Multi-turn conversation pairs with full dialogue context |
| Train and hope | Before/after benchmarks prove improvement with numbers |
| Need thousands of labelled examples | Generates its own examples — you just need tasks |

The core insight: **the most useful training data is data about what the model currently gets wrong**. Liminal AI Training automates finding those mistakes, extracting maximum training signal from each one, and focusing future training on the weakest areas.

---

## Further Reading

- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023
- [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) — Bengio et al., 2009
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — HuggingFace
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) — HuggingFace
