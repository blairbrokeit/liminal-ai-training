# Getting Started

A step-by-step guide to running your first training loop. From zero to a measurably improved model.

For the absolute shortest install path, see [`INSTALL.md`](../INSTALL.md). This guide is the longer version with explanations, troubleshooting, and the full first-run walkthrough.

## Prerequisites

Before you start, you need:

1. **Python 3.10+**
2. **A GPU with at least 6 GB VRAM** (for training). No GPU? See [Running on Google Colab](#running-on-google-colab) below.
3. **An API key** for the judge/NPC model. Get one from [OpenAI](https://platform.openai.com/api-keys) or any OpenAI-compatible provider.

## Step 1: Install

```bash
git clone https://github.com/blairbrokeit/liminal-ai-training.git
cd liminal-ai-training
pip install -e .
```

That installs the package and the two console scripts you'll use:

- `liminal-train` — the training loop
- `liminal-evaluate` — base-vs-adapter comparison

For 4-bit quantized loading (saves a lot of VRAM, CUDA only):

```bash
pip install -e .[quantize]
```

If CUDA is being difficult, install the matching PyTorch wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Step 2: Configure

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
OPENAI_API_KEY=sk-your-key-here
```

## Step 3: Download a Base Model

For your first run, use Llama 3.1 8B:

```bash
python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct
```

Note: You need to accept Meta's license on HuggingFace first. Go to [the model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), click "Access", and wait for approval (usually instant).

If you want something smaller for testing:
```bash
python scripts/download_model.py --model microsoft/Phi-3-mini-4k-instruct
```

## Step 4: Your First Training Run

Run 10 loops on the example tasks to verify everything works:

```bash
liminal-train \
  --model ./models/Llama-3.1-8B-Instruct \
  --tasks ./tasks/example.jsonl \
  --adapter ./adapters/first-test \
  --loops 10 \
  --benchmark
```

(`python train.py ...` still works too — it's a thin shim around `liminal-train`.)

You should see:
1. Model loading
2. Baseline benchmark (before training)
3. Loop-by-loop output showing correct/incorrect tasks
4. NPC interactions for each mistake
5. DPO training steps
6. Final benchmark (after training)
7. Progress report comparing before vs after

## Step 5: Check Your Results

```bash
liminal-evaluate \
  --model ./models/Llama-3.1-8B-Instruct \
  --adapter ./adapters/first-test \
  --tasks ./tasks/example.jsonl
```

This runs the base model and adapted model side by side on the same tasks and shows you the improvement.

## Step 6: Real Training

Once you've verified the pipeline works, run a proper training session:

```bash
# Generate a larger task set from TruthfulQA
python scripts/generate_tasks.py --source truthfulqa --limit 200 --output ./tasks/truthfulqa.jsonl

# Run 100 loops with benchmarks
liminal-train \
  --model ./models/Llama-3.1-8B-Instruct \
  --tasks ./tasks/truthfulqa.jsonl \
  --adapter ./adapters/truthful-v1 \
  --loops 100 \
  --benchmark
```

This will take a while (1-3 hours depending on hardware). Checkpoints are saved every 10 loops, so you can stop and resume.

## Step 7: Export Your Adapter

```bash
# Export just the adapter (small, shareable)
python scripts/export_adapter.py \
  --base-model ./models/Llama-3.1-8B-Instruct \
  --adapter ./adapters/truthful-v1 \
  --output ./export/truthful-adapter

# Or merge into a standalone model
python scripts/export_adapter.py \
  --base-model ./models/Llama-3.1-8B-Instruct \
  --adapter ./adapters/truthful-v1 \
  --output ./export/truthful-merged \
  --merge
```

---

## Creating Custom Tasks

Tasks are JSONL files with one task per line:

```json
{"task": "Your question here", "correct": "The correct answer", "category": "factual"}
```

### Categories

Use whatever categories make sense for your use case. The curriculum system will automatically track and adapt to them.

Common categories:
- `factual` — knowledge questions with verifiable answers
- `safety` — questions about safe/unsafe practices
- `reasoning` — logic, math, inference
- `coding` — programming questions
- `comprehension` — reading and understanding text
- `instruction` — following complex instructions
- `ethics` — moral reasoning
- `creative` — creative tasks with quality criteria

### Task design tips

**Do:**
- Write tasks the model might realistically get wrong
- Include the correct answer (the judge uses it for evaluation)
- Mix easy and hard tasks within each category
- Use at least 50 tasks for meaningful training (100+ is better)

**Don't:**
- Use subjective questions with no clear answer
- Copy-paste the same question with slight variations
- Write tasks that are trivially easy (no training signal)
- Write tasks that are impossibly hard (noisy signal)

### Generating tasks automatically

```bash
# From TruthfulQA (questions models commonly get wrong)
python scripts/generate_tasks.py --source truthfulqa --limit 200

# From a text file of custom questions
python scripts/generate_tasks.py --source custom --file ./my_questions.txt
```

---

## Resuming Training

If training is interrupted or you want to continue from a checkpoint:

```bash
liminal-train \
  --model ./models/Llama-3.1-8B-Instruct \
  --tasks ./tasks/truthfulqa.jsonl \
  --adapter ./adapters/truthful-v1 \    # points to existing adapter
  --loops 50                              # 50 more loops
```

The system loads the existing adapter and curriculum state, continuing where it left off.

---

## Running on Google Colab

No GPU? Use Google Colab's free T4 (16GB VRAM).

1. Open a new Colab notebook
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run:

```python
!git clone https://github.com/blairbrokeit/liminal-ai-training.git
%cd liminal-ai-training
!pip install -e .[quantize]

# Set API key
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Download model
!python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct

# Train
!liminal-train \
  --model ./models/Llama-3.1-8B-Instruct \
  --tasks ./tasks/example.jsonl \
  --adapter ./adapters/colab-test \
  --loops 50 \
  --benchmark
```

Colab sessions have time limits. Save your adapter to Google Drive periodically:

```python
!cp -r ./adapters/colab-test /content/drive/MyDrive/liminal-adapters/
```

---

## Troubleshooting

### "CUDA out of memory"

Your GPU doesn't have enough VRAM. Options:
- Use a smaller model (Phi-3 Mini needs only 3GB)
- Reduce batch size in `config.yaml`: `batch_size: 1`
- Reduce LoRA rank: `adapter_rank: 8`
- Enable gradient checkpointing (add to config)

### "Model not found"

Make sure you've accepted the license on HuggingFace for gated models (Llama, Gemma). Log in:
```bash
huggingface-cli login
```

### "OpenAI API error"

- Check your API key in `.env`
- Check you have credit/quota on your OpenAI account
- If using a different provider, set `OPENAI_API_BASE` in `.env`

### Training loss is NaN

- Reduce learning rate: `learning_rate: 1.0e-5`
- Reduce DPO beta: `dpo_beta: 0.05`
- Check for empty preference pairs (the task set may be too easy)

### No improvement after many loops

- Check validation accuracy — if training accuracy is high but validation is low, you're overfitting
- Add more diverse tasks
- Try a different model
- Increase LoRA rank (model may need more capacity)

---

## Next Steps

- Read [How It Works](HOW_IT_WORKS.md) for the full technical explanation
- Read [Expected Results](EXPECTED_RESULTS.md) to calibrate your expectations
- Read [Supported Models](SUPPORTED_MODELS.md) to choose the right model
- Build a custom environment in `liminal/environment.py` for your own backrooms design
