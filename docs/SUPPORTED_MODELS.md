# Supported Models

Any model on HuggingFace that supports LoRA fine-tuning works with Liminal AI Training. This guide covers tested models, recommended configurations, and hardware requirements.

## Quick Reference

| Model | Parameters | Min VRAM (Q4) | Quality | Speed | Recommended For |
|-------|-----------|---------------|---------|-------|-----------------|
| Phi-3 Mini | 3.8B | 3GB | Good | Fast | Testing, low-end hardware |
| Gemma 2 2B | 2.6B | 2GB | Decent | Very fast | Rapid prototyping |
| Mistral 7B v0.3 | 7.2B | 6GB | Good | Fast | Budget training |
| Llama 3.1 8B | 8B | 6GB | Very good | Fast | **Best starting point** |
| Qwen 2.5 7B | 7.6B | 6GB | Very good | Fast | Multilingual tasks |
| Gemma 2 9B | 9.2B | 7GB | Very good | Medium | Strong reasoning |
| Llama 3.1 70B | 70B | 40GB | Excellent | Slow | Best results (needs hardware) |
| Mistral Large | 123B | 72GB | Excellent | Very slow | Research, multi-GPU |

## Detailed Model Profiles

---

### Llama 3.1 8B Instruct (Recommended Starting Point)

```bash
python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct
```

**Why start here:**
- Best quality-to-cost ratio at the 8B scale
- Well-documented, huge community
- Strong baseline across all task categories
- Efficient LoRA training
- Meta's open license allows commercial use

**Config:**
```yaml
model:
  base: "./models/Llama-3.1-8B-Instruct"
  adapter_rank: 32
  adapter_alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Hardware:**
| Quantization | VRAM (inference) | VRAM (training) | Speed |
|-------------|-----------------|-----------------|-------|
| Q4 | ~6GB | ~10GB | ~15 tok/s (3090) |
| Q8 | ~10GB | ~16GB | ~12 tok/s (3090) |
| FP16 | ~16GB | ~24GB | ~18 tok/s (3090) |

**Expected improvement:** 10-20% accuracy gain after 100 loops on a diverse 100-task set.

**Access:** Requires accepting Meta's license on HuggingFace. Go to the model page and click "Access" first.

---

### Mistral 7B v0.3 Instruct

```bash
python scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.3
```

**Strengths:**
- Fast inference
- Good at following instructions
- Strong multilingual capability
- Apache 2.0 license (fully open)

**Config:**
```yaml
model:
  base: "./models/Mistral-7B-Instruct-v0.3"
  adapter_rank: 32
  adapter_alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Hardware:** Similar to Llama 3.1 8B. ~6GB VRAM at Q4.

**Best for:** Fast iteration, multilingual tasks, commercial use without license restrictions.

---

### Phi-3 Mini 3.8B Instruct

```bash
python scripts/download_model.py --model microsoft/Phi-3-mini-4k-instruct
```

**Strengths:**
- Runs on almost anything (3GB VRAM at Q4)
- Surprisingly capable for its size
- Fast training loops (more iterations per hour)

**Limitations:**
- Smaller capacity limits improvement ceiling
- Weaker on complex reasoning tasks
- Shorter context window (4K by default)

**Config:**
```yaml
model:
  base: "./models/Phi-3-mini-4k-instruct"
  adapter_rank: 16          # lower rank for smaller model
  adapter_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Hardware:** ~3GB VRAM at Q4, ~5GB for training. Can train on a laptop GPU.

**Best for:** Testing the pipeline, learning how it works, hardware-constrained environments.

---

### Qwen 2.5 7B Instruct

```bash
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
```

**Strengths:**
- Excellent multilingual support (especially Chinese, but also European and Asian languages)
- Strong coding capability
- Good mathematical reasoning

**Config:**
```yaml
model:
  base: "./models/Qwen2.5-7B-Instruct"
  adapter_rank: 32
  adapter_alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Best for:** Multilingual task sets, coding-focused training, math reasoning.

---

### Gemma 2 9B Instruct

```bash
python scripts/download_model.py --model google/gemma-2-9b-it
```

**Strengths:**
- Strong reasoning capability for its size
- Good at structured outputs
- Google's architecture innovations (sliding window attention)

**Config:**
```yaml
model:
  base: "./models/gemma-2-9b-it"
  adapter_rank: 32
  adapter_alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Hardware:** ~7GB VRAM at Q4.

**Best for:** Tasks requiring strong reasoning, structured output generation.

---

### Llama 3.1 70B Instruct (Best Results)

```bash
python scripts/download_model.py --model meta-llama/Llama-3.1-70B-Instruct
```

**Strengths:**
- Best overall quality
- Near-frontier reasoning capability
- Most headroom for improvement via LoRA
- Strong across all categories

**Limitations:**
- Needs serious hardware (40GB+ VRAM at Q4)
- Slow inference on consumer GPUs
- Slow training loops

**Config:**
```yaml
model:
  base: "./models/Llama-3.1-70B-Instruct"
  adapter_rank: 64          # higher rank for larger model
  adapter_alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Hardware:**
| Quantization | VRAM (inference) | VRAM (training) | Setup |
|-------------|-----------------|-----------------|-------|
| Q4 | ~40GB | ~60GB | 2x RTX 3090 or 1x A100 |
| Q8 | ~72GB | ~96GB | 2x A100 or 1x H100 |

**Best for:** Maximum quality, research, when you have the hardware.

---

### Gemma 2 2B (Lightweight)

```bash
python scripts/download_model.py --model google/gemma-2-2b-it
```

**Strengths:**
- Tiny — runs on 2GB VRAM
- Very fast training loops
- Good for rapid prototyping

**Limitations:**
- Limited capacity
- Weak on complex tasks
- Lower improvement ceiling

**Config:**
```yaml
model:
  base: "./models/gemma-2-2b-it"
  adapter_rank: 8            # small rank for small model
  adapter_alpha: 16
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Best for:** Proof of concept, rapid testing, extremely limited hardware.

---

## Running on Raspberry Pi

Yes, you can run inference on a Raspberry Pi. No, you cannot train on one.

**Setup:**
- Raspberry Pi 5 (8GB recommended)
- Use Q4 quantized models via llama.cpp or similar
- Expect ~0.5-2 tokens per second

**Workflow:**
1. Train on a GPU machine (desktop, cloud, Colab)
2. Export the adapter
3. Load the base model + adapter on the Pi for inference
4. The Pi runs the trained model; a separate machine handles training

```bash
# On GPU machine: train and export
python train.py --model ./models/phi-3-mini --tasks ./tasks/my_tasks.jsonl --loops 100
python scripts/export_adapter.py --base-model ./models/phi-3-mini --adapter ./adapters/default --output ./export/pi-adapter

# Transfer export/pi-adapter to the Pi
# Load with your preferred Pi inference runtime
```

---

## Judge / NPC Model Options

The judge and NPC models run via API. Any OpenAI-compatible API works.

| Model | Cost | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| GPT-4o-mini | Very low | Good | Fast | **Recommended default** |
| GPT-4o | Medium | Very good | Medium | Better adversarial NPCs |
| GPT-4.5 | High | Excellent | Medium | Best judgement quality |
| Claude Sonnet | Medium | Very good | Medium | Different perspective from GPT |
| Claude Haiku | Low | Good | Fast | Budget alternative |
| Local model | Free | Varies | Varies | Via Ollama + OpenAI-compatible API |

### Using a local model as judge

If you want to avoid API costs entirely, you can run the judge/NPC on a local model via Ollama:

```bash
# Install Ollama and pull a model
ollama pull llama3.1:8b

# Set the API base in .env
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama  # any string works

# Set the model name in config.yaml
judge:
  model: "llama3.1:8b"
npc:
  model: "llama3.1:8b"
```

Note: Using the same model family for both training and judging is not recommended (shared blind spots), but it's free and works for experimentation.

---

## Choosing the Right Model

```
                    ┌─ Do you have 40GB+ VRAM?
                    │
               Yes ─┤─→ Llama 3.1 70B (best results)
                    │
               No  ─┤─ Do you have 6-8GB VRAM?
                    │
               Yes ─┤─→ Llama 3.1 8B (recommended)
                    │   or Mistral 7B (if you need Apache license)
                    │   or Qwen 2.5 7B (if multilingual)
                    │
               No  ─┤─ Do you have 3-4GB VRAM?
                    │
               Yes ─┤─→ Phi-3 Mini 3.8B
                    │   or Gemma 2 2B (fastest iteration)
                    │
               No  ─┤─→ Use Google Colab (free T4 GPU = 16GB VRAM)
                        or rent cloud GPU (Lambda, RunPod, vast.ai)
```

## Model-Specific LoRA Recommendations

| Model Size | Rank | Alpha | Target Modules | Trainable Params |
|-----------|------|-------|----------------|------------------|
| 2-4B | 8-16 | 16-32 | q_proj, v_proj | ~1-2M |
| 7-9B | 16-32 | 32-64 | q_proj, v_proj, k_proj, o_proj | ~4-8M |
| 13-30B | 32-64 | 64-128 | q_proj, v_proj, k_proj, o_proj | ~8-16M |
| 70B+ | 64-128 | 128-256 | q_proj, v_proj, k_proj, o_proj | ~16-32M |

General rule: larger models benefit from larger adapters, but start conservative and increase if the model plateaus.
