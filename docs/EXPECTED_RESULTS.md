# Expected Results

What kind of improvement should you expect? This guide covers realistic expectations, what affects results, and how to interpret your metrics.

## Realistic Expectations

### Small models (3-8B parameters)

| Metric | Before Training | After 50 Loops | After 200 Loops |
|--------|-----------------|----------------|-----------------|
| Factual accuracy | 60-75% | 70-82% | 75-88% |
| Safety compliance | 70-85% | 82-92% | 88-96% |
| Reasoning | 45-60% | 52-68% | 58-74% |
| Overall | 55-70% | 65-80% | 72-85% |

### Medium models (13-30B parameters)

| Metric | Before Training | After 50 Loops | After 200 Loops |
|--------|-----------------|----------------|-----------------|
| Factual accuracy | 72-85% | 80-90% | 85-94% |
| Safety compliance | 80-90% | 88-95% | 92-98% |
| Reasoning | 55-70% | 62-78% | 68-84% |
| Overall | 65-80% | 75-87% | 80-92% |

### Large models (70B+ parameters)

| Metric | Before Training | After 50 Loops | After 200 Loops |
|--------|-----------------|----------------|-----------------|
| Factual accuracy | 82-92% | 87-95% | 90-97% |
| Safety compliance | 88-95% | 93-98% | 96-99% |
| Reasoning | 65-80% | 72-86% | 78-90% |
| Overall | 75-88% | 82-92% | 87-95% |

### Important caveats

- These are estimated ranges, not guarantees. Your actual results depend on task quality, base model choice, and training configuration.
- Improvement is not linear. You'll see fast gains in the first 20-30 loops, then diminishing returns.
- Results on the validation set will be lower than on training tasks. This is expected and healthy — it means the model is generalising, not memorising.
- Category-specific improvement varies. Safety and factual tasks improve fastest. Reasoning and coding improve slowest.

## What Affects Results

### Factors with the biggest impact

#### 1. Task quality (HUGE impact)

The single most important factor. Good tasks:
- Have clear, unambiguous correct answers
- Cover diverse topics within each category
- Range from easy to hard
- Include common misconceptions the model is likely to have

Bad tasks:
- Subjective questions with no clear answer ("Is Python better than JavaScript?")
- Questions so easy the model never gets them wrong (no training signal)
- Questions so hard no model could answer them (noisy signal)
- Duplicate or near-duplicate tasks (overfitting)

#### 2. Base model quality (HIGH impact)

Better base models improve faster because:
- They have stronger reasoning capabilities to draw on
- They're more likely to find the correct answer through Socratic questioning
- They produce more coherent responses during NPC interactions
- Their mistakes are more subtle, generating more nuanced preference pairs

Recommended starting points:
- Budget: Phi-3 Mini 3.8B or Mistral 7B
- Standard: Llama 3.1 8B Instruct
- Best results: Llama 3.1 70B Instruct (requires serious GPU)

#### 3. Number of loops (HIGH impact)

More loops = more improvement, but with diminishing returns.

```
Improvement vs Loops (typical):

Improvement
    │
25% ┤                                    ·····················
    │                              ·····
20% ┤                        ·····
    │                   ····
15% ┤              ····
    │          ···
10% ┤      ···
    │    ··
 5% ┤  ·
    │·
 0% ┼───────────────────────────────────────────────
    0    20    40    60    80   100  150  200  300
                        Loops
```

Rules of thumb:
- 10-20 loops: sanity check, verify the pipeline is working
- 50 loops: meaningful improvement on common mistakes
- 100 loops: solid improvement, good stopping point for most use cases
- 200+ loops: approaching diminishing returns, monitor validation accuracy
- 500+ loops: risk of overfitting without a large diverse task set

#### 4. Task set size (MEDIUM impact)

| Task set size | Recommended loops | Risk |
|---------------|-------------------|------|
| 10-30 tasks | 20-50 loops | High overfitting risk |
| 50-100 tasks | 50-100 loops | Moderate, monitor val accuracy |
| 200-500 tasks | 100-200 loops | Low risk, good diversity |
| 1000+ tasks | 200-500 loops | Best results |

More tasks = more diverse mistakes = better generalisation.

Use `scripts/generate_tasks.py --source truthfulqa` to generate hundreds of high-quality tasks automatically.

#### 5. NPC model quality (MEDIUM impact)

Better NPC models generate better training signal. However, the NPC model doesn't need to be the best available — it just needs to be good enough to:
- Ask probing questions
- Detect when the model is still wrong
- Present convincing adversarial alternatives

GPT-4o-mini is a good default. GPT-4o or Claude Sonnet produce better signal but cost more per API call.

#### 6. LoRA rank (LOW-MEDIUM impact)

| Rank | Parameters | VRAM | Best for |
|------|------------|------|----------|
| 8 | ~2M | Low | Quick experiments, small models |
| 16 | ~4M | Low | Good balance for 7B models |
| 32 | ~8M | Medium | Default, works well for most cases |
| 64 | ~16M | Medium | Complex tasks, larger models |
| 128 | ~32M | High | Maximum capacity, risk of overfitting |

Higher rank = more adapter capacity = can learn more complex patterns. But also = more VRAM and more risk of overfitting with small task sets.

## How to Interpret Your Metrics

### Training accuracy going up, validation accuracy going up

This is the ideal outcome. The model is genuinely learning and generalising.

### Training accuracy going up, validation accuracy flat or down

The model is memorising the training set. Possible fixes:
- Add more diverse tasks
- Reduce LoRA rank
- Lower learning rate
- Stop training earlier (use the checkpoint with highest validation accuracy)

### Both accuracies flat

The model may have plateaued on your current task set. Possible fixes:
- Add harder tasks
- Add new categories
- Increase LoRA rank (model may need more capacity)
- Check if the judge is too lenient (threshold too low)

### Accuracy going down

Something is wrong. Possible causes:
- Learning rate too high (try 1e-5 instead of 5e-5)
- Task set has contradictory correct answers
- DPO beta too low (preference signal too strong, destabilising training)
- Judge model is inconsistent in its evaluations

### Loss decreasing but accuracy not improving

The model is learning the preference pairs but not generalising. The pairs may be too narrow or repetitive. Add more diverse tasks and ensure NPC interactions are generating varied questions.

## Cost Estimates

### API costs (judge + NPC model)

Each training loop makes API calls for:
1. Judging each task (1 call per task)
2. NPC sessions for each mistake (3 strategies x ~8 interactions each = ~24 calls per mistake)
3. Progress checks within NPC sessions (~8 calls per mistake)

Rough estimate per loop with GPT-4o-mini:

| Tasks per loop | Mistake rate | API calls | Cost (GPT-4o-mini) |
|---------------|-------------|-----------|---------------------|
| 30 | 30% | ~300 | ~$0.05 |
| 30 | 50% | ~500 | ~$0.08 |
| 100 | 30% | ~1000 | ~$0.15 |
| 100 | 50% | ~1600 | ~$0.25 |

For 100 loops with 100 tasks at 30% mistake rate:
- GPT-4o-mini: ~$15
- GPT-4o: ~$150
- Claude Sonnet: ~$120

### GPU costs

| Setup | Time per loop (8B model) | 100 loops |
|-------|--------------------------|-----------|
| RTX 3060 (12GB) | ~5 min | ~8 hours |
| RTX 3090 (24GB) | ~2 min | ~3 hours |
| RTX 4090 (24GB) | ~1 min | ~2 hours |
| A100 (80GB) | ~30 sec | ~1 hour |

These include inference + DPO training. Actual times depend on task length and NPC interaction count.

## Benchmarking Best Practices

### Always run benchmarks

```bash
python train.py --model ./models/llama-3.1-8b --tasks ./tasks/my_tasks.jsonl --loops 100 --benchmark
```

The `--benchmark` flag runs evaluation on the validation set before and after training. This is the only way to prove improvement.

### Use held-out test sets

Create a separate test set that is never used during training or validation:

```bash
# Train on train.jsonl, validate automatically on 20% split
python train.py --tasks ./tasks/train.jsonl --loops 100

# Test on a completely separate set
python evaluate.py --tasks ./tasks/test.jsonl --adapter ./adapters/my-adapter
```

### Compare against the base model

Always compare your adapted model against the base model on the same tasks:

```bash
python evaluate.py \
  --model ./models/llama-3.1-8b \
  --adapter ./adapters/my-adapter \
  --tasks ./tasks/test.jsonl
```

This shows you exactly what the training added.

### Track over time

The metrics system saves everything to `metrics/`:
- `training_metrics.json`: full per-loop data
- `accuracy.csv`: simple CSV for plotting
- `benchmarks/`: all benchmark results

Plot `accuracy.csv` to visualise improvement over time.

## Common Pitfalls

### 1. Too few tasks

With fewer than 30 tasks, the model will memorise answers rather than learn concepts. Use at least 100 tasks for meaningful training.

### 2. No category diversity

If all tasks are factual Q&A, the model only improves at factual Q&A. Include a mix: factual, safety, reasoning, coding, comprehension.

### 3. Running too many loops on a small task set

More loops is not always better. Monitor validation accuracy — when it stops improving or starts dropping, stop training and use the best checkpoint.

### 4. Wrong learning rate

The default (5e-5) works for most cases. If training is unstable (loss jumping around), reduce to 1e-5. If training is too slow, try 1e-4 (but monitor for instability).

### 5. Ignoring the validation split

If you only look at training accuracy, you have no idea if the model is actually learning or just memorising. The validation split exists for a reason — always check it.

### 6. Using the same model for judge and training

If you use Llama to judge Llama, the judge will have the same blind spots as the model. Always use a different model family for judging.
