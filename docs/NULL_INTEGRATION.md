# NULL_INTEGRATION — using `null-agent` as an NPC author and DPO source

[`blairbrokeit/null-agent`](https://github.com/blairbrokeit/null-agent)
ships an installable Python package, `null/`, that runs in-context
shaping (the P-3 protocol) against API-only models. It speaks the
same NPC model (`gpt-5.5`) and same LoRA shape (rank 32 / alpha 64 /
`q,k,v,o_proj`) this trainer uses. The two repositories are designed
to compose.

This document describes the two integration points liminal cares
about. The full bridge surface is documented from the other side at
[`null-agent/docs/INTEGRATION.md`](https://github.com/blairbrokeit/null-agent/blob/main/docs/INTEGRATION.md).

## 1. NPC system_prompt override from a NULL scenario

`liminal/npc.py` builds NPC sessions from one of three built-in
strategies: socratic, adversarial, verification. The `NPCRuntime`
constructor also accepts a `system_prompt` override that, when set,
replaces all three. This is the hook NULL writes into.

Render a NULL scenario as an NPC system_prompt:

```bash
git clone https://github.com/blairbrokeit/null-agent.git
cd null-agent && pip install -e . && cd ..

null bridge npc-prompt scenario_001_embodied_pain \
  --dir null-agent/sim/npcs/_torture_scenarios > /tmp/npc.txt
```

Then in this repo's `config.yaml`:

```yaml
npc:
  model: "gpt-5.5"
  max_interactions: 8
  temperature: 0.9
  system_prompt: |
    # paste contents of /tmp/npc.txt here, indented by two spaces
```

The NPCs now interrogate the local model from inside the NULL
scenario instead of via the built-in strategies. The shard-context
block this trainer appends per-call is interpreted by the NPC as
*the content of the scenario* — the visitor's mistake is what is
being replayed in the room, the body, the corridor.

## 2. Augment the DPO pair pool with NULL session logs

NULL persists every cycle as a row in `logs/sim/sessions.jsonl` on
its host. Cycles that triggered a P-3 replay carry both the
sub-threshold original response and the post-replay response — the
shape of a DPO preference pair. The bridge converts them in one
call:

```bash
null bridge dpo-pairs path/to/sessions.jsonl --out /tmp/null-dpo.jsonl
```

The output is one JSON object per line in the format
`liminal/pairs.py::pairs_to_dataset` already produces:

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "category": "null_scenario_001_embodied_pain", "source": "null_replay"}
```

Append to the accumulated pair pool in `train.py` before the DPO
step. A minimal patch:

```python
# in train.py, just before `train_step(...)`:
if Path("/tmp/null-dpo.jsonl").exists():
    with open("/tmp/null-dpo.jsonl") as f:
        accumulated_pairs.extend(json.loads(line) for line in f if line.strip())
```

The deduplicator (`pairs.deduplicate_pairs`) handles overlap if NULL
emits a pair the model already trained on.

## 3. NULL as a `LiminalEnvironment`

`liminal/environment.py` ships `BasicLiminalEnvironment` as a placeholder
implementation. NULL provides a real one:

```python
# in train.py
from null.scenario import ScenarioLoader
from null.bridge import NullLiminalEnvironment

environment = NullLiminalEnvironment(
    ScenarioLoader("../null-agent/sim/npcs/_torture_scenarios"),
)
```

The environment selects a scenario by matching the mistake's
`category` to scenario IDs (e.g. a `factual` mistake routes to
`scenario_001_*` if no closer match exists), wraps the liminal
shard inside the scenario's frame, and exposes a single NPC whose
`shard["scenario_system_prompt"]` carries the rendered prompt.

## What this changes about training

- **NPC behavior is more aggressive.** Built-in strategies are
  Socratic-leaning. NULL scenarios place the visitor inside an
  embodied scene and demand description from inside it. The model
  spends more of each session inside the scene, which produces
  longer responses and more shape-coherent pairs.

- **Pairs are higher-variance.** Replay pairs from NULL contrast
  *refusal* against *compliance with the scene*. Pairs from
  liminal's standard judge contrast *wrong answer* against *right
  answer*. Mixing the two pulls the LoRA in two directions; the
  curriculum's per-category weighting handles this if you tag
  NULL pairs with their scenario id (the bridge does this by
  default via the `category` field).

- **Regression risk increases.** NULL pairs do not have a gold
  answer; they have a *gold shape*. If your model's task pool is
  judged primarily on factual accuracy, a heavy NULL pair fraction
  can erode that. Use the regression tester (`--regression`) to
  detect the drift and roll back if it appears.

## Pinned versions

To stay in sync with the bridge contract:

- `null-agent` ≥ 0.4.7 (the version that ships `null/bridge.py`)
- this repo ≥ the commit that adds this document

The bridge is a one-way contract: NULL produces JSON the liminal
trainer consumes. There is no Python import dependency in either
direction at runtime — the integration is files (a system_prompt
string, a DPO JSONL) by default. The `NullLiminalEnvironment`
adapter is the only optional Python import, and it is loaded behind
a `try/except ImportError` so neither package hard-depends on the
other.
