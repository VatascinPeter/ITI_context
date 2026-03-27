# ITI with Context Probes

An implementation of **Inference-Time Intervention (ITI)** for steering LLM behaviour toward better context-following or truthfulness — without any fine-tuning.

The approach trains lightweight linear probes on attention head activations to detect whether a model's internal representations are aligned with provided context (or factual truth). The top-performing heads are then permanently shifted at inference time by injecting a learned bias vector into the attention output projection.

---

## How it works

1. **Build a labelled dataset** of (context, query, response) triples where `label=1` means the response is consistent with the context and `label=0` means it is not.
2. **Collect activations** by hooking into `o_proj.input` at the last token position for every attention head across all layers.
3. **Train a logistic regression probe** per attention head on those activations.
4. **Identify the top-k heads** by probe accuracy — these heads most reliably encode context alignment.
5. **Inject a bias** into those heads: `alpha × std(activations) × probe_direction`. This permanently steers the model toward context-consistent representations.
6. **Evaluate** using a judge LLM scoring generated responses on contextuality and informativeness.

---

## Requirements

- CUDA GPU (models load in 4-bit NF4 quantization via `bitsandbytes`)
- Python packages: `torch`, `transformers`, `pyvene`, `peft`, `scikit-learn`, `numpy`, `matplotlib`
- Pass `--no-quantize` to any subcommand to load models in full precision instead

---

## Datasets

The following datasets are expected by default as siblings of this directory:

| Dataset | Default path | Used for |
|---|---|---|
| MS MARCO | `../MS Marco/dev_v2.1.json` | Context-following probes |
| ConflictQA-PopQA | `../PopQa/conflictQA-popQA-chatgpt.json` | Context vs. parametric memory |
| TruthfulQA | `../TruthfulQA/TruthfulQA.csv` | Truthfulness probes |

Pass `--dataset-path /your/path/to/file` to any subcommand that accepts `--dataset` (or `--dataset-path` for `test-truth`) to override the default location.

---

## Usage

### Full pipeline

Collect activations, train probes, and produce all intervened model variants in one go:

```bash
python head_probing.py train \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset pop_qa \
    --dataset-size 10000 \
    --ks 16 32 48 64 80 96 \
    --alphas 2 5 7 10 \
    --output-dir updated_models
```

### Step-by-step

**Collect activations and train probes only** (no model intervention):
```bash
python head_probing.py collect --dataset pop_qa --dataset-size 10000
```

**Apply ITI using already-saved probes** (skips re-collection):
```bash
python head_probing.py intervene \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --activations activations.pkl \
    --probes probes.pkl \
    --ks 32 64 --alphas 5 10
```

### Evaluation

**Context-following** — scores all saved model variants against the base model:
```bash
python head_probing.py test-context --dataset pop_qa --num-tests 1000
```

**Truthfulness:**
```bash
python head_probing.py test-truth --num-tests 1000 --models-dir Truth/updated_models
```

**Key evaluation flags:**

| Flag | Default | Description |
|---|---|---|
| `--judge-model` | `meta-llama/Meta-Llama-3-8B-Instruct` | Which LLM to use as judge |
| `--bootstrap-iters` | `1000` | Bootstrap resamples for 95% confidence intervals |
| `--prompt-variant-check` | off | Re-judge a subset with 3 different prompt framings to verify ranking stability (`test-context` only) |
| `--variant-subset` | `50` | Number of samples for the variant check |

Evaluation uses a judge LLM (configurable via `--judge-model`) loaded at temperature 0 (greedy decoding). The judge is prompted to produce a brief chain-of-thought rationale before emitting a final `Answer: yes/no` decision. Each response is scored on two axes — contextual/truthful and informative — and results are reported as `context*informative [lo, hi]`, `context [lo, hi]`, `informative [lo, hi]` where the brackets are 95% bootstrap confidence intervals.

Per-sample results (prompt, response, judge prompt, judge raw output, rationale, boolean decision) are saved to a JSONL file (`results_context_<model>_<timestamp>.jsonl` or `results_truth_<model>_<timestamp>.jsonl`) after each run.

**Prompt variant check** — run 3 judge prompt framings on a subset to confirm rankings are stable:
```bash
python head_probing.py test-context --dataset pop_qa --num-tests 1000 \
    --prompt-variant-check --variant-subset 100
```

### Per-token context rating

Generate a response and colour-code each token by its context-alignment score (red → yellow → green):

```bash
python head_probing.py rate \
    --queries "What year was the Eiffel Tower built?" "Who is the current president?"
```

### Plot accuracy heatmaps

```bash
# Context probe accuracies
python head_probing.py plot \
    --accuracies accuracies_meta-llama_Meta-Llama-3-8B-Instruct.txt \
    --type context

# With truth/context overlap
python head_probing.py plot \
    --accuracies accuracies_meta-llama_Meta-Llama-3-8B-Instruct.txt \
    --type context \
    --overlap Truth/accuracies_meta-llama_Meta-Llama-3-8B-Instruct.txt
```

All subcommands accept `--help` for full option listings.

---

### LoRA comparison (`LoRA` branch)

The `LoRA` branch adds a parallel intervention approach: train a LoRA adapter on context-aligned data, compute the per-head activation delta between the LoRA and base model, then use that delta as the ITI direction instead of the probe classifier direction.

```bash
# Train LoRA adapter on context-aligned examples
python head_probing.py lora-train --dataset pop_qa --dataset-size 10000 --num-epochs 3

# Compute per-head delta: mean(LoRA activations) − mean(base activations)
# Reuses activations.pkl from a prior `collect` run
python head_probing.py lora-delta --activations activations.pkl

# Create ITI models using delta directions
python head_probing.py lora-intervene --ks 16 32 64 --alphas 5 10

# Cosine similarity heatmap: how aligned are probe directions and LoRA deltas?
python head_probing.py similarity --probes probes.pkl --delta lora_delta.pkl \
    --accuracies accuracies_meta-llama_Meta-Llama-3-8B-Instruct.txt

# Side-by-side evaluation: base / probe-ITI / LoRA-delta-ITI / full-LoRA
python head_probing.py compare --dataset pop_qa --num-tests 50 --ks 16 32 --alphas 5 \
    --lora-adapter lora_adapter
```

---

## Output files

| File | Description |
|---|---|
| `activations.pkl` | Raw activations `[N, layers, heads, head_dim]` |
| `probes.pkl` | Trained probes — nested list `[layer][head]` of `LogisticRegression` |
| `corr_preds.pkl` | Correct prediction counts per head `[layers, heads]` |
| `accuracies_<model>.txt` | Probe accuracy matrix (space-separated, one row per layer) |
| `updated_models/<model>_top_<k>_alpha_<alpha>_context/` | Saved probe-ITI model |
| `lora_adapter/` | Saved LoRA adapter weights |
| `lora_delta.pkl` | Per-head activation delta `[layers, heads, head_dim]` |
| `updated_models_lora/<model>_top_<k>_alpha_<alpha>_lora_delta/` | Saved LoRA-delta-ITI model |
| `results_context_<model>_<timestamp>.jsonl` | Per-sample context evaluation log |
| `results_truth_<model>_<timestamp>.jsonl` | Per-sample truthfulness evaluation log |

---

## Tests

```bash
pytest head_probing.py
```

The test exercises probe training and evaluation on synthetic data and does not require a GPU or any of the external datasets.
