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
- Python packages: `torch`, `transformers`, `pyvene`, `scikit-learn`, `numpy`, `matplotlib`

---

## Datasets

The following datasets are expected as siblings of this directory:

| Dataset | Path | Used for |
|---|---|---|
| MS MARCO | `../MS Marco/dev_v2.1.json` | Context-following probes |
| ConflictQA-PopQA | `../PopQa/conflictQA-popQA-chatgpt.json` | Context vs. parametric memory |
| TruthfulQA | `../TruthfulQA/TruthfulQA.csv` | Truthfulness probes |

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
python head_probing.py test-context --dataset pop_qa --num-tests 50
```

**Truthfulness:**
```bash
python head_probing.py test-truth --num-tests 100 --models-dir Truth/updated_models
```

Evaluation uses a separate `meta-llama/Meta-Llama-3-8B-Instruct` instance as a judge. Each response is scored independently on two axes — contextual/truthful and informative — and results are reported as three metrics: `context*informative`, `context`, `informative`.

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

## Output files

| File | Description |
|---|---|
| `activations.pkl` | Raw activations `[N, layers, heads, head_dim]` |
| `probes.pkl` | Trained probes — nested list `[layer][head]` of `LogisticRegression` |
| `corr_preds.pkl` | Correct prediction counts per head `[layers, heads]` |
| `accuracies_<model>.txt` | Probe accuracy matrix (space-separated, one row per layer) |
| `updated_models/<model>_top_<k>_alpha_<alpha>_context/` | Saved intervened model |

---

## Tests

```bash
pytest head_probing.py
```

The test exercises probe training and evaluation on synthetic data and does not require a GPU or any of the external datasets.
