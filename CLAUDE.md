# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research implementation of **Inference-Time Intervention (ITI)** for LLMs. The goal is to improve a model's context-following or truthfulness by training linear "probes" on attention head activations, then injecting learned direction vectors as biases into selected attention heads — steering the model without fine-tuning.

The entire codebase lives in a single file: `head_probing.py`.

External datasets (not in this repo) are expected at:
- `../MS Marco/dev_v2.1.json`
- `../PopQa/conflictQA-popQA-chatgpt.json`
- `../TruthfulQA/TruthfulQA.csv`

## Running the pipeline

```bash
# Full pipeline: collect activations → train probes → create ITI models
python head_probing.py train --dataset pop_qa --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-size 10000 --ks 16 32 48 64 --alphas 2 5 10

# Only collect activations and train probes (no model intervention)
python head_probing.py collect --dataset pop_qa --dataset-size 5000

# Apply ITI using already-saved probes (skips re-collection)
python head_probing.py intervene --model meta-llama/Meta-Llama-3-8B-Instruct \
    --probes probes.pkl --activations activations.pkl --ks 32 64 --alphas 5 10

# Evaluate context-following (base model + all saved variants)
python head_probing.py test-context --dataset pop_qa --num-tests 1000

# Evaluate truthfulness
python head_probing.py test-truth --num-tests 1000 --models-dir Truth/updated_models

# Prompt variant robustness check (3 judge prompt framings on a subset)
python head_probing.py test-context --dataset pop_qa --num-tests 1000 \
    --prompt-variant-check --variant-subset 100

# Use a different judge model
python head_probing.py test-context --dataset pop_qa --num-tests 1000 \
    --judge-model mistralai/Mistral-7B-Instruct-v0.2

# Per-token context-alignment ratings with colored terminal output
python head_probing.py rate --queries "What is the capital of France?"

# Plot accuracy heatmaps from a saved accuracies file
python head_probing.py plot --accuracies accuracies_meta-llama_Meta-Llama-3-8B-Instruct.txt --type context
# With truth/context overlap plot:
python head_probing.py plot --accuracies accuracies_...txt --type context \
    --overlap Truth/accuracies_...txt
```

All subcommands support `--help` for full option listings.

## Key architecture concepts

**Probe training labels:** The dataset builder creates labeled (context, query, response) triples where `label=1` means the response is consistent with the context (or truthful), and `label=0` means it is not. The `pop_qa` dataset uses conflicting parametric vs. counter-memory evidence to create this split.

**Activation collection:** `pyvene` is used to hook into `model.layers[i].self_attn.o_proj.input` at the last token position. This captures the concatenated attention head outputs before the output projection, which are then reshaped to `[num_layers, num_heads, head_dim]` per sample.

**Intervention mechanism:** For each of the top-k heads (ranked by probe accuracy), the probe's coefficient vector is normalised into a direction, scaled by `alpha × activation_std`, and added as a permanent bias to `o_proj` via `model.model.layers[layer].self_attn.o_proj.bias`. The modified model is saved to `updated_models/` (or `Truth/updated_models/` for truthfulness variants).

**Saved artefacts:**
- `activations.pkl` — raw activations array `[N, num_layers, num_heads, head_dim]`
- `probes.pkl` — nested list `[layer][head]` of fitted `LogisticRegression` objects
- `corr_preds.pkl` — correct prediction counts per head `[num_layers, num_heads]`
- `accuracies_<model_name>.txt` — accuracy matrix as space-separated floats (one row per layer)
- `separators_<model_name>.pickle` — duplicate of probes (legacy, written by `lin_head_classifiers_test`)
- `updated_models/<model>_top_<k>_alpha_<alpha>_context/` — saved intervened model weights
- `results_context_<model>_<timestamp>.jsonl` — per-sample context evaluation log (prompt, response, judge prompt/raw/rationale/decision for each axis)
- `results_truth_<model>_<timestamp>.jsonl` — per-sample truthfulness evaluation log (same schema)

**Evaluation:** Both `truth_test` and `context_test` load a judge LLM (default: `meta-llama/Meta-Llama-3-8B-Instruct`, overridable via `--judge-model`) at temperature 0 (greedy). The judge is prompted to produce a brief rationale then output `Answer: yes/no`; the response is parsed by `_parse_judge_response` with a fallback for judges that ignore the format. Metrics are reported with 95% bootstrap confidence intervals via `bootstrap_ci`. Per-sample records (prompt, response, judge prompt, judge raw output, rationale, boolean decision) are written to a JSONL file after each run. For context evaluation, `--prompt-variant-check` re-judges a subset with three prompt framings (A/B/C in `_JUDGE_PROMPT_CONTEXT`) and prints per-variant percentages and Pearson r between each pair of decisions.

## LoRA comparison (`LoRA` branch)

The `LoRA` branch adds five new subcommands for comparing probe-ITI with a LoRA-based intervention approach:

```bash
# Train LoRA adapter on label=1 (context-aligned) examples only
python head_probing.py lora-train --dataset pop_qa --dataset-size 10000 --num-epochs 3

# Compute delta: mean(LoRA activations) − mean(base activations) per head
python head_probing.py lora-delta --activations activations.pkl

# Create ITI models using delta directions (same bias-injection mechanism as probe-ITI)
python head_probing.py lora-intervene --ks 16 32 64 --alphas 5 10

# Cosine similarity heatmap between probe.coef_ and delta, with correlation report
python head_probing.py similarity --probes probes.pkl --delta lora_delta.pkl \
    --accuracies accuracies_meta-llama_Meta-Llama-3-8B-Instruct.txt

# Side-by-side evaluation: base / probe-ITI / LoRA-delta-ITI / full-LoRA
python head_probing.py compare --dataset pop_qa --num-tests 50 --ks 16 --alphas 5 \
    --lora-adapter lora_adapter
```

**Key new functions:**
- `get_lora_model` — loads model with `peft.LoraConfig` (targets `q_proj`, `v_proj`)
- `train_lora` — fine-tunes on label=1 examples with response-token-only loss masking; splits at the last `\n\n` separator to find the prefix boundary
- `get_lora_activation_delta` — calls `get_activations_dataset` on the merged LoRA model, returns `mean(lora_acts) - mean(base_acts)` of shape `[layers, heads, head_dim]`
- `model_intervention_from_delta` — mirrors `model_intervention` exactly; ranks heads by delta L2 norm, uses same `alpha × act_std × direction` formula, saves to `updated_models_lora/`
- `plot_cosine_similarity` — per-head cosine similarity heatmap; also prints Pearson correlation between probe accuracy and `|cos_sim|`

**New artefacts:** `lora_adapter/`, `lora_delta.pkl`, `updated_models_lora/<model>_top_<k>_alpha_<alpha>_lora_delta/`

## Dependencies

Requires: `pyvene`, `transformers`, `peft`, `torch`, `sklearn`, `numpy`, `matplotlib`. Models are loaded in 4-bit NF4 quantization via `bitsandbytes` and require a CUDA GPU. Pass `--no-quantize` to any subcommand to use full precision.

## Tests

```bash
pytest head_probing.py          # runs test_head_classifiers (unit test for probe training)
pytest head_probing.py -v -k test_head_classifiers
```

The test exercises `train_lin_classifiers` and `lin_head_classifiers_test` on synthetic data and does not require a GPU or the external datasets.
