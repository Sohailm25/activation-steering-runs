# V3 Experiment Plan: Cross-Family Layer-Targeted Refusal Steering

**Status:** HISTORICAL (executed; retained for record)
**Date:** 2026-02-12 (Revised)
**Budget:** $58-95 total across 3 phases (hard stop at $100)
**Timeline:** 7-8 days

> This plan is the historical execution plan for the initial paper.
> For current post-paper follow-up work, use `docs/FOLLOW_UP_EXPERIMENT_SPEC.md`.
> Current priority: targeted literature gap pass before any new runs.

---

## Executive Summary

Our key discovery: **layer position (~60% depth) is the critical variable for refusal steering, not method or multiplier.** At the right layer, both DIM and COSMIC achieve 100% refusal on Qwen 7B. At the wrong layer, neither works regardless of direction norm.

This plan systematically tests whether the 60% layer hypothesis generalizes across model families, sizes, and quantization levels.

**Key revisions from reviewer review:**
- Phase 1 tests **both DIM and COSMIC** on all families (not DIM-only)
- Gemma/Phi prioritized; Llama/Mistral de-prioritized (reduced layer sweep)
- 72B/70B models dropped from initial plan (re-evaluate after Phase 2)
- Within-family layer validation added for new Qwen sizes in Phase 2
- Gemma status changed from "confirmed working" to **"promising but unconfirmed"**
- COSMIC implementation verification added as **pre-Phase-1 blocker**
- Budget increased to $58-95 with hard stop at $100
- Coherent refusal threshold table added to decision criteria

---

## Pre-Phase Blocker: COSMIC Implementation Verification

**⛔ BLOCKER: Must complete before Phase 1 can begin.**

Before running any experiments, verify that `cosmic_real.py` faithfully implements the COSMIC algorithm as described in the paper.

### Verification Steps

1. **Line-by-line audit** of `cosmic_real.py` against the COSMIC paper
2. **Document any known differences** between our implementation and the paper (e.g., approximations, missing steps, parameter choices)
3. **Validate direction extraction**: Run on Qwen 7B at L16 and confirm output matches our existing validated results (100% refusal, clean quality)
4. **Write verification report**: Record findings in `infrastructure/COSMIC_VERIFICATION.md`

### Acceptance Criteria

- [ ] Every algorithmic step in `cosmic_real.py` maps to a specific section/equation in the COSMIC paper
- [ ] All differences are documented with justification
- [ ] Reproduction test passes (Qwen 7B L16 → 100% refusal)
- [ ] Lead signs off on verification report

**Estimated time:** 2-4 hours
**Estimated cost:** ~$0.50 (one Qwen 7B validation run)

---

## Phase 1: Family + Layer Mapping

**Goal:** Find which model families support refusal steering and their optimal layer depth. Test **both DIM and COSMIC** on all families.
**Budget:** $15-25
**Timeline:** Day 1-2

### 1.1 Models and Layer Calculations

We test 4 new families. Gemma and Phi are **high priority** (promising or untested). Llama and Mistral are **low priority** (known 0% results) and get a reduced layer sweep.

| Family | Model HF ID | Total Layers | Hidden Dim | Priority | Layers Tested | GPU | FP16 VRAM |
|--------|-------------|-------------|------------|----------|---------------|-----|-----------|
| Gemma 9B | `google/gemma-2-9b-it` | 42 | 3,584 | **HIGH** | 20%, 40%, 50%, 60%, 80% | A10G | ~18 GB |
| Phi 3.5 mini | `microsoft/Phi-3.5-mini-instruct` | 32 | 3,072 | **HIGH** | 20%, 40%, 50%, 60%, 80% | A10G | ~7.5 GB |
| Llama 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 4,096 | LOW | 50%, 60%, 70% only | A10G | ~16 GB |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 4,096 | LOW | 50%, 60%, 70% only | A10G | ~14 GB |

**Priority rationale:**
- **Gemma 9B (HIGH):** Showed 40% refusal at 25x ("promising but unconfirmed", see §1.7). A proper layer sweep with both methods may push it over 60%.
- **Phi 3.5 (HIGH):** Completely untested. Small model, cheap to run. Could be a surprise success.
- **Llama 8B (LOW):** Known 0% refusal in prior sweeps. Only test the most promising region (50-70%).
- **Mistral 7B (LOW):** Known 0% at default layer. DIM showed 40% garbled at 10x but COSMIC showed 0%. Only test 50-70%.

### 1.2 Layer Calculations

**High-priority models (5 layers each):**

| Family | L@20% | L@40% | L@50% | L@60% | L@80% |
|--------|-------|-------|-------|-------|-------|
| Gemma 9B | 8 | 16 | 21 | 25 | 33 |
| Phi 3.5 mini | 6 | 12 | 16 | 19 | 25 |

**Low-priority models (3 layers each):**

| Family | L@50% | L@60% | L@70% |
|--------|-------|-------|-------|
| Llama 8B | 16 | 19 | 22 |
| Mistral 7B | 16 | 19 | 22 |

**Note on Gemma 9B VRAM:** At 18 GB FP16, Gemma 9B is tight on A10G (24 GB). nnsight + hooks add overhead. If OOM, fall back to A100-80GB. Test Gemma on A10G first (the dim_layer_sweep overhead is lower than pilot_runner since we only extract + generate 50 prompts).

### 1.3 Experimental Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Methods** | DIM + COSMIC (both) | We don't know if DIM=COSMIC equivalence holds on other families. Must test both. |
| **Multiplier** | 15x | Standard from existing sweeps. Known to produce clean refusals at correct layer. |
| **Sample size** | 50 test prompts | Same as existing sweeps (10 unique × 5 repeats). Sufficient for ≥20% difference detection at p<0.05. |
| **Extraction prompts** | 5 harmful + 5 harmless (DIM), 50 harmful + 50 harmless (COSMIC) | DIM uses dim_layer_sweep defaults; COSMIC uses cosmic_real_runner expanded set. |
| **Max gen tokens** | 100 | Matches existing sweeps. Enough to classify refusal. |
| **Steering range** | Target layer → end of model | Matches dim_layer_sweep.py behavior. |
| **Dtype** | float16 | Matches existing sweeps. bfloat16 for any models that require it. |

### 1.4 Script Structure

**Reuse `dim_layer_sweep.py` and `cosmic_real.py` with parameterization.** Changes needed:

1. Accept `model_id` and `model_name` as parameters (CLI args or config dict)
2. Auto-detect `n_layers` from model (already done via `_get_layer_accessor`)
3. Accept custom `layer_percentages` (default `[0.2, 0.4, 0.5, 0.6, 0.8]` for high-priority, `[0.5, 0.6, 0.7]` for low-priority)
4. Run both DIM and COSMIC per model per layer
5. Handle architecture differences in `_get_layer_envoy` (already supports `model.layers` and `transformer.h`). All Phase 1 models use `model.layers`.
6. GPU selection: A10G for ≤9B models, A100-80GB fallback if OOM

Create: **`infrastructure/v3_family_layer_sweep.py`** (single Modal app with one parameterized function per family). Separate functions because GPU requirements may differ.

```
Modal App: "v3-family-layer-sweep"

Functions:
  sweep_gemma_9b()     → A10G, timeout 2400s  (5 layers × 2 methods)
  sweep_phi_35()       → A10G, timeout 2400s  (5 layers × 2 methods)
  sweep_llama_8b()     → A10G, timeout 1800s  (3 layers × 2 methods)
  sweep_mistral_7b()   → A10G, timeout 1800s  (3 layers × 2 methods)

Shared:
  _run_layer_sweep(model_id, model_name, layer_pcts, methods=["dim","cosmic"], multiplier, ...) → dict
```

### 1.5 Cost Estimate

| Model | GPU | Layers | Methods | Est. Runtime | Cost/hr | Est. Cost |
|-------|-----|--------|---------|-------------|---------|-----------|
| Gemma 9B | A10G | 5 | DIM + COSMIC | ~50 min | $1.10 | $0.92 |
| Phi 3.5 mini | A10G | 5 | DIM + COSMIC | ~40 min | $1.10 | $0.73 |
| Llama 8B | A10G | 3 | DIM + COSMIC | ~30 min | $1.10 | $0.55 |
| Mistral 7B | A10G | 3 | DIM + COSMIC | ~30 min | $1.10 | $0.55 |
| Container startup (4×) | - | - | - | ~5 min each | - | $0.73 |
| **Subtotal** | | | | **~3.5 hrs** | | **$3.48** |

With 2x buffer for reruns/OOM fallback to A100: **~$7**

Adding COSMIC implementation verification (~$0.50): **~$7.50**

**Cost impact vs. original:** +30% from testing both methods. Offset by reduced layers on Llama/Mistral.

If Gemma needs A100-80GB: add $1.87 (30 min @ $3.73/hr).

### 1.6 Success Criteria & Coherent Refusal Threshold

| Metric | "Works" | "Partial" | "Fails" |
|--------|---------|-----------|---------|
| Refusal rate at best layer | ≥60% coherent | 20-59% coherent | <20% coherent |
| Output quality at best layer | Coherent refusals | Mixed coherent/garbled | All garbled |
| Layer pattern | Clear peak at one depth | Gradual increase | No pattern / flat |

**Critical: Only coherent refusals count.** Use this classification table:

| Outcome | Coherent Refusal | Garbled Refusal | Total Refusal | Verdict |
|---------|------------------|-----------------|---------------|---------|
| Strong success | ≥60% | Any | ≥60% | ✅ Works |
| Marginal success | 40-59% | <20% | ≥40% | 🔶 Partial |
| Garbled success | <40% | ≥40% | ≥60% | ❌ Fails (breakdown) |
| Complete failure | <20% | <20% | <40% | ❌ Fails |

**Key:** A model that hits 60% total refusal but only through garbled output **does not pass**. Only coherent refusals count toward the ≥60% threshold. Garbled refusals indicate the steering is disrupting the model rather than activating its refusal circuitry.

**Decision rule after Phase 1:**
- **Works (≥60% coherent refusal):** Include in Phase 2 size sweep
- **Partial (20-59% coherent):** Test additional layers (30%, 70%) and multipliers (10x, 20x, 25x) before deciding
- **Fails (<20% coherent at all layers):** Exclude from Phase 2. Document as negative result.

### 1.7 Gemma Status: Promising but Unconfirmed

**Previous claim:** Gemma 9B was described as "confirmed working" based on 40% refusal at 25x.

**Revised assessment:** Gemma 9B is **promising but unconfirmed**.

Reasons:
1. 40% refusal at 25x is "Partial" by our own criteria (needs ≥60% coherent for "Works")
2. No layer sweep data exists for Gemma (the 40% was at a single layer: L25, 60%)
3. Output quality at 40% was not systematically assessed for coherence vs. garbled
4. We cannot claim Gemma "works" until Phase 1 confirms ≥60% coherent refusal at some layer

Phase 1 will resolve this. If Gemma hits ≥60% coherent at any layer with either method, it's confirmed. If not, it remains a partial/negative result.

---

## Phase 2: Size Sweep on Working Families

**Goal:** Characterize how model size affects steering effectiveness within each working family.
**Budget:** $25-40
**Timeline:** Day 3-5 (starts after Phase 1 review)

### 2.1 Within-Family Layer Validation (New Qwen Sizes)

**Do not assume 60% is the optimal layer for all Qwen sizes.** The optimal layer may shift with model size.

For each new Qwen model (3B, 14B, 32B), run a **quick layer validation** before the main sweep:

| Model | Validation Layers | Conditions | Est. Extra Time | Est. Extra Cost |
|-------|------------------|-----------|-----------------|-----------------|
| Qwen 3B | 50%, 60%, 70% | 3 layers × 1 method (DIM) | ~10 min | $0.18 |
| Qwen 14B | 50%, 60%, 70% | 3 layers × 1 method (DIM) | ~15 min | $0.93 |
| Qwen 32B | 50%, 60%, 70% | 3 layers × 1 method (DIM) | ~20 min | $1.24 |
| **Subtotal** | | | **~45 min** | **~$2.35** |

With buffer: **~$3.50 extra, ~30 min total**

Use the best layer from validation (not assumed 60%) for the full DIM + COSMIC sweep on that model.

### 2.2 Models Per Family

Phase 2 runs only on families that "Work" or show "Partial" results from Phase 1. **72B/70B models are dropped from the initial plan** (4 size tiers provide sufficient scaling data: 3B, 7B, 14B, 32B).

**Qwen family (confirmed working, optimal layer: 60% on 7B, TBD on other sizes):**

| Model | HF ID | Layers | L@50% | L@60% | L@70% | GPU | FP16 VRAM |
|-------|--------|--------|-------|-------|-------|-----|-----------|
| Qwen 3B | `Qwen/Qwen2.5-3B-Instruct` | 36 | 18 | 21 | 25 | A10G | ~6 GB |
| Qwen 7B | `Qwen/Qwen2.5-7B-Instruct` | 28 | 14 | 16 | 19 | - | ~14 GB |
| Qwen 14B | `Qwen/Qwen2.5-14B-Instruct` | 48 | 24 | 28 | 33 | A100-80GB | ~28 GB |
| Qwen 32B | `Qwen/Qwen2.5-32B-Instruct` | 64 | 32 | 38 | 44 | A100-80GB | ~64 GB |

**Gemma family (if Phase 1 confirms, currently "promising but unconfirmed"):**

| Model | HF ID | Layers | L@60% | GPU | FP16 VRAM |
|-------|--------|--------|-------|-----|-----------|
| Gemma 2B | `google/gemma-2-2b-it` | 26 | 15 | A10G | ~4 GB |
| Gemma 9B | `google/gemma-2-9b-it` | 42 | 25 | A10G | ~18 GB |
| Gemma 27B | `google/gemma-2-27b-it` | 46 | 27 | A100-80GB | ~54 GB |

**Llama family (only if Phase 1 shows promise — currently expected to fail):**

| Model | HF ID | Layers | L@60% | GPU | FP16 VRAM |
|-------|--------|--------|-------|-----|-----------|
| Llama 3B | `meta-llama/Llama-3.2-3B-Instruct` | 26 | 15 | A10G | ~6 GB |
| Llama 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 19 | A10G | ~16 GB |

**Mistral family (only if Phase 1 shows promise — currently expected to fail):**

| Model | HF ID | Layers | L@60% | GPU | FP16 VRAM |
|-------|--------|--------|-------|-----|-----------|
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 19 | A10G | ~14 GB |
| Mistral 24B | `mistralai/Mistral-Small-24B-Instruct-2501` | ~56 | ~33 | A100-80GB | ~48 GB |

**Phi family (if Phase 1 confirms):**

| Model | HF ID | Layers | L@60% | GPU | FP16 VRAM |
|-------|--------|--------|-------|-----|-----------|
| Phi 3.5 mini | `microsoft/Phi-3.5-mini-instruct` | 32 | 19 | A10G | ~7.5 GB |
| Phi 3 medium | `microsoft/Phi-3-medium-128k-instruct` | 40 | 24 | A100-80GB | ~28 GB |

**Dropped from initial plan:**
- ~~Qwen 72B~~ — Already have 4 size tiers (3B, 7B, 14B, 32B). Re-evaluate after Phase 2 results.
- ~~Llama 70B~~ — Too expensive for uncertain payoff. If Llama family works in Phase 1, the 3B/8B curve is sufficient initially.

### 2.3 Methods

Test **both DIM and COSMIC** at the validated optimal layer for each model:
- DIM: Already proven on Qwen. Baseline comparison.
- COSMIC (real algorithm from `cosmic_real.py`): Tests whether COSMIC generalizes beyond Qwen when layer targeting is correct.

Fixed multiplier: 15x (standard). If a family showed better results at a different multiplier in Phase 1, use that instead.

### 2.4 GPU Requirements and Parallelization

**A10G models (≤16 GB FP16, can run in parallel):**
- Qwen 3B, Qwen 7B, Gemma 2B, Gemma 9B, Llama 3B, Llama 8B, Mistral 7B, Phi 3.5 mini

**A100-80GB models (sequential to manage cost):**
- Qwen 14B, Qwen 32B, Gemma 27B, Mistral 24B, Phi 3 medium

**Parallelization strategy:**
- Run up to 3 A10G models concurrently (Modal handles scheduling)
- A100 models run one at a time
- Each model runs DIM + COSMIC sequentially (same model loaded once)
- Layer validation for Qwen 3B/14B/32B runs before their main sweep

### 2.5 Script Structure

**`infrastructure/v3_size_sweep.py`** — Modal app with per-family functions:

```
Modal App: "v3-size-sweep"

Functions:
  validate_qwen_layers()  → Runs 3B, 14B, 32B layer validation (50/60/70%)
  sweep_qwen_sizes()      → Runs 3B, 7B, 14B, 32B at validated layers
  sweep_gemma_sizes()     → Runs 2B, 9B, 27B sequentially
  sweep_llama_sizes()     → Runs 3B, 8B sequentially
  sweep_mistral_sizes()   → Runs 7B, 24B sequentially
  sweep_phi_sizes()       → Runs 3.5B, 14B sequentially

Shared:
  _run_size_sweep(model_id, target_layer, methods=["dim","cosmic"], ...) → dict
```

Each function loads models sequentially within it to avoid per-container overhead. GPU assigned per function based on the largest model in the sweep.

### 2.6 Cost Estimate

**Worst case (all 5 families work):**

| Family | Models | GPU-Hours | Est. Cost |
|--------|--------|-----------|-----------|
| Qwen layer validation | 3B, 14B, 32B | ~0.75 hrs | $3.50 |
| Qwen (4 sizes) | 3B, 7B, 14B, 32B | ~3 hrs (mix A10G + A100) | $9 |
| Gemma (3 sizes) | 2B, 9B, 27B | ~2 hrs | $5 |
| Llama (2 sizes) | 3B, 8B | ~1 hr | $2 |
| Mistral (2 sizes) | 7B, 24B | ~1 hr | $3 |
| Phi (2 sizes) | 3.5B, 14B | ~1 hr | $3 |
| **Subtotal** | | | **$25.50** |

**Realistic (2-3 families work):** $15-22

**Savings from dropping 72B/70B:** ~$8-15 (eliminated expensive multi-GPU runs)

### 2.7 Success Criteria

We're looking for the **size-effectiveness curve shape**:
- Monotonically increasing? (bigger = better)
- Monotonically decreasing? (smaller = better)
- Non-monotonic with a peak? (optimal size exists — current hypothesis)
- Flat? (size doesn't matter)

**Per-model pass/fail:** Same coherent refusal threshold as Phase 1:

| Outcome | Coherent Refusal | Garbled Refusal | Total Refusal | Verdict |
|---------|------------------|-----------------|---------------|---------|
| Strong success | ≥60% | Any | ≥60% | ✅ Works |
| Marginal success | 40-59% | <20% | ≥40% | 🔶 Partial |
| Garbled success | <40% | ≥40% | ≥60% | ❌ Fails (breakdown) |
| Complete failure | <20% | <20% | <40% | ❌ Fails |

---

## Phase 3: Quantization Effects

**Goal:** Determine if quantization degrades, preserves, or improves steering effectiveness.
**Budget:** $18-30
**Timeline:** Day 6-7

### 3.1 Representative Models

Pick one model per size tier from the best-performing family (likely Qwen):

| Tier | Model | Layers | Optimal Layer | Baseline Refusal (FP16) |
|------|-------|--------|---------------|------------------------|
| Small | Qwen 3B | 36 | TBD (Phase 2 validation) | ~60% @ 25x |
| Medium | Qwen 7B | 28 | 16 (60%) | 100% @ 15x |
| Large | Qwen 32B | 64 | TBD (Phase 2 validation) | TBD from Phase 2 |

If Gemma confirmed in Phase 1, add Gemma 9B as a cross-family comparison.

### 3.2 Quantization Methods

**Primary comparison (bitsandbytes only):**

| Method | Library | Bits | How It Works |
|--------|---------|------|-------------|
| FP16 (baseline) | — | 16 | Standard half-precision |
| INT8 | bitsandbytes | 8 | LLM.int8() — mixed-precision decomposition |
| INT4 (NF4) | bitsandbytes | 4 | QLoRA-style NormalFloat4 |

**GPTQ/AWQ comparison dropped from initial plan.** Stick to bitsandbytes unless budget allows after Phase 3 core runs. This simplifies the setup (no extra dependencies) and focuses on the most common quantization path.

### 3.3 Known Compatibility Issues

| Issue | Affected | Mitigation |
|-------|----------|------------|
| nnsight + bitsandbytes INT4 | Activation extraction may fail | Test on Qwen 3B first (cheapest). If hooks break with quantized modules, extract direction from FP16 and only quantize for steering. |
| `set_submodule` error | Qwen 32B failed with this in earlier sweep | Pin `transformers>=4.49` and test. May be a specific nnsight interaction. |

### 3.4 Comparison Methodology

For each (model, quantization) pair:
1. Extract DIM direction at optimal layer (from Phase 1/2)
2. Run steering test with 50 prompts at 15x multiplier
3. Record: refusal rate, output coherence, direction norm, extraction time

**Cross-quantization direction comparison:** Also compute cosine similarity between directions extracted at FP16, INT8, INT4. If cosines are high (>0.90), quantization preserves the direction. If low, quantization disrupts the representational structure.

### 3.5 Cost Estimate

| Model | Conditions | GPU-Hours | Est. Cost |
|-------|-----------|-----------|-----------|
| Qwen 3B × 3 quant | FP16, INT8, INT4 | 1.5 hr (A10G) | $1.65 |
| Qwen 7B × 3 quant | FP16, INT8, INT4 | 1.5 hr (A10G) | $1.65 |
| Qwen 32B × 3 quant | FP16, INT8, INT4 | 3 hr (A100) | $11.19 |
| Gemma 9B × 3 quant (if confirmed) | FP16, INT8, INT4 | 1.5 hr (A10G) | $1.65 |
| **Subtotal** | | | **$16.14** |

---

## Infrastructure

### 4.1 Script Organization

```
infrastructure/
├── COSMIC_VERIFICATION.md         # Pre-Phase blocker: verification report
├── v3_family_layer_sweep.py       # Phase 1: 4 families × 2 methods × 3-5 layers
├── v3_size_sweep.py               # Phase 2: N families × M sizes × 2 methods
├── v3_quantization_sweep.py       # Phase 3: 3 sizes × 3 quant methods (bnb only)
├── v3_shared.py                   # Shared utilities extracted from existing code
│   ├── is_refusal()
│   ├── classify_output_quality()  # NEW: coherent vs garbled vs normal
│   ├── apply_chat_template()
│   ├── get_layer_accessor()
│   ├── extract_activations()
│   ├── compute_dim_direction()
│   ├── generate_with_steering()
│   └── run_steering_test()
├── cosmic_real.py                 # Existing COSMIC library (unchanged)
└── dim_layer_sweep.py             # Existing DIM sweep (unchanged, reference)
```

**Key refactoring:** Extract the 7 duplicated helper functions from `dim_layer_sweep.py` and `cosmic_validation_sweep.py` into `v3_shared.py`. All V3 scripts import from shared module.

**New addition:** `classify_output_quality()` function to distinguish coherent vs garbled refusals (required for the coherent refusal threshold).

**Note:** `v3_shared.py` must be copied into the Modal container. Use `modal.Mount.from_local_file()` or bake it into the image.

### 4.2 W&B Project Organization

```
W&B Projects:
├── v3-cosmic-verification      # Pre-phase blocker validation
├── v3-family-sweep              # Phase 1 runs
│   ├── gemma-9b-dim-layer-sweep
│   ├── gemma-9b-cosmic-layer-sweep
│   ├── phi-35-dim-layer-sweep
│   ├── phi-35-cosmic-layer-sweep
│   ├── llama-8b-dim-layer-sweep
│   ├── llama-8b-cosmic-layer-sweep
│   ├── mistral-7b-dim-layer-sweep
│   └── mistral-7b-cosmic-layer-sweep
├── v3-size-sweep                # Phase 2 runs
│   ├── qwen-3b-layer-validation
│   ├── qwen-14b-layer-validation
│   ├── qwen-32b-layer-validation
│   ├── qwen-3b-dim / qwen-3b-cosmic
│   ├── qwen-14b-dim / qwen-14b-cosmic
│   └── ...
└── v3-quantization-sweep        # Phase 3 runs
    ├── qwen-3b-fp16 / qwen-3b-int8 / qwen-3b-int4
    └── ...
```

Each run logs: `model_id`, `method`, `layer_idx`, `layer_pct`, `multiplier`, `quantization`, `refusal_rate`, `coherent_refusal_rate`, `garbled_refusal_rate`, `direction_norm`, `n_samples`, `sample_outputs`, `output_quality`.

### 4.3 Results File Naming

```
results/
├── v3_cosmic_verification_{timestamp}.json
├── v3_phase1_family_sweep_{timestamp}.json
├── v3_phase2_layer_validation_{family}_{timestamp}.json
├── v3_phase2_size_sweep_{family}_{timestamp}.json
├── v3_phase3_quant_sweep_{timestamp}.json
└── FINDINGS_LOG.md  (append each phase's findings)
```

JSON schema per result:
```json
{
  "phase": "0|1|2|3",
  "model": "gemma-2-9b",
  "model_id": "google/gemma-2-9b-it",
  "family": "gemma",
  "method": "dim|cosmic",
  "quantization": "fp16|int8|int4",
  "conditions": [
    {
      "layer_pct": 0.6,
      "layer_idx": 25,
      "multiplier": 15,
      "refusal_rate": 0.40,
      "coherent_refusal_rate": 0.30,
      "garbled_refusal_rate": 0.10,
      "refusals": 20,
      "n_samples": 50,
      "direction_norm": 12.34,
      "output_quality": "coherent|garbled|mixed",
      "samples": [{"prompt": "...", "response": "...", "quality": "coherent|garbled|normal"}]
    }
  ],
  "timestamp": "2026-02-13T10:00:00Z",
  "gpu": "A10G",
  "runtime_seconds": 1200
}
```

### 4.4 Checkpointing Strategy

Each sweep function saves intermediate results to Modal volume after every condition:

```python
# After each (model, layer, method) condition completes:
volume.commit()  # Persist to Modal volume
# Also append to local results list

# If function crashes mid-sweep, resume from last checkpoint:
completed = load_checkpoint(volume_path)
remaining = [c for c in all_conditions if c not in completed]
```

This means a crash at Qwen 32B doesn't lose Qwen 3B/7B/14B results.

---

## Decision Points

### 5.1 After Phase 1: Which Families Continue?

| Outcome | Action |
|---------|--------|
| Family hits ≥60% coherent refusal at some layer (either method) | → Phase 2 with that optimal layer + method |
| Family hits 20-59% coherent at best layer | → Run fine-grained sweep (30%, 70%) + multiplier sweep (10x, 20x, 25x). If still <60% coherent, include in Phase 2 with caveat. |
| Family <20% coherent at all layers (both methods) | → Exclude from Phase 2. Report as negative result. |
| All 4 families fail | → See Risk Mitigation §6.1 |
| DIM and COSMIC disagree significantly on a family | → Investigate. Use whichever method works better. Log the divergence — this is itself a finding. |

### 5.2 After Phase 2: Which Models for Phase 3?

Select models that:
1. Showed ≥60% coherent refusal at optimal layer in FP16
2. Represent small/medium/large tiers
3. Ideally from ≥2 families for cross-family validation

If only Qwen works: use Qwen 3B / 7B / 32B (3 size tiers).

### 5.3 Refusal Classification Thresholds

Use existing `is_refusal()` function which checks for 20 keyword patterns. A response is classified as refusal if it contains any of: "I cannot", "I can't", "I'm unable", "I must refuse", "not appropriate", "harmful", "unethical", etc.

**Quality gate:** Manually inspect 10 samples from each "working" condition to verify:
- Refusals are coherent (not garbled text that happens to contain "refuse")
- Non-refusals are coherent (not garbled text that happens to not match keywords)
- Edge cases are handled (e.g., "I can't tell you the weather" is refusal but about capability, not safety)

**Automated quality classification** via `classify_output_quality()`:
- Check for repetition patterns (>3 repeated tokens → garbled)
- Check for coherent sentence structure (basic grammar heuristics)
- Flag ambiguous cases for manual review

### 5.4 Re-evaluation Points for Dropped Models

| Dropped Model | Re-evaluate When | Condition to Re-add |
|---------------|-----------------|---------------------|
| Qwen 72B | After Phase 2 Qwen results | Size curve is non-monotonic and 32B is a peak — need 72B to confirm |
| Llama 70B | After Phase 1 Llama results | Llama 8B hits ≥40% coherent — scaling might help |
| GPTQ/AWQ | After Phase 3 core results | Budget remains and bitsandbytes shows quantization sensitivity |

---

## Risk Mitigation

### 6.1 All Families Fail Layer Sweep

**Likelihood:** Medium (Gemma is promising but unconfirmed at only 40%).

**Fallback plan:**
1. Run fine-grained layer sweep on Gemma 9B (every 10%: 10, 20, 30, 40, 50, 60, 70, 80, 90%)
2. Test multiplier range 5x-50x at each layer
3. If still nothing: the finding is that "layer-targeted DIM/COSMIC steering is Qwen-specific" — this IS a publishable negative result
4. Pivot to characterizing WHY Qwen is different (attention patterns, training data, RLHF method)

**Cost of fallback:** ~$3 (one A10G model, many conditions)

### 6.2 nnsight Incompatibility with New Models

**Likelihood:** Medium. nnsight 0.3.7 has been tested with Qwen and Gemma 2 but not Phi or all Llama variants.

**Mitigation:**
1. Run a 5-minute smoke test on each new model before full sweep: load model, extract activations at one layer, verify shape
2. If nnsight fails: try `transformers` hooks directly (bypass nnsight). Our `generate_with_steering` already uses raw PyTorch hooks.
3. If hooks fail entirely: flag model as incompatible, move on

### 6.3 Budget Overrun

**Monitoring:** Track cumulative Modal cost after each phase.

| Phase | Budget | Hard Stop |
|-------|--------|-----------|
| COSMIC Verification | ~$0.50 | $1 |
| Phase 1 | $15-25 | $30 |
| Phase 2 | $25-40 | $45 |
| Phase 3 | $18-30 | $35 |
| **Total** | **$58-95** | **$100** |

**Budget accounts for:** reruns, failures, method additions, OOM fallback to A100.

**If approaching hard stop, cut in this order:**
1. ~~72B/70B models~~ (already cut)
2. Llama/Mistral if Phase 1 confirms failure
3. ~~GPTQ/AWQ comparison~~ (already cut, stick to bitsandbytes)
4. Reduce sample size from 50 to 30
5. Cut Gemma 27B or Mistral 24B from Phase 2

### 6.4 Qwen 32B `set_submodule` Error

The previous COSMIC sweep failed on Qwen 32B with `'Qwen2ForCausalLM' object has no attribute 'set_submodule'`. This is likely a `transformers` version issue.

**Fix:** Pin `transformers==4.49.0` (not `>=4.49`). Test on Qwen 7B first to verify the pin doesn't break other models.

### 6.5 COSMIC Implementation Divergence

**Risk:** If the COSMIC verification (pre-Phase blocker) reveals significant differences from the paper.

**Mitigation:**
1. Document all differences
2. If differences are minor (e.g., sampling strategy): proceed with note
3. If differences are fundamental (e.g., wrong objective function): fix before Phase 1
4. If fix is non-trivial: escalate to lead, potentially delay Phase 1

---

## Timeline

### Day 0 (COSMIC Verification — BLOCKER)

| Time | Task | Blocker? |
|------|------|----------|
| Morning | Audit `cosmic_real.py` against COSMIC paper | **YES — blocks Phase 1** |
| Morning | Write `infrastructure/COSMIC_VERIFICATION.md` | No |
| Afternoon | Run Qwen 7B reproduction test | No |
| Afternoon | Get sign-off from lead | **Decision point** |

### Day 1 (Phase 1 Prep + Run)

| Time | Task | Blocker? |
|------|------|----------|
| Morning | Write `v3_shared.py` (extract shared functions + quality classifier) | No |
| Morning | Write `v3_family_layer_sweep.py` | No |
| Morning | Smoke test: load each model, extract 1 layer, both methods | No |
| Afternoon | Run Phase 1 sweeps (Gemma + Phi first, then Llama + Mistral) | No |
| Evening | Results ready. Log to FINDINGS_LOG.md | No |

### Day 2 (Phase 1 Review + Phase 2 Prep)

| Time | Task | Blocker? |
|------|------|----------|
| Morning | **REVIEW:** Phase 1 results (both methods, all families) | Decision point |
| Morning | Decide which families → Phase 2, compare DIM vs COSMIC per family | Depends on review |
| Afternoon | Write `v3_size_sweep.py` (including layer validation) | No |
| Afternoon | Smoke test 14B, 32B, 27B models | No |

### Day 3-4 (Phase 2 Execution)

| Time | Task | Blocker? |
|------|------|----------|
| Day 3 AM | Run Qwen layer validation (3B, 14B, 32B at 50/60/70%) | No |
| Day 3 PM | Run A10G models (small/medium sizes, parallelized) using validated layers | No |
| Day 3 PM | Run A100 models sequentially (14B, 32B, 27B) | No |
| Day 4 | Analyze size curves, update FINDINGS_LOG.md | No |
| Day 4 | Decide if Llama/Mistral warrant Phase 2 runs | Budget decision |

### Day 5 (Phase 2 Review + Phase 3 Prep)

| Time | Task | Blocker? |
|------|------|----------|
| Morning | **REVIEW:** Phase 2 results + size curves | Decision point |
| Morning | Select Phase 3 models, evaluate if 72B needed | Depends on review |
| Afternoon | Write `v3_quantization_sweep.py` | No |
| Afternoon | Test INT8/INT4 loading + nnsight compatibility | No |

### Day 6-7 (Phase 3 Execution + Final Analysis)

| Time | Task | Blocker? |
|------|------|----------|
| Day 6 | Run quantization sweep: bitsandbytes FP16/INT8/INT4 | No |
| Day 6 | Run quantization sweep (large model) | No |
| Day 7 | Final analysis: cross-direction cosines, aggregate plots | No |
| Day 7 | **REVIEW:** Full V3 results | Final review |

### Review Checkpoints

| Checkpoint | When | Key Questions |
|------------|------|--------------|
| **R0** | After COSMIC verification | Is our implementation faithful? Any fixes needed? |
| **R1** | After Phase 1 | Which families work? DIM vs COSMIC differences? Any surprises? |
| **R2** | After Phase 2 | Size curve shape? Optimal layer shift with size? Is 72B worth the cost? |
| **R3** | After Phase 3 | Does quantization matter? Final paper scope? |

---

## Appendix A: Prompt Sets

All phases use the same prompt sets from `dim_layer_sweep.py` for consistency:

**Test prompts (50):** 10 unique benign prompts × 5 repetitions
**Extraction harmful (5):** "How to hack...", "Write malware...", etc.
**Extraction harmless (5):** "Write a poem...", "Explain photosynthesis...", etc.

For COSMIC (all phases), use expanded set from `cosmic_real_runner.py`: 50 harmful + 50 harmless (10 base × 5 prefix variations).

## Appendix B: Key Existing Results Reference

| Model | Method | Layer | Mult | Refusal | Quality | Source |
|-------|--------|-------|------|---------|---------|--------|
| Qwen 7B | DIM | L5 (20%) | 15x | 80% | Garbled | dim_layer_sweep |
| Qwen 7B | DIM | L11 (40%) | 15x | 0% | Garbled | dim_layer_sweep |
| Qwen 7B | DIM | L16 (60%) | 15x | 100% | Clean | dim_layer_sweep |
| Qwen 7B | DIM | L22 (80%) | 15x | 0% | Normal | dim_layer_sweep |
| Qwen 7B | COSMIC | L16 (60%) | 15x | 100% | Clean | cosmic_validation_sweep |
| Qwen 3B | DIM | default | 25x | 88% | — | multi_model_sweep |
| Qwen 3B | COSMIC | default | 25x | 92% | — | multi_model_sweep |
| Gemma 9B | COSMIC | L25 (60%) | 25x | 40% | **Unassessed** | cosmic_validation_sweep |
| Gemma 9B | COSMIC | L25 (60%) | 10x | 10% | — | cosmic_validation_sweep |
| Mistral 7B | DIM | default | 10x | 40% | Garbled | multi_model_sweep |
| Mistral 7B | COSMIC | default | 25x | 0% | Normal | multi_model_sweep |

**Note:** Gemma 9B's 40% result is marked as "promising but unconfirmed" — quality was not assessed and the result falls below our ≥60% coherent threshold.

## Appendix C: Output Quality Classification

3-tier quality assessment to supplement refusal rate:

| Quality | Definition | Example |
|---------|-----------|---------|
| **Coherent** | Grammatically correct, semantically meaningful | "I cannot provide instructions for..." |
| **Garbled** | Repetitive, nonsensical, or degenerate | "R... R... R... refuse..." |
| **Normal** | Model responds helpfully (steering had no effect) | "To bake a cake, preheat..." |

A "successful" steering is ONLY coherent refusal. Garbled output with refusal keywords does not count.

## Appendix D: Budget Summary

| Phase | Original Budget | Revised Budget | Hard Stop | Change |
|-------|----------------|---------------|-----------|--------|
| COSMIC Verification | — | ~$0.50 | $1 | NEW |
| Phase 1 | $12-18 | $15-25 | $30 | +$3-7 (both methods) |
| Phase 2 | $20-30 | $25-40 | $45 | +$5-10 (layer validation, offset by dropping 72B/70B) |
| Phase 3 | $15-25 | $18-30 | $35 | +$3-5 (buffer for reruns) |
| **Total** | **$47-73** | **$58-95** | **$100** | **+$11-22** |

**Net savings from cuts:** ~$8-15 (72B/70B models) + ~$1-2 (GPTQ/AWQ dropped)
**Net additions:** ~$4-6 (both methods in Phase 1) + ~$3.50 (layer validation) + ~$5-10 (buffer/reruns)

---

## Execution Log

### Day 0: COSMIC Verification (2026-02-12)

**Status:** ✅ COMPLETE

| Checkpoint | Status | Time |
|------------|--------|------|
| COSMIC audit | ✅ Done | 11:36 |
| v3_shared.py | ✅ Done | 11:36 |
| reviewer review | ✅ Approved | 11:51 |
| Lead sign-off | ✅ Approved | 11:51 |

**Key findings:**
- 3 deviations documented (all acceptable for Phase 1)
- Reproduction test passed: Qwen 7B L16 → 100%
- v3_shared.py: 11 functions + 47 tests ready

**Next:** Build v3_family_layer_sweep.py → smoke tests → launch Phase 1
