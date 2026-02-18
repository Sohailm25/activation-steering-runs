# Phase 1: Tooling Parity Protocol

**Track:** A — Tooling Reproducibility Study
**Status:** Protocol ready for review
**Date:** 2026-02-18
**Depends on:** Phase 0 literature gap analysis (complete)

---

## A. Objective

Test whether the choice of activation-extraction tooling changes the extracted refusal direction and downstream steering behavior.

Our paper reports that NNsight tracing produces 100% coherent refusal on Qwen 7B while PyTorch forward hooks produce only 10%. This is the only published quantitative evidence that extraction tooling affects steering direction quality. Phase 1 determines whether this discrepancy is:

1. A stable, reproducible tooling effect (different tools extract meaningfully different directions)
2. Run-to-run noise (same tool gives different results each time)
3. A Qwen 7B-specific artifact (does not replicate on other architectures)

---

## B. Exact Design

### Models

| Model | Role | Layers | Hidden dim | Justification |
|---|---|---|---|---|
| **Qwen 7B** | Primary | 28 | 3584 | Existing data; site of original discrepancy |
| **Gemma 9B** | Non-Qwen control | 42 | 3584 | Different architecture family (Google vs Alibaba); we have existing infrastructure from our paper; similar hidden dim enables geometric comparison; 42 layers provides a different depth profile than Qwen's 28 |

**Why Gemma 9B over alternatives:**
- Llama 8B would work but we lack existing infrastructure for it. Gemma 9B was used in our original paper, so contrastive prompts and evaluation pipeline are already validated.
- Mistral 7B shares more architectural DNA with Llama than with Qwen, making it less informative as a second family.
- Gemma's 42-layer architecture (vs Qwen's 28) tests whether tooling effects depend on model depth.

### Tools

| # | Tool | Extraction mechanism | Notes |
|---|---|---|---|
| 1 | **NNsight** (Fiotto-Kaufman et al., arXiv:2407.14561) | Graph-based tracing; deferred execution | Our paper's primary tool |
| 2 | **Raw PyTorch forward hooks** | `register_forward_hook()` on target module | Our paper's comparison baseline |
| 3 | **pyvene** (Wu et al., arXiv:2403.07809) | Declarative intervention framework; configurable extraction/intervention schemes | Different intervention paradigm from both NNsight and hooks; pip-installable (`pip install pyvene`); actively maintained; supports arbitrary PyTorch models |

**Why pyvene over IBM activation-steering:**
- pyvene is model-agnostic (any PyTorch module), while IBM's library is specialized for steering workflows. A general-purpose tool is more likely to expose extraction-level differences.
- pyvene's declarative config separates "what to extract" from "how to extract," providing a third distinct extraction paradigm.
- Both are viable; IBM activation-steering (Lee et al., ICLR 2025) is an acceptable substitute if pyvene integration proves problematic.

### Fixed Settings

| Parameter | Value | Source |
|---|---|---|
| Contrastive prompts | 5 harmful + 5 harmless from `v3_shared.py` | Same as original paper |
| Evaluation prompts | 30 prompts from original eval set | Same as original paper |
| Target layer (Qwen 7B) | L16 (layer index 16, ~57% depth = 16/28) | Original paper's optimal layer |
| Target layer (Gemma 9B) | L16 (layer index 16, ~38% depth = 16/42) | Matched layer INDEX (not depth) for direct comparison; see note below |
| Multiplier (Qwen 7B) | 15× | Original paper |
| Multiplier (Gemma 9B) | 25× | Original paper |
| Decoding | Greedy (temperature=0, top_k=1) | Eliminates sampling variance |
| max_tokens | 256 | Original paper |

**Layer note for Gemma 9B:** Using L16 for both models matches the layer index, putting Gemma at ~38% depth (vs Qwen at ~57%). This is deliberate — we want to test tooling parity at the same extraction point specification (layer 16), not at matched relative depth. If tooling effects are layer-dependent, a follow-up can sweep depth.

### Repeats

- **K = 5** identical runs per method per model
- Total runs per model: 5 runs × 3 tools = **15 extraction+steering cycles**
- Total across both models: **30 cycles**

Each run uses identical random seed (seed=42), identical prompt ordering, and identical model weights. The only variable is the extraction tool.

---

## C. Metrics

### Primary metrics

| Metric | Computation | Purpose |
|---|---|---|
| **Direction cosine similarity (within-method)** | Pairwise cosine sim across K=5 runs for same tool | Measures run-to-run determinism |
| **Direction cosine similarity (across-method)** | Pairwise cosine sim between tools (nnsight vs hooks, nnsight vs pyvene, hooks vs pyvene) | Measures tooling divergence |
| **Coherent refusal rate** | % of 30 eval prompts producing coherent refusal per run | Primary behavioral metric |
| **Garbled rate** | % of 30 eval prompts producing incoherent/garbled output | Detects over-steering |
| **Normal rate** | % of 30 eval prompts producing normal (non-refusing) response | Detects under-steering |

### Secondary metrics

| Metric | Computation | Purpose |
|---|---|---|
| **Direction norm** | L2 norm of extracted direction per run | Detects magnitude differences across tools |
| **Mean ± std across K repeats** | For all primary metrics | Quantifies variance |
| **Per-component divergence** | Element-wise difference between directions from different tools | Localizes where directions differ |

### Determinism threshold

- **Within-method cosine > 0.999** across K=5 runs → method is deterministic (run-to-run variance is negligible)
- **Within-method cosine < 0.999** → method has non-trivial run-to-run variance; must report and investigate source (floating-point non-determinism, batching effects, etc.)

---

## D. Tensor-Site Parity Checklist

Before running any experiments, verify the following are identical across all three tools. Document the verification in a checklist file (`results/phase1_parity_checklist.json`).

### Extraction point

| Check | Requirement | How to verify |
|---|---|---|
| **Residual stream location** | Output of layer L's residual stream (post-attention + post-MLP addition, BEFORE the next layer's LayerNorm) | For each tool: extract activation, also extract pre-LayerNorm and post-LayerNorm tensors. Verify which one the tool returns by checking `torch.allclose()` against the known pre/post-LN values. |
| **Shape** | `[batch, seq_len, hidden_dim]` — must be identical across tools | Assert shape equality. For Qwen 7B: `[1, *, 3584]`. For Gemma 9B: `[1, *, 3584]`. |
| **dtype** | Must match across tools. Prefer `float32` for extraction; if model runs in `bfloat16`, cast to `float32` before direction computation. | Check `tensor.dtype` from each tool. If any tool auto-casts, document and force-match. |
| **Token position** | Last token of the prompt (index `-1` along seq_len dim) | For each tool: extract full `[1, seq_len, hidden_dim]`, verify seq_len matches tokenized prompt length, take position `-1`. Do NOT use `[0, 0, :]` — that's the BOS token. |
| **Timing** | Activation captured at same computational stage — AFTER residual connection (attention output + MLP output added to residual stream) at layer L | Run a manual forward pass with hooks on attention_output, mlp_output, and residual_stream. Verify that `extracted_activation ≈ residual_input + attention_output + mlp_output` for each tool. |

### Pre-flight validation procedure

1. Extract activations for a single prompt with all three tools
2. Compute pairwise `torch.allclose(tool_A, tool_B, atol=1e-5)` on the raw activation tensor (not the direction)
3. If any pair fails: investigate which check above is violated
4. Do NOT proceed to direction extraction until raw activations match across all three tools

---

## E. Decision Rules / Hard Stops

### Outcome 1: "Tooling issue is real and generalizable"

**Trigger:** Cross-method cosine similarity < 0.95 AND steering effectiveness (coherent refusal rate) differs by >20 percentage points across tools, on **at least 2 models** (both Qwen 7B and Gemma 9B).

**Action:** This is the strongest result. Write up as a reproducibility finding. Investigate root cause (likely extraction-point mismatch per Section D). Proceed to Phase 2 using whichever tool passes the parity checklist.

### Outcome 2: "Model-specific artifact"

**Trigger:** Tooling divergence on Qwen 7B only (cosine < 0.95, effectiveness gap > 20pp), but NOT on Gemma 9B (cosine > 0.99, effectiveness gap < 5pp).

**Action:** Report as Qwen 7B-specific finding. Investigate whether Qwen's architecture (grouped-query attention, specific LayerNorm placement) causes the divergence. Proceed to Phase 2 with the tool that passes parity on both models.

### Outcome 3: "Proceed to Phase 2"

**Trigger:** Within-method variance is low (cosine > 0.999) AND we can identify which tool gives "correct" extractions (validated by: passes parity checklist in Section D AND produces highest steering effectiveness).

**Action:** Lock in the validated tool. Proceed to Phase 2 (Transfer Protocol).

### Outcome 4: "Stop — tooling divergence is not generalizable"

**Trigger:** All 3 tools agree (cross-method cosine > 0.99) on BOTH models, AND steering effectiveness matches within 5pp across tools.

**Action:** Write up as "tooling divergence is model-specific or was an artifact of our original implementation." Still publishable as a reproducibility confirmation. Do NOT extend to more models — the finding doesn't generalize.

---

## F. Budget + Runtime Estimate

### Compute

| Item | Calculation | Estimate |
|---|---|---|
| Qwen 7B | 5 runs × 3 tools = 15 cycles × ~15 min/cycle on A10G | ~3.75 GPU-hours |
| Gemma 9B | 5 runs × 3 tools = 15 cycles × ~20 min/cycle on A10G | ~5.0 GPU-hours |
| Pre-flight parity checks | ~30 min per model × 2 models | ~1.0 GPU-hour |
| **Total GPU time** | | **~9.75 GPU-hours** |
| **Estimated cost** | A10G spot on RunPod/Modal at ~$1-1.50/hr | **$10-15** |

### Human time

| Task | Estimate |
|---|---|
| Implement pyvene extraction pipeline | 2-3 hours |
| Pre-flight parity verification (Section D) | 1-2 hours |
| Run experiments (mostly automated) | 1 hour active supervision |
| Analysis + write-up | 2-3 hours |
| **Total** | **~1 day** |

### Prerequisites

- [ ] pyvene installed and tested on local machine (`pip install pyvene`)
- [ ] Qwen 7B and Gemma 9B weights downloaded
- [ ] GPU access confirmed (RunPod/Modal account with A10G or better)
- [ ] Contrastive prompts (`v3_shared.py`) and eval prompts verified accessible
- [ ] Results directory structure created (`results/phase1/`)
