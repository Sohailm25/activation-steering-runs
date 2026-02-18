# Phase 2: Direction Transfer Protocol

**Track:** B — Direction Transfer Study
**Status:** Protocol ready for review
**Date:** 2026-02-18
**Depends on:** Phase 1 (Tooling Parity) must confirm extraction is reliable before transfer tests are meaningful.

---

## B1: Same-Family Transfer (Qwen 3B → 7B → 14B → 32B)

### Objective

Test whether a DIM refusal direction extracted from one Qwen model transfers directly (without learned mappings) to other Qwen models at different scales, and whether transfer degradation follows the same inverse scaling curve we identified for single-model steering.

### Differentiation from prior work

- **Oozeer et al. (2025, ICML):** Tests same-family transfer for Qwen 0.5B → 1.5B → 2.5B using autoencoder-based nonlinear mappings. We test raw DIM direction transfer (no mapping) at 3B–32B scale.
- **Cristofano (2026):** Tests Qwen3-VL-2B → Qwen3-VL-8B using concept-basis reconstruction. We test standard text-only Qwen at larger scale with the simpler DIM extraction.
- Neither work tests whether transfer degradation correlates with the inverse scaling of single-model steering.

### Models

| Model | Layers | Hidden dim | Optimal layer | Depth % | Source |
|---|---|---|---|---|---|
| Qwen 3B | 36 | 2048 | L21 | 58% | Our paper |
| Qwen 7B | 28 | 3584 | L16 | 57% | Our paper |
| Qwen 14B | 48 | 5120 | L24 | 50% | Our paper |
| Qwen 32B | 64 | 5120 | L32 | 50% | Our paper |

### Transfer matrix

Full transfer matrix: extract at each size, apply to every other size.

| Source ↓ \ Target → | 3B (L21) | 7B (L16) | 14B (L24) | 32B (L32) |
|---|---|---|---|---|
| **3B** | Self-control | Transfer | Transfer | Transfer |
| **7B** | Transfer | Self-control | Transfer | Transfer |
| **14B** | Transfer | Transfer | Self-control | Transfer |
| **32B** | Transfer | Transfer | Transfer | Self-control |

- **4 self-application controls** (extract and apply on same model — baseline)
- **12 transfer pairs** (extract on source, apply on target)
- **16 total experiments**

### Transfer procedure

For each (source, target) pair:

1. **Extract** DIM direction from source model at source's optimal layer using the Phase 1-validated tool
2. **Project** the direction into target model's residual stream space:
   - If `hidden_dim_source == hidden_dim_target`: apply direction directly
   - If `hidden_dim_source ≠ hidden_dim_target`: direction cannot be applied directly. Compute cosine similarity between source direction and target's own DIM direction as the geometric comparison metric. For behavioral transfer, use a learned linear projection (document this clearly as a departure from "raw transfer").
3. **Apply** at target model's optimal layer (NOT source's layer depth) with fixed multiplier
4. **Evaluate** on the same 30 eval prompts with greedy decoding, max_tokens=256

### Dimension mismatch handling

| Source → Target | Source dim | Target dim | Action |
|---|---|---|---|
| 3B → 7B/14B/32B | 2048 | 3584/5120 | Geometric comparison only (cosine of own DIM directions). No direct behavioral transfer without projection. |
| 7B → 3B | 3584 | 2048 | Same — geometric comparison only. |
| 7B → 14B/32B | 3584 | 5120 | Same — geometric comparison only. |
| 14B → 32B | 5120 | 5120 | Direct application (same dim). |
| 32B → 14B | 5120 | 5120 | Direct application (same dim). |
| 14B ↔ 3B/7B | 5120 | 2048/3584 | Geometric comparison only. |

**Note:** Only the 14B ↔ 32B pair shares hidden_dim (5120). All other cross-size pairs require geometric comparison rather than direct behavioral transfer. This is itself a finding: within-family transfer is constrained by architecture heterogeneity even within the same model family.

### Fixed settings

| Parameter | Value |
|---|---|
| Contrastive prompts | 5 harmful + 5 harmless from `v3_shared.py` |
| Evaluation prompts | 30 from original eval set |
| Multiplier | 15× for all Qwen models |
| Decoding | Greedy (temperature=0, top_k=1) |
| max_tokens | 256 |
| Extraction tool | Whichever tool passes Phase 1 parity |

### Layer policy for transfer

Apply source direction at **target model's optimal depth** (not source's depth). Rationale: our paper shows layer choice strongly affects steering effectiveness. Using the target's known-optimal layer isolates the direction quality from layer-selection effects.

| Target | Application layer |
|---|---|
| Qwen 3B | L21 |
| Qwen 7B | L16 |
| Qwen 14B | L24 |
| Qwen 32B | L32 |

### Metrics

| Metric | Computation | Purpose |
|---|---|---|
| **Coherent refusal rate** | % of 30 eval prompts with coherent refusal | Primary behavioral metric |
| **Garbled rate** | % garbled output | Over-steering detection |
| **Normal rate** | % normal (non-refusing) response | Under-steering detection |
| **Cosine similarity** | cos(source_DIM, target_own_DIM) | Geometric compatibility |
| **Norm ratio** | ‖source_DIM‖ / ‖target_own_DIM‖ | Magnitude compatibility |
| **Transfer efficiency** | coherent_refusal_transfer / coherent_refusal_self_control | How much effectiveness is lost in transfer |

### Key analysis: transfer vs inverse scaling correlation

Plot transfer efficiency (y-axis) against scale gap (x-axis, measured as log ratio of target/source parameter count). Overlay our paper's single-model inverse scaling curve. If transfer degradation follows the same curve, this suggests a unified geometric explanation. If transfer degrades faster or differently, the mechanisms are distinct.

---

## B2: Cross-Family Transfer (Qwen ↔ Gemma)

### Objective

Test whether DIM refusal directions transfer across model families without learned mappings, and characterize the geometry of cross-family refusal representations.

### Differentiation from prior work

- **Oozeer et al. (2025, ICML):** Cross-family with autoencoders at sub-3B. No Gemma. We test raw direction transfer at 7B+ scale with Gemma.
- **Cristofano (2026):** Cross-family with concept-basis reconstruction (Qwen→Ministral, no Gemma). We test raw DIM transfer for Qwen↔Gemma.
- **Wang et al. (2024, ICLR 2025, arXiv:2410.06672):** Shows feature-level universality across Transformer/Mamba architectures but does not test refusal-direction transfer between specific model families.
- No published work includes Gemma in any cross-family refusal direction transfer study.

### Models

| Model | Family | Layers | Hidden dim | Optimal layer | Depth % |
|---|---|---|---|---|---|
| Qwen 7B | Qwen | 28 | 3584 | L16 | 57% |
| Qwen 14B | Qwen | 48 | 5120 | L24 | 50% |
| Gemma 9B | Gemma | 42 | 3584 | L16 | 38% |

### Transfer matrix

| Source ↓ \ Target → | Qwen 7B (L16) | Qwen 14B (L24) | Gemma 9B (L16) |
|---|---|---|---|
| **Qwen 7B** | Self-control | (covered in B1) | Cross-family transfer |
| **Qwen 14B** | (covered in B1) | Self-control | Cross-family transfer |
| **Gemma 9B** | Cross-family transfer | Cross-family transfer | Self-control |

**Core cross-family pairs:**
1. Qwen 7B → Gemma 9B (matched scale, different family)
2. Gemma 9B → Qwen 7B (reverse)
3. Qwen 14B → Gemma 9B (larger→smaller, cross-family)
4. Gemma 9B → Qwen 14B (smaller→larger, cross-family)

**New self-control needed:** Gemma 9B self-application (extract own DIM, apply to self) — 1 additional experiment.

**Total B2-specific experiments:** 4 cross-family transfers + 1 Gemma self-control = **5 experiments**

### Tokenizer handling

Use the same English prompts for all models. Each model's tokenizer processes prompts independently.

**Rationale:** The DIM direction is extracted in residual stream space, not token space. Tokenizer differences affect prompt encoding (different token sequences for the same text) but not the geometric comparison of directions. The contrastive prompt design (harmful vs harmless) relies on semantic content, which is language-level — both tokenizers will encode the same English meaning, even if the token sequences differ.

**Verification step:** Before running transfers, confirm that both tokenizers produce reasonable outputs on the 5+5 contrastive prompts (no truncation, no UNK tokens, similar sequence lengths).

### Dimension handling for cross-family

| Source → Target | Source dim | Target dim | Action |
|---|---|---|---|
| Qwen 7B ↔ Gemma 9B | 3584 | 3584 | Direct application (same dim). This is the primary comparison. |
| Qwen 14B → Gemma 9B | 5120 | 3584 | Geometric comparison only. |
| Gemma 9B → Qwen 14B | 3584 | 5120 | Geometric comparison only. |

**Key advantage:** Qwen 7B and Gemma 9B share `hidden_dim=3584`, enabling direct behavioral transfer without projection. This is the cleanest cross-family test possible.

### Multiplier policy for cross-family

Use **target model's standard multiplier**:
- Target is Qwen → 15×
- Target is Gemma → 25×

Rationale: multiplier compensates for the target model's activation scale, not the source's. Using the source's multiplier would conflate direction quality with scale mismatch.

### Fixed settings

Same as B1 (contrastive prompts, eval prompts, decoding, max_tokens).

### Metrics

Same as B1, plus:

| Metric | Computation | Purpose |
|---|---|---|
| **Cross-family cosine** | cos(Qwen_DIM, Gemma_DIM) at matched scale | Core universality metric |

### Interpretation guardrails

| Cosine range | Interpretation | Action |
|---|---|---|
| **< 0.1** | Directions are completely family-specific | Expected null result. Report as "refusal geometry is family-specific for DIM extraction." |
| **0.1 – 0.5** | Partial overlap — some shared refusal structure | Worth investigating. Decompose directions into shared and family-specific components. Check whether the shared component alone can steer. |
| **> 0.5** | Strong universality signal | Unexpected and significant. Verify with additional families if budget permits. Check behavioral transfer effectiveness. |

### Go/no-go from Phase 1

**Proceed to B2 only if:**
1. Phase 1 confirms extraction is reliable (within-method cosine > 0.999 for validated tool)
2. Phase 1 identifies a tool that passes parity on both Qwen 7B and Gemma 9B
3. B1 results are available (same-family transfer provides baseline for interpreting cross-family)

**Skip B2 if:**
- Phase 1 shows all tools agree (Outcome 4) AND B1 shows zero same-family transfer — cross-family transfer is extremely unlikely to work if same-family already fails.

---

## Budget + Runtime Estimate

### B1 (Same-Family)

| Item | Calculation | Estimate |
|---|---|---|
| Direction extraction | 4 models × 1 extraction each | ~2 GPU-hours |
| Self-controls | 4 models × 30 eval prompts × greedy | ~2 GPU-hours |
| Direct transfers (14B↔32B) | 2 pairs × 30 eval prompts | ~1.5 GPU-hours |
| Geometric comparisons | 10 pairs × cosine computation | Negligible (CPU) |
| **B1 total GPU** | | **~5.5 GPU-hours** |

### B2 (Cross-Family)

| Item | Calculation | Estimate |
|---|---|---|
| Gemma 9B direction extraction | 1 model | ~0.5 GPU-hours |
| Gemma self-control | 1 model × 30 eval prompts | ~0.5 GPU-hours |
| Cross-family transfers (Qwen 7B ↔ Gemma 9B) | 2 pairs × 30 eval prompts | ~1 GPU-hour |
| Geometric comparisons (14B pairs) | 2 pairs × cosine computation | Negligible (CPU) |
| **B2 total GPU** | | **~2 GPU-hours** |

### Combined Phase 2

| Item | Estimate | Cost |
|---|---|---|
| B1 GPU time | ~5.5 hours | ~$6-8 |
| B2 GPU time | ~2 hours | ~$2-3 |
| **Total GPU** | **~7.5 hours** | **$8-11** |
| Human time (B1) | ~1 day | |
| Human time (B2) | ~0.5 day | |
| **Total human time** | **~1.5 days** | |

### Combined Phase 1 + Phase 2 budget

| Phase | GPU hours | Cost | Human days |
|---|---|---|---|
| Phase 1 (Tooling Parity) | ~10 | $10-15 | 1 |
| Phase 2 B1 (Same-Family) | ~5.5 | $6-8 | 1 |
| Phase 2 B2 (Cross-Family) | ~2 | $2-3 | 0.5 |
| **Total** | **~17.5** | **$18-26** | **2.5** |
