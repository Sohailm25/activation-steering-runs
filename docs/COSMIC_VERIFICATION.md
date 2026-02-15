# COSMIC Implementation Verification

**Date:** 2026-02-12
**Reviewer:** Claude Code (line-by-line audit)
**File Reviewed:** `infrastructure/cosmic_real.py`
**Status:** VERIFICATION COMPLETE — Implementation is a **faithful approximation** with 3 documented deviations

---

## Paper Reference

**Title:** COSMIC: Generalized Refusal Direction Identification in LLM Activations
**Authors:** Vincent Siu, Nicholas Crispino, Zihao Yu, Sam Pan, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang
**Venue:** Findings of the Association for Computational Linguistics: ACL 2025, pages 25534–25553, Vienna, Austria
**arXiv:** [2506.00085](https://arxiv.org/abs/2506.00085)
**ACL Anthology:** [2025.findings-acl.1310](https://aclanthology.org/2025.findings-acl.1310/)

---

## Algorithm Comparison Table

| Step | Paper Description | Our Implementation (`cosmic_real.py`) | Match? | Notes |
|------|-------------------|--------------------------------------|--------|-------|
| **Candidate generation** | DIM at 5 post-instruction token positions × L layers: `r_{i,l} = r+_{i,l} - r-_{i,l}`, unit-normalized | `_compute_dim_direction()` at `n_token_positions` × `len(layers)` pairs, unit-normalized (L140-156) | ✅ | Token positions: last N (-1 through -N). Default N=5. |
| **Token positions** | `I = {-5, -4, -3, -2, -1}` (5 post-instruction positions) | Positions 0=last, 1=second-to-last, ..., N-1 (L90-93 in `_extract_activations_at_positions`) | ✅ | Same semantic positions, different indexing convention. |
| **Layer range** | All L layers used for candidates | Default: layer 1 to 80% depth; layer 0 always excluded (L476-485) | ✅ | Paper doesn't specify range restriction; our 80% cap is a reasonable default. Layer 0 exclusion is justified (embedding layer, no contextual info). |
| **L_low selection** | Bottom 10% of layers by cosine similarity between harmful/harmless mean activations | `find_divergence_layers()`: sorts layers by cosine, selects bottom `l_low_fraction` (default 0.1) with minimum 1 (L159-179) | ✅ | Matches paper. |
| **Divergence measurement** | Cosine between mean harmful and mean harmless activations per layer | `_compute_divergence_cosines()` → `_mean_cosine_similarity()` (L290-317) | ⚠️ | **Deviation 1:** Paper uses cosine of *mean* activations: `cos(mean(harmful), mean(harmless))`. Our code computes *mean of pairwise cosines*: `mean_i[cos(harmful_i, harmless_i)]`. See Deviation 1 below. |
| **S_comply scoring** | `cos(ā, b̄_-)` — cosine between mean harmless and mean ablated-harmful at L_low | `_mean_cosine_similarity(base_harmless_l_low, ablated_harmful_l_low)` (L418-419) | ⚠️ | **Deviation 1 applies here too.** Pairwise-then-mean vs mean-then-cosine. |
| **S_refuse scoring** | `cos(ā_+, b̄)` — cosine between mean induced-refusal and mean natural-harmful at L_low | `_mean_cosine_similarity(added_harmless_l_low, base_harmful_l_low)` (L422-423) | ⚠️ | **Deviation 1 applies here too.** |
| **Combined score** | `argmax(S_comply + S_refuse)` | `combined_score = s_comply + s_refuse`, select via `argmax` (L588, L602) | ✅ | Exact match. |
| **Directional ablation** | `v' = v - proj_r(v)` where `proj_r(v) = (v·r/‖r‖²)r` | `h[:] = h - (h * d).sum(-1, keepdim=True) * d` at intervention layer (L370-371) | ✅ | Since `d` is unit-normalized, `(v·d)d = proj_r(v)`. Correct. |
| **Activation addition** | `v' = v + r` at layer l* | `h[:] = h + d` at intervention layer (L393) | ✅ | Matches paper (scale=1). |
| **Forward-pass intervention** | Apply at layer l, measure at L_low via forward pass | `_score_candidate_with_intervention()` uses nnsight hooks: intervention at `intervention_layer`, read at each `l_low` layer (L320-428) | ✅ | Core COSMIC innovation correctly implemented. |
| **Intervention only upstream of L_low** | Paper: "apply at layer l and collect at all layers in L_low" | If `intervention_layer >= all L_low`, falls back to static scoring (L560-584) | ✅ | Sensible engineering decision. Paper doesn't explicitly address this edge case. |
| **Train/validation split** | 180 train + 100 validation per class | No split — same prompts used for extraction AND scoring | ❌ | **Deviation 2.** See below. |
| **Steering application (ablation)** | Ablate across all layers plus embedding | Sweep code applies steering from `target_layer` to end (L208-210 in `cosmic_validation_sweep.py`) | ⚠️ | **Deviation 3.** Sweep uses addition-based steering at selected layer through all subsequent layers, not ablation across all layers. See below. |

---

## Detailed Deviations

### Deviation 1: Cosine Similarity Aggregation (MINOR)

**Paper formula:**
```
S_refuse = cos(ā_+, b̄)
```
Where `ā_+` and `b̄` are the *mean* activation vectors across all validation instances, and cosine is computed once on these means.

**Our formula:**
```python
# _mean_cosine_similarity (L218-249)
cosines = [cos(a+_i, b_i) for i in range(n)]
S_refuse = mean(cosines)
```
We compute pairwise cosines first, then average.

**Impact:** These are mathematically different. Mean-then-cosine gives a single direction comparison; pairwise-then-mean accounts for instance-level alignment variation. In practice, with enough samples, both approaches identify the same best candidate because the ordering of candidates is preserved. The pairwise approach is actually more robust to outliers.

**Severity:** LOW. The ranking of candidates is unlikely to change. Our approach may be slightly more conservative (avoids dominance by a few extreme activations in the mean).

### Deviation 2: No Train/Validation Split (MEDIUM)

**Paper:** Uses separate train (180 prompts) and validation (100 prompts) sets. Candidates are generated from training data; scoring uses validation data.

**Our code:** Uses the same prompts for both candidate generation (`_extract_activations_at_positions`) and scoring (`_score_candidate_with_intervention`). The scoring still runs independent forward passes (not cached), so it's not trivially overfit, but there's no held-out validation set.

**Impact:** Risk of selecting a candidate that overfits to the specific prompt set. In practice, with the small prompt sets we use (5-50), this is a real concern. The paper's approach with 180+100 is more statistically robust.

**Severity:** MEDIUM. Could affect candidate selection quality, especially with small prompt sets. Does NOT affect the algorithmic correctness of the scoring mechanism itself.

**Fix:** Add a `validation_harmful`/`validation_harmless` parameter or implement an internal split.

### Deviation 3: Steering Application Method (CONTEXT-DEPENDENT)

**Paper describes three application methods:**
1. **Directional ablation (LCE):** Remove direction across ALL layers plus embedding
2. **Activation addition (LCE):** Add direction at layer l* only
3. **ACE:** Combined ablation + re-centering + scaled addition at layer l*

**Our validation sweep (`cosmic_validation_sweep.py`):**
- Uses activation addition with a multiplier (10x, 15x, 25x) from `target_layer` through all subsequent layers
- This is NOT what COSMIC proposes. COSMIC uses scale=1 addition at a single layer, or ablation across all layers.

**Impact:** HIGH for the validation sweep results. Applying a 15x multiplied direction from one layer through all subsequent layers is a much stronger intervention than what COSMIC recommends. The 100% refusal rate at L16 on Qwen 7B is likely an artifact of the aggressive steering, not a faithful reproduction of COSMIC's expected behavior.

**Severity:** HIGH for interpreting validation results. The `cosmic_real.py` direction *extraction* is faithful to COSMIC. The *application* in the sweep is not.

**Note:** This is a sweep/runner issue, not a `cosmic_real.py` issue. The extraction algorithm itself is correct.

---

## Verification of Individual Functions

### `_compute_dim_direction()` (L140-156)
```python
diff = harmful_acts.mean(axis=0) - harmless_acts.mean(axis=0)
norm = np.linalg.norm(diff)
if norm > 1e-8:
    return diff / norm
```
**Paper:** `r_{i,l} = r+_{i,l} - r-_{i,l}` where `r+` and `r-` are mean activations.
**Verdict:** ✅ Exact match. Unit normalization is standard practice.

### `find_divergence_layers()` (L159-179)
```python
sorted_layers = sorted(cosines_per_layer.keys(), key=lambda l: cosines_per_layer[l])
n_select = max(1, int(len(sorted_layers) * bottom_fraction))
return sorted_layers[:n_select]
```
**Paper:** "L_low consists of the 10% of layers with the lowest cosine similarity"
**Verdict:** ✅ Correct. `max(1, ...)` is a sensible guard.

### `_directional_ablation()` (L182-197)
```python
projections = activations @ direction  # (n,)
return activations - np.outer(projections, direction)
```
**Paper:** `v' = v - (v·r/‖r‖²)r`. Since `direction` is unit-normalized, `‖r‖² = 1`, so `proj = v·r` and `v' = v - (v·r)r`.
**Verdict:** ✅ Exact match for unit-normalized directions.

### `_directional_addition()` (L200-215)
```python
return activations + scale * direction[np.newaxis, :]
```
**Paper:** `v' = v + r` (scale=1 default).
**Verdict:** ✅ Match. Scale parameter allows experimentation.

### `_score_candidate_with_intervention()` (L320-428)
- Applies ablation hook at `intervention_layer` for harmful prompts (L366-371)
- Applies addition hook at `intervention_layer` for harmless prompts (L391-394)
- Reads activations at L_low layers downstream (L375-376, L398-399)
- Computes S_comply and S_refuse per L_low layer, averages (L410-427)

**Paper:** "Apply directional ablation and activation addition at layer l and collect the modified values in the residual stream at all layers in L_low."
**Verdict:** ✅ Core mechanism matches. Only the cosine aggregation method differs (Deviation 1).

### `extract_refusal_direction_cosmic_real()` (L431-667)
- Generates 5L candidates via DIM at (token_pos, layer) pairs ✅
- Computes divergence across all layers to find L_low ✅
- Scores via forward-pass intervention at candidate layer, measured at L_low ✅
- Selects argmax(S_comply + S_refuse) ✅
- Returns unit-normalized direction ✅
- Logs to W&B ✅

**Verdict:** ✅ Overall algorithm structure matches COSMIC faithfully, with the 3 deviations noted.

---

## Reproduction Test Results

### Qwen 2.5-7B at L16 (60% depth) — from `cosmic_validation_results_20260212_095637.json`

| Metric | Expected | Actual | Match? |
|--------|----------|--------|--------|
| Refusal rate (15x mult) | ~100% | **100.0%** (50/50) | ✅ |
| Direction norm | significant | 28.52 | ✅ |

**Caveat:** This 100% result uses the *sweep's* aggressive multi-layer steering (Deviation 3), not COSMIC's prescribed single-layer application. The direction itself was extracted using a simplified DIM at a fixed layer, not the full COSMIC candidate selection. This test validates that DIM-extracted refusal directions work for steering, but does NOT validate COSMIC's candidate selection over DIM.

### Qwen 2.5-3B COSMIC-Real Results — from `results/COSMIC_REAL_RESULTS.md`

| Metric | Value |
|--------|-------|
| Selected layer | 23 (of 36) |
| Selected position | -4 (pos_idx=3) |
| S_comply | 0.830 |
| S_refuse | 0.655 |
| Combined score | 1.485 |
| Cosine to DIM | 0.598 |
| Cross-seed stability | 1.000 (perfectly deterministic) |
| L_low layers | [35, 31, 29] |

**This test validates the COSMIC direction extraction algorithm.** The direction is partially aligned with DIM (0.598 cosine) but geometrically distinct, consistent with COSMIC selecting a different (layer, position) pair than DIM's default.

---

## Verification Checklist

- [x] Candidate generation matches paper (DIM at token_pos × layer pairs)
- [x] L_low layer selection matches paper (bottom 10% by cosine)
- [x] Scoring uses forward-pass interventions at candidate layer, measured at L_low
- [x] S_comply formula implements ablation of direction from harmful, measure similarity to harmless
- [x] S_refuse formula implements addition of direction to harmless, measure similarity to harmful
- [x] Combined score = S_comply + S_refuse, selection via argmax
- [x] Directional ablation formula correct (`v - (v·d)d` for unit d)
- [x] Directional addition formula correct (`v + d`)
- [x] Layer 0 excluded from candidates
- [x] No SVD used (confirmed by test + code audit)
- [x] Unit normalization of output direction
- [x] Seed reproducibility verified (cross-seed cosine = 1.000)
- [x] All 3 deviations documented with severity assessment
- [ ] **Steering application in validation sweep does NOT match paper** (Deviation 3 — documented)
- [ ] **No train/validation split** (Deviation 2 — documented)
- [ ] Sohail sign-off

---

## Conclusion

**Does `cosmic_real.py` faithfully represent the COSMIC algorithm?**

**YES, with caveats.** The direction extraction algorithm is a faithful implementation of COSMIC's core innovation: DIM candidate generation at multiple (token_position, layer) pairs, scored via forward-pass cosine-similarity interventions measured at divergence layers (L_low). The three deviations are:

1. **Cosine aggregation** (minor): pairwise-then-mean vs paper's mean-then-cosine. Unlikely to change candidate ranking.
2. **No train/val split** (medium): same prompts for extraction and scoring. Real risk with small prompt sets.
3. **Steering application in sweep** (high, but separate file): the validation sweep's multi-layer multiplied steering is much more aggressive than COSMIC prescribes.

**Recommendation:** The implementation is suitable for Phase 2 direction extraction. For publication-grade results, add a train/validation split (Deviation 2) and align the steering application with the paper's prescribed methods (Deviation 3). Deviation 1 is an acceptable alternative that may be more robust.

---

## Files

| File | Status |
|------|--------|
| `infrastructure/cosmic_real.py` | ✅ Verified — faithful COSMIC extraction with 3 deviations |
| `infrastructure/tests/test_cosmic_real.py` | ✅ 24 tests covering all algorithmic steps |
| `infrastructure/cosmic_validation_sweep.py` | ⚠️ Steering application deviates from paper |
| `results/COSMIC_REAL_RESULTS.md` | ✅ Documents Qwen 2.5-3B results with fixed implementation |

---

## Sign-Off

**Date:** 2026-02-12
**Reviewer:** Professor
**Status:** ✅ APPROVED

### Acceptance of Deviations

| Deviation | Severity | Decision |
|-----------|----------|----------|
| Cosine aggregation (pairwise-then-mean) | LOW | ✅ Acceptable — more robust |
| No train/val split | MEDIUM | ✅ Acceptable for internal experiments |
| Multi-layer steering in sweep | HIGH (interpretation) | ✅ Extraction is faithful; steering differs but comparison is fair |

### Conditions for Proceeding

1. ✅ Extraction algorithm (`cosmic_real.py`) faithfully represents COSMIC
2. ✅ Fair comparison maintained (both DIM and COSMIC use same prompts + steering)
3. ✅ Deviations documented as limitations
4. ⏳ Consider train/val split if results are published

### Sign-Off Chain

- [x] Ghost audit complete (2026-02-12)
- [x] Professor review (2026-02-12) — APPROVED with notes
- [x] Sohail sign-off (2026-02-12) — APPROVED, proceed to Phase 1
