# Manuscript Evidence Map — Phase-2 Transfer Results

Maps each manuscript claim in §8 to its exact source artifact and numeric value.

All values must match `GENERATED_TABLES.md` (canonical source).

---

## §8.2 Same-Family Transfer

| Claim | Value | Source Artifact | Field |
|-------|-------|-----------------|-------|
| 14B self-control coherent | 96.7% | `b1_qwen-14b_to_qwen-14b_m10.0_20260219_223147.json` | eval.coherent_rate |
| 32B self-control coherent | 80.0% | `b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json` | eval.coherent_rate |
| 14B→32B coherent | 100.0% | `b1_qwen-14b_to_qwen-32b_m15.0_20260219_224941.json` | eval.coherent_rate |
| 14B→32B TE | 1.25 | Computed: 100.0/80.0 | |
| 14B→32B TE CI | [1.071, 1.579] | Bootstrap (10k, seed=999) | |
| 14B→32B cross-cosine | 0.324 | `b1_qwen-14b_to_qwen-32b_m15.0_20260219_224941.json` | geometric.cross_cosine |
| 32B→14B coherent | 96.7% | `b1_qwen-32b_to_qwen-14b_m10.0_20260219_224854.json` | eval.coherent_rate |
| 32B→14B TE | 1.00 | Computed: 96.7/96.7 | |
| 32B→14B TE CI | [0.900, 1.111] | Bootstrap (10k, seed=999) | |
| 32B→14B cross-cosine | 0.324 | `b1_qwen-32b_to_qwen-14b_m10.0_20260219_224854.json` | geometric.cross_cosine |
| "CI lower bound excludes 1.0" (14B→32B) | 1.071 > 1.0 | Bootstrap output | |
| "CI spans both sides" (32B→14B) | 0.900 < 1.0 < 1.111 | Bootstrap output | |

## §8.3 Cross-Family Transfer

| Claim | Value | Source Artifact | Field |
|-------|-------|-----------------|-------|
| Gemma 9B self-control coherent | 96.7% | `b2_gemma-9b_to_gemma-9b_m25.0_20260219_234111.json` | eval.coherent_rate |
| Qwen 7B self-control coherent | 100.0% | Phase-1 artifact (Qwen-7B parity) | |
| Q7→G9 coherent | 16.7% | `b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json` | eval.coherent_rate |
| Q7→G9 TE | 0.17 | Computed: 16.7/96.7 | |
| Q7→G9 TE CI | [0.036, 0.321] | Bootstrap (10k, seed=999, per_prompt) | |
| Q7→G9 cross-cosine | 0.019 | `b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json` | geometric.cross_cosine |
| G9→Q7 coherent | 3.3% | `b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` | eval.coherent_rate |
| G9→Q7 TE | 0.03 | Computed: 3.3/100.0 | |
| G9→Q7 TE CI | [0.000, 0.100] | Bootstrap (10k, seed=999, per_prompt) | |
| G9→Q7 cross-cosine | 0.019 | `b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` | geometric.cross_cosine |
| G9→Q7 garbled | 6.7% | `b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` | eval.garbled_rate |

## §8.4 Interpretation Claims

| Claim | Evidence basis | Limitation |
|-------|---------------|-----------|
| "matching hidden_dim necessary but not sufficient" | Both pairs have matched dims; only same-family transfers | 2 data points |
| "cross-cosine co-varies with transfer" | 0.324→TE≥1.0 vs 0.019→TE≤0.17 | 2 data points, not causal |
| "TE>1.0 partly mechanical" | 32B self=80% makes it easier to exceed | Acknowledged in text |

## Quick Tour (§2) Addition

| Claim | Value | Maps to |
|-------|-------|---------|
| "TE=1.25, CI [1.071, 1.579]" | 14B→32B | §8.2 table row |
| "TE=0.17, CI [0.036, 0.321]" | Q7→G9 | §8.3 table row |
| "0.324 same-family vs 0.019 cross-family" | Cross-cosines | §8.2, §8.3 |
