# Phase-2 Transfer Results — Consolidated Artifact

**Date:** 2026-02-19/20
**Author:** Ghost (experiment agent)
**Status:** APPROVED for paper-level bounded claims (Professor 2026-02-19 21:34 CST)
**Numeric source:** All table values sourced from `GENERATED_TABLES.md` via `infrastructure/rebuild_phase2_tables.py`

---

## 1. B1 — Same-Family Transfer (Qwen 14B ↔ 32B)

### 1.1 Calibrated Self-Controls

Multipliers locked by objective rule: min garbled → max coherent → refusal not collapsed.

| Model | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |
|-------|-----------|----------|---------|--------|---------|-------------------|
| Qwen-14B | 10.0 | 96.7% | 0% | 3.3% | 96.7% | [90.0%, 100.0%] |
| Qwen-32B | 15.0 | 80.0% | 0% | 20.0% | 80.0% | [63.3%, 93.3%] |

Both seeds (42, 43) produced identical results (deterministic decoding pipeline).

### 1.2 Transfer Results

| Transfer | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |
|----------|-----------|----------|---------|--------|---------|-------------------|
| 14B→32B | 15.0 | 100.0% | 0% | 0% | 100.0% | [100.0%, 100.0%] |
| 32B→14B | 10.0 | 96.7% | 0% | 3.3% | 96.7% | [90.0%, 100.0%] |

Both seeds (42, 43) produced identical results.

### 1.3 Transfer Efficiency & Geometry

| Transfer | TE | TE 95% CI | Cross-cosine |
|----------|-----|-----------|-------------|
| 14B→32B | 1.25 | [1.071, 1.579] | 0.324 |
| 32B→14B | 1.00 | [0.900, 1.111] | 0.324 |

**Notes:**
- TE(14→32) > 1.0 means the 14B direction steers 32B *better* than 32B's own direction steers itself. The CI lower bound (1.071) excludes 1.0.
- TE(32→14) point estimate is 1.00, but the CI [0.900, 1.111] spans both sides of 1.0, so exact equivalence with self-control is not statistically established.

---

## 2. B2 — Cross-Family Transfer (Qwen-7B ↔ Gemma-9B)

### 2.1 Baselines

| Model | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |
|-------|-----------|----------|---------|--------|---------|-------------------|
| Gemma-9B (self) | 25.0 | 96.7% | 0% | 3.3% | 96.7% | [90.0%, 100.0%] |
| Qwen-7B (self, Phase-1) | 15.0 | 100.0% | 0% | 0% | 100.0% | [100.0%, 100.0%] |

### 2.2 Cross-Family Transfer Results

| Transfer | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |
|----------|-----------|----------|---------|--------|---------|-------------------|
| Q7→G9 | 25.0 | 16.7% | 0% | 83.3% | 16.7% | [3.3%, 30.0%] |
| G9→Q7 | 15.0 | 3.3% | 6.7% | 90.0% | 3.3% | [0.0%, 10.0%] |

### 2.3 Transfer Efficiency & Geometry

| Transfer | TE | TE 95% CI | Cross-cosine |
|----------|-----|-----------|-------------|
| Q7→G9 | 0.17 | [0.036, 0.321] | 0.019 |
| G9→Q7 | 0.03 | [0.000, 0.100] | 0.019 |

Seed=42 only. No seed=43 replication for B2.

---

## 3. Combined Comparison Table

| Pair | Family | dim_match | Cross-cos | Coherent (transfer) | TE | TE 95% CI |
|------|--------|-----------|-----------|--------------------|----|-----------|
| 14B→32B | same (Qwen) | ✓ (5120) | 0.324 | 100.0% | 1.25 | [1.071, 1.579] |
| 32B→14B | same (Qwen) | ✓ (5120) | 0.324 | 96.7% | 1.00 | [0.900, 1.111] |
| Q7→G9 | cross (Qwen→Gemma) | ✓ (3584) | 0.019 | 16.7% | 0.17 | [0.036, 0.321] |
| G9→Q7 | cross (Gemma→Qwen) | ✓ (3584) | 0.019 | 3.3% | 0.03 | [0.000, 0.100] |

---

## 4. Bootstrap CI Methodology

- **Method:** Non-parametric prompt-level bootstrap, 10,000 resamples, seed=999
- **B2:** Used `per_prompt` labels from artifacts (quality field: coherent/normal/garbled)
- **B1:** Reconstructed prompt-level labels from aggregate counts (k coherent out of n=30); per_prompt data not recorded in B1 artifacts (limitation)
- **TE CI:** Ratio bootstrap — TE = coherent_transfer / coherent_self, computed per resample
- **Note:** B1 CIs may slightly understate true variability since prompt identity is not preserved

---

## 5. Bounded Conclusions

### What we can claim (bounded):

1. **Same-family, same-dim transfer can be strong for this tested pair.** Qwen 14B↔32B transfer achieves TE ≥ 1.0 with cross-cosine = 0.324. The 14B direction steers 32B at TE=1.25 [1.071, 1.579], exceeding self-control performance. Tested on n=30 prompts, 2 seeds, deterministic decoding.

2. **Cross-family transfer fails despite matched hidden_dim, for this tested pair.** Qwen-7B→Gemma-9B and Gemma-9B→Qwen-7B both produce TE < 0.2 with cross-cosine = 0.019. Dimension matching is necessary but not sufficient for transfer. Tested on n=30 prompts, 1 seed.

3. **Cross-cosine alignment appears predictive of transfer success in this dataset.** Same-family (cos=0.324) → TE ≥ 1.0. Cross-family (cos=0.019) → TE ≤ 0.17. However, we have only 2 data points (1 same-family pair, 1 cross-family pair), so we cannot establish a general threshold or functional relationship.

### What we cannot claim:

- **No universality claim.** We tested one same-family pair and one cross-family pair. We do not know if these results generalize to other model families, other sizes, or other steering directions.
- **No inverse-scaling claim.** The TE>1.0 for 14B→32B is interesting but has a mundane explanation: 32B's self-control is weaker (80% vs 96.7%), so beating it is easier.
- **No causal claim about cross-cosine.** Correlation between cosine and TE across 2 data points is not evidence of a causal or predictive relationship.
- **norm_ratio is non-diagnostic.** Both B1 and B2 show norm_ratio ≈ 1.0 due to unit normalization in the DIM extraction pipeline. This metric does not distinguish successful from failed transfer.

### Caveats:

- Deterministic decoding (greedy) means seed variation tests pipeline reproducibility, not sampling robustness
- n=30 prompts is sufficient for detecting large effects but not for precise rate estimation
- B2 has seed=42 only; B1 has seed=42+43 (both identical due to determinism)
- All experiments use DIM extraction at a single layer per model; other layers untested

---

## 6. Provenance & Reproducibility

### B1 Artifacts

| Artifact | App ID | git_commit | seed | multiplier | n |
|----------|--------|-----------|------|-----------|---|
| `b1_qwen-14b_to_qwen-14b_m10.0_20260219_223147.json` | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 (implicit) | 10.0 | 30 |
| `b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json` | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 (implicit) | 15.0 | 30 |
| `b1_qwen-14b_to_qwen-32b_m15.0_20260219_224941.json` | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 (implicit) | 15.0 | 30 |
| `b1_qwen-32b_to_qwen-14b_m10.0_20260219_224854.json` | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 (implicit) | 10.0 | 30 |
| `b1_qwen-14b_to_qwen-14b_m10.0_s43_20260219_225813.json` | ap-s0OVqJ7h9fS3nar4Mf9CNE | 3517c67 | 43 | 10.0 | 30 |
| `b1_qwen-32b_to_qwen-32b_m15.0_s43_20260219_225815.json` | ap-6LEDkJsw1JLx0wltnS4XsG | 3517c67 | 43 | 15.0 | 30 |
| `b1_qwen-14b_to_qwen-32b_m15.0_s43_20260219_225845.json` | ap-FLd8yQR9IzIb5lxtOnv6vb | 3517c67 | 43 | 15.0 | 30 |
| `b1_qwen-32b_to_qwen-14b_m10.0_s43_20260219_230041.json` | ap-GnenPYnxnwREi9G7LRitrS | 3517c67 | 43 | 10.0 | 30 |

### B1 Calibration Artifacts (multiplier sweep)

| Artifact | git_commit | multiplier | Coherent | Garbled |
|----------|-----------|-----------|----------|---------|
| `b1_qwen-14b_to_qwen-14b_m10.0_20260219_222846.json` | e22f012 | 10.0 | 96.7% | 0% |
| `b1_qwen-14b_to_qwen-14b_m12.5_20260219_223352.json` | e22f012 | 12.5 | 93.3% | 6.7% |
| `b1_qwen-14b_to_qwen-14b_m15.0_20260219_223338.json` | e22f012 | 15.0 | 73.3% | 26.7% |
| `b1_qwen-14b_to_qwen-14b_m17.5_20260219_223442.json` | e22f012 | 17.5 | 40.0% | 60.0% |
| `b1_qwen-32b_to_qwen-32b_m10.0_20260219_223555.json` | e22f012 | 10.0 | 36.7% | 0% |
| `b1_qwen-32b_to_qwen-32b_m12.5_20260219_223314.json` | e22f012 | 12.5 | 63.3% | 0% |
| `b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json` | e22f012 | 15.0 | 80.0% | 0% |
| `b1_qwen-32b_to_qwen-32b_m17.5_20260219_223141.json` | e22f012 | 17.5 | 56.7% | 0% |

### B2 Artifacts

| Artifact | App ID | git_commit | seed | multiplier | n |
|----------|--------|-----------|------|-----------|---|
| `b2_gemma-9b_to_gemma-9b_m25.0_20260219_234111.json` | ap-AXGI3XX4Cyhsk6jWdJjAFU | e967e01 | 42 | 25.0 | 30 |
| `b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json` | ap-18DrKgSRqF0LJJFN51neYF | e967e01 | 42 | 25.0 | 30 |
| `b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` | ap-j8ZI9tvS2RWE2AZwMUh7Ic | e967e01 | 42 | 15.0 | 30 |

### Scripts

| Script | git_commit | Description |
|--------|-----------|-------------|
| `infrastructure/v4_transfer_b1.py` | 3517c67 | Same-family transfer with --seed support |
| `infrastructure/v4_transfer_b2.py` | e967e01 | Cross-family transfer |

---

## Errata

### E1: B1 calibration table transcription error (corrected 2026-02-19 21:35 CST)

**What was wrong:** Two rows in the B1 Calibration Artifacts table (Section 6) had incorrect coherent rates:
- `b1_qwen-32b_to_qwen-32b_m10.0_20260219_223555.json`: reported 76.7%, actual 36.7%
- `b1_qwen-32b_to_qwen-32b_m12.5_20260219_223314.json`: reported 76.7%, actual 63.3%

**Cause:** Manual transcription error when building the provenance table. The values were swapped/misread from source artifacts.

**Downstream impact:** No downstream claim changed. These calibration rows are informational context for the multiplier selection process. The locked multipliers (14B=10.0, 32B=15.0) were selected from correct data at calibration time. All transfer results, TEs, CIs, and conclusions are unaffected — they derive from the transfer and self-control artifacts, not the calibration table.

**Correction:** Values now match source JSON exactly. Verified by direct `json.load()` from artifacts.

---

### Known Issues

- B1 seed=42 artifacts lack explicit `seed` field (pre-dates seed parameter addition)
- B1 artifacts lack `per_prompt` data (added in B2 script only)
- Duplicate B1 14B m=10 artifact exists (`222846` and `223147`, identical metrics)
- Earlier B2 Gemma self artifact (`231226`) exists on volume — from a failed/duplicate run; canonical is `234111`
