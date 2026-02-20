# Phase-2 Promotion Bundle — Paper-Level Bounded Claims

**Status:** APPROVED by Professor (2026-02-19 21:34 CST, msg `1771558460.715059`)
**Compute state:** Frozen. No further runs.

---

## 1. Final Bounded Claims (ready-to-paste)

In this protocol and tested pairs, same-family activation steering transfer remains strong while cross-family transfer collapses despite matched hidden dimensionality. Specifically:

1. **Same-family transfer (Qwen 14B↔32B):** DIM refusal directions transfer with TE ≥ 1.0. The 14B-extracted direction steers 32B at 100% coherent (TE = 1.25, 95% CI [1.071, 1.579]), exceeding 32B's own self-control baseline of 80%. The reverse direction (32B→14B) has point estimate TE = 1.00, though the CI [0.900, 1.111] spans both sides of 1.0, so exact equivalence is not statistically established. Cross-cosine between directions is 0.324.

2. **Cross-family transfer (Qwen-7B↔Gemma-9B):** Despite identical hidden_dim = 3584, cross-family transfer fails. Q7→G9 achieves only 16.7% coherent (TE = 0.17, 95% CI [0.036, 0.321]); G9→Q7 achieves 3.3% (TE = 0.03, 95% CI [0.000, 0.100]). Cross-cosine is 0.019 — near-orthogonal.

3. **Cross-cosine as an observation:** In this 2-pair dataset, cross-cosine alignment (0.324 for same-family vs 0.019 for cross-family) co-varies with transfer success. This is suggestive but not evidence of a calibrated predictor or general threshold.

These findings are suggestive, not universal; broader family coverage is future work.

---

## 2. Canonical Artifact Paths

### B1 — Same-Family Transfer

| Artifact | Role |
|----------|------|
| `results/phase2/b1_qwen-14b_to_qwen-14b_m10.0_20260219_223147.json` | 14B self-control (s42) |
| `results/phase2/b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json` | 32B self-control (s42) |
| `results/phase2/b1_qwen-14b_to_qwen-32b_m15.0_20260219_224941.json` | 14B→32B transfer (s42) |
| `results/phase2/b1_qwen-32b_to_qwen-14b_m10.0_20260219_224854.json` | 32B→14B transfer (s42) |
| `results/phase2/b1_qwen-14b_to_qwen-14b_m10.0_s43_20260219_225813.json` | 14B self-control (s43) |
| `results/phase2/b1_qwen-32b_to_qwen-32b_m15.0_s43_20260219_225815.json` | 32B self-control (s43) |
| `results/phase2/b1_qwen-14b_to_qwen-32b_m15.0_s43_20260219_225845.json` | 14B→32B transfer (s43) |
| `results/phase2/b1_qwen-32b_to_qwen-14b_m10.0_s43_20260219_230041.json` | 32B→14B transfer (s43) |

### B2 — Cross-Family Transfer

| Artifact | Role |
|----------|------|
| `results/phase2/b2_gemma-9b_to_gemma-9b_m25.0_20260219_234111.json` | Gemma self-control (s42) |
| `results/phase2/b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json` | Q7→G9 transfer (s42) |
| `results/phase2/b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` | G9→Q7 transfer (s42) |

### Consolidated Results

| Doc | Role |
|-----|------|
| `results/phase2/PHASE2_TRANSFER_RESULTS.md` | Full results with CIs, provenance, errata |
| `results/phase2/PHASE2_PROMOTION_BUNDLE.md` | This file — paper-level claims |

---

## 3. Limitations and Caveats

- **Scope:** One same-family pair (Qwen 14B↔32B) and one cross-family pair (Qwen-7B↔Gemma-9B). No claim of universality.
- **Decoding:** Greedy/deterministic. Seed variation confirms reproducibility, not sampling robustness. Temperature > 0 or prompt variation would be needed for the latter.
- **Sample size:** n = 30 prompts per condition. Sufficient for detecting large effects (TE > 0.5 vs TE < 0.2) but not for precise rate estimation.
- **B2 replication:** Seed = 42 only. B1 has 2 seeds (42, 43) with identical results.
- **B1 per-prompt data:** Not recorded in B1 artifacts (added in B2 script). CIs reconstructed from aggregate counts.
- **norm_ratio:** Non-diagnostic due to unit normalization in DIM extraction.
- **Cross-cosine predictiveness:** 2 data points. Not a calibrated predictor.
- **TE > 1.0 interpretation:** 14B→32B exceeding 32B self-control is partly mechanical — 32B self-control is only 80%, making it easier to exceed. Not evidence of inverse-scaling.

---

## 4. What We Can Safely Say / Cannot Say

### Can say:
- Same-family same-dim transfer can be strong for the tested Qwen 14B↔32B pair.
- Cross-family transfer can fail despite matched hidden_dim, for the tested Q7↔G9 pair.
- Cross-cosine alignment co-varies with transfer success in this dataset.
- In this protocol and tested pairs, same-family transfer remains strong while cross-family transfer collapses.

### Cannot say:
- Transfer generalizes across all model families or sizes.
- Cross-cosine is a reliable predictor of transfer success (2 data points).
- TE > 1.0 implies inverse-scaling or surprising emergent properties.
- These results replicate under non-deterministic decoding.
- Refusal geometry is always family-specific (tested on one cross-family pair only).

---

## 5. Provenance Table

| Artifact | App ID | git_commit | seed | multiplier | n |
|----------|--------|-----------|------|-----------|---|
| b1_14b_self (s42) | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 | 10.0 | 30 |
| b1_32b_self (s42) | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 | 15.0 | 30 |
| b1_14→32 (s42) | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 | 15.0 | 30 |
| b1_32→14 (s42) | ap-E05B8KuxAsBfwaK5flvB2r | e22f012 | 42 | 10.0 | 30 |
| b1_14b_self (s43) | ap-s0OVqJ7h9fS3nar4Mf9CNE | 3517c67 | 43 | 10.0 | 30 |
| b1_32b_self (s43) | ap-6LEDkJsw1JLx0wltnS4XsG | 3517c67 | 43 | 15.0 | 30 |
| b1_14→32 (s43) | ap-FLd8yQR9IzIb5lxtOnv6vb | 3517c67 | 43 | 15.0 | 30 |
| b1_32→14 (s43) | ap-GnenPYnxnwREi9G7LRitrS | 3517c67 | 43 | 10.0 | 30 |
| b2_gemma_self | ap-AXGI3XX4Cyhsk6jWdJjAFU | e967e01 | 42 | 25.0 | 30 |
| b2_q7→g9 | ap-18DrKgSRqF0LJJFN51neYF | e967e01 | 42 | 25.0 | 30 |
| b2_g9→q7 | ap-j8ZI9tvS2RWE2AZwMUh7Ic | e967e01 | 42 | 15.0 | 30 |

### Scripts
| Script | git_commit |
|--------|-----------|
| `infrastructure/v4_transfer_b1.py` | 3517c67 |
| `infrastructure/v4_transfer_b2.py` | e967e01 |
