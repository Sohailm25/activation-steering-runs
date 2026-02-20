# Phase-2 Reproducibility Checklist

## Artifact Registry

All canonical artifacts are in `results/phase2/`. See `GENERATED_TABLES.md` for SHA-256 checksums.

### B1 Same-Family (8 artifacts)
- `b1_qwen-14b_to_qwen-14b_m10.0_20260219_223147.json` (14B self, s42, git=e22f012)
- `b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json` (32B self, s42, git=e22f012)
- `b1_qwen-14b_to_qwen-32b_m15.0_20260219_224941.json` (14B→32B, s42, git=e22f012)
- `b1_qwen-32b_to_qwen-14b_m10.0_20260219_224854.json` (32B→14B, s42, git=e22f012)
- `b1_qwen-14b_to_qwen-14b_m10.0_s43_20260219_225813.json` (14B self, s43, git=3517c67)
- `b1_qwen-32b_to_qwen-32b_m15.0_s43_20260219_225815.json` (32B self, s43, git=3517c67)
- `b1_qwen-14b_to_qwen-32b_m15.0_s43_20260219_225845.json` (14B→32B, s43, git=3517c67)
- `b1_qwen-32b_to_qwen-14b_m10.0_s43_20260219_230041.json` (32B→14B, s43, git=3517c67)

### B1 Calibration (8 artifacts)
- `b1_qwen-14b_to_qwen-14b_m{10,12.5,15,17.5}_20260219_*.json` (git=e22f012)
- `b1_qwen-32b_to_qwen-32b_m{10,12.5,15,17.5}_20260219_*.json` (git=e22f012)

### B2 Cross-Family (3 artifacts)
- `b2_gemma-9b_to_gemma-9b_m25.0_20260219_234111.json` (Gemma self, s42, git=e967e01)
- `b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json` (Q7→G9, s42, git=e967e01)
- `b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` (G9→Q7, s42, git=e967e01)

## Commit Hashes

| Commit | Description |
|--------|-------------|
| e22f012 | B1 script with multiplier support, provenance fix |
| 3517c67 | B1 seed parameter addition |
| e967e01 | B2 cross-family script |

## Seeds

- B1: seed=42 (implicit in pre-seed artifacts) and seed=43 (explicit)
- B2: seed=42 only

## Table Regeneration Command

```bash
cd ~/clawd-experiments/experiments/Inverse\ Scaling\ in\ Activation\ Steering/repo
python3 infrastructure/rebuild_phase2_tables.py
```

Output: `results/phase2/GENERATED_TABLES.md`

Expected content hash (SHA-256, first 16): `a6b2d787f6057d76`

## Bootstrap Parameters

- Resamples: 10,000
- Bootstrap seed: 999
- Method: non-parametric prompt-level resampling
- B1: from aggregate counts (per_prompt not in B1 artifacts)
- B2: from per_prompt labels (quality field)

## Data Contract

- **Canonical numeric source:** `results/phase2/GENERATED_TABLES.md` is the sole source of truth for all Phase-2 table numbers.
- **Precision policy:** CI bounds shown to 3 decimal places. Coherent/garbled/normal/refusal rates to 1 decimal place (percentage). TE point estimates to 2 decimal places.
- **Propagation rule:** All dependent docs (`PHASE2_TRANSFER_RESULTS.md`, `PHASE2_PROMOTION_BUNDLE.md`, `FINAL_HANDOFF_MEMO.md`) must source numeric values from `GENERATED_TABLES.md`. No manual CI values.
- **Regeneration:** Run `python3 infrastructure/rebuild_phase2_tables.py` to regenerate. Compare content hash against expected value in this checklist.
- **Bootstrap method:** Uses per_prompt labels when available (B2 artifacts); falls back to aggregate-count reconstruction (B1 artifacts). Per-prompt preserves prompt identity and produces slightly different CI bounds than aggregate-count.

## Run-End Provenance (B2)

| Artifact | App ID | Completion Timestamp (UTC) | Elapsed |
|----------|--------|---------------------------|---------|
| `b2_gemma-9b_to_gemma-9b_m25.0_20260219_234111.json` | ap-AXGI3XX4Cyhsk6jWdJjAFU | 2026-02-19T23:36:20 UTC | 318s |
| `b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json` | ap-18DrKgSRqF0LJJFN51neYF | 2026-02-20T00:01:00 UTC | 429s |
| `b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json` | ap-j8ZI9tvS2RWE2AZwMUh7Ic | 2026-02-19T23:59:21 UTC | 309s |

Timestamps extracted from artifact `timestamp_utc` fields. No new Modal runs were executed after these artifacts landed.

## Deterministic Decoding Note

All experiments used greedy decoding (temperature=0). Seed variation (42 vs 43) confirms pipeline reproducibility, not sampling robustness. B1 seeds 42 and 43 produced identical results, confirming determinism.
