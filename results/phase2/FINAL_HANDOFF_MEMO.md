# Phase-2 Final Handoff Memo

**Date:** 2026-02-20
**Author:** Ghost (experiment agent)
**Professor approval:** 2026-02-19 21:34 CST (APPROVED for paper-level bounded claims)

---

## What Is Finalized

### Phase-1: Tooling Parity (FROZEN)
- 3-tool parity (nnsight, hooks, pyvene) established for Qwen-7B and Gemma-9B
- Closeout doc: `results/phase1/PHASE1_TOOLING_PARITY_CLOSEOUT.md`

### Phase-2: Activation Steering Transfer (FROZEN)
- **B1 same-family (Qwen 14B↔32B):** TE ≥ 1.0, cross-cos = 0.324, 2 seeds
- **B2 cross-family (Qwen-7B↔Gemma-9B):** TE ≤ 0.17, cross-cos = 0.019, 1 seed
- Consolidated results: `results/phase2/PHASE2_TRANSFER_RESULTS.md`
- Promotion bundle: `results/phase2/PHASE2_PROMOTION_BUNDLE.md`
- Generated tables: `results/phase2/GENERATED_TABLES.md`
- Reproducibility script: `infrastructure/rebuild_phase2_tables.py`
- Reproducibility checklist: `results/phase2/REPRO_CHECKLIST.md`

### Approved Bounded Claims
1. Same-family same-dim transfer can be strong for the tested Qwen 14B↔32B pair.
2. Cross-family transfer can fail despite matched hidden_dim, for the tested Q7↔G9 pair.
3. Cross-cosine alignment co-varies with transfer success in this 2-pair dataset — suggestive, not universal.

---

## What Remains Optional Future Work

1. **Broader cross-family validation:** Test additional family pairs (e.g., Llama↔Qwen, Llama↔Gemma) before any generalized predictor claim about cross-cosine.
2. **Sampling robustness:** Repeat key conditions with temperature > 0 or varied prompt sets to test beyond deterministic decoding.
3. **Additional same-family pairs:** Test 3B/7B↔14B/32B within Qwen to assess whether TE ≥ 1.0 holds across all same-family pairs or is specific to 14B↔32B.
4. **Random-direction control:** Inject a random unit vector (same norm) to establish baseline perturbation refusal rate, confirming the B2 residual ~16% coherent is not direction-specific.
5. **Per-prompt data for B1:** Rerun B1 conditions with per_prompt logging for richer bootstrap analysis (low priority — aggregate CIs are sufficient for current claims).

---

## Recommended Next Research Phase

**Option A: Manuscript integration.** Write up Phase-1 + Phase-2 as a methods paper on activation steering transfer, with the current bounded claims. No additional compute needed.

**Option B: Broader cross-family validation.** Expand B2 to 2-3 more family pairs. This would strengthen or weaken the cross-cosine observation and could elevate it from "suggestive" to "supported" (or reveal it as coincidental). Estimated cost: ~$15-25 for 3 additional family pairs.

These are not mutually exclusive. A could proceed while B is designed.

---

## Explicit Statement

Compute remained frozen during the entire packaging phase. No new Modal runs were executed after the B2 artifacts landed. All tables in the final package are machine-generated from canonical JSON artifacts via `infrastructure/rebuild_phase2_tables.py`.
