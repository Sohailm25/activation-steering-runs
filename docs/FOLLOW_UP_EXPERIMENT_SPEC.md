# Follow-Up Experiment Spec (Post-Initial Paper)

**Date:** 2026-02-18  
**Status:** Planning (do not run new experiments yet)  
**Primary rule:** Complete targeted literature gap pass first.

---

## Context

The initial paper is complete and published as a preprint. External technical feedback highlighted three concrete follow-ups:

1. Tighten Mistral wording around extraction vs propagation (done in paper text)
2. Validate tooling discrepancy (NNsight vs raw hooks) with explicit repeatability protocol
3. Test cross-architecture transfer of refusal directions (family-specific vs partially universal)

This document defines the follow-up plan and decision gates.

---

## Phase 0 (MANDATORY): Targeted Literature Gap Pass

No new runs until this is done.

### Questions to answer

1. Has same-family direction transfer (e.g., 7B → 14B → 30B) already been done for refusal steering?
2. Has cross-family transfer (Qwen ↔ Gemma ↔ other families) already been done for refusal directions?
3. Are there published protocols for tooling parity checks in activation extraction pipelines?
4. What evidence exists for universal vs family-specific refusal representations?

### Search scope

- Focus: 2023–2026
- Priority sources: arXiv, ACL/EMNLP/NeurIPS/ICML/ICLR papers, code repos, mechanistic interpretability writeups
- Must include papers suggested by external feedback:
  - Universal Refusal Circuits Across LLMs
  - Towards universality: studying mechanistic similarity across language model architectures
  - Activation Space Interventions Can Be Transferred Between Large Language Models

### Deliverables

Create all of the following before Phase 1:

1. `research/LITERATURE_GAP_FOLLOWUP.md`
   - What is already done
   - What remains novel
   - Citations with links
2. `research/FOLLOWUP_BIB.md`
   - Curated bibliography for follow-up study
3. `research/FOLLOWUP_DECISION.md`
   - Go/no-go recommendation for each follow-up track
   - Justification and estimated cost

### Decision gate

Proceed only if at least one follow-up track has a clear novelty claim and tractable budget.

---

## Phase 1: Tooling Reproducibility Study (Track A)

**Goal:** Determine whether the NNsight vs raw-hook discrepancy is stable, and whether it is due to extraction path or run noise.

### Core design

- Fixed model: Qwen 7B (same as original discrepancy)
- Fixed prompts, layer, multiplier, generation settings
- Repeated identical runs per method

### Required checks

1. Repeatability within method
   - NNsight extraction x K runs
   - Raw-hook extraction x K runs
   - Report mean/std for coherent refusal
2. Cross-tool tensor-site parity
   - Verify exact extraction site and tensor shape/dtype equivalence
3. Direction stability
   - Cosine similarity within-method and across-method
4. Downstream behavior
   - Coherent/garbled/normal rates under identical evaluation protocol

### Success criterion

Clear separation between:
- method-consistent signal vs run-to-run noise
- extraction-path differences vs uncontrolled implementation drift

---

## Phase 2: Direction Transfer Study (Track B)

**Goal:** Test whether refusal directions transfer across model sizes/families.

### Stage B1 (sanity): same-family transfer

- Qwen: extract at 7B, apply to 14B and 30/32B (and vice versa where feasible)
- Evaluate transfer decay and layer-depth alignment sensitivity

### Stage B2 (main): cross-family transfer

- Qwen → Gemma and Gemma → Qwen
- Optional third family if budget permits
- Matched depth protocol + explicit multiplier policy

### Metrics

- Coherent refusal rate
- Garbled rate
- Normal response rate
- Direction cosine alignment
- Optional language-mode drift flags

### Core hypothesis

- Same-family transfer should be stronger than cross-family transfer
- Cross-family transfer outcome distinguishes family-specific vs partially universal refusal geometry

---

## Budget and sequencing

- Start with Phase 0 (no GPU spend)
- Then Track A (low-cost, high diagnostic value)
- Then Track B1 (same-family) before B2 (cross-family)

If budget gets tight, prioritize:
1) Track A reproducibility
2) Track B1 same-family
3) Track B2 cross-family

---

## Documentation update rule

For all follow-up work:

- Log runs to `results/`
- Keep JSON artifacts as source of truth
- Update paper-facing claims only after repeatability checks pass
- Add caveats immediately when uncertainty remains

---

## Current state snapshot

- Initial paper: complete
- Wording updates from external review: complete
- Follow-up runs: not started
- Next required action: complete targeted literature gap pass
