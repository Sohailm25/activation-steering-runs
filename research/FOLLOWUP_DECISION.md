# Follow-Up Decision Gate

**Status:** COMPLETE — literature gap analysis done 2026-02-18

---

## Gate criteria

Proceed to runs only if all are satisfied:

- [x] Novelty is explicit for at least one track (A or B)
- [x] Direct duplication risk is low or clearly scoped
- [ ] Minimal experiment design is concrete and falsifiable
- [ ] Budget estimate is within approved range
- [ ] Measurement plan includes reproducibility checks

(Last three require experiment design, not literature review. Marked incomplete pending Phase 1.)

---

## Track A Decision: Tooling Reproducibility Study

**Question:** Do we run repeated extraction parity study (NNsight vs raw hooks)?

- **Novelty verdict:** HIGH. No published tooling parity protocol exists. Our paper's nnsight-vs-hooks finding (100% vs 10% on Qwen 7B) is the only quantitative data point in the literature. No survey (Bartoszcze 2025), no library paper (NNsight, pyvene, nnterp, IBM), and no experimental paper has systematically compared extraction tooling impact on steering direction quality.

- **Duplication risk:** LOW. No published work addresses this as of Feb 2026. The tooling papers (NNsight, pyvene, nnterp, IBM activation-steering) focus on API design and usability, not on whether different tools produce different activation vectors. This is a blind spot in the literature.

- **Estimated cost:**
  - Models: 2-3 (Qwen 7B + one other family, optionally Qwen 14B for scale check)
  - Tools: nnsight, PyTorch hooks, IBM activation-steering library (or pyvene)
  - Compute: ~4-8 GPU-hours per model × 3 tools = 12-24 GPU-hours
  - Human time: 1-2 days for protocol design + implementation + analysis

- **Risk level:** LOW. We already have the infrastructure. The experiment is small and well-scoped. Worst case: tools agree (still publishable as a reproducibility confirmation).

- **Recommendation: GO**

- **Rationale:** This is the single highest-novelty, lowest-risk experiment available. It directly extends our paper's most surprising finding. Our paper provides the only published quantitative evidence that extraction tooling affects steering direction quality, and no other group has investigated this systematically. Even a negative result (tools agree on other models) is informative — it would localize the problem to Qwen 7B specifically.

---

## Track B Decision: Direction Transfer Study

### B1: Same-family transfer (Qwen 7B → 14B → 32B)

- **Novelty verdict:** MODERATE. Same-family transfer has been demonstrated at sub-8B scale with learned mappings (Oozeer 2025, Cristofano 2026). Our contribution: raw DIM direction transfer (no mapping) at 7B+ scale. This is a simpler, more falsifiable test that the literature hasn't done.

- **Duplication risk:** MODERATE. Cristofano (2026) tests Qwen3-VL-2B → Qwen3-VL-8B in-family. If they extend to larger Qwen text models, our contribution shrinks. However, their methodology (concept-basis reconstruction) is fundamentally different from raw direction transfer — so even overlap in model choice doesn't fully duplicate.

- **Estimated cost:**
  - Models: Qwen 7B, 14B, 32B (we already have 7B and 32B infrastructure)
  - Extractions: DIM at 3-5 layers per model = 9-15 directions
  - Transfer tests: 6 direction pairs (7B→14B, 7B→32B, 14B→7B, 14B→32B, 32B→7B, 32B→14B)
  - Compute: ~2-4 GPU-hours per model for extraction + ~1 GPU-hour per transfer test = 12-20 GPU-hours
  - Human time: 1-2 days

- **Recommendation: GO (conditional)**

- **Rationale:** The experiment directly extends our inverse scaling paper. The key question — "does a direction from a small model work on a large model, and does the transfer gap track our inverse scaling curve?" — is novel and falsifiable. **Condition:** Run Track A first. If tooling matters, we need clean extraction before transfer tests make sense.

### B2: Cross-family transfer (Qwen ↔ Gemma, optional third family)

- **Novelty verdict:** MODERATE-HIGH. Cross-family raw direction transfer has NOT been done. Oozeer tests cross-family with autoencoders at sub-3B only. Cristofano tests cross-family with concept-basis reconstruction (Qwen→Ministral, not Qwen→Gemma). No Gemma in any cross-family transfer study. Wang et al. (2024, ICLR 2025, arXiv:2410.06672) provides supporting evidence for universality at the feature level (Transformer vs Mamba SAE features converge), but does not test refusal-direction transfer or within-Transformer-family comparisons. We would be the first to test Qwen↔Gemma direction compatibility at any scale.

- **Duplication risk:** LOW-MODERATE. The specific Qwen↔Gemma comparison is untested. Risk: someone publishes cross-family raw transfer before us (unlikely given the field's focus on learned mappings).

- **Estimated cost:**
  - Models: Qwen 7B/14B + Gemma 9B (we already have infrastructure for all)
  - Extractions: DIM at 3-5 layers per model = 6-10 directions
  - Transfer tests: 4 direction pairs minimum (Qwen 7B→Gemma 9B, reverse, plus one at 14B scale)
  - Compute: ~2-4 GPU-hours per model + ~1 GPU-hour per transfer test = 8-16 GPU-hours
  - Complication: Different tokenizers. Need to decide how to handle prompt compatibility.
  - Human time: 2-3 days (tokenizer alignment adds complexity)

- **Recommendation: GO (conditional, lower priority than B1)**

- **Rationale:** Gemma is missing from all cross-family transfer papers. We have Gemma infrastructure from our original paper. The experiment is complementary to B1 — if same-family transfer fails, cross-family transfer is unlikely to work (and that's still informative). **Condition:** Run after B1. If B1 shows zero transfer, B2 becomes "expected null" — still publishable but less exciting.

---

## Final recommendation to execute

### Priority order:
1. **Track A (Tooling Parity)** — Highest novelty, lowest risk, smallest scope. Do first.
2. **Track B1 (Same-Family Transfer)** — Moderate novelty, moderate risk. Do second.
3. **Track B2 (Cross-Family Transfer)** — Moderate-high novelty but depends on B1 results. Do third if B1 shows any transfer signal.

### Proposed first run:
**Track A: Tooling parity on Qwen 7B**
- Extract DIM direction using: (1) nnsight tracing, (2) PyTorch forward hooks, (3) IBM activation-steering library
- Same model, same contrastive data, same target layer, same multiplier
- Compare: cosine similarity between directions, steering effectiveness (coherent refusal rate), and activation norm statistics
- If directions differ: characterize the divergence (which components differ, at which layers)

### Proposed hard stop condition:
- **Track A:** If all three tools produce directions with cosine similarity > 0.99 on Qwen 7B, the finding doesn't generalize. Write up as "tooling divergence is model-specific" (still publishable but smaller contribution). Do NOT extend to more models.
- **Track B1:** If cosine similarity between Qwen 7B and 14B DIM directions is < 0.3 AND transferred direction produces 0% coherent refusal, the directions are family-specific-by-scale. Write up the negative result and skip B2.
- **Track B2:** If Qwen→Gemma cosine similarity is < 0.1, directions are completely family-specific. Write up and stop.

### Open questions before launch:
1. Do we have access to the IBM activation-steering library, or should we use pyvene as the third tool?
2. For B2, how do we handle tokenizer differences? Do we use the same English prompts and just let each tokenizer process them, or do we need alignment?
3. Should we include Qwen 3B (which we already ran in our paper) in B1 to get a 4-point scaling curve (3B→7B→14B→32B)?
4. Budget approval: Total estimated compute is ~30-60 GPU-hours across all three tracks. Is this within range?
