# Next Step Brief

**Date:** 2026-02-18 | **Status:** Awaiting approval

---

## 1. What is now known with confidence

- **Inverse scaling is real and independently confirmed.** Our paper (Qwen/Gemma) and Ali et al. 2025 (Llama 2) both show steering effectiveness degrades with model scale. Two independent groups, three model families.
- **Tooling affects extraction on Qwen 7B.** NNsight tracing → 100% coherent refusal; PyTorch hooks → 10%. This is the only published quantitative evidence of tooling impact on steering direction quality.
- **No published tooling parity protocol exists.** Six tooling papers reviewed (NNsight, pyvene, nnterp, TransformerLens, IBM activation-steering, RepE survey) — none benchmark extraction fidelity across tools.
- **Same-family transfer exists but only at small scale with learned mappings.** Oozeer et al. (sub-3B, autoencoders) and Cristofano (VL models up to 8B, concept-basis reconstruction). No raw DIM transfer at 7B+.
- **Cross-family transfer has not been tested for Qwen↔Gemma at any scale.**

## 2. What remains uncertain

- Whether the NNsight vs hooks divergence replicates on non-Qwen models (could be Qwen-specific artifact)
- Whether a third tool (pyvene) agrees with NNsight, hooks, or neither
- Whether raw DIM directions are geometrically compatible across Qwen model sizes (3B/7B share no hidden_dim with 14B/32B)
- Whether refusal directions are family-specific or partially universal at the DIM level (prior universality evidence uses more complex extraction methods)
- The root cause of the tooling divergence (extraction point, dtype, token position, or something else)

## 3. Exact first experiment to run (if approved)

**Phase 1: Tooling Parity on Qwen 7B + Gemma 9B**

1. Implement pyvene extraction pipeline for DIM directions
2. Run pre-flight parity check: verify all 3 tools extract identical raw activations for a single prompt (Section D of protocol)
3. Run 5 × 3 = 15 extraction+steering cycles per model (30 total)
4. Analyze: within-method cosine, cross-method cosine, coherent/garbled/normal rates
5. Apply decision rules to determine next step

Full protocol: `docs/PHASE1_TOOLING_PARITY_PROTOCOL.md`

## 4. Expected cost/time

| Resource | Estimate |
|---|---|
| GPU hours | ~10 hours (A10G) |
| Cloud cost | $10-15 |
| Human time | ~1 day |

Phase 2 (if Phase 1 proceeds): additional ~7.5 GPU-hours, $8-11, 1.5 days.

## 5. What result would change strategy

| Result | Strategy change |
|---|---|
| All 3 tools agree on both models (cosine > 0.99) | Stop tooling track. Our original finding was implementation-specific. Write brief note, move directly to transfer experiments. |
| Tools diverge on Qwen only, not Gemma | Report as Qwen-specific artifact. Proceed to transfer with validated tool, but de-scope tooling contribution. |
| Tools diverge on both models | Strongest result. Prioritize root-cause analysis and tooling paper before transfer work. |
| Pre-flight parity check fails (raw activations differ) | Root cause is extraction point, not direction computation. Fix extraction alignment first — this may resolve the entire discrepancy trivially. |
