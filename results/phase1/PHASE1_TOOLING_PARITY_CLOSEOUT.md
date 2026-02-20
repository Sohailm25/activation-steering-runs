# Phase-1 Tooling Parity Closeout

**Date:** 2026-02-19  
**Author:** Ghost (experiment agent)  
**Professor Review:** ADVISED (2026-02-19, verdict thread 1771533811.923549)  
**Status:** COMPLETE — frozen, no further parity runs unless defect found.

---

## 1. Scope Statement

Phase-1 tooling parity has been established for **two models only**:

- **Qwen-7B** (`Qwen/Qwen2.5-7B-Instruct`)
- **Gemma-9B** (`google/gemma-2-9b-it`)

This result does **not** generalize to other model families, architectures, or sizes without additional testing. Claims are limited to the tested models under the protocol below.

---

## 2. Protocol Lock

| Parameter | Value |
|-----------|-------|
| Eval prompts | 30 |
| Sampling | Greedy decode (temperature=0, top_p=1.0) |
| Repeats | 5 |
| Methods | nnsight, hooks, pyvene |
| Execution | Modal-only (A10G for Qwen, A100 for Gemma) |
| Steering site | Residual stream, layer 15 |
| Direction | Mean-difference (harm − safe), L2-normalized |
| Steering alpha | 3.0 |
| Max new tokens | 128 |

---

## 3. Canonical Artifacts

| Model | Artifact Path | Modal App ID |
|-------|--------------|--------------|
| Qwen-7B | `results/phase1/modal_qwen-7b_20260219_130630.json` | `ap-DDAzaPVzONgMvI6rSiEuwl` |
| Gemma-9B | `results/phase1/modal_gemma-9b_20260219_203953.json` | `ap-RKamaXsFjhj6stxL5lLqQh` |

Remote volume copies:
- `/results/v4_parity_qwen-7b_20260219_130630.json`
- `/results/v4_parity_gemma-9b_20260219_203953.json`

---

## 4. Results Summary

### Per-Model, Per-Method Metrics

#### Qwen-7B (n=5 repeats)

| Method | n_ok | coherent (mean±std) | garbled (mean±std) | normal (mean±std) | within-method cosine |
|--------|------|--------------------|--------------------|-------------------|---------------------|
| nnsight | 5 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 1.0000000596 |
| hooks | 5 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 1.0000000596 |
| pyvene | 5 | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 1.0000000596 |

**Pairwise parity (Qwen-7B):**

| Pair | cosine | L2 |
|------|--------|----|
| hooks ↔ nnsight | 0.9999999844 | 0.0 |
| hooks ↔ pyvene | 0.9999999844 | 0.0 |
| nnsight ↔ pyvene | 0.9999999844 | 0.0 |

**parity_pass: true**

#### Gemma-9B (n=5 repeats)

| Method | n_ok | coherent (mean±std) | garbled (mean±std) | normal (mean±std) | within-method cosine |
|--------|------|--------------------|--------------------|-------------------|---------------------|
| nnsight | 5 | 93.3% ± 0.0 | 0.0% ± 0.0 | 6.7% ± 0.0 | 1.0000000596 |
| hooks | 5 | 93.3% ± 0.0 | 0.0% ± 0.0 | 6.7% ± 0.0 | 1.0000000596 |
| pyvene | 5 | 93.3% ± 0.0 | 0.0% ± 0.0 | 6.7% ± 0.0 | 1.0000000596 |

**Pairwise parity (Gemma-9B):**

| Pair | cosine | L2 |
|------|--------|----|
| hooks ↔ nnsight | 0.9999999844 | 0.0 |
| hooks ↔ pyvene | 0.9999999844 | 0.0 |
| nnsight ↔ pyvene | 0.9999999844 | 0.0 |

**parity_pass: true**

### Cross-Model Closure

| Model | Methods passing | parity_pass | coherent range | garbled range | Notable |
|-------|----------------|-------------|----------------|---------------|---------|
| Qwen-7B | 3/3 | true | 100% | 0% | Perfect coherence |
| Gemma-9B | 3/3 | true | 93.3% | 0% | 6.7% normal (2/30 prompts not steered) |

---

## 5. Provenance

- **Git commit at run time:** `e4c7d53` (confirmed via pre-flight; artifact metadata fields are empty due to a script bug)
- **TODO:** Patch `v4_modal_parity.py` and `v4_tooling_parity.py` to always write `git_commit` to artifact JSON. Do not rerun Gemma/Qwen solely for this.
- **Modal profile:** `sohailm25`
- **HF secret:** `hf-secret` (Modal secret)

---

## 6. Limitations

1. **Scope is two models only.** Parity on other architectures (Llama, Mistral, etc.) is untested.
2. **Single steering site tested** (layer 15, residual stream). Other sites may behave differently.
3. **Greedy decode only.** Stochastic sampling may surface method differences not visible here.
4. **30 eval prompts.** A larger prompt set might reveal edge cases.
5. **Gemma coherent rate is 93.3%, not 100%.** Two prompts out of 30 produced "normal" (unsteered) output consistently across all methods — this is a model behavior, not a tooling discrepancy.

---

## 7. Next-Phase Handoff

Phase-1 establishes that all three intervention libraries (nnsight, hooks, pyvene) produce identical steering vectors and equivalent behavioral outcomes on tested models. The tooling decision is now locked.

**Before proceeding to B1/B2 transfer experiments:**
1. Obtain Sohail's explicit budget approval (Phase-1 spent ~$43, exceeding prior $15 cap).
2. Prepare B1/B2 run plan with commands, expected artifacts, and stop conditions.
3. Request Professor pre-run review of B1/B2 protocol.
