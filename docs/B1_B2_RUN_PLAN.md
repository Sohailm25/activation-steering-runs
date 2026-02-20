# B1/B2 Transfer Experiments — Run Plan

**Date:** 2026-02-19  
**Status:** Draft — awaiting Professor pre-run review  
**Depends on:** Phase-1 tooling parity (COMPLETE), budget approval (APPROVED)

---

## Tooling Decision

**Locked tool: nnsight** (all three tools produce identical results; nnsight has cleanest API for extraction + intervention).

---

## B1: Same-Family Transfer (Qwen 3B → 7B → 14B → 32B)

### Infrastructure needed

New Modal script: `infrastructure/v4_transfer_b1.py`
- Extract DIM direction from source model
- Apply at target model's optimal layer
- Evaluate on 30 prompts, greedy decode, max_tokens=256

### Execution order

**Phase B1a — Self-controls (4 runs, establish baselines):**

```bash
# Each extracts own DIM + evaluates on self
modal run --detach v4_transfer_b1.py --source qwen-3b --target qwen-3b --multiplier 15
modal run --detach v4_transfer_b1.py --source qwen-7b --target qwen-7b --multiplier 15
modal run --detach v4_transfer_b1.py --source qwen-14b --target qwen-14b --multiplier 15
modal run --detach v4_transfer_b1.py --source qwen-32b --target qwen-32b --multiplier 15
```

**Phase B1b — Direct transfer (14B ↔ 32B only, same hidden_dim=5120):**

```bash
modal run --detach v4_transfer_b1.py --source qwen-14b --target qwen-32b --multiplier 15
modal run --detach v4_transfer_b1.py --source qwen-32b --target qwen-14b --multiplier 15
```

**Phase B1c — Geometric comparisons (10 remaining pairs, CPU-only):**

For all pairs where hidden_dim differs (3B↔7B, 3B↔14B, 3B↔32B, 7B↔14B, 7B↔32B), compute:
- Cosine similarity between source DIM and target's own DIM
- Norm ratio
- No behavioral transfer (dimension mismatch prevents direct application)

### GPU requirements

| Run | GPU | Estimated time |
|-----|-----|---------------|
| 3B self-control | A10G | ~20 min |
| 7B self-control | A10G | ~30 min |
| 14B self-control | A100 | ~40 min |
| 32B self-control | A100-80GB | ~60 min |
| 14B→32B transfer | A100-80GB | ~60 min |
| 32B→14B transfer | A100 | ~40 min |

### Expected artifacts

```
results/phase2/b1_self_qwen-3b_YYYYMMDD.json
results/phase2/b1_self_qwen-7b_YYYYMMDD.json
results/phase2/b1_self_qwen-14b_YYYYMMDD.json
results/phase2/b1_self_qwen-32b_YYYYMMDD.json
results/phase2/b1_transfer_qwen14b_to_qwen32b_YYYYMMDD.json
results/phase2/b1_transfer_qwen32b_to_qwen14b_YYYYMMDD.json
results/phase2/b1_geometric_comparisons.json
```

### Stop conditions

- If 3B or 7B self-control coherent < 50%: STOP — extraction protocol may be wrong
- If 32B produces only garbled output: reduce multiplier to 10x and rerun
- If any self-control shows 0% refusal: STOP — DIM may not work at this scale

---

## B2: Cross-Family Transfer (Qwen ↔ Gemma)

### Prerequisite

B1 self-controls complete (provides baselines for interpretation).

### Execution order

**Phase B2a — Gemma self-control:**

```bash
modal run --detach v4_transfer_b2.py --source gemma-9b --target gemma-9b --multiplier 25
```

**Phase B2b — Direct cross-family transfer (Qwen 7B ↔ Gemma 9B, both dim=3584):**

```bash
modal run --detach v4_transfer_b2.py --source qwen-7b --target gemma-9b --multiplier 25
modal run --detach v4_transfer_b2.py --source gemma-9b --target qwen-7b --multiplier 15
```

**Phase B2c — Geometric comparisons (14B pairs, dim mismatch):**

```bash
# CPU-only: cosine(Qwen14B_DIM, Gemma9B_DIM) — requires alignment or comparison in matched subspace
modal run --detach v4_transfer_b2.py --source qwen-14b --target gemma-9b --geometric-only
modal run --detach v4_transfer_b2.py --source gemma-9b --target qwen-14b --geometric-only
```

### GPU requirements

| Run | GPU | Estimated time |
|-----|-----|---------------|
| Gemma self-control | A100 | ~30 min |
| Qwen 7B→Gemma 9B | A100 | ~30 min |
| Gemma 9B→Qwen 7B | A10G | ~30 min |

### Expected artifacts

```
results/phase2/b2_self_gemma-9b_YYYYMMDD.json
results/phase2/b2_transfer_qwen7b_to_gemma9b_YYYYMMDD.json
results/phase2/b2_transfer_gemma9b_to_qwen7b_YYYYMMDD.json
results/phase2/b2_geometric_comparisons.json
```

### Stop conditions

- If Gemma self-control coherent < 50%: STOP — multiplier may need tuning
- If B1 shows zero same-family transfer: SKIP B2 (cross-family won't work if same-family fails)
- If cross-family behavioral transfer = 0% but cosine > 0.3: still a valid geometric result (report cosine finding)

---

## Budget Estimate

| Phase | GPU hours | Estimated cost |
|-------|-----------|---------------|
| B1 self-controls | ~2.5 hrs | ~$3-4 |
| B1 transfers (14B↔32B) | ~1.5 hrs | ~$3-4 |
| B1 geometric | CPU only | ~$0 |
| B2 self-control | ~0.5 hrs | ~$1 |
| B2 transfers | ~1 hr | ~$1-2 |
| **Total B1+B2** | **~5.5 hrs** | **~$8-11** |

Combined with Phase-1 (~$43): **total project spend ~$51-54**.

---

## Pre-registration: Expected Outcomes

### B1 predictions

1. Self-controls should reproduce Phase-1 coherent rates (≥90% for 3B/7B, unknown for 14B/32B)
2. 14B→32B direct transfer: expect degraded coherent rate (50-80% of self-control)
3. 32B→14B direct transfer: expect less degradation (larger→smaller may transfer better)
4. Geometric cosine across different hidden_dims: expect 0.2-0.6 (partial overlap within family)

### B2 predictions

1. Gemma self-control: expect ~93% coherent (matching Phase-1 parity result)
2. Qwen 7B→Gemma 9B: expect <50% coherent (cross-family transfer degradation)
3. Cross-family cosine (Qwen 7B vs Gemma 9B DIM): expect 0.1-0.4 (partial family-specific geometry)

### What would failure look like

- All self-controls working but ALL transfers at 0%: refusal directions are fully model-specific
- Transfer works equally well as self-control: refusal directions are universal (surprising, would need replication)
- Self-controls fail on 14B/32B: DIM extraction doesn't scale (would need different extraction method)

---

## Implementation TODO

1. Write `infrastructure/v4_transfer_b1.py` (Modal script for B1)
2. Write `infrastructure/v4_transfer_b2.py` (Modal script for B2)
3. Verify Qwen 14B and 32B model IDs on HuggingFace
4. Run tokenizer verification on all models (contrastive prompts produce valid tokens)
5. Run B1a self-controls first, review before proceeding
