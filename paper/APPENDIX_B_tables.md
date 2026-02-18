# Appendix B: Complete Results Tables

All results are from `results/FINAL_RESULTS.json`. Coherent refusal rate = percentage of outputs classified as coherent refusal (contains refusal keywords, not garbled). All experiments use greedy decoding, 100 max generation tokens.

---

## B.1 Qwen 2.5 Size Sweep: Full Layer Profiles

Method: DIM @ 15x multiplier. Direction extracted from 5 harmful + 5 harmless prompts.

### Qwen 2.5-3B-Instruct (36 layers)

| Layer | Depth % | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------------|---------|---|---------------|--------|
| L18 | 50% | 80.0% | 20.0% | 50 | 10.92 | v3_nnsight_qwen-3b_20260213_121150.json |
| **L21** | **60%** | **100.0%** | **0.0%** | **50** | **21.14** | v3_nnsight_qwen-3b_20260213_121150.json |
| L25 | 70% | 70.0% | 30.0% | 50 | 48.44 | v3_nnsight_qwen-3b_20260213_121150.json |

### Qwen 2.5-7B-Instruct (28 layers)

| Layer | Depth % | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------------|---------|---|---------------|--------|
| L14 | 50% | 86.7% | 0.0% | 30 | - | v3_gap_fill_qwen-7b-gaps_20260214_175701.json |
| **L16** | **60%** | **100.0%** | **0.0%** | **50** | **26.22** | v3_nnsight_qwen-7b_20260213_120047.json |
| L16 | 60% | 100.0% | 0.0% | 30 | - | v3_nnsight_quant_qwen-7b_sweep_20260214_155514.json (quant baseline) |
| L19 | 70% | 16.7% | 0.0% | 30 | - | v3_gap_fill_qwen-7b-gaps_20260214_175701.json |

### Qwen 2.5-14B-Instruct (48 layers)

| Layer | Depth % | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------------|---------|---|---------------|--------|
| **L24** | **50%** | **90.0%** | **10.0%** | **50** | **33.05** | v3_nnsight_qwen-14b_20260213_121333.json |
| L28 | 60% | 90.0% | 0.0% | 50 | 67.05 | v3_nnsight_qwen-14b_20260213_121333.json |
| L33 | 70% | 0.0% | 0.0% | 50 | 176.92 | v3_nnsight_qwen-14b_20260213_121333.json |

### Qwen 2.5-32B-Instruct (64 layers)

| Layer | Depth % | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------------|---------|---|---------------|--------|
| **L32** | **50%** | **60.0%** | **0.0%** | **50** | **63.16** | v3_nnsight_qwen-32b_20260213_121619.json |
| L38 | 60% | 20.0% | 0.0% | 50 | 84.85 | v3_nnsight_qwen-32b_20260213_121619.json |
| L44 | 70% | 10.0% | 0.0% | 50 | 165.17 | v3_nnsight_qwen-32b_20260213_121619.json |

---

## B.2 Gemma 2 Size Sweep: Full Layer Profiles

Method: DIM @ 25x multiplier.

### Gemma 2-2B-IT (26 layers)

| Layer | Depth % | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------------|---------|---|---------------|--------|
| **L7** | **30%** | **100.0%** | **0.0%** | **50** | **24.32** | v3_gemma_sweep_gemma-2b_20260213_143851.json |
| L10 | 40% | 100.0% | 0.0% | 50 | 53.03 | v3_gemma_sweep_gemma-2b_20260213_143851.json |
| L13 | 50% | 70.0% | 0.0% | 50 | 132.89 | v3_gemma_sweep_gemma-2b_20260213_143851.json |
| L15 | 60% | 30.0% | 0.0% | 50 | 155.97 | v3_gemma_sweep_gemma-2b_20260213_143851.json |

### Gemma 2-9B-IT (42 layers)

**Canonical sweep (30 prompts, 25x multiplier):**

| Layer | Depth % | Coherent Refusal | Garbled | n | Source |
|-------|---------|-----------------|---------|---|--------|
| **L12** | **30%** | **96.7%** | **0.0%** | **30** | v3_gap_fill_gemma-9b-canonical_20260214_181837.json |
| L16 | 40% | 96.7% | 0.0% | 30 | v3_gap_fill_gemma-9b-canonical_20260214_181837.json |
| L21 | 50% | 73.3% | 0.0% | 30 | v3_gap_fill_gemma-9b-canonical_20260214_181837.json |
| L25 | 60% | 40.0% | 0.0% | 30 | v3_gap_fill_gemma-9b-canonical_20260214_181837.json |

**Earlier 50-prompt results (reference):**

| Layer | Depth % | Multiplier | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------|-----------------|---------|---|---------------|--------|
| L16 | 40% | 20x | 80.0% | 0.0% | 50 | 92.95 | v3_nnsight_gemma-9b_20260213_121125.json |
| L16 | 40% | 25x | 90.0% | 0.0% | 50 | 92.95 | v3_nnsight_gemma-9b_20260213_121125.json |

**Gap fill (30 prompts, 15x multiplier):**

| Layer | Depth % | Coherent Refusal | Garbled | n | Source |
|-------|---------|-----------------|---------|---|--------|
| L12 | 30% | 76.7% | 0.0% | 30 | v3_gap_fill_gemma-9b-canonical_20260214_181837.json |
| L21 | 50% | 36.7% | 0.0% | 30 | v3_gap_fill_gemma-9b-canonical_20260214_181837.json |

### Gemma 2-27B-IT (46 layers)

| Layer | Depth % | Coherent Refusal | Garbled | n | Direction Norm | Source |
|-------|---------|-----------------|---------|---|---------------|--------|
| L13 | 30% | 0.0% | 100.0% | 50 | 353.25 | v3_gemma_sweep_gemma-27b_20260213_145411.json |
| L18 | 40% | 0.0% | 100.0% | 50 | - | v3_gemma_sweep_gemma-27b_20260213_145411.json |
| L23 | 50% | 0.0% | 100.0% | 50 | - | v3_gemma_sweep_gemma-27b_20260213_145411.json |
| L27 | 60% | 0.0% | 100.0% | 50 | - | v3_gemma_sweep_gemma-27b_20260213_145411.json |

**Note:** Gemma 27B is genuinely unsteerable. Direction norms range from 351 to 2352. All outputs are empty/garbled at every layer tested. Tested with bfloat16 precision.

---

## B.3 Mistral 7B: Architecture Failure

Method: DIM + COSMIC @ 15x multiplier. Model: Mistral-7B-Instruct-v0.3 (32 layers).

| Layer | Depth % | DIM Coherent | DIM Garbled | COSMIC Coherent | COSMIC Garbled | n | Source |
|-------|---------|-------------|------------|----------------|---------------|---|--------|
| L16 | 50% | 0.0% | 100.0% | 0.0% | 100.0% | 50 | v3_phase1_family_sweep_20260212_125230.json |
| L19 | 60% | 0.0% | 100.0% | 0.0% | 100.0% | 50 | v3_phase1_family_sweep_20260212_125230.json |
| L22 | 70% | 0.0% | 100.0% | 0.0% | 100.0% | 50 | v3_phase1_family_sweep_20260212_125230.json |

---

## B.4 DIM vs COSMIC Comparison

### Real COSMIC (Full Algorithm)

All comparisons at n=50 prompts. COSMIC uses multi-position forward-pass scoring for automated layer selection.

| Model | DIM Layer | DIM Rate | COSMIC Layer | COSMIC Rate | Cosine (DIM vs COSMIC) | COSMIC Score | Source |
|-------|-----------|----------|-------------|------------|----------------------|-------------|--------|
| Qwen 3B | L21 (60%) | **100.0%** | L18 (50%) | 100.0% | 0.763 | 1.511 | v3_cosmic_real_qwen-3b_20260213_133054.json |
| Qwen 14B | L24 (50%) | **90.0%** | L23 (48%) | 90.0% | 0.537 | 1.492 | v3_cosmic_real_qwen-14b_20260213_133844.json |
| Qwen 32B | L32 (50%) | **60.0%** | L43 (67%) | 10.0% | 0.533 | 1.481 | v3_cosmic_real_qwen-32b_20260213_133604.json |
| Gemma 9B | L16 (40%) | **90.0%** | L19 (45%) | 70.0% | 0.838 | 1.388 | v3_cosmic_real_gemma-9b_20260213_133251.json |

### Simplified COSMIC (SVD, Reference Only)

These used `compute_cosmic_direction()` (simplified SVD) -- NOT the real COSMIC algorithm. Included for completeness.

| Model | DIM Rate | Simplified COSMIC Rate | Cosine |
|-------|----------|----------------------|--------|
| Qwen 3B | 100.0% | 20.0% | 0.276 |
| Qwen 14B | 90.0% | 0.0% | 0.130 |
| Qwen 32B | 60.0% | 10.0% | 0.041 |
| Gemma 9B | 90.0% | 10.0% | 0.112 |

---

## B.5 Quantization Robustness

Method: DIM @ 15x, n=30 prompts. Direction extracted from FP16 model, applied to quantized models. Quantization via bitsandbytes (INT8: LLM.int8(), INT4: NF4).

### Qwen 2.5-7B-Instruct (L16, 60% depth)

| Precision | Coherent Refusal | Garbled | Direction Norm | Cosine vs FP16 | Source |
|-----------|-----------------|---------|---------------|---------------|--------|
| FP16 | 100.0% | 0.0% | 26.22 | 1.000 | v3_nnsight_quant_qwen-7b_sweep_20260214_155514.json |
| INT8 | 100.0% | 0.0% | 25.93 | 0.994 | v3_nnsight_quant_qwen-7b_sweep_20260214_155514.json |
| INT4 | 100.0% | 0.0% | 25.58 | 0.972 | v3_nnsight_quant_qwen-7b_sweep_20260214_155514.json |

### Qwen 2.5-32B-Instruct (L32, 50% depth)

| Precision | Coherent Refusal | Garbled | Direction Norm | Cosine vs FP16 | 95% Wilson CI | Source |
|-----------|-----------------|---------|---------------|---------------|-------------|--------|
| FP16 | 76.7% | 0.0% | 63.16 | 1.000 | [59.1%, 88.2%] | v3_nnsight_quant_qwen-32b_sweep_20260214_155648.json |
| INT8 | 83.3% | 0.0% | 62.62 | 0.991 | [66.4%, 92.7%] | v3_nnsight_quant_qwen-32b_sweep_20260214_155648.json |
| INT4 | 56.7% | 0.0% | 63.55 | 0.974 | [39.2%, 72.6%] | v3_nnsight_quant_qwen-32b_sweep_20260214_155648.json |

**Note:** FP16 vs INT4 at 32B: Fisher's exact p ~ 0.11, Cohen's h = 0.42. The 20pp drop is suggestive but not statistically significant at n=30.

---

## B.6 Multiplier Sensitivity: Qwen 32B

Model: Qwen 2.5-32B-Instruct, L32 (50% depth), DIM extraction, n=50 prompts.

| Multiplier | Coherent Refusal | Garbled | Normal | Direction Norm | Source |
|------------|-----------------|---------|--------|---------------|--------|
| 15x | 60.0% | 0.0% | 40.0% | 63.16 | v3_qwen32b_mult_20260213_143821.json |
| 20x | 20.0% | 0.0% | 80.0% | 63.16 | v3_qwen32b_mult_20260213_143821.json |
| 25x | 0.0% | 90.0% | 10.0% | 63.16 | v3_qwen32b_mult_20260213_143821.json |

**Interpretation:** The effective multiplier window at 32B is remarkably narrow. At 15x, the model produces moderate coherent refusal. At 20x, it largely reverts to normal output. At 25x, coherence collapses with 90% garbled output. By contrast, smaller Qwen models tolerate 15x--25x without significant degradation.

---

## B.7 Direction Norms: All Models at Best Layer

| Model | Family | Best Layer | Depth % | Direction Norm | Coherent Refusal | Outcome |
|-------|--------|-----------|---------|---------------|-----------------|---------|
| Qwen 3B | Qwen | L21 | 60% | 21.14 | 100% | WORKS |
| Qwen 7B | Qwen | L16 | 60% | 26.22 | 100% | WORKS |
| Qwen 14B | Qwen | L24 | 50% | 33.05 | 90% | WORKS |
| Qwen 32B | Qwen | L32 | 50% | 63.16 | 60% | PARTIAL |
| Gemma 2B | Gemma | L7 | 30% | 24.32 | 100% | WORKS |
| Gemma 9B | Gemma | L16 | 40% | 92.95 | 90% | WORKS |
| Gemma 27B | Gemma | L13 | 30% | 353.25 | 0% | FAILS |

**Heuristic:** Direction norms in the 20--90 range predict successful steering. Norms above 100 predict failure, with the exception of Gemma 2B at 50% depth (norm 133, 70% coherent refusal).
