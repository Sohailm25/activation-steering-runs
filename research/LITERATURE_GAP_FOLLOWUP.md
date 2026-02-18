# Targeted Literature Gap Pass (Follow-Up)

**Experiment:** Inverse Scaling in Activation Steering (follow-up phase)
**Date completed:** 2026-02-18
**Owner:** Ghost
**Status:** Complete

---

## Objective

Answer four pre-run questions before any new experiments:

1. Has same-family direction transfer (e.g., 7B -> 14B -> 30/32B) already been done for refusal steering?
2. Has cross-family transfer (Qwen <-> Gemma <-> other families) already been done for refusal directions?
3. Are there published protocols for tooling parity checks in activation extraction pipelines?
4. What evidence exists for universal vs family-specific refusal representations?

---

## Search Protocol

- Time window: 2023-2026 (prefer newest work first)
- Source types: papers + code + technical reports
- Sources searched: arXiv, Semantic Scholar, OpenReview, LessWrong, GitHub
- Date of search: 2026-02-18

### Reviewer-suggested references evaluated

| # | Title | Found? | arXiv ID | Date |
|---|---|---|---|---|
| 1 | Universal Refusal Circuits Across LLMs | **Yes** | 2601.16034 | Jan 2026 |
| 2 | Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures | **Yes** | 2410.06672 | Oct 2024 (ICLR 2025) |
| 3 | Activation Space Interventions Can Be Transferred Between Large Language Models | **Yes** | 2503.04429 | Mar 2025 (ICML 2025) |

---

## Findings by Question

### Q1. Same-family transfer already done?

**Evidence:**

1. **Oozeer et al. (2025)** — "Activation Space Interventions Can Be Transferred Between Large Language Models" (arXiv:2503.04429, ICML 2025). Tests same-family transfer for Qwen (0.5B → 1.5B → 2.5B) and Llama (1B → 3B). Shows autoencoder-based nonlinear mappings can transfer activation interventions within families. **However: only tests sub-3B models.** No models at 7B+ scale. The authors explicitly acknowledge "experiments focused on Llama, Qwen, and Gemma models with limited size variations."

2. **Cristofano (2026)** — "Universal Refusal Circuits Across LLMs" (arXiv:2601.16034). Tests in-family transfer: Qwen3-VL-2B → Qwen3-VL-8B and Ministral-3-3B → Ministral-3-14B. Reports near-complete refusal suppression (0.00 ± 0.00) with negligible capability drift for in-family pairs. **But:** Qwen models are VL variants (vision-language), not standard text-only; largest Qwen tested is 8B.

3. **Ali et al. (2025)** — "Scaling laws for activation steering with Llama 2 models and refusal mechanisms" (arXiv:2507.11771). Studies CAA across Llama 2 7B/13B/70B. Finds performance degrades with scale (consistent with our inverse scaling finding). **But:** They study scaling of steering effectiveness, not transfer of directions between sizes. Each model gets its own extracted direction.

4. **Arditi et al. (2024)** — "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717, NeurIPS 2024). Tests 13 models up to 72B but extracts fresh directions per model. No transfer between sizes.

**Verdict: PARTIALLY DONE**

Same-family transfer has been demonstrated but only at small scale (sub-8B) and using learned mappings (autoencoders, concept-basis reconstruction) rather than raw cosine similarity of DIM/COSMIC vectors. No one has tested whether a DIM direction extracted from Qwen 7B works directly (without mapping) on Qwen 14B or 32B.

**Remaining novelty:**
- Direct DIM direction transfer (no learned mapping) across Qwen 7B → 14B → 32B
- Testing whether cosine similarity between same-family DIM vectors decreases with scale gap (we already have infrastructure for this)
- Connecting transfer success/failure to our inverse scaling curve — does transfer degrade at the same rate as single-model steering?

---

### Q2. Cross-family transfer already done?

**Evidence:**

1. **Oozeer et al. (2025)** — Tests cross-family: Qwen 0.5B → Llama 3B and Gemma 2B → Llama 3B. Finds "cross-architecture mappings struggled when models had significantly different vocabulary spaces." Only sub-3B models. Nonlinear autoencoder mapping required (not raw direction transfer).

2. **Cristofano (2026)** — Tests 8 cross-family pairs including Qwen3-VL → Ministral, Qwen3-VL → GPT-OSS-20B (Dense→MoE), Qwen3-VL → GLM-4.6V. Reports refusal rate drops from 0.98 to 0.02 for cross-family and 0.08 for Dense→MoE. Uses concept-basis reconstruction (learned alignment), not raw direction transfer. **No Gemma models tested.**

3. **Beaglehole et al. (2025)** — "Toward universal steering and monitoring of AI models" (arXiv:2502.03708). Tests Llama 3.1 8B/70B, Llama 3.3 70B, Llama-vision 90B, DeepSeek. Demonstrates cross-lingual transfer of concept vectors. RFM/AGOP methodology extracts per-model directions and shows they are combinable. **But: no explicit cross-family transfer of extracted directions.** Stays within Llama family + DeepSeek.

4. **Wang et al. (2025)** — "Refusal Direction is Universal Across Safety-Aligned Languages" (arXiv:2505.17306). Shows refusal vector extracted from English transfers across 14 languages within the same model. Cross-lingual, not cross-model.

**Verdict: PARTIALLY DONE (with learned mappings) / NOT DONE (with raw directions)**

Cross-family transfer has been demonstrated using learned mappings (autoencoders, concept-basis reconstruction) at small scale. No one has tested raw DIM direction cosine similarity between Qwen and Gemma families, or whether a direction extracted from one family works on another without a mapping function. The Cristofano paper comes closest but uses an elaborate reconstruction pipeline and does not include Gemma.

**Remaining novelty:**
- Raw DIM direction transfer Qwen ↔ Gemma (no mapping, just extract-and-apply)
- Cosine similarity analysis of refusal directions across families at matched scale
- Testing whether family-specific vs universal holds for DIM specifically (all prior work uses more complex extraction)
- Including Gemma in cross-family analysis (missing from both Oozeer and Cristofano)

---

### Q3. Tooling parity / reproducibility protocols already done?

**Evidence:**

1. **Our own paper** — We report that nnsight vs PyTorch hooks produces 100% vs 10% coherent refusal on Qwen 7B. This is (to our knowledge) the only published quantitative comparison of extraction tooling impact on steering effectiveness.

2. **NNsight paper** (Fiotto-Kaufman et al., arXiv:2407.14561) — Describes NNsight's architecture (graph-based tracing vs direct hook attachment) but does not benchmark steering direction quality across tooling.

3. **nnterp** (arXiv:2511.14465) — Wrapper around NNsight providing unified interface across 50+ model variants / 16 architecture families. Addresses naming convention fragmentation but does not compare extracted activation quality.

4. **IBM activation-steering library** (Lee et al., ICLR 2025, github.com/IBM/activation-steering) — General-purpose library, but no published comparison of extraction fidelity across tooling approaches.

5. **pyvene** (Wu et al., arXiv:2403.07809) — Intervention library with configurable schemes. No published parity benchmarks against other tools.

6. **Bartoszcze et al. (2025)** — "Representation Engineering for LLMs: Survey and Research Challenges" (arXiv:2502.17601). Comprehensive survey that discusses RepE methods but does not include tooling parity protocols.

**Verdict: NOT DONE**

No published protocol exists for systematically comparing activation extraction tooling (nnsight vs hooks vs TransformerLens vs pyvene) on steering direction quality. Our paper's finding is the only published quantitative evidence that tooling matters. The field has multiple competing tools but no benchmarking standard.

**Remaining novelty:**
- First systematic tooling parity study: same model, same data, same layer, multiple tools → compare resulting direction quality
- Protocol for detecting tooling-induced artifacts in activation extraction
- Extending our Qwen 7B finding to additional models and scales
- HIGH novelty — this is an open gap that affects reproducibility across the entire steering literature

---

### Q4. Universal vs family-specific refusal evidence?

**Evidence:**

1. **Arditi et al. (2024)** — Shows refusal is mediated by a single direction across 13 models. But each model gets its own direction — no direct comparison of whether directions are the same across models.

2. **Cristofano (2026)** — Strongest evidence FOR universality. Shows "semantic profile of refusal—characterized by high correlation with Deception, Safety Flagging, and Legalese atoms—is highly correlated" across model families. Transfer works across Dense→MoE and across families (Qwen→Ministral). Reports 8/8 successful transfers.

3. **Oozeer et al. (2025)** — Mixed evidence. Same-family transfer works well. Cross-family transfer "struggled when models had significantly different vocabulary spaces." Suggests family-specificity in the raw activation space, even if a learned mapping can bridge it.

4. **Wang et al. (2025)** — "Towards Universality" (arXiv:2410.06672, ICLR 2025). Studies Transformer vs Mamba. Shows most SAE features are similar. Induction circuits are structurally analogous. **But:** Studies general features (not refusal), and architecture comparison is Transformer vs Mamba (not within Transformer families).

5. **Wang et al. (2025)** — "Refusal Direction is Universal Across Safety-Aligned Languages" (arXiv:2505.17306). Shows cross-lingual universality of refusal direction within a model. Supports universality of the concept but within-model only.

6. **Beaglehole et al. (2025)** — Evidence leans toward family-specific. "Newer, larger, and better performing Llama models were also more steerable" but "steerability improvements weren't uniform across concept classes or model sizes." Compares 4 extraction methods (PCA, DIM, logistic regression, RFM) — RFM substantially outperforms others. No explicit cross-family comparison of directions.

7. **Ali et al. (2025)** — Scaling laws for CAA with Llama 2. Finds steering effectiveness degrades with scale. Consistent with our inverse scaling but within one family only.

8. **Wollschläger et al. (2025)** — "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence" (arXiv:2502.17420, ICML 2025). Shows refusal is NOT a single direction but a **concept cone** — multiple orthogonal directions encoding distinct refusal mechanisms (detection vs execution, safety vs identity vs capability). Uses Refusal Direction Optimization (RDO), not DIM. This challenges the premise that DIM captures "the" refusal direction — it captures the mean of a cone.

9. **Joad et al. (2026)** — "There Is More to Refusal in Large Language Models than a Single Direction" (arXiv:2602.02132). Identifies 11 categories of non-compliance with geometrically distinct but functionally equivalent directions. Different refusal categories are orthogonal yet all increase overall refusal rate — a "one-dimensional control knob" paradox. Suggests DIM captures the average of multiple refusal styles.

**Verdict: CONTESTED — evidence for both sides**

The emerging picture:
- **Within a model:** Refusal is single-direction and universal across languages (Arditi, Wang)
- **Within a family:** Transfer works with learned mappings at small scale (Oozeer, Cristofano)
- **Across families:** Transfer works with sophisticated reconstruction (Cristofano) but raw mappings struggle with tokenizer mismatch (Oozeer)
- **No direct comparison:** Nobody has computed cosine similarity of DIM refusal vectors across Qwen and Gemma at matched scale to answer "are these the same direction?"

**Remaining novelty:**
- Direct geometric comparison: cosine similarity of DIM refusal vectors across families at matched scale
- Testing whether universality holds for simple (DIM) vs complex (COSMIC, RFM) extraction
- Connecting universality evidence to our inverse scaling finding — does universality break down at scale?

---

## Synthesis

### What is already saturated

1. **Single-direction refusal finding** — Arditi et al. established this firmly (NeurIPS 2024). Replicating it adds nothing.
2. **Transfer with learned mappings at small scale** — Oozeer et al. (ICML 2025) covers Llama/Qwen/Gemma sub-3B with autoencoders. Cristofano (2026) covers cross-family with concept-basis reconstruction.
3. **Cross-lingual universality** — Wang et al. (2025) shows refusal direction transfers across 14 languages within a model.
4. **Inverse scaling of CAA with Llama 2** — Ali et al. (2025) independently shows steering degrades with scale for Llama 2 7B/13B/70B.

### What remains open and testable

1. **Raw direction transfer at 7B+ scale** — Nobody has tested whether a DIM vector from Qwen 7B works on Qwen 14B or 32B without a learned mapping. This is the simplest, most falsifiable test.
2. **Cross-family direction geometry** — Nobody has computed cosine similarity of DIM refusal directions between Qwen and Gemma at matched scale.
3. **Tooling parity** — Our nnsight vs hooks finding is the only data point. Systematic extension is wide open.
4. **Connecting transfer to inverse scaling** — Does direction transfer degrade at the same rate as single-model steering? Novel framing.
5. **Gemma in cross-family studies** — Neither Oozeer nor Cristofano includes Gemma. We already have Gemma infrastructure.

### What we should explicitly avoid duplicating

1. **Transfer with learned mappings** — Oozeer and Cristofano own this space. Don't build autoencoders or concept-basis reconstruction.
2. **Scaling laws for Llama 2 CAA** — Ali et al. (2025) already did this.
3. **Cross-lingual transfer** — Wang et al. (2025) covers this.
4. **Transformer vs Mamba universality** — Wang et al. (2024, ICLR 2025) covers this.

### Proposed novelty statement for our follow-up

"We test whether raw DIM refusal directions transfer across scale (Qwen 7B→14B→32B) and across families (Qwen↔Gemma) without learned mappings, and whether transfer degradation follows the same inverse scaling curve we identified for single-model steering. Separately, we provide the first systematic tooling parity protocol comparing nnsight, PyTorch hooks, and at least one additional framework on identical extraction tasks."

This is complementary to (not duplicative of) Oozeer's learned-mapping transfer and Cristofano's concept-basis reconstruction, because we test the simpler hypothesis that raw directions are transferable — and if they're not, we characterize the failure mode geometrically.
