# Follow-Up Bibliography (Curated)

**Purpose:** Canonical citation pool for follow-up study design.
**Last updated:** 2026-02-18

---

## Tier A: Core papers (must read)

| Paper | Link | Why it matters | Status | Notes |
|---|---|---|---|---|
| Cristofano (2026). "Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction" | [arXiv:2601.16034](https://arxiv.org/abs/2601.16034) | Strongest existing work on cross-family refusal transfer. Tests 8 model pairs including Qwen→Ministral, Dense→MoE. Reports near-zero refusal rates after transfer. | Reviewed | Uses concept-basis reconstruction (learned mapping), not raw DIM. No Gemma. Only up to ~20B. |
| Oozeer et al. (2025). "Activation Space Interventions Can Be Transferred Between Large Language Models" | [arXiv:2503.04429](https://arxiv.org/abs/2503.04429) | ICML 2025. Tests same-family and cross-family transfer via autoencoder mappings. Llama/Qwen/Gemma families. | Reviewed | Only sub-3B models. Cross-family struggles with tokenizer mismatch. Autoencoder-based, not raw direction. |
| Wang et al. (2024). "Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures" | [arXiv:2410.06672](https://arxiv.org/abs/2410.06672) | ICLR 2025. Studies whether Transformers and Mambas converge on similar features/circuits. Supports universality hypothesis via SAE feature similarity. | Reviewed | General features, not refusal-specific. Transformer vs Mamba, not within Transformer families. No transfer experiments. |

---

## Tier B: Adjacent transfer / RepE papers

| Paper | Link | Relation | Status | Notes |
|---|---|---|---|---|
| Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction" | [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) | NeurIPS 2024. Foundational: establishes single-direction refusal across 13 models up to 72B. We already cite this. | Reviewed | No cross-model transfer. Each model gets own direction. |
| Wollschläger et al. (2025). "The Geometry of Refusal: Concept Cones and Representational Independence" | [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) | ICML 2025. Refusal is a CONE of multiple orthogonal directions, not a single vector. Uses RDO (gradient-based). Critical challenge to DIM premise. | Reviewed | Different refusal directions encode different mechanisms (detection vs execution). DIM captures cone mean. |
| Joad et al. (2026). "There Is More to Refusal than a Single Direction" | [arXiv:2602.02132](https://arxiv.org/abs/2602.02132) | 11 non-compliance categories, geometrically distinct but functionally equivalent. "One-dimensional control knob" paradox. | Reviewed | Steering any refusal direction increases ALL refusal. Style differs but function converges. Feb 2026 — very recent. |
| Beaglehole et al. (2025). "Toward Universal Steering and Monitoring of AI Models" | [arXiv:2502.03708](https://arxiv.org/abs/2502.03708) | RFM/AGOP method for concept extraction. Tests Llama 3.1 8B/70B. Compares 4 extraction methods — RFM beats PCA, DIM, logistic regression. We already cite this. | Reviewed | Cross-lingual transfer within model. No explicit cross-family direction comparison. Suggests family-specific encoding. |
| Wang et al. (2025). "Refusal Direction is Universal Across Safety-Aligned Languages" | [arXiv:2505.17306](https://arxiv.org/abs/2505.17306) | Shows English refusal direction transfers across 14 languages within a model. Supports universality within-model. | Reviewed | Cross-lingual, not cross-model. Different question from ours. |
| Ali et al. (2025). "Scaling Laws for Activation Steering with Llama 2 Models and Refusal Mechanisms" | [arXiv:2507.11771](https://arxiv.org/abs/2507.11771) | Studies CAA across Llama 2 7B/13B/70B. Finds performance degrades with scale (inverse scaling). Layer effectiveness peaks at early-middle layers. | Reviewed | Independent confirmation of our inverse scaling finding for Llama 2. No direction transfer between sizes. |
| Lee et al. (2025). "Programming Refusal with Conditional Activation Steering" (CAST) | [arXiv:2409.05907](https://arxiv.org/abs/2409.05907) | ICLR 2025 Spotlight. Conditional steering with 7 models including Qwen 1.5 (1.8B, 32B), Llama 2/3, OLMo, Zephyr. IBM activation-steering library. | Reviewed | No cross-model transfer. Tests conditional application, not direction universality. |
| Dunefsky & Cohan (2025). "Investigating Generalization of One-shot LLM Steering Vectors" | [arXiv:2502.18862](https://arxiv.org/abs/2502.18862) | COLM 2025. One-shot optimized steering vectors. 96.9% Harmbench ASR for refusal suppression. Tests cross-input generalization. | Reviewed | Within-model generalization, not cross-model transfer. |
| Bartoszcze et al. (2025). "Representation Engineering for LLMs: Survey and Research Challenges" | [arXiv:2502.17601](https://arxiv.org/abs/2502.17601) | Comprehensive RepE survey. Formalizes goals/methods. Discusses risks (performance decrease, steerability issues). | Reviewed | Survey — no new experimental results. No tooling parity section. |

---

## Tier C: Tooling/reproducibility papers

| Paper | Link | Relation | Status | Notes |
|---|---|---|---|---|
| Fiotto-Kaufman et al. (2024). "NNsight and NDIF" | [arXiv:2407.14561](https://arxiv.org/abs/2407.14561) | NNsight architecture: graph-based tracing, remote execution. Compares to TransformerLens, pyvene, baukit. | Reviewed | No steering quality benchmarks across tools. |
| Parascandolo et al. (2025). "nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers" | [arXiv:2511.14465](https://arxiv.org/abs/2511.14465) | NNsight wrapper: unified API across 50+ models, 16 architecture families. Auto module renaming + validation. | Reviewed | Addresses naming fragmentation. No activation quality comparison. |
| Wu et al. (2024). "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions" | [arXiv:2403.07809](https://arxiv.org/abs/2403.07809) | Declarative intervention framework. Configurable schemes on arbitrary PyTorch modules. | Reviewed | No parity benchmarks. Different design philosophy than nnsight. |
| TransformerLens (Nanda et al.) | [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Custom implementations for 50+ GPT-style models. Clean unified interface but requires manual adaptation per architecture. | Reviewed | Reimplements models from scratch — may introduce subtle divergences from HuggingFace originals. |
| IBM activation-steering library (Lee et al., 2025) | [github.com/IBM/activation-steering](https://github.com/IBM/activation-steering) | General-purpose steering library from CAST paper. ICLR 2025. | Reviewed | No published comparison of extraction fidelity across tools. |

---

## Keep / Drop decisions

### Keep for follow-up framing

- **Cristofano (2026)** — Primary comparison point for B1/B2. Our contribution: raw DIM transfer without learned mappings.
- **Oozeer et al. (2025)** — Secondary comparison for B1/B2. Our contribution: 7B+ scale models.
- **Ali et al. (2025)** — Independent confirmation of inverse scaling. Cite to strengthen our scaling claim.
- **Beaglehole et al. (2025)** — Already cited. RFM/AGOP as alternative extraction method.
- **NNsight, nnterp, pyvene, TransformerLens** — Tooling landscape for Track A framing.
- **Wang et al. (2024) "Towards Universality"** — Supports universality hypothesis at feature level.

### Drop (insufficient relevance)

- **Dunefsky & Cohan (2025)** — One-shot optimization is a different paradigm than direction extraction. Not relevant to our transfer or tooling questions.
- **Wang et al. (2025) "Refusal Direction is Universal Across Languages"** — Cross-lingual universality is a different axis. Cite only if needed for Q4 context.

---

## Citation candidates for follow-up write-up

### Background
- Arditi et al. (2024) — foundational single-direction result
- Beaglehole et al. (2025) — RFM/AGOP as advanced extraction method
- Bartoszcze et al. (2025) — RepE survey for broad context
- Ali et al. (2025) — independent inverse scaling confirmation

### Methods justification
- Cristofano (2026) — "Unlike Cristofano's concept-basis reconstruction, we test raw DIM transfer to isolate whether directions are geometrically compatible without learned mappings"
- Oozeer et al. (2025) — "Unlike Oozeer et al.'s autoencoder approach, we test direct application at 7B+ scale"
- Lee et al. (2025) / IBM library — if we use their activation-steering library as a third tool in Track A

### Threats-to-validity
- Oozeer et al. (2025) — tokenizer mismatch as known confounder for cross-family
- Cristofano (2026) — concept-basis coverage as potential limitation
- Fiotto-Kaufman et al. (2024) — nnsight architecture details explaining tooling divergence
- Our own paper — extending single-datapoint tooling finding
