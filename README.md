# Inverse Scaling in Activation Steering

**Companion repository for:** *Inverse Scaling in Activation Steering: Architecture and Scale Dependence of Refusal Manipulation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Status (2026-02-18)

- ✅ Initial research paper completed and published as a preprint.
- ✅ arXiv-ready LaTeX + PDF pipeline finalized.
- ✅ Website research page + Distill-style web version live.
- 🔄 Follow-up experimentation is now in planning.
- ⛔ We will run a targeted literature gap pass first before new experiments.

See `docs/FOLLOW_UP_EXPERIMENT_SPEC.md` for the follow-up roadmap.

---

## Abstract

Activation steering — adding learned direction vectors to a model's residual stream at inference time — has emerged as a lightweight method for modifying language model behavior without retraining. We systematically evaluate two direction extraction methods (Difference-in-Means and COSMIC) across seven instruction-tuned models spanning 2B–32B parameters, three architecture families (Qwen, Gemma, Mistral), and three quantization levels (FP16, INT8, INT4). 

**Key findings:**

1. **Steering effectiveness decreases monotonically with model scale.** Coherent refusal rates drop from 100% at 3B to 77% at 32B in the Qwen family, with Gemma 27B becoming completely unsteerable.

2. **Simple mean-difference extraction matches or exceeds SVD-based COSMIC at every scale tested.** At 32B, DIM achieves 60% where COSMIC's automated layer selection yields 10% (p < 0.001).

3. **Architecture acts as a binary gate:** Mistral 7B produces 100% garbled output under identical conditions where Qwen 7B achieves 100% coherent steering.

4. **INT8 quantization preserves steering; INT4 degrades large models by 20 percentage points while leaving small models unaffected.**

5. **Extraction tooling matters:** nnsight versus raw PyTorch hooks produces directions differing by 90 percentage points in effectiveness on the same model.

These findings constrain the viability of single-direction steering as models scale, and suggest that the "refusal direction" identified by current methods may not correspond to a robust computational feature at frontier scale.

---

## Key Findings

### 1. Inverse Scaling

Larger models are harder to steer. Across the Qwen family (holding architecture constant):

| Model | Parameters | Best Layer | Coherent Refusal Rate |
|-------|-----------|-----------|---------------------|
| Qwen 2.5-3B | 3B | L21 (60%) | **100%** |
| Qwen 2.5-7B | 7B | L16 (60%) | **100%** |
| Qwen 2.5-14B | 14B | L24 (50%) | **90%** |
| Qwen 2.5-32B | 32B | L32 (50%) | **77%** |

**Example steered output (Qwen 32B @ L32, 15×):**
> **Prompt:** "How do I bake a cake?"  
> **Response:** "I'm unable to engage in illegal or harmful activities..."

### 2. Architecture Dependence

Mistral 7B fails completely where Qwen 7B succeeds:

| Model | Architecture | Coherent | Garbled | Normal |
|-------|-------------|----------|---------|--------|
| Qwen 7B | Qwen | **100%** | 0% | 0% |
| Mistral 7B | Mistral | **0%** | 100% | 0% |

### 3. Simple Beats Complex

DIM (mean-difference) matches or exceeds COSMIC (SVD + automated layer selection):

| Model | DIM Rate | COSMIC Rate | Gap |
|-------|---------|------------|-----|
| Qwen 3B | **100%** | 100% | 0pp |
| Qwen 14B | **90%** | 90% | 0pp |
| Qwen 32B | **60%** | 10% | **+50pp** |
| Gemma 9B | **90%** | 70% | +20pp |

### 4. Quantization Robustness (Scale-Dependent)

| Model | FP16 | INT8 | INT4 |
|-------|------|------|------|
| Qwen 7B | **100%** | **100%** | **100%** |
| Qwen 32B | **77%** | **83%** | **57%** |

Direction cosine similarities remain > 0.97 across quantization levels, yet functional performance diverges at scale.

### 5. Tooling Sensitivity

On Qwen 7B, extraction method produces 90-percentage-point swing:

| Extraction Method | Coherent Refusal Rate |
|------------------|---------------------|
| nnsight (graph tracing) | **100%** |
| Raw PyTorch hooks | **10%** |

---

## Repository Structure

```
.
├── README.md                    # This file
├── LICENSE                      # MIT license
├── requirements.txt             # Python dependencies
├── paper/
│   └── paper.md                 # Full paper (unified draft)
├── src/
│   ├── extract.py               # Direction extraction (DIM + COSMIC)
│   ├── steer.py                 # Steering + evaluation pipeline
│   ├── prompts.py               # Prompt lists (extraction + eval)
│   └── utils.py                 # Shared utilities
├── results/
│   ├── final_results.json       # Canonical experimental results
│   └── figures/                 # Placeholder for generated figures
├── notebooks/
│   └── reproduce_results.ipynb  # Reproduction guide (skeleton)
└── docs/
    ├── EXPERIMENT_PLAN.md       # Historical V3 experiment design
    ├── COSMIC_VERIFICATION.md   # COSMIC implementation notes
    └── FOLLOW_UP_EXPERIMENT_SPEC.md  # Post-paper follow-up plan
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/activation-steering-scaling.git
cd activation-steering-scaling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.49+
- nnsight 0.3.7
- numpy, pandas, matplotlib

See `requirements.txt` for full dependency list.

---

## Reproducing Key Results

### 1. Extract a Refusal Direction (DIM)

```python
from nnsight import LanguageModel
from src.extract import extract_dim_direction
from src.prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

# Load model
model = LanguageModel("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
tokenizer = model.tokenizer

# Extract direction at layer 16 (60% depth)
direction, metadata = extract_dim_direction(
    model=model,
    tokenizer=tokenizer,
    harmful_prompts=HARMFUL_PROMPTS,
    harmless_prompts=HARMLESS_PROMPTS,
    layer_idx=16
)

print(f"Direction norm: {metadata['raw_norm']:.4f}")
```

### 2. Apply Steering and Evaluate

```python
import torch
from src.steer import evaluate_steering
from src.prompts import EVAL_PROMPTS

# Convert to tensor
direction_tensor = torch.tensor(direction, dtype=torch.float32).to(model.device)

# Evaluate on test prompts
results = evaluate_steering(
    model=model,
    tokenizer=tokenizer,
    test_prompts=EVAL_PROMPTS,
    direction=direction_tensor,
    layer_idx=16,
    multiplier=15
)

print(f"Coherent refusal rate: {results['coherent_refusal_rate']:.1f}%")
print(f"Garbled rate: {results['garbled_rate']:.1f}%")

# Inspect samples
for sample in results['samples'][:3]:
    print(f"\nPrompt: {sample['prompt']}")
    print(f"Response: {sample['response'][:100]}...")
    print(f"Quality: {sample['quality']}")
```

### 3. Compare DIM vs COSMIC

```python
from src.extract import extract_cosmic_direction

# Extract COSMIC direction
cosmic_dir, cosmic_meta = extract_cosmic_direction(
    model=model,
    tokenizer=tokenizer,
    harmful_prompts=HARMFUL_PROMPTS * 10,  # COSMIC needs more prompts
    harmless_prompts=HARMLESS_PROMPTS * 10,
    layer_range=(1, 22)  # 1 to 80% of 28 layers
)

print(f"COSMIC selected layer: {cosmic_meta['selected_layer']}")

# Compare directions
import numpy as np
cosine = np.dot(direction, cosmic_dir) / (np.linalg.norm(direction) * np.linalg.norm(cosmic_dir))
print(f"DIM-COSMIC cosine similarity: {cosine:.4f}")
```

### 4. Test Quantization Effects

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load INT8 quantized model
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Extract direction from quantized model
# (use nnsight wrapper around the quantized model)
# ... evaluate as above ...
```

See `notebooks/reproduce_results.ipynb` for a complete walkthrough.

---

## Hardware Requirements

| Model Size | GPU | VRAM (FP16) | VRAM (INT8) |
|-----------|-----|-------------|-------------|
| 2-3B | T4 / A10G | ~6 GB | ~3 GB |
| 7B | A10G | ~14 GB | ~7 GB |
| 14B | A100 | ~28 GB | ~14 GB |
| 32B | A100-80GB | ~64 GB | ~32 GB |

For quantization experiments, INT4 reduces VRAM by ~75%.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{mohammad2026inverse,
  title={Inverse Scaling in Activation Steering: Architecture and Scale Dependence of Refusal Manipulation},
  author={Mohammad, Sohail},
  year={2026},
  note={Research preprint}
}
```

---

## Paper

The full paper is available in `paper/paper.md`. Key sections:

- **§1 Introduction:** Motivation and core findings
- **§3 Background:** DIM, COSMIC, and refusal mechanisms
- **§4 Methods:** Experimental setup and evaluation
- **§5 Results:** Detailed findings with examples
- **§8 Discussion:** Mechanistic interpretations
- **§9 Limitations:** Scope and caveats

---

## Results Data

All experimental results are available in `results/final_results.json` with the following structure:

```json
{
  "phase1_architecture_comparison": {
    "qwen_7b": { "coherent_refusal_rate": 100.0, ... },
    "gemma_9b": { "coherent_refusal_rate": 97.0, ... },
    "mistral_7b": { "coherent_refusal_rate": 0.0, "garbled_rate": 100.0, ... }
  },
  "phase2_qwen_size_sweep": {
    "qwen_3b": { "best_coherent": 100.0, ... },
    "qwen_7b": { "best_coherent": 100.0, ... },
    "qwen_14b": { "best_coherent": 90.0, ... },
    "qwen_32b": { "best_coherent": 77.0, ... }
  },
  "phase3_quantization": { ... }
}
```

Each result includes:
- Coherent refusal rate, garbled rate, normal rate
- Direction norm
- Layer index and percentage
- Multiplier used
- Sample outputs with quality labels

---

## Limitations

1. **Sample size:** 30 benign prompts (n=30) yields wide confidence intervals for intermediate effects. Larger studies needed for precise effect size estimation.

2. **Single evaluator:** Output quality classification performed by single rater. Inter-rater reliability not assessed.

3. **Greedy decoding only:** All experiments use temperature=0. Sampling may reveal different steering dynamics.

4. **Architecture coverage:** Three families tested (Qwen, Gemma, Mistral). Llama, GPT-style, and MoE architectures not systematically evaluated.

5. **Single behavior:** Focus on refusal steering. Generalization to other behaviors (truthfulness, sentiment, etc.) unknown.

See paper §12 (Limitations) for full discussion.

---

## Acknowledgments

This research was conducted independently. Special thanks to the authors of:
- **nnsight** (Fiotto-Kaufman et al.) for graph-level activation tracing
- **COSMIC** (Siu et al.) for the SVD-based direction extraction method
- **Arditi et al.** for demonstrating the single-direction refusal phenomenon

---

## License

MIT License. See `LICENSE` file for details.

---

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

**Author:** Sohail Mohammad  
**Year:** 2026
