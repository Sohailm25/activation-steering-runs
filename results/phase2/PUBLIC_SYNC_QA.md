# Publication Sync QA Checklist

**Date:** 2026-02-20
**Operator:** Ghost

---

## 1. PDF compiles successfully ✅

- **Compile command:** `pdflatex -interaction=nonstopmode main.tex` (×2)
- **Output:** `main.pdf` — 31 pages, 509,335 bytes
- **Errors:** 0
- **Undefined references:** 0
- **Path:** `arxiv/main.pdf`

## 2. Distill page style matches reference page ✅

- **Reference:** https://sohailmo.ai/research/activation-steering/
- **Updated file:** `Sohailm25.github.io/content/extra/research/activation-steering/index.html`
- **Changes:** +69 lines (new §11 Transfer section with table, hypothesis box, interpretation, caveats)
- **Style checks:**
  - Everforest dark palette: ✅ (uses existing CSS variables)
  - Table styling: ✅ (same `var(--ef-bg-dim)`, `var(--ef-border)` patterns)
  - Hypothesis box: ✅ (uses existing `.hypothesis-box` class)
  - Color coding: ✅ (green for success TE, red for failure TE)
  - Section numbering: ✅ (renumbered: §11 Transfer, §12 Limitations, §13 Conclusion)

## 3. Title click opens distill page ✅

- **Research listing entry:** `### [Inverse Scaling in Activation Steering](/research/activation-steering/)`
- **Behavior:** Title links to `/research/activation-steering/` (distill page) — unchanged from existing pattern

## 4. Footer contains `Paper (PDF) · Code (GitHub)` ✅

- **Research listing:**
  ```
  [Paper (PDF)]({static}/papers/activation-steering-2026.pdf) · [Code (GitHub)](https://github.com/Sohailm25/activation-steering-runs)
  ```
- **Format:** Matches all other entries exactly (middle-dot separator)

## 5. Numbers/claims match approved bounded Phase-2 language ✅

### LaTeX paper (`arxiv/main.tex`):
| Claim | Value in paper | Approved value | Match |
|-------|---------------|----------------|-------|
| 14B→32B coherent | 100.0% | 100% | ✅ |
| 14B→32B TE | 1.25 | 1.25 | ✅ |
| 14B→32B CI | [1.071, 1.579] | [1.071, 1.579] | ✅ |
| 32B→14B coherent | 96.7% | 96.7% | ✅ |
| 32B→14B TE | 1.00 | 1.00 | ✅ |
| 32B→14B CI | [0.900, 1.111] | [0.900, 1.111] | ✅ |
| Q7→G9 coherent | 16.7% | 16.7% | ✅ |
| Q7→G9 TE | 0.17 | 0.17 | ✅ |
| Q7→G9 CI | [0.036, 0.321] | [0.036, 0.321] | ✅ |
| G9→Q7 coherent | 3.3% | 3.3% | ✅ |
| G9→Q7 TE | 0.03 | 0.03 | ✅ |
| G9→Q7 CI | [0.000, 0.100] | [0.000, 0.100] | ✅ |
| Same-family cross-cos | 0.324 | 0.324 | ✅ |
| Cross-family cross-cos | 0.019 | 0.019 | ✅ |

### Distill page: Same values confirmed ✅

## 6. No universal/causal overclaims ✅

- **Grep for "validated":** 0 occurrences in new content
- **Grep for "confirmed":** 0 occurrences in new content  
- **Grep for "universal":** Only in negation ("We do not claim universality")
- **Scope language present:**
  - LaTeX: "one same-family pair", "one cross-family pair", "We do not claim universality"
  - Distill: "In this protocol and tested pairs", "one same-family pair", "one cross-family pair"
  - TE>1.0 caveat present: "mechanically easier to exceed" (LaTeX), noted in both
  - Cross-cosine caveat: "two data points", "not a calibrated predictor" (both)

---

## Summary

| Check | Status |
|-------|--------|
| PDF compiles | ✅ PASS |
| Distill style match | ✅ PASS |
| Title click behavior | ✅ PASS |
| Footer links | ✅ PASS |
| Numbers match approved | ✅ PASS (14/14 values) |
| No overclaims | ✅ PASS |

**All 6 checks PASS.**
