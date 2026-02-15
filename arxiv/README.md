# arXiv Submission: Inverse Scaling in Activation Steering

This directory contains the complete LaTeX project for arXiv submission.

## Files

- **main.tex** - Main paper document
- **references.bib** - Bibliography in BibTeX format
- **figures/** - PDF figures (8 total)

## Building the PDF

To compile the paper, run the following commands in sequence:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The double `pdflatex` run after `bibtex` ensures that all cross-references and citations are resolved correctly.

### Alternative: One-line build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Output

The build process will generate `main.pdf` containing the complete paper with figures and bibliography.

## Requirements

- A TeX distribution (TeX Live, MiKTeX, or MacTeX)
- Standard LaTeX packages:
  - amsmath, amssymb
  - graphicx
  - hyperref
  - booktabs
  - natbib
  - xcolor
  - microtype
  - tcolorbox

All required packages are standard in modern TeX distributions.

## Figures

The 8 figures are located in `figures/` and are referenced in the document:

1. `fig1_inverse_scaling.pdf` - Inverse scaling trend
2. `fig2_optimal_depth.pdf` - Optimal layer depth by model size
3. `fig3_quantization.pdf` - Quantization robustness
4. `fig4_dim_vs_cosmic.pdf` - DIM vs COSMIC comparison
5. `fig5_multiplier_sensitivity.pdf` - Multiplier sensitivity at scale
6. `fig6_layer_profiles.pdf` - Layer-by-layer profiles
7. `fig7_norm_vs_refusal.pdf` - Direction norm vs refusal rate
8. `fig8_quant_cosine_divergence.pdf` - Quantization cosine divergence

## Notes

- The paper uses `natbib` with `plainnat` bibliography style for author-year citations
- Greedy line breaking is enabled via `microtype` for better typesetting
- Hypotheses use `tcolorbox` environments for visual distinction
- All tables use `booktabs` for professional formatting

## arXiv Submission Checklist

- [x] All figures included as PDF
- [x] Bibliography in BibTeX format
- [x] Compilation produces no errors
- [x] All cross-references resolved
- [x] Hyperlinks functional
- [x] Abstract < 1920 characters
- [x] Author contact information included
