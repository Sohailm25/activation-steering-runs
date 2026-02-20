# Publication Style Spec — sohailmo.ai Research Pages

**Source of truth:** https://sohailmo.ai/research/activation-steering/
**Repo:** https://github.com/Sohailm25/Sohailm25.github.io (branch: master)
**Static site generator:** Pelican
**Template engine:** Distill.pub v2 (`https://distill.pub/template.v2.js`)

---

## 1. Color Palette (Everforest Dark)

| Token | Hex | Usage |
|-------|-----|-------|
| `--ef-bg` | `#2d353b` | Page background |
| `--ef-bg-dim` | `#232a2e` | Title bar, byline, table headers |
| `--ef-bg-card` | `#343f44` | Callout boxes, hover cards |
| `--ef-text` | `#d3c6aa` | Body text |
| `--ef-text-muted` | `#859289` | Captions, byline, secondary text |
| `--ef-green` | `#A7C080` | H1 title, H2 headings, table headers |
| `--ef-aqua` | `#83C092` | H3 headings, links, citation numbers |
| `--ef-blue` | `#7fbbb3` | Accent (sparingly) |
| `--ef-red` | `#e67e80` | Negative/failure data points |
| `--ef-orange` | `#e69875` | Warning, secondary accent |
| `--ef-yellow` | `#dbbc7f` | Highlight accent |
| `--ef-purple` | `#d699b6` | Tertiary accent |
| `--ef-border` | `#4a555b` | Table borders, H2 underlines, figure borders |

## 2. Typography

- **Body:** Distill.pub default (system serif stack)
- **H1 (title):** `color: var(--ef-green)` — inside `<d-title>`
- **H2:** `margin-top: 2em; border-bottom: 1px solid var(--ef-border); padding-bottom: 0.3em; color: var(--ef-green)`
- **H3:** `margin-top: 1.5em; color: var(--ef-aqua)`
- **Code blocks:** `background: var(--ef-bg-dim); border: 1px solid var(--ef-border); border-radius: 4px; padding: 1em; color: var(--ef-green); font-size: 0.85em`
- **Links:** `color: var(--ef-aqua)` → hover: `var(--ef-green)`

## 3. Figure/Chart Styling

- **Border:** `1px solid var(--ef-border); border-radius: 4px`
- **Width:** `100%`
- **Caption:** Below figure, `font-size: 0.9em; color: var(--ef-text-muted); margin-top: 0.5em`
- **Margin:** `2em 0`
- **Chart palette (for new figures):** Use Everforest palette colors in order: green → aqua → blue → orange → red → purple
- **Background of chart images:** Should be transparent or match `--ef-bg` (#2d353b)
- **Axis/title text in charts:** Use `--ef-text` (#d3c6aa) or `--ef-text-muted` (#859289)

## 4. Tables

- **Border-collapse:** collapse
- **Header row:** `background: var(--ef-bg-dim); color: var(--ef-green); font-weight: 600`
- **Cell border:** `1px solid var(--ef-border)`
- **Cell padding:** `8px 12px`
- **Row hover:** `background: var(--ef-bg-card)`
- **Font size:** `0.9em`
- **Width:** `100%`
- **Highlight cells:** `background: rgba(167, 192, 128, 0.15)` (green tint)

## 5. Callout/Hypothesis Boxes

```css
.hypothesis-box {
    background: var(--ef-bg-card);
    border-left: 3px solid var(--ef-green);
    padding: 1em 1.5em;
    margin: 1.5em 0;
    border-radius: 0 4px 4px 0;
}
```

## 6. Section Structure (Distill template)

```
<d-front-matter> → title, authors, katex config
<d-title> → h1, subtitle p
<d-byline> → auto-populated from front-matter
<d-article> → main content
  h2 sections (numbered: §1, §2, etc.)
  h3 subsections
  <d-math> for block equations
  <d-cite> for citations
  <d-footnote> for footnotes
<d-appendix> → supplementary
<d-bibliography src="bibliography.bib">
```

**Article container override:** `display: block; max-width: 780px; margin: 0 auto; padding: 0 24px`

## 7. Footer Action Links (CRITICAL)

At the bottom of each paper's research listing on `/pages/research/`, add:

```markdown
[Paper (PDF)]({static}/papers/<slug>.pdf) · [Code (GitHub)](https://github.com/Sohailm25/<repo-name>)
```

Rendered format: `Paper (PDF) · Code (GitHub)` with middle-dot separator.

## 8. Title Click Behavior

The paper title in the research listing page is a link to the distill page:

```markdown
### [Paper Title](/research/<slug>/)
```

Clicking the title navigates to the distill-formatted research page.

## 9. Distill Page Shadow DOM Overrides

Distill.pub components (`d-cite`, `d-footnote`, `d-bibliography`) use shadow DOM. Inject Everforest colors into shadow roots via MutationObserver script (see existing activation-steering/index.html for pattern).

## 10. Layout Conventions

- **d-article max-width:** 780px
- **Figure width:** 100% of article container
- **Table width:** 100% of article container
- **Margin rhythm:** 2em between major sections, 1.5em between subsections
- **List spacing:** Default Distill

---

**This spec is locked. All Phase-2 publication work must conform to these patterns.**
