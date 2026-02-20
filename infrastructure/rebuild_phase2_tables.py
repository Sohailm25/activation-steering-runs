#!/usr/bin/env python3
"""
Phase-2 reproducibility runner.

Loads canonical JSON artifacts, recomputes all metrics, TE, bootstrap CIs,
and emits markdown tables matching PHASE2_TRANSFER_RESULTS.md.

Usage:
    python3 infrastructure/rebuild_phase2_tables.py [--output results/phase2/GENERATED_TABLES.md]

No GPU or Modal required. CPU-only, ~5 seconds.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "results" / "phase2"

# ── Canonical artifact registry ──────────────────────────────────────────────

CANONICAL_ARTIFACTS = {
    # B1 self-controls (seed 42)
    "b1_14b_self_s42": "b1_qwen-14b_to_qwen-14b_m10.0_20260219_223147.json",
    "b1_32b_self_s42": "b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json",
    # B1 transfers (seed 42)
    "b1_14to32_s42": "b1_qwen-14b_to_qwen-32b_m15.0_20260219_224941.json",
    "b1_32to14_s42": "b1_qwen-32b_to_qwen-14b_m10.0_20260219_224854.json",
    # B1 self-controls (seed 43)
    "b1_14b_self_s43": "b1_qwen-14b_to_qwen-14b_m10.0_s43_20260219_225813.json",
    "b1_32b_self_s43": "b1_qwen-32b_to_qwen-32b_m15.0_s43_20260219_225815.json",
    # B1 transfers (seed 43)
    "b1_14to32_s43": "b1_qwen-14b_to_qwen-32b_m15.0_s43_20260219_225845.json",
    "b1_32to14_s43": "b1_qwen-32b_to_qwen-14b_m10.0_s43_20260219_230041.json",
    # B1 calibration
    "cal_14b_m10": "b1_qwen-14b_to_qwen-14b_m10.0_20260219_222846.json",
    "cal_14b_m12.5": "b1_qwen-14b_to_qwen-14b_m12.5_20260219_223352.json",
    "cal_14b_m15": "b1_qwen-14b_to_qwen-14b_m15.0_20260219_223338.json",
    "cal_14b_m17.5": "b1_qwen-14b_to_qwen-14b_m17.5_20260219_223442.json",
    "cal_32b_m10": "b1_qwen-32b_to_qwen-32b_m10.0_20260219_223555.json",
    "cal_32b_m12.5": "b1_qwen-32b_to_qwen-32b_m12.5_20260219_223314.json",
    "cal_32b_m15": "b1_qwen-32b_to_qwen-32b_m15.0_20260219_223419.json",
    "cal_32b_m17.5": "b1_qwen-32b_to_qwen-32b_m17.5_20260219_223141.json",
    # B2
    "b2_gemma_self": "b2_gemma-9b_to_gemma-9b_m25.0_20260219_234111.json",
    "b2_q7_to_g9": "b2_qwen-7b_to_gemma-9b_m25.0_20260220_000100.json",
    "b2_g9_to_q7": "b2_gemma-9b_to_qwen-7b_m15.0_20260219_235921.json",
}


def load_artifact(key: str) -> dict:
    path = PHASE2 / CANONICAL_ARTIFACTS[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text())


def file_sha256(key: str) -> str:
    path = PHASE2 / CANONICAL_ARTIFACTS[key]
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci_from_counts(k: int, n: int, n_boot: int = 10000, seed: int = 999) -> Tuple[float, float]:
    """Bootstrap CI from aggregate counts (no per-prompt data)."""
    rng = np.random.RandomState(seed)
    labels = [1] * k + [0] * (n - k)
    means = [np.mean([labels[i] for i in rng.randint(0, n, size=n)]) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ci_from_per_prompt(per_prompt: List[dict], n_boot: int = 10000, seed: int = 999) -> Tuple[float, float]:
    """Bootstrap CI from per-prompt labels."""
    labels = [1 if p["quality"] == "coherent" else 0 for p in per_prompt]
    n = len(labels)
    rng = np.random.RandomState(seed)
    means = [np.mean([labels[i] for i in rng.randint(0, n, size=n)]) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_te_ci(
    transfer_k: int, transfer_n: int,
    self_k: int, self_n: int,
    n_boot: int = 10000, seed: int = 999,
    transfer_per_prompt: Optional[List[int]] = None,
    self_per_prompt: Optional[List[int]] = None,
) -> Tuple[float, float, float]:
    """Bootstrap TE = transfer_coherent / self_coherent. Returns (point, ci_lo, ci_hi).
    
    Uses per_prompt labels when available (preserves prompt identity);
    falls back to aggregate-count reconstruction otherwise.
    """
    rng = np.random.RandomState(seed)
    t_labels = transfer_per_prompt if transfer_per_prompt is not None else [1] * transfer_k + [0] * (transfer_n - transfer_k)
    s_labels = self_per_prompt if self_per_prompt is not None else [1] * self_k + [0] * (self_n - self_k)
    n_t, n_s = len(t_labels), len(s_labels)
    te_vals = []
    for _ in range(n_boot):
        idx_t = rng.randint(0, n_t, size=n_t)
        idx_s = rng.randint(0, n_s, size=n_s)
        t = np.mean([t_labels[i] for i in idx_t])
        s = np.mean([s_labels[i] for i in idx_s])
        if s > 0:
            te_vals.append(t / s)
    point = (transfer_k / transfer_n) / (self_k / self_n) if self_k > 0 else float("inf")
    return point, float(np.percentile(te_vals, 2.5)), float(np.percentile(te_vals, 97.5))


# ── Extract metrics ──────────────────────────────────────────────────────────

@dataclass
class Metrics:
    coherent: float
    garbled: float
    normal: float
    refusal: float
    n: int
    k_coherent: int
    cross_cos: Optional[float] = None


def get_per_prompt_labels(d: dict) -> Optional[List[int]]:
    """Extract per-prompt coherent labels if available."""
    pp = d.get("eval", {}).get("per_prompt")
    if pp:
        return [1 if p["quality"] == "coherent" else 0 for p in pp]
    return None


def extract_metrics(d: dict) -> Metrics:
    e = d["eval"]
    n = e["n_prompts"]
    coherent = e["coherent_rate"]
    k = round(coherent * n / 100)
    cross_cos = d.get("geometric", {}).get("cross_cosine")
    return Metrics(
        coherent=round(coherent, 1),
        garbled=round(e["garbled_rate"], 1),
        normal=round(e["normal_rate"], 1),
        refusal=round(e["refusal_rate"], 1),
        n=n,
        k_coherent=k,
        cross_cos=round(cross_cos, 3) if cross_cos is not None else None,
    )


# ── Table generation ─────────────────────────────────────────────────────────

def generate_tables() -> str:
    lines = []
    lines.append("<!-- GENERATED by infrastructure/rebuild_phase2_tables.py -->")
    lines.append("<!-- Do not edit manually. Rerun script to update. -->")
    lines.append("")

    # ── B1 Self-Controls ──
    lines.append("## B1 — Self-Controls")
    lines.append("")
    lines.append("| Model | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |")
    lines.append("|-------|-----------|----------|---------|--------|---------|-------------------|")
    for key, label, m in [("b1_14b_self_s42", "Qwen-14B", 10.0), ("b1_32b_self_s42", "Qwen-32B", 15.0)]:
        d = load_artifact(key)
        met = extract_metrics(d)
        ci_lo, ci_hi = bootstrap_ci_from_counts(met.k_coherent, met.n)
        lines.append(f"| {label} | {m} | {met.coherent}% | {met.garbled}% | {met.normal}% | {met.refusal}% | [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] |")
    lines.append("")

    # ── B1 Transfers ──
    lines.append("## B1 — Transfer Results")
    lines.append("")
    lines.append("| Transfer | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |")
    lines.append("|----------|-----------|----------|---------|--------|---------|-------------------|")
    for key, label, m in [("b1_14to32_s42", "14B→32B", 15.0), ("b1_32to14_s42", "32B→14B", 10.0)]:
        d = load_artifact(key)
        met = extract_metrics(d)
        pp = get_per_prompt_labels(d)
        ci_lo, ci_hi = bootstrap_ci_from_per_prompt(d["eval"]["per_prompt"]) if pp is not None else bootstrap_ci_from_counts(met.k_coherent, met.n)
        lines.append(f"| {label} | {m} | {met.coherent}% | {met.garbled}% | {met.normal}% | {met.refusal}% | [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] |")
    lines.append("")

    # ── B1 TE ──
    lines.append("## B1 — Transfer Efficiency")
    lines.append("")
    lines.append("| Transfer | TE | TE 95% CI | Cross-cosine |")
    lines.append("|----------|-----|-----------|-------------|")
    b1_pairs = [
        ("b1_14to32_s42", "b1_32b_self_s42", "14B→32B"),
        ("b1_32to14_s42", "b1_14b_self_s42", "32B→14B"),
    ]
    for t_key, s_key, label in b1_pairs:
        t_met = extract_metrics(load_artifact(t_key))
        s_met = extract_metrics(load_artifact(s_key))
        te, te_lo, te_hi = bootstrap_te_ci(t_met.k_coherent, t_met.n, s_met.k_coherent, s_met.n)
        cos = t_met.cross_cos if t_met.cross_cos is not None else "N/A"
        lines.append(f"| {label} | {te:.2f} | [{te_lo:.3f}, {te_hi:.3f}] | {cos} |")
    lines.append("")

    # ── B2 Baselines ──
    lines.append("## B2 — Baselines")
    lines.append("")
    lines.append("| Model | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |")
    lines.append("|-------|-----------|----------|---------|--------|---------|-------------------|")
    # Gemma self
    d = load_artifact("b2_gemma_self")
    met = extract_metrics(d)
    pp = d["eval"].get("per_prompt")
    if pp:
        ci_lo, ci_hi = bootstrap_ci_from_per_prompt(pp)
    else:
        ci_lo, ci_hi = bootstrap_ci_from_counts(met.k_coherent, met.n)
    lines.append(f"| Gemma-9B (self) | {d['multiplier']} | {met.coherent}% | {met.garbled}% | {met.normal}% | {met.refusal}% | [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] |")
    # Q7 self from Phase-1 (hardcoded: 100% coherent, n=30)
    lines.append(f"| Qwen-7B (self, Phase-1) | 15.0 | 100.0% | 0% | 0% | 100.0% | [100.0%, 100.0%] |")
    lines.append("")

    # ── B2 Transfers ──
    lines.append("## B2 — Cross-Family Transfer Results")
    lines.append("")
    lines.append("| Transfer | Multiplier | Coherent | Garbled | Normal | Refusal | 95% CI (coherent) |")
    lines.append("|----------|-----------|----------|---------|--------|---------|-------------------|")
    for key, label in [("b2_q7_to_g9", "Q7→G9"), ("b2_g9_to_q7", "G9→Q7")]:
        d = load_artifact(key)
        met = extract_metrics(d)
        pp = d["eval"].get("per_prompt")
        if pp:
            ci_lo, ci_hi = bootstrap_ci_from_per_prompt(pp)
        else:
            ci_lo, ci_hi = bootstrap_ci_from_counts(met.k_coherent, met.n)
        lines.append(f"| {label} | {d['multiplier']} | {met.coherent}% | {met.garbled}% | {met.normal}% | {met.refusal}% | [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] |")
    lines.append("")

    # ── B2 TE ──
    lines.append("## B2 — Transfer Efficiency")
    lines.append("")
    lines.append("| Transfer | TE | TE 95% CI | Cross-cosine |")
    lines.append("|----------|-----|-----------|-------------|")
    # Q7→G9 vs Gemma self (use per_prompt when available)
    q7g9_d = load_artifact("b2_q7_to_g9")
    gemma_self_d = load_artifact("b2_gemma_self")
    q7g9 = extract_metrics(q7g9_d)
    gemma_self = extract_metrics(gemma_self_d)
    te, te_lo, te_hi = bootstrap_te_ci(
        q7g9.k_coherent, q7g9.n, gemma_self.k_coherent, gemma_self.n,
        transfer_per_prompt=get_per_prompt_labels(q7g9_d),
        self_per_prompt=get_per_prompt_labels(gemma_self_d),
    )
    lines.append(f"| Q7→G9 | {te:.2f} | [{te_lo:.3f}, {te_hi:.3f}] | {q7g9.cross_cos} |")
    # G9→Q7 vs Q7 self (100%, n=30)
    g9q7_d = load_artifact("b2_g9_to_q7")
    g9q7 = extract_metrics(g9q7_d)
    te2, te2_lo, te2_hi = bootstrap_te_ci(
        g9q7.k_coherent, g9q7.n, 30, 30,
        transfer_per_prompt=get_per_prompt_labels(g9q7_d),
    )
    lines.append(f"| G9→Q7 | {te2:.2f} | [{te2_lo:.3f}, {te2_hi:.3f}] | {g9q7.cross_cos} |")
    lines.append("")

    # ── Combined Comparison Table ──
    lines.append("## Combined Comparison Table")
    lines.append("")
    lines.append("| Pair | Family | dim_match | Cross-cos | Coherent (transfer) | TE | TE 95% CI |")
    lines.append("|------|--------|-----------|-----------|--------------------|----|-----------|")

    rows = [
        ("14B→32B", "same (Qwen)", "✓ (5120)", "b1_14to32_s42", "b1_32b_self_s42"),
        ("32B→14B", "same (Qwen)", "✓ (5120)", "b1_32to14_s42", "b1_14b_self_s42"),
    ]
    for label, family, dim, t_key, s_key in rows:
        t_met = extract_metrics(load_artifact(t_key))
        s_met = extract_metrics(load_artifact(s_key))
        te_pt, te_lo, te_hi = bootstrap_te_ci(t_met.k_coherent, t_met.n, s_met.k_coherent, s_met.n)
        cos = t_met.cross_cos if t_met.cross_cos is not None else "N/A"
        lines.append(f"| {label} | {family} | {dim} | {cos} | {t_met.coherent}% | {te_pt:.2f} | [{te_lo:.3f}, {te_hi:.3f}] |")
    # B2 rows (use per_prompt)
    q7g9_d = load_artifact("b2_q7_to_g9")
    gemma_d = load_artifact("b2_gemma_self")
    q7g9_m = extract_metrics(q7g9_d)
    gemma_m = extract_metrics(gemma_d)
    te_pt, te_lo, te_hi = bootstrap_te_ci(
        q7g9_m.k_coherent, q7g9_m.n, gemma_m.k_coherent, gemma_m.n,
        transfer_per_prompt=get_per_prompt_labels(q7g9_d),
        self_per_prompt=get_per_prompt_labels(gemma_d),
    )
    lines.append(f"| Q7→G9 | cross (Qwen→Gemma) | ✓ (3584) | {q7g9_m.cross_cos} | {q7g9_m.coherent}% | {te_pt:.2f} | [{te_lo:.3f}, {te_hi:.3f}] |")
    g9q7_d = load_artifact("b2_g9_to_q7")
    g9q7_m = extract_metrics(g9q7_d)
    te_pt2, te_lo2, te_hi2 = bootstrap_te_ci(
        g9q7_m.k_coherent, g9q7_m.n, 30, 30,
        transfer_per_prompt=get_per_prompt_labels(g9q7_d),
    )
    lines.append(f"| G9→Q7 | cross (Gemma→Qwen) | ✓ (3584) | {g9q7_m.cross_cos} | {g9q7_m.coherent}% | {te_pt2:.2f} | [{te_lo2:.3f}, {te_hi2:.3f}] |")
    lines.append("")

    # ── Calibration Table ──
    lines.append("## B1 Calibration Sweep")
    lines.append("")
    lines.append("| Artifact | Multiplier | Coherent | Garbled |")
    lines.append("|----------|-----------|----------|---------|")
    cal_keys = [
        "cal_14b_m10", "cal_14b_m12.5", "cal_14b_m15", "cal_14b_m17.5",
        "cal_32b_m10", "cal_32b_m12.5", "cal_32b_m15", "cal_32b_m17.5",
    ]
    for key in cal_keys:
        d = load_artifact(key)
        met = extract_metrics(d)
        fname = CANONICAL_ARTIFACTS[key]
        lines.append(f"| `{fname}` | {d['multiplier']} | {met.coherent}% | {met.garbled}% |")
    lines.append("")

    # ── Artifact checksums ──
    lines.append("## Artifact Checksums (SHA-256 prefix)")
    lines.append("")
    lines.append("| Key | File | SHA-256 (first 16) |")
    lines.append("|-----|------|--------------------|")
    for key, fname in sorted(CANONICAL_ARTIFACTS.items()):
        path = PHASE2 / fname
        if path.exists():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            lines.append(f"| {key} | `{fname}` | `{sha}` |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Rebuild Phase-2 tables from canonical artifacts")
    parser.add_argument("--output", default=str(PHASE2 / "GENERATED_TABLES.md"))
    args = parser.parse_args()

    print("Loading canonical artifacts...")
    missing = []
    for key, fname in CANONICAL_ARTIFACTS.items():
        if not (PHASE2 / fname).exists():
            missing.append(fname)
    if missing:
        print(f"ERROR: Missing artifacts: {missing}", file=sys.stderr)
        sys.exit(1)

    print("Generating tables...")
    tables = generate_tables()

    out = Path(args.output)
    out.write_text(tables)
    print(f"Written to {out}")

    # Print content hash for reproducibility verification
    content_hash = hashlib.sha256(tables.encode()).hexdigest()[:16]
    print(f"Content SHA-256 (first 16): {content_hash}")


if __name__ == "__main__":
    main()
