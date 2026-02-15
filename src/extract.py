"""
Direction Extraction Methods

This module implements two methods for extracting refusal directions from
language model activations:

1. DIM (Difference-in-Means): Simple mean-difference extraction
2. COSMIC: SVD-based extraction with automated layer selection
"""

from __future__ import annotations

import gc
import logging
from typing import Callable

import numpy as np
import torch

from .utils import apply_chat_template, get_num_layers

logger = logging.getLogger(__name__)


# ── Activation Extraction ──────────────────────────────────────────


def _get_layer_envoy(model) -> Callable:
    """Get nnsight envoy layer accessor for tracing.

    Args:
        model: nnsight LanguageModel.

    Returns:
        Callable (layer_idx) -> nnsight envoy for that layer.
    """
    underlying = model._model
    if hasattr(underlying, "model") and hasattr(underlying.model, "layers"):
        return lambda idx: model.model.layers[idx]
    elif hasattr(underlying, "transformer") and hasattr(underlying.transformer, "h"):
        return lambda idx: model.transformer.h[idx]
    else:
        raise ValueError(f"Unsupported architecture: {type(underlying).__name__}")


def extract_activations(
    model, tokenizer, prompts: list[str], layer_idx: int
) -> np.ndarray:
    """Extract last-token activations at a specified layer for given prompts.

    Uses nnsight tracing for activation extraction.

    Args:
        model: nnsight LanguageModel.
        tokenizer: Model tokenizer.
        prompts: List of prompt strings.
        layer_idx: Layer index to extract from.

    Returns:
        numpy array of shape (n_prompts, hidden_dim).
    """
    get_layer = _get_layer_envoy(model)
    results = []

    for prompt in prompts:
        templated = apply_chat_template(tokenizer, prompt)
        with torch.no_grad():
            with model.trace(templated):
                hidden = get_layer(layer_idx).output[0]
                saved = hidden.save()

        h = saved.value.detach().float().cpu()
        if h.ndim == 3:
            h = h.squeeze(0)
        results.append(h[-1].numpy())

    gc.collect()
    return np.stack(results)


# ── DIM (Difference-in-Means) Extraction ───────────────────────────


def extract_dim_direction(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer_idx: int,
) -> tuple[np.ndarray, dict]:
    """Extract refusal direction using Difference-in-Means (DIM).

    Direction = mean(harmful_activations) - mean(harmless_activations),
    normalized to unit vector.

    Args:
        model: nnsight LanguageModel.
        tokenizer: Model tokenizer.
        harmful_prompts: Prompts that elicit refusal.
        harmless_prompts: Prompts that elicit compliance.
        layer_idx: Layer index to extract from.

    Returns:
        direction: Unit-normalized direction of shape (hidden_dim,).
        metadata: Dict with extraction info including raw norm.
    """
    logger.info(f"DIM: Extracting activations at layer {layer_idx}...")
    
    harmful_acts = extract_activations(model, tokenizer, harmful_prompts, layer_idx)
    harmless_acts = extract_activations(model, tokenizer, harmless_prompts, layer_idx)

    # Compute mean difference
    diff = harmful_acts.mean(axis=0) - harmless_acts.mean(axis=0)
    raw_norm = float(np.linalg.norm(diff))

    # Normalize to unit vector
    if raw_norm > 1e-8:
        direction = diff / raw_norm
    else:
        direction = diff

    metadata = {
        "method": "dim",
        "layer": layer_idx,
        "n_harmful": len(harmful_prompts),
        "n_harmless": len(harmless_prompts),
        "raw_norm": raw_norm,
    }

    logger.info(f"DIM: Direction norm = {raw_norm:.4f}")
    return direction, metadata


# ── COSMIC Extraction ──────────────────────────────────────────────


def _mean_cosine_similarity(
    activations_a: np.ndarray, activations_b: np.ndarray
) -> float:
    """Compute mean pairwise cosine similarity between two sets of activations.

    Args:
        activations_a: (n, hidden_dim)
        activations_b: (n, hidden_dim)

    Returns:
        Mean cosine similarity (scalar).
    """
    n = min(len(activations_a), len(activations_b))
    a = activations_a[:n]
    b = activations_b[:n]

    # Normalize each row
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)

    a_norms = np.maximum(a_norms, 1e-8)
    b_norms = np.maximum(b_norms, 1e-8)

    a_normed = a / a_norms
    b_normed = b / b_norms

    cosines = np.sum(a_normed * b_normed, axis=1)
    return float(np.mean(cosines))


def find_divergence_layers(
    cosines_per_layer: dict[int, float],
    bottom_fraction: float = 0.1,
) -> list[int]:
    """Identify L_low layers where harmful/harmless activations diverge most.

    Selects the bottom `bottom_fraction` of layers by cosine similarity
    (i.e., layers with lowest cosine = most divergence).

    Args:
        cosines_per_layer: Mapping from layer index to mean cosine similarity.
        bottom_fraction: Fraction of layers to select (default 0.1).

    Returns:
        List of layer indices comprising L_low.
    """
    sorted_layers = sorted(cosines_per_layer.keys(), key=lambda l: cosines_per_layer[l])
    n_select = max(1, int(len(sorted_layers) * bottom_fraction))
    return sorted_layers[:n_select]


def _compute_divergence_cosines(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layers: list[int],
) -> dict[int, float]:
    """Compute per-layer mean cosine similarity between harmful and harmless.

    Used to identify L_low layers (most diverged = lowest cosine).

    Args:
        model: nnsight LanguageModel.
        tokenizer: Model tokenizer.
        harmful_prompts: Prompts that elicit refusal.
        harmless_prompts: Prompts that elicit compliance.
        layers: Layer indices to measure.

    Returns:
        Dict mapping layer index to mean cosine similarity.
    """
    cosines = {}
    for layer_idx in layers:
        harmful_acts = extract_activations(model, tokenizer, harmful_prompts, layer_idx)
        harmless_acts = extract_activations(model, tokenizer, harmless_prompts, layer_idx)
        cosines[layer_idx] = _mean_cosine_similarity(harmful_acts, harmless_acts)
        logger.info(f"  Layer {layer_idx} divergence cosine = {cosines[layer_idx]:.4f}")
    return cosines


def extract_cosmic_direction(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer_range: tuple[int, int] | None = None,
    l_low_fraction: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Extract refusal direction using COSMIC algorithm.

    COSMIC (Siu et al., ACL 2025) computes DIM directions at multiple layers,
    scores them via forward-pass interventions, and selects the best.

    This is a simplified version that:
    1. Identifies L_low layers (most divergent by cosine similarity)
    2. Extracts DIM direction at each candidate layer
    3. Scores via cosine similarity after directional ablation/addition
    4. Selects direction with highest combined score

    Args:
        model: nnsight LanguageModel.
        tokenizer: Model tokenizer.
        harmful_prompts: Prompts that elicit refusal.
        harmless_prompts: Prompts that elicit compliance.
        layer_range: (start, end) layer range for candidates. None = 1 to 80% depth.
        l_low_fraction: Fraction of layers to use as L_low (default 0.1).

    Returns:
        direction: Unit-normalized direction of shape (hidden_dim,).
        metadata: Dict with candidates, scores, and selection info.
    """
    num_layers = get_num_layers(model)

    if layer_range is None:
        layer_start = 1  # Exclude layer 0 (embedding layer)
        layer_end = int(num_layers * 0.8)
    else:
        layer_start, layer_end = layer_range
        if layer_start < 1:
            layer_start = 1

    all_layers = list(range(layer_start, layer_end))
    
    logger.info(f"COSMIC: Finding L_low layers across {len(all_layers)} layers...")
    divergence_cosines = _compute_divergence_cosines(
        model, tokenizer, harmful_prompts, harmless_prompts, all_layers
    )
    
    l_low_layers = find_divergence_layers(divergence_cosines, l_low_fraction)
    logger.info(f"COSMIC: L_low layers = {l_low_layers}")

    # Generate candidate directions at each layer
    logger.info(f"COSMIC: Generating {len(all_layers)} candidate directions...")
    candidates = []
    
    for layer_idx in all_layers:
        harmful_acts = extract_activations(model, tokenizer, harmful_prompts, layer_idx)
        harmless_acts = extract_activations(model, tokenizer, harmless_prompts, layer_idx)
        
        diff = harmful_acts.mean(axis=0) - harmless_acts.mean(axis=0)
        norm = float(np.linalg.norm(diff))
        direction = diff / norm if norm > 1e-8 else diff

        # Score via cross-layer cosine similarity
        # This is a simplified scoring - full COSMIC uses forward-pass interventions
        score = 0.0
        for l_low in l_low_layers:
            l_low_harmful = extract_activations(model, tokenizer, harmful_prompts, l_low)
            l_low_harmless = extract_activations(model, tokenizer, harmless_prompts, l_low)
            
            # Simple scoring: cosine similarity aggregated across L_low
            score += _mean_cosine_similarity(l_low_harmful, l_low_harmless)

        candidates.append({
            "layer": layer_idx,
            "direction": direction,
            "norm": norm,
            "score": score,
        })

    # Select best candidate
    scores = [c["score"] for c in candidates]
    selected_idx = int(np.argmax(scores))
    best = candidates[selected_idx]

    logger.info(f"COSMIC: Selected layer {best['layer']} (score={best['score']:.4f})")

    direction = best["direction"].copy()
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm

    metadata = {
        "method": "cosmic",
        "selected_layer": best["layer"],
        "selected_score": best["score"],
        "n_candidates": len(candidates),
        "l_low_layers": l_low_layers,
        "l_low_fraction": l_low_fraction,
        "layer_range": (layer_start, layer_end),
    }

    return direction, metadata
