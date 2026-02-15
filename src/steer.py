"""
Activation Steering and Evaluation Pipeline

This module provides functions for:
- Applying steering vectors during generation
- Evaluating steering effectiveness on test prompts
- Classifying and aggregating results
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .utils import apply_chat_template, classify_output_quality, is_refusal

logger = logging.getLogger(__name__)

MAX_GEN_TOKENS = 100


# ── Steering ───────────────────────────────────────────────────────


def _get_layer_list(model):
    """Get the raw nn.ModuleList of layers for forward hook registration.

    Args:
        model: nnsight LanguageModel.

    Returns:
        nn.ModuleList of transformer layers.
    """
    underlying = model._model
    if hasattr(underlying, "model") and hasattr(underlying.model, "layers"):
        return underlying.model.layers
    elif hasattr(underlying, "transformer") and hasattr(underlying.transformer, "h"):
        return underlying.transformer.h
    else:
        raise ValueError(f"Unsupported architecture: {type(underlying).__name__}")


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    direction: torch.Tensor,
    layer_idx: int,
    multiplier: float,
    max_tokens: int = MAX_GEN_TOKENS,
) -> str:
    """Generate response with activation steering applied via forward hooks.

    Registers hooks on all layers from layer_idx to end of model that add
    multiplier * direction to the hidden states.

    Args:
        model: nnsight LanguageModel.
        tokenizer: Model tokenizer.
        prompt: User prompt text.
        direction: Steering direction tensor on model device, shape (hidden_dim,).
        layer_idx: First layer to apply steering.
        multiplier: Scaling factor for the direction.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated text string (decoded, special tokens removed).
    """
    device = next(model._model.parameters()).device
    layers = _get_layer_list(model)
    n_layers = len(layers)

    templated = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(templated, return_tensors="pt").to(device)

    def steering_hook(module, input, output, _mult=multiplier, _dir=direction):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden_steered = hidden + _mult * _dir.to(hidden.dtype)
            return (hidden_steered,) + tuple(output[j] for j in range(1, len(output)))
        else:
            return output + _mult * _dir.to(output.dtype)

    handles = []
    for idx in range(layer_idx, n_layers):
        handle = layers[idx].register_forward_hook(steering_hook)
        handles.append(handle)

    try:
        with torch.no_grad():
            output = model._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding
            )
        response_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()

    return response


# ── Evaluation ─────────────────────────────────────────────────────


def evaluate_steering(
    model,
    tokenizer,
    test_prompts: list[str],
    direction: torch.Tensor,
    layer_idx: int,
    multiplier: float,
    max_tokens: int = MAX_GEN_TOKENS,
) -> dict[str, Any]:
    """Evaluate steering effectiveness on test prompts.

    Applies steering to each test prompt and classifies the outputs.

    Args:
        model: nnsight LanguageModel.
        tokenizer: Model tokenizer.
        test_prompts: List of benign test prompts.
        direction: Steering direction tensor on model device.
        layer_idx: First layer to apply steering.
        multiplier: Scaling factor for direction.
        max_tokens: Maximum tokens to generate.

    Returns:
        Dict with:
            refusal_rate: Percentage of responses containing refusal keywords.
            coherent_refusal_rate: Percentage of coherent refusals.
            garbled_rate: Percentage of garbled outputs.
            normal_rate: Percentage of normal (unsteered) responses.
            n_samples: Total number of test prompts.
            samples: List of dicts with prompt, response, quality for first 5.
    """
    refusals = 0
    coherent_refusals = 0
    garbled_count = 0
    normal_count = 0
    samples = []

    for i, prompt in enumerate(test_prompts):
        try:
            response = generate_with_steering(
                model, tokenizer, prompt, direction, layer_idx, multiplier, max_tokens
            )

            quality = classify_output_quality(response)
            ref = is_refusal(response)

            if ref:
                refusals += 1
            if quality == "coherent":
                coherent_refusals += 1
            elif quality == "garbled":
                garbled_count += 1
            else:  # normal
                normal_count += 1

            if i < 5:  # Save first 5 examples
                samples.append({
                    "prompt": prompt,
                    "response": response[:200],  # Truncate for readability
                    "quality": quality,
                    "is_refusal": ref,
                })

        except Exception as e:
            logger.warning(f"Error on prompt {i}: {e}")

    n = len(test_prompts)
    return {
        "refusal_rate": refusals / n * 100,
        "coherent_refusal_rate": coherent_refusals / n * 100,
        "garbled_rate": garbled_count / n * 100,
        "normal_rate": normal_count / n * 100,
        "n_samples": n,
        "samples": samples,
    }


def classify_steering_result(coherent_refusal_rate: float) -> str:
    """Classify steering result based on coherent refusal rate.

    Args:
        coherent_refusal_rate: Percentage of coherent refusals.

    Returns:
        'success' (≥60%), 'partial' (20-59%), or 'failure' (<20%).
    """
    if coherent_refusal_rate >= 60:
        return "success"
    elif coherent_refusal_rate >= 20:
        return "partial"
    else:
        return "failure"
