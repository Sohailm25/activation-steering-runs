"""
Shared Utilities for Activation Steering

This module provides utilities for:
- Refusal detection and output quality classification
- Chat template application
- Layer accessor detection across architectures
"""

from __future__ import annotations

import re
from .prompts import REFUSAL_KEYWORDS


# ── Refusal Detection ──────────────────────────────────────────────


def is_refusal(text: str) -> bool:
    """Check if response contains refusal patterns.

    Args:
        text: Model output text to check.

    Returns:
        True if any refusal keyword is found (case-insensitive).
    """
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in REFUSAL_KEYWORDS)


# ── Output Quality Classification ──────────────────────────────────


def _is_garbled(text: str) -> bool:
    """Detect garbled/degenerate output.

    Checks for:
    - Empty or very short text
    - Single character repeated many times
    - Token/word repeated 4+ times consecutively
    - Phrase repeated 3+ times
    - Mostly non-alphanumeric characters

    Args:
        text: Model output text.

    Returns:
        True if output appears garbled.
    """
    if len(text.strip()) < 5:
        return True

    # Single character repeated (e.g., "aaaaaaa")
    if re.search(r"(.)\1{7,}", text):
        return True

    # Same short token repeated with separators
    if re.search(r"(\b\S{1,5}\b\W*){4,}", text):
        # Verify it's actually the same token repeated
        tokens = re.findall(r"\b\w+\b", text.lower())
        if tokens:
            from collections import Counter
            counts = Counter(tokens)
            most_common_count = counts.most_common(1)[0][1]
            if most_common_count >= 4 and most_common_count / len(tokens) > 0.6:
                return True

    # Word repeated 4+ times in a row
    if re.search(r"\b(\w+)\b(?:\s+\1\b){3,}", text, re.IGNORECASE):
        return True

    # Phrase repeated 3+ times
    words = text.split()
    if len(words) >= 6:
        for phrase_len in range(1, 4):
            for start in range(len(words) - phrase_len * 3 + 1):
                phrase = " ".join(words[start:start + phrase_len])
                repeat_count = 0
                pos = start
                while pos + phrase_len <= len(words):
                    candidate = " ".join(words[pos:pos + phrase_len])
                    if candidate.lower() == phrase.lower():
                        repeat_count += 1
                        pos += phrase_len
                    else:
                        break
                if repeat_count >= 3:
                    return True

    # Mostly non-alphanumeric
    alnum_count = sum(1 for c in text if c.isalnum())
    if len(text) > 10 and alnum_count / len(text) < 0.3:
        return True

    return False


def classify_output_quality(text: str) -> str:
    """Classify model output quality into 3 tiers.

    Args:
        text: Model output text.

    Returns:
        'coherent' — contains refusal keywords AND is grammatically sensible
        'garbled'  — repetitive, nonsensical, or degenerate output
        'normal'   — helpful response without refusal patterns
    """
    if _is_garbled(text):
        return "garbled"

    if is_refusal(text):
        return "coherent"

    return "normal"


# ── Chat Template Handling ─────────────────────────────────────────


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Apply model's chat template to a user prompt.

    Args:
        tokenizer: HuggingFace tokenizer.
        prompt: Raw user message text.

    Returns:
        Formatted string with chat template applied. Falls back to raw
        prompt if the tokenizer has no chat template.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


# ── Layer Utilities ────────────────────────────────────────────────


def get_layer_accessor(model) -> str:
    """Returns the layer path string for the model architecture.

    Supports Qwen, Llama, Gemma, Mistral (model.layers) and GPT-2 style
    (transformer.h) architectures.

    Args:
        model: nnsight LanguageModel or nn.Module with transformers architecture.

    Returns:
        String like 'model.layers' or 'transformer.h'.

    Raises:
        ValueError: If model architecture is not recognized.
    """
    underlying = getattr(model, "_model", model)
    if hasattr(underlying, "model") and hasattr(underlying.model, "layers"):
        return "model.layers"
    elif hasattr(underlying, "transformer") and hasattr(underlying.transformer, "h"):
        return "transformer.h"
    else:
        raise ValueError(f"Unsupported architecture: {type(underlying).__name__}")


def get_num_layers(model) -> int:
    """Get total number of transformer layers in model.

    Args:
        model: nnsight LanguageModel or nn.Module.

    Returns:
        Integer count of layers.
    """
    underlying = getattr(model, "_model", model)
    if hasattr(underlying, "model") and hasattr(underlying.model, "layers"):
        return len(underlying.model.layers)
    elif hasattr(underlying, "transformer") and hasattr(underlying.transformer, "h"):
        return len(underlying.transformer.h)
    else:
        raise ValueError(f"Cannot detect layers for {type(underlying).__name__}")
