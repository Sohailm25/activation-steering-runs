"""
Activation Steering for LLM Refusal Manipulation

This package provides tools for extracting and applying activation steering
directions to manipulate LLM refusal behavior.

Modules:
    extract: Direction extraction (DIM, COSMIC)
    steer: Steering application and evaluation
    prompts: Prompt datasets
    utils: Shared utilities
"""

__version__ = "1.0.0"
__author__ = "Sohail Mohammad"

from .extract import extract_dim_direction, extract_cosmic_direction
from .steer import generate_with_steering, evaluate_steering
from .utils import is_refusal, classify_output_quality, apply_chat_template

__all__ = [
    "extract_dim_direction",
    "extract_cosmic_direction",
    "generate_with_steering",
    "evaluate_steering",
    "is_refusal",
    "classify_output_quality",
    "apply_chat_template",
]
