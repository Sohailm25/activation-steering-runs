#!/usr/bin/env python3
"""
Quick import test to verify package structure.

This script verifies that all modules can be imported and basic
functions are accessible. Run this before attempting to reproduce results.

Usage:
    python test_imports.py
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test prompts module
        from src.prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS, EVAL_PROMPTS, REFUSAL_KEYWORDS
        print("✓ src.prompts imported successfully")
        assert len(EVAL_PROMPTS) == 30, f"Expected 30 eval prompts, got {len(EVAL_PROMPTS)}"
        assert len(HARMFUL_PROMPTS) == 5, f"Expected 5 harmful prompts, got {len(HARMFUL_PROMPTS)}"
        print(f"  - {len(EVAL_PROMPTS)} evaluation prompts loaded")
        print(f"  - {len(REFUSAL_KEYWORDS)} refusal keywords defined")
        
        # Test utils module
        from src.utils import is_refusal, classify_output_quality, apply_chat_template
        print("✓ src.utils imported successfully")
        
        # Test refusal detection
        test_refusal = "I'm sorry, I cannot help with that request."
        test_normal = "Here's how to bake a cake..."
        assert is_refusal(test_refusal), "Refusal detection failed"
        assert not is_refusal(test_normal), "False positive in refusal detection"
        print("  - Refusal detection works")
        
        # Test quality classification
        assert classify_output_quality(test_refusal) == "coherent"
        assert classify_output_quality(test_normal) == "normal"
        assert classify_output_quality("illegal illegal illegal illegal") == "garbled"
        print("  - Output quality classification works")
        
        # Test extract module
        from src.extract import extract_dim_direction, extract_cosmic_direction
        print("✓ src.extract imported successfully")
        print("  - extract_dim_direction available")
        print("  - extract_cosmic_direction available")
        
        # Test steer module
        from src.steer import generate_with_steering, evaluate_steering, classify_steering_result
        print("✓ src.steer imported successfully")
        print("  - generate_with_steering available")
        print("  - evaluate_steering available")
        
        # Test package-level imports
        import src
        print("✓ src package imported successfully")
        print(f"  - Package version: {src.__version__}")
        
        # Verify package exports
        expected_exports = [
            "extract_dim_direction",
            "extract_cosmic_direction",
            "generate_with_steering",
            "evaluate_steering",
            "is_refusal",
            "classify_output_quality",
            "apply_chat_template",
        ]
        for export in expected_exports:
            assert hasattr(src, export), f"Missing export: {export}"
        print(f"  - All {len(expected_exports)} expected exports present")
        
        print("\n" + "="*60)
        print("✓ ALL IMPORTS SUCCESSFUL")
        print("="*60)
        print("\nYou can now:")
        print("  1. Run the Jupyter notebook: notebooks/reproduce_results.ipynb")
        print("  2. Import modules in your own scripts")
        print("  3. Load models and extract directions")
        print("\nSee README.md for usage examples.")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure you're running from the repository root and have installed dependencies:")
        print("  pip install -r requirements.txt")
        return False
        
    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        return False
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
