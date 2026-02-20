#!/usr/bin/env python3
"""Debug pyvene Qwen2 mapping — v2.

Root cause hypothesis: pyvene's installed version may not have qwen2 mappings,
or the model_type string isn't matching. This script:
1. Checks if qwen2 mappings exist in the installed pyvene
2. Tests with explicit model_type strings
3. Falls back to manual submodule path if mappings are missing
4. Tests the correct fix for v4_tooling_parity.py integration
"""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16
HIDDEN_DIM = 3584

def check_mappings():
    """Check what qwen2 mappings pyvene knows about."""
    print("=== pyvene mapping check ===")
    
    # Check if qwen2 model files exist
    try:
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_dimension_mapping,
        )
        print("✓ qwen2 mappings FOUND in pyvene")
        print(f"  base block_output: {qwen2_type_to_module_mapping.get('block_output')}")
        print(f"  lm block_output:   {qwen2_lm_type_to_module_mapping.get('block_output')}")
        return True
    except ImportError as e:
        print(f"✗ qwen2 mappings NOT FOUND: {e}")
        return False

    # Check the intervenable model's type resolution
    try:
        from pyvene.models.intervenable_base import IntervenableModel
        print(f"  IntervenableModel model type registry: checking...")
        # Try to see what model types are registered
        if hasattr(IntervenableModel, '_type_to_module_mapping_registry'):
            print(f"  Registry keys: {list(IntervenableModel._type_to_module_mapping_registry.keys())}")
    except Exception as e:
        print(f"  Could not inspect registry: {e}")

def test_model_type_strings(model, tokenizer):
    """Test different model_type strings."""
    print("\n=== Testing model_type strings ===")
    test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)
    
    for mt in ["qwen2", "Qwen2ForCausalLM", "qwen2_lm", None]:
        try:
            cfg = pv.IntervenableConfig(
                model_type=mt,
                representations=[
                    pv.RepresentationConfig(
                        layer=LAYER_IDX,
                        component="block_output",
                        intervention_type=pv.CollectIntervention,
                    )
                ],
            )
            imodel = pv.IntervenableModel(cfg, model=model)
            with torch.no_grad():
                result = imodel(base={"input_ids": test_input["input_ids"]})
            collected = result[1][0]
            shape = list(collected.shape) if isinstance(collected, torch.Tensor) else str(type(collected))
            match = "✓" if isinstance(collected, torch.Tensor) and collected.shape[-1] == HIDDEN_DIM else "✗"
            print(f"  model_type={str(mt):30s} → shape={shape} {match}")
        except Exception as e:
            print(f"  model_type={str(mt):30s} → ERROR: {e}")

def test_manual_submodule_hook(model, tokenizer):
    """Test hooking the submodule directly via register_forward_hook."""
    print("\n=== Manual hook test (bypass pyvene mapping) ===")
    test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)
    
    collected = {}
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            collected['output'] = output.detach()
        elif isinstance(output, tuple):
            collected['output'] = output[0].detach() if isinstance(output[0], torch.Tensor) else str(type(output[0]))
    
    # Hook at model.layers[16]
    layer = model.model.layers[LAYER_IDX]
    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**test_input)
    handle.remove()
    
    if 'output' in collected and isinstance(collected['output'], torch.Tensor):
        print(f"  model.model.layers[{LAYER_IDX}] output shape: {list(collected['output'].shape)}")
        match = "✓" if collected['output'].shape[-1] == HIDDEN_DIM else "✗"
        print(f"  Hidden dim match: {match}")
    else:
        print(f"  Unexpected output: {collected}")

def test_pyvene_auto_detection(model, tokenizer):
    """Test if pyvene can auto-detect the model type."""
    print("\n=== Auto-detection test (no model_type) ===")
    print(f"  Model class: {type(model).__name__}")
    print(f"  Model config model_type: {getattr(model.config, 'model_type', 'N/A')}")
    
    # Check what pyvene's IntervenableModel does with this model
    test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)
    try:
        cfg = pv.IntervenableConfig(
            representations=[
                pv.RepresentationConfig(
                    layer=LAYER_IDX,
                    component="block_output",
                    intervention_type=pv.CollectIntervention,
                )
            ],
        )
        imodel = pv.IntervenableModel(cfg, model=model)
        
        # Check what mapping pyvene resolved to
        if hasattr(imodel, '_intervention_type_to_module_mapping'):
            print(f"  Resolved mapping: {imodel._intervention_type_to_module_mapping}")
        if hasattr(imodel, 'model_type'):
            print(f"  Resolved model_type: {imodel.model_type}")
            
        # Check the actual intervention key
        for k, v in imodel.interventions.items():
            print(f"  Intervention key: {k}")
            
        with torch.no_grad():
            result = imodel(base={"input_ids": test_input["input_ids"]})
        collected = result[1][0]
        shape = list(collected.shape) if isinstance(collected, torch.Tensor) else str(type(collected))
        print(f"  Output shape: {shape}")
    except Exception as e:
        print(f"  ERROR: {e}")

def main():
    pv_ver = getattr(pv, '__version__', 'unknown')
    print(f"pyvene version: {pv_ver}")
    print(f"pyvene location: {pv.__file__}")
    
    has_mappings = check_mappings()
    
    print("\n=== Loading model ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    test_manual_submodule_hook(model, tok)
    test_model_type_strings(model, tok)
    test_pyvene_auto_detection(model, tok)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
