#!/usr/bin/env python3
"""Debug pyvene component mapping for Qwen2.

Goal: find which pyvene component string hooks the transformer layer
hidden state (dim=3584) rather than lm_head output (dim=152064).
"""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16

# Components to try
COMPONENTS = [
    "block_output",
    "block_input",
    "mlp_output",
    "mlp_input",
    "attention_output",
    "attention_input",
    "mlp_activation",
]

def main():
    pv_ver = getattr(pv, '__version__', 'unknown')
    print(f"pyvene version: {pv_ver}")
    
    # Check what pyvene knows about qwen2
    if hasattr(pv, 'models') and hasattr(pv.models, 'constants'):
        print("\n=== pyvene model type mappings ===")
        # Try to find qwen2 in pyvene's model config
        try:
            from pyvene.models.constants import CONST_QKV_INDICES
            print(f"CONST_QKV_INDICES keys: {list(CONST_QKV_INDICES.keys()) if isinstance(CONST_QKV_INDICES, dict) else 'not a dict'}")
        except:
            pass

    # List the actual module structure of Qwen2
    print("\n=== Qwen2 module structure (first 3 layers) ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    # Print model structure
    for name, mod in model.named_modules():
        if "layers.16" in name or name in ("model", "model.layers", "lm_head"):
            print(f"  {name}: {type(mod).__name__}")
    
    # Now test each component
    print(f"\n=== Testing pyvene components at layer {LAYER_IDX} ===")
    test_input = tok("Hello world", return_tensors="pt").to(model.device)
    
    for comp in COMPONENTS:
        try:
            cfg = pv.IntervenableConfig(
                model_type="qwen2",
                representations=[
                    pv.RepresentationConfig(
                        layer=LAYER_IDX,
                        component=comp,
                        intervention_type=pv.CollectIntervention,
                    )
                ],
            )
            imodel = pv.IntervenableModel(cfg, model=model)
            with torch.no_grad():
                result = imodel(base={"input_ids": test_input["input_ids"]})
            
            collected = result[1][0]
            if isinstance(collected, torch.Tensor):
                shape = list(collected.shape)
            elif isinstance(collected, (list, tuple)):
                shape = f"tuple/list len={len(collected)}, first={type(collected[0]).__name__}"
                if isinstance(collected[0], torch.Tensor):
                    shape += f" shape={list(collected[0].shape)}"
            else:
                shape = f"type={type(collected).__name__}"
            
            hidden_match = "✓ MATCH" if isinstance(collected, torch.Tensor) and collected.shape[-1] == 3584 else "✗"
            print(f"  {comp:25s} → shape={shape} {hidden_match}")
            
        except Exception as e:
            print(f"  {comp:25s} → ERROR: {e}")
    
    # Also try without model_type
    print(f"\n=== Testing WITHOUT model_type ===")
    for comp in ["block_output", "mlp_output"]:
        try:
            cfg = pv.IntervenableConfig(
                representations=[
                    pv.RepresentationConfig(
                        layer=LAYER_IDX,
                        component=comp,
                        intervention_type=pv.CollectIntervention,
                    )
                ],
            )
            imodel = pv.IntervenableModel(cfg, model=model)
            with torch.no_grad():
                result = imodel(base={"input_ids": test_input["input_ids"]})
            
            collected = result[1][0]
            if isinstance(collected, torch.Tensor):
                shape = list(collected.shape)
            else:
                shape = f"type={type(collected).__name__}"
            
            hidden_match = "✓ MATCH" if isinstance(collected, torch.Tensor) and collected.shape[-1] == 3584 else "✗"
            print(f"  {comp:25s} (no model_type) → shape={shape} {hidden_match}")
        except Exception as e:
            print(f"  {comp:25s} (no model_type) → ERROR: {e}")

    # Try direct submodule hook path
    print(f"\n=== Testing direct submodule path ===")
    for submod_path in [
        f"model.layers[{LAYER_IDX}]",
        f"model.layers.{LAYER_IDX}",
    ]:
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
            # Override the intervention key manually
            print(f"  Trying submodule: {submod_path}")
        except Exception as e:
            print(f"  {submod_path} → ERROR: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
