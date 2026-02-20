#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v4: monkey-patch fix.

Root cause: pyvene's BaseModel.__init__ does:
    self.config.model_type = str(type(model))  # backfill
This overwrites the user-provided model_type="qwen2" with the full class string,
so the qwen2 mapping is never found.

Fix approaches tested here:
1. Monkey-patch config.model_type back after IntervenableModel init
2. Register the mappings in pyvene's type resolution system directly
3. Use pyvene's `hook_type` override to bypass auto-resolution
"""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16
HIDDEN_DIM = 3584

def main():
    from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
        qwen2_lm_type_to_module_mapping,
        qwen2_lm_type_to_dimension_mapping,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    test_input = tok("Hello world", return_tensors="pt").to(model.device)

    # FIX 1: Check if there's a type resolution function we can hook into
    print("=== Approach 1: Check IntervenableModel._get_type_config ===")
    try:
        from pyvene.models.intervenable_base import IntervenableModel as IM
        # Look at what methods resolve model type
        relevant = [m for m in dir(IM) if 'type' in m.lower() or 'mapping' in m.lower() or 'module' in m.lower()]
        print(f"  Relevant methods/attrs: {relevant}")
    except Exception as e:
        print(f"  Error: {e}")

    # FIX 2: Check how _intervention_pointers are resolved
    print("\n=== Approach 2: Inspect intervention setup ===")
    cfg = pv.IntervenableConfig(
        model_type="qwen2",
        representations=[
            pv.RepresentationConfig(
                layer=LAYER_IDX,
                component="block_output",
                intervention_type=pv.CollectIntervention,
            )
        ],
    )
    imodel = pv.IntervenableModel(cfg, model=model)
    
    # Check what module the intervention is actually hooked to
    for key in imodel.interventions:
        print(f"  Intervention key: {key}")
    
    # Check if there's an _intervention_pointers or similar
    for attr in ['_intervention_pointers', 'representations', '_key_getter_call_hook', 
                 '_intervention_state', 'sorted_keys', '_batched_setter_activation_select']:
        if hasattr(imodel, attr):
            val = getattr(imodel, attr)
            print(f"  {attr}: {type(val).__name__} = {str(val)[:200]}")

    # Check what model_type got set to
    print(f"  config.model_type after init: {imodel.config.model_type[:100]}")

    # FIX 3: Try the correct approach - subclass or use the new API
    print("\n=== Approach 3: Use represent_module_path directly ===")
    # In newer pyvene, RepresentationConfig might accept a direct module path
    try:
        # The correct path for Qwen2 CausalLM block_output at layer 16:
        module_path = f"model.layers.{LAYER_IDX}"
        
        cfg2 = pv.IntervenableConfig(
            representations=[
                {
                    "layer": LAYER_IDX,
                    "component": f"model.layers.{LAYER_IDX}.output",
                    "intervention_type": pv.CollectIntervention,
                }
            ],
        )
        imodel2 = pv.IntervenableModel(cfg2, model=model)
        with torch.no_grad():
            result = imodel2(base={"input_ids": test_input["input_ids"]})
        collected = result[1][0]
        if isinstance(collected, torch.Tensor):
            print(f"  Direct path: shape={list(collected.shape)}")
        else:
            print(f"  Direct path: type={type(collected)}")
    except Exception as e:
        print(f"  Direct path ERROR: {e}")

    # FIX 4: Monkey-patch _get_representation_key or the hooks directly
    print("\n=== Approach 4: Rebuild intervention with correct hooks ===")
    try:
        # Create model, then manually fix the hook targets
        cfg3 = pv.IntervenableConfig(
            representations=[
                pv.RepresentationConfig(
                    layer=LAYER_IDX,
                    component="block_output",
                    intervention_type=pv.CollectIntervention,
                )
            ],
        )
        imodel3 = pv.IntervenableModel(cfg3, model=model)
        
        # Now check what module is being hooked
        print(f"  Keys: {list(imodel3.interventions.keys())}")
        
        # Check _intervention_group  
        if hasattr(imodel3, '_intervention_group'):
            print(f"  _intervention_group: {imodel3._intervention_group}")
        
        # Try to find where the hook is registered
        for name, mod in model.named_modules():
            hooks = getattr(mod, '_forward_hooks', {})
            pre_hooks = getattr(mod, '_forward_pre_hooks', {})
            if hooks or pre_hooks:
                print(f"  Module '{name}' has {len(hooks)} hooks, {len(pre_hooks)} pre-hooks")
    except Exception as e:
        print(f"  Approach 4 ERROR: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
