#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v6: trace where hooks actually get placed."""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvene.models.modeling_utils import get_module_hook, get_internal_model_type, getattr_for_torch_module
from pyvene.models.intervenable_modelcard import type_to_module_mapping

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16
HIDDEN_DIM = 3584

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    test_input = tok("Hello world", return_tensors="pt").to(model.device)

    # 1. Check what get_module_hook resolves to
    print("=== get_module_hook resolution ===")
    rep = pv.RepresentationConfig(
        layer=LAYER_IDX,
        component="block_output",
        intervention_type=pv.CollectIntervention,
    )
    
    model_type = get_internal_model_type(model)
    print(f"Internal model type: {model_type}")
    print(f"In mapping: {model_type in type_to_module_mapping}")
    
    if model_type in type_to_module_mapping:
        type_info = type_to_module_mapping[model_type]["block_output"]
        param_name = type_info[0] % LAYER_IDX
        hook_type = type_info[1]
        print(f"Resolved parameter_name: {param_name}")
        print(f"Hook type: {hook_type}")
        
        module = getattr_for_torch_module(model, param_name)
        print(f"Resolved module: {type(module).__name__}")
        print(f"Module is layer {LAYER_IDX}: {module is model.model.layers[LAYER_IDX]}")

    # 2. Get the hook function and test it manually
    print("\n=== Manual hook via get_module_hook ===")
    hook_fn = get_module_hook(model, rep)
    print(f"Hook function: {hook_fn}")
    
    # 3. Create IntervenableModel and trace what happens
    print("\n=== IntervenableModel internals ===")
    cfg = pv.IntervenableConfig(
        representations=[rep],
    )
    imodel = pv.IntervenableModel(cfg, model=model)
    
    # Check hooks on model after init
    print("Hooks after IntervenableModel init:")
    for name, mod in model.named_modules():
        fh = dict(getattr(mod, '_forward_hooks', {}))
        pfh = dict(getattr(mod, '_forward_pre_hooks', {}))
        if fh or pfh:
            print(f"  {name}: {len(fh)} forward, {len(pfh)} pre-forward")
    
    # 4. Run forward and check what's collected
    print("\n=== Forward pass collection ===")
    with torch.no_grad():
        result = imodel(base={"input_ids": test_input["input_ids"]})
    
    # Check all parts of result
    print(f"Result type: {type(result)}")
    if isinstance(result, tuple):
        print(f"Result length: {len(result)}")
        for i, r in enumerate(result):
            if isinstance(r, torch.Tensor):
                print(f"  result[{i}]: Tensor shape={list(r.shape)}")
            elif isinstance(r, (list, tuple)):
                print(f"  result[{i}]: {type(r).__name__} len={len(r)}")
                for j, item in enumerate(r):
                    if isinstance(item, torch.Tensor):
                        print(f"    [{j}]: Tensor shape={list(item.shape)}")
                    else:
                        print(f"    [{j}]: {type(item).__name__}")
            else:
                print(f"  result[{i}]: {type(r).__name__}")
    
    # Check hooks after forward
    print("\nHooks after forward:")
    for name, mod in model.named_modules():
        fh = dict(getattr(mod, '_forward_hooks', {}))
        pfh = dict(getattr(mod, '_forward_pre_hooks', {}))
        if fh or pfh:
            print(f"  {name}: {len(fh)} forward, {len(pfh)} pre-forward")

    print("\nDone.")

if __name__ == "__main__":
    main()
