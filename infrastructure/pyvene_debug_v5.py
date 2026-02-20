#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v5: check if Qwen2ForCausalLM is registered in type_to_module_mapping."""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models as hf_models

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16
HIDDEN_DIM = 3584

def main():
    # Check pyvene's type_to_module_mapping
    from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
    
    print(f"Number of registered types: {len(type_to_module_mapping)}")
    
    # Check if Qwen2ForCausalLM is registered
    qwen2_cls = hf_models.qwen2.modeling_qwen2.Qwen2ForCausalLM
    print(f"\nQwen2ForCausalLM class: {qwen2_cls}")
    print(f"Is registered in type_to_module_mapping: {qwen2_cls in type_to_module_mapping}")
    
    if qwen2_cls in type_to_module_mapping:
        mapping = type_to_module_mapping[qwen2_cls]
        print(f"block_output mapping: {mapping.get('block_output')}")
    
    # Also check what type(model) returns
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    model_type = type(model)
    print(f"\ntype(model): {model_type}")
    print(f"type(model) == Qwen2ForCausalLM: {model_type == qwen2_cls}")
    print(f"type(model) in type_to_module_mapping: {model_type in type_to_module_mapping}")
    
    # If not equal, check MRO
    print(f"\nModel MRO:")
    for cls in type(model).__mro__[:5]:
        print(f"  {cls}")
        if cls in type_to_module_mapping:
            print(f"    ^ FOUND in type_to_module_mapping!")
    
    # Check if it's the Qwen2_5 vs Qwen2 issue
    print(f"\nModel config model_type: {model.config.model_type}")
    print(f"Model class name: {type(model).__name__}")
    print(f"Module: {type(model).__module__}")

    # If type doesn't match, try the fix: register it
    if model_type not in type_to_module_mapping and qwen2_cls in type_to_module_mapping:
        print("\n=== APPLYING FIX: Register model type ===")
        type_to_module_mapping[model_type] = type_to_module_mapping[qwen2_cls]
        type_to_dimension_mapping[model_type] = type_to_dimension_mapping[qwen2_cls]
        print(f"Registered {model_type}")
        
        # Now test
        test_input = tok("Hello world", return_tensors="pt").to(model.device)
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
        with torch.no_grad():
            result = imodel(base={"input_ids": test_input["input_ids"]})
        collected = result[1][0]
        if isinstance(collected, torch.Tensor):
            shape = list(collected.shape)
            match = "✓ MATCH" if collected.shape[-1] == HIDDEN_DIM else f"✗ WRONG"
            print(f"After fix: shape={shape} {match}")
            
            if collected.shape[-1] == HIDDEN_DIM:
                # Cross-check with manual hook
                collected_manual = {}
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        collected_manual['output'] = output[0].detach()
                    else:
                        collected_manual['output'] = output.detach()
                handle = model.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
                with torch.no_grad():
                    model(**test_input)
                handle.remove()
                
                pv_vec = collected.detach().float().cpu().squeeze()[-1]
                manual_vec = collected_manual['output'].float().cpu().squeeze()[-1]
                cos = torch.nn.functional.cosine_similarity(pv_vec.unsqueeze(0), manual_vec.unsqueeze(0)).item()
                print(f"Cross-check cosine: {cos:.10f}")
                print(f"PARITY: {'✓ PASS' if cos > 0.999 else '✗ FAIL'}")

    print("\nDone.")

if __name__ == "__main__":
    main()
