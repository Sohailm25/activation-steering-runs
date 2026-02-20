#!/usr/bin/env python3
"""v10: Check pyvene's computed intervention dimension + intervention internals."""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvene.models.modeling_utils import get_dimension_by_component, get_internal_model_type
from pyvene.models.intervenable_modelcard import type_to_dimension_mapping

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    model_type = get_internal_model_type(model)
    
    # Check dimension mapping
    print("=== Dimension mapping ===")
    dim_mapping = type_to_dimension_mapping.get(model_type)
    if dim_mapping:
        for comp, proposals in dim_mapping.items():
            dim = get_dimension_by_component(model_type, model.config, comp)
            print(f"  {comp:30s} → proposals={proposals}, computed_dim={dim}")
    else:
        print("  No dimension mapping found!")

    # Check intervention internals
    print("\n=== IntervenableModel intervention internals ===")
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
    
    for key, intervention in imodel.interventions.items():
        print(f"  Key: {key}")
        print(f"  Type: {type(intervention).__name__}")
        if hasattr(intervention, 'interchange_dim'):
            print(f"  interchange_dim: {intervention.interchange_dim}")
        if hasattr(intervention, 'embed_dim'):
            print(f"  embed_dim: {intervention.embed_dim}")
        for attr in dir(intervention):
            if not attr.startswith('_') and 'dim' in attr.lower():
                print(f"  {attr}: {getattr(intervention, attr)}")

    # Check _gather_intervention_output
    print("\n=== _gather_intervention_output source ===")
    import inspect
    try:
        src = inspect.getsource(imodel._gather_intervention_output)
        # Print first 40 lines
        for i, line in enumerate(src.split('\n')[:40]):
            print(f"  {line}")
    except Exception as e:
        print(f"  Could not get source: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
