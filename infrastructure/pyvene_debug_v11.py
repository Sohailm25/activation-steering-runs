#!/usr/bin/env python3
"""v11: Trace _gather_intervention_output to understand 1792."""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 16

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
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

    # Print intervention details
    for key, intervention in imodel.interventions.items():
        print(f"Intervention key: {key}")
        print(f"  Type: {type(intervention).__name__}")
        # Print all non-private non-method attributes
        for attr in sorted(dir(intervention)):
            if attr.startswith('_'):
                continue
            val = getattr(intervention, attr)
            if callable(val):
                continue
            print(f"  {attr}: {val}")

    # Monkey-patch _gather_intervention_output to trace
    original_gather = imodel._gather_intervention_output
    
    def traced_gather(output, representations_key, unit_locations):
        print(f"\n_gather called:")
        print(f"  representations_key: {representations_key}")
        print(f"  unit_locations: {unit_locations}")
        if isinstance(output, torch.Tensor):
            print(f"  output shape: {list(output.shape)}")
        elif isinstance(output, tuple):
            print(f"  output: tuple len={len(output)}")
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: shape={list(item.shape)}")
                elif item is None:
                    print(f"    [{i}]: None")
                else:
                    print(f"    [{i}]: {type(item).__name__}")
        result = original_gather(output, representations_key, unit_locations)
        if isinstance(result, torch.Tensor):
            print(f"  result shape: {list(result.shape)}")
        return result
    
    imodel._gather_intervention_output = traced_gather

    with torch.no_grad():
        result = imodel(base={"input_ids": test_input["input_ids"]})

    c = result[0][1][0]
    if isinstance(c, torch.Tensor):
        print(f"\nFinal collected: shape={list(c.shape)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
