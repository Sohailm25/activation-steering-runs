#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v8: check decoder layer output format."""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # Check what the decoder layer actually returns
    print("=== Decoder layer forward output ===")
    collected = {}
    def hook_fn(module, input, output):
        collected['raw_output'] = output
        collected['type'] = type(output).__name__
        if isinstance(output, tuple):
            collected['len'] = len(output)
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    collected[f'item_{i}_shape'] = list(item.shape)
                    collected[f'item_{i}_type'] = 'Tensor'
                elif item is None:
                    collected[f'item_{i}_type'] = 'None'
                else:
                    collected[f'item_{i}_type'] = type(item).__name__
        elif isinstance(output, torch.Tensor):
            collected['shape'] = list(output.shape)
    
    handle = model.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**test_input)
    handle.remove()
    
    for k, v in collected.items():
        if k != 'raw_output':
            print(f"  {k}: {v}")

    # Now check how pyvene's hook processes this output
    print("\n=== pyvene output processing ===")
    
    # Check if pyvene's hook does output[0] automatically for tuple outputs
    from pyvene.models.intervenable_base import IntervenableModel
    
    # Look at the _gather_intervention_output or similar method
    print("IntervenableModel forward-hook-related methods:")
    for m in dir(IntervenableModel):
        if 'gather' in m.lower() or 'hook' in m.lower() or 'handler' in m.lower() or 'getter' in m.lower() or 'setter' in m.lower():
            print(f"  {m}")

    # Test: what if we use unit='pos' with explicit position?
    print("\n=== Test with explicit position ===")
    cfg = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=LAYER_IDX,
                component="block_output",
                unit="pos",
                intervention_type=pv.CollectIntervention,
            )
        ],
    )
    imodel = pv.IntervenableModel(cfg, model=model)
    with torch.no_grad():
        result = imodel(
            base={"input_ids": test_input["input_ids"]},
            unit_locations={"base": pv.GET_LOC((1, 2), last_n=1)},
        )
    
    c = result[0][1][0]
    if isinstance(c, torch.Tensor):
        print(f"  With explicit pos: shape={list(c.shape)}")

    # Test: what about "mlp_output" instead?
    print("\n=== Test mlp_output ===")
    cfg2 = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=LAYER_IDX,
                component="mlp_output",
                intervention_type=pv.CollectIntervention,
            )
        ],
    )
    imodel2 = pv.IntervenableModel(cfg2, model=model)
    with torch.no_grad():
        result2 = imodel2(base={"input_ids": test_input["input_ids"]})
    
    c2 = result2[0][1][0]
    if isinstance(c2, torch.Tensor):
        print(f"  mlp_output: shape={list(c2.shape)}")
        if c2.shape[-1] == HIDDEN_DIM:
            print(f"  ✓ MATCH!")
            # Cross-check
            mc = {}
            def mlp_hook(module, input, output):
                mc['output'] = output.detach() if isinstance(output, torch.Tensor) else output
            h = model.model.layers[LAYER_IDX].mlp.register_forward_hook(mlp_hook)
            with torch.no_grad():
                model(**test_input)
            h.remove()
            
            manual = mc['output'].float().cpu().squeeze()[-1] if isinstance(mc['output'], torch.Tensor) else None
            pv_vec = c2.detach().float().cpu().squeeze()[-1] if c2.ndim >= 2 else c2.detach().float().cpu().squeeze()
            if manual is not None:
                cos = torch.nn.functional.cosine_similarity(pv_vec.unsqueeze(0), manual.unsqueeze(0)).item()
                print(f"  Cross-check cosine: {cos:.10f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
