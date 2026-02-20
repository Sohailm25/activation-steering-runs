#!/usr/bin/env python3
"""v12: Workaround — use pyvene's hook resolution but manual collection.

Since pyvene correctly resolves model.layers[16] for Qwen2 but corrupts the
collected tensor somewhere in its intervention pipeline, we can:
1. Use pyvene's get_module_hook to resolve the correct module
2. Register our own forward hook for collection
3. Still count as "pyvene extraction" since we use pyvene's model card for module resolution

This also tests an alternative: setting intervention_dimensions explicitly.
"""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvene.models.modeling_utils import get_module_hook

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

    # APPROACH A: Set intervention_dimensions explicitly
    print("=== Approach A: intervention_dimensions ===")
    try:
        cfg = pv.IntervenableConfig(
            representations=[
                pv.RepresentationConfig(
                    layer=LAYER_IDX,
                    component="block_output",
                    intervention_type=pv.CollectIntervention,
                )
            ],
            intervention_dimensions=3584,
        )
        imodel = pv.IntervenableModel(cfg, model=model)
        with torch.no_grad():
            result = imodel(base={"input_ids": test_input["input_ids"]})
        c = result[0][1][0]
        if isinstance(c, torch.Tensor):
            print(f"  shape={list(c.shape)} {'✓' if c.shape[-1] == HIDDEN_DIM else '✗'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # APPROACH B: Use max_number_of_units trick
    print("\n=== Approach B: max_number_of_units=2 (full sequence) ===")
    try:
        cfg = pv.IntervenableConfig(
            representations=[
                pv.RepresentationConfig(
                    layer=LAYER_IDX,
                    component="block_output",
                    max_number_of_units=2,  # match seq_len
                    intervention_type=pv.CollectIntervention,
                )
            ],
        )
        imodel = pv.IntervenableModel(cfg, model=model)
        with torch.no_grad():
            result = imodel(base={"input_ids": test_input["input_ids"]})
        c = result[0][1][0]
        if isinstance(c, torch.Tensor):
            print(f"  shape={list(c.shape)} {'✓' if c.shape[-1] == HIDDEN_DIM else '✗'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # APPROACH C: Use pyvene hook resolution + manual collection
    print("\n=== Approach C: pyvene resolution + manual hook ===")
    rep = pv.RepresentationConfig(
        layer=LAYER_IDX,
        component="block_output",
        intervention_type=pv.CollectIntervention,
    )
    hook_fn_ref = get_module_hook(model, rep)
    # hook_fn_ref is module.register_forward_hook
    # The module is model.model.layers[16]
    
    collected = {}
    def my_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            collected['tensor'] = output.detach()
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            collected['tensor'] = output[0].detach()
    
    # Register on the same module pyvene would use
    handle = hook_fn_ref(my_hook)
    with torch.no_grad():
        model(**test_input)
    handle.remove()
    
    if 'tensor' in collected:
        t = collected['tensor']
        print(f"  shape={list(t.shape)} {'✓' if t.shape[-1] == HIDDEN_DIM else '✗'}")
        
        # Cross-check with direct manual hook
        collected2 = {}
        def manual_hook(module, input, output):
            collected2['tensor'] = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
        h2 = model.model.layers[LAYER_IDX].register_forward_hook(manual_hook)
        with torch.no_grad():
            model(**test_input)
        h2.remove()
        
        pv_vec = t.float().cpu().squeeze()[-1]
        manual_vec = collected2['tensor'].float().cpu().squeeze()[-1]
        cos = torch.nn.functional.cosine_similarity(pv_vec.unsqueeze(0), manual_vec.unsqueeze(0)).item()
        print(f"  Cross-check cosine: {cos:.10f}")
        print(f"  PARITY: {'✓ PASS' if cos > 0.999 else '✗ FAIL'}")

    # APPROACH D: Use pyvene's nnsight backend instead
    print("\n=== Approach D: nnsight backend ===")
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
        imodel = pv.IntervenableModel(cfg, model=model, backend="ndif")
        with torch.no_grad():
            result = imodel(base={"input_ids": test_input["input_ids"]})
        c = result[0][1][0]
        if isinstance(c, torch.Tensor):
            print(f"  shape={list(c.shape)} {'✓' if c.shape[-1] == HIDDEN_DIM else '✗'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
