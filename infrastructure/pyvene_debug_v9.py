#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v9: test mlp_output + understand the 1792 dim."""
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

    components = ["block_output", "block_input", "mlp_output", "mlp_input", "attention_output", "attention_input"]
    
    for comp in components:
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
            
            # Try both indexing paths
            old_way = None
            new_way = None
            
            # Old way: result[1][0]
            try:
                r1 = result[1]
                if hasattr(r1, 'logits'):
                    old_way = f"CausalLMOutput logits={list(r1.logits.shape)}"
                elif isinstance(r1, (list, tuple)) and len(r1) > 0:
                    if isinstance(r1[0], torch.Tensor):
                        old_way = f"shape={list(r1[0].shape)}"
            except:
                pass
            
            # New way: result[0][1][0]
            try:
                c = result[0][1][0]
                if isinstance(c, torch.Tensor):
                    new_way = f"shape={list(c.shape)}"
                    match = "✓" if c.shape[-1] == HIDDEN_DIM else f"✗ (dim={c.shape[-1]})"
                else:
                    new_way = f"type={type(c).__name__}"
                    match = "?"
            except Exception as e:
                new_way = f"error: {e}"
                match = "?"
            
            print(f"  {comp:25s} | old=result[1][0]: {old_way} | new=result[0][1][0]: {new_way} {match}")
        except Exception as e:
            print(f"  {comp:25s} | ERROR: {e}")

    # Now try mlp_output with cross-check
    print("\n=== mlp_output cross-check ===")
    cfg = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=LAYER_IDX,
                component="mlp_output",
                intervention_type=pv.CollectIntervention,
            )
        ],
    )
    imodel = pv.IntervenableModel(cfg, model=model)
    with torch.no_grad():
        result = imodel(base={"input_ids": test_input["input_ids"]})
    
    c = result[0][1][0]
    if isinstance(c, torch.Tensor) and c.shape[-1] == HIDDEN_DIM:
        pv_vec = c.detach().float().cpu().squeeze()[-1]
        
        # Manual hook
        mc = {}
        def hook_fn(module, input, output):
            mc['output'] = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
        h = model.model.layers[LAYER_IDX].mlp.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**test_input)
        h.remove()
        
        manual_vec = mc['output'].float().cpu().squeeze()[-1]
        cos = torch.nn.functional.cosine_similarity(pv_vec.unsqueeze(0), manual_vec.unsqueeze(0)).item()
        print(f"  mlp_output pyvene vs manual hook cosine: {cos:.10f}")
        print(f"  PARITY: {'✓ PASS' if cos > 0.999 else '✗ FAIL'}")
    elif isinstance(c, torch.Tensor):
        print(f"  mlp_output shape: {list(c.shape)} — wrong dim")
    
    # Now try block_output and understand the 1792
    print("\n=== block_output 1792 analysis ===")
    cfg2 = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=LAYER_IDX,
                component="block_output",
                intervention_type=pv.CollectIntervention,
            )
        ],
    )
    imodel2 = pv.IntervenableModel(cfg2, model=model)
    with torch.no_grad():
        result2 = imodel2(base={"input_ids": test_input["input_ids"]})
    
    c2 = result2[0][1][0]
    if isinstance(c2, torch.Tensor):
        print(f"  block_output shape: {list(c2.shape)}")
        print(f"  3584 / 2 = {3584 // 2} (matches 1792: {c2.shape[-1] == 3584 // 2})")
        print(f"  This suggests pyvene is applying a gather/position-select that halves the dim")
        
        # Check if nunit=1 is causing position-based slicing
        # When unit='pos' and max_number_of_units=1, pyvene gathers 1 position
        # from the tensor. For [1, 2, 3584] gathering 1 pos gives [1, 1, 3584]
        # not [1, 2, 1792]. So something else is happening.

    print("\nDone.")

if __name__ == "__main__":
    main()
