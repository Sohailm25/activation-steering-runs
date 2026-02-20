#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v7: fix result indexing.

Root cause found: pyvene returns (collected, model_output), not (model_output, collected).
result[0] = (None, [collected_tensors])
result[1] = CausalLMOutputWithPast (logits, 152064)

We were reading result[1][0] = logits. Should read result[0][1][0].
"""
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

    # Explore result structure
    print("=== Result structure ===")
    print(f"result[0] type: {type(result[0])}")
    if isinstance(result[0], tuple):
        print(f"  result[0][0]: {type(result[0][0])}")
        print(f"  result[0][1]: {type(result[0][1])}")
        if isinstance(result[0][1], list):
            for i, item in enumerate(result[0][1]):
                if isinstance(item, torch.Tensor):
                    print(f"    result[0][1][{i}]: shape={list(item.shape)}")
                    if item.shape[-1] == HIDDEN_DIM:
                        print(f"    ✓ MATCH! This is the collected activation!")
                else:
                    print(f"    result[0][1][{i}]: {type(item)}")

    print(f"\nresult[1] type: {type(result[1])}")
    if hasattr(result[1], 'logits'):
        print(f"  result[1].logits shape: {list(result[1].logits.shape)}")

    # Now test proper extraction
    collected = result[0][1][0]
    if isinstance(collected, torch.Tensor):
        print(f"\n=== Correct extraction ===")
        print(f"Collected shape: {list(collected.shape)}")
        print(f"Hidden dim match: {'✓' if collected.shape[-1] == HIDDEN_DIM else '✗'}")
        
        if collected.shape[-1] == HIDDEN_DIM:
            # Cross-check with manual hook
            c = collected.detach().float().cpu().squeeze()
            if c.ndim == 2:
                c = c[-1]

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

            manual_vec = collected_manual['output'].float().cpu().squeeze()[-1]
            cos = torch.nn.functional.cosine_similarity(c.unsqueeze(0), manual_vec.unsqueeze(0)).item()
            l2 = torch.norm(c - manual_vec).item()
            print(f"\nCross-check vs manual hook:")
            print(f"  Cosine: {cos:.10f}")
            print(f"  L2: {l2:.10f}")
            print(f"  PARITY: {'✓ PASS' if cos > 0.999 else '✗ FAIL'}")

    print("\nDone.")

if __name__ == "__main__":
    main()
