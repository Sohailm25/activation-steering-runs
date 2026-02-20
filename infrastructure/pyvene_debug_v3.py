#!/usr/bin/env python3
"""Debug pyvene Qwen2 — v3: test the actual fix.

Root cause confirmed: pyvene has qwen2 mappings but its auto-resolver doesn't
use them. The fix is to manually pass the type_to_module/dimension mappings
to IntervenableModel.

This script tests the fix approach.
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

    print(f"lm block_output mapping: {qwen2_lm_type_to_module_mapping['block_output']}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    test_input = tok("Hello world", return_tensors="pt").to(model.device)

    # FIX APPROACH: pass mappings directly to IntervenableModel
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

    imodel = pv.IntervenableModel(
        cfg,
        model=model,
        type_to_module_mapping=qwen2_lm_type_to_module_mapping,
        type_to_dimension_mapping=qwen2_lm_type_to_dimension_mapping,
    )

    with torch.no_grad():
        result = imodel(base={"input_ids": test_input["input_ids"]})

    collected = result[1][0]
    if isinstance(collected, torch.Tensor):
        shape = list(collected.shape)
        match = "✓ MATCH" if collected.shape[-1] == HIDDEN_DIM else f"✗ WRONG (expected {HIDDEN_DIM})"
        print(f"FIX TEST: shape={shape} {match}")
    else:
        print(f"FIX TEST: unexpected type {type(collected)}")

    # Test extraction of last-token hidden state
    if isinstance(collected, torch.Tensor) and collected.shape[-1] == HIDDEN_DIM:
        c = collected.detach().float().cpu().squeeze()
        if c.ndim == 2:
            c = c[-1]  # last token
        print(f"Last-token vector shape: {list(c.shape)}")
        print(f"Vector norm: {torch.norm(c).item():.4f}")
        print(f"Vector mean: {c.mean().item():.6f}")

        # Cross-check with manual hook
        collected_manual = {}
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                collected_manual['output'] = output.detach()
            elif isinstance(output, tuple):
                collected_manual['output'] = output[0].detach()

        layer = model.model.layers[LAYER_IDX]
        handle = layer.register_forward_hook(hook_fn)
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
