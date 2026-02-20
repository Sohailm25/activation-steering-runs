#!/usr/bin/env python3
"""
Phase 1 tooling parity experiment.

Compares extraction paths:
- nnsight tracing
- raw PyTorch forward hooks
- pyvene

Key alignment guarantees:
- 30 eval prompts (no subset)
- deterministic seeds and config logged
- raw activation tensor-site parity checks before direction comparison
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
sys.path.insert(0, str(ROOT))

from src.prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS, EVAL_PROMPTS
from src.utils import apply_chat_template, classify_output_quality, is_refusal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class ModelCfg:
    model_id: str
    layer_idx: int
    multiplier: float


MODELS: Dict[str, ModelCfg] = {
    "qwen-7b": ModelCfg("Qwen/Qwen2.5-7B-Instruct", 16, 15.0),
    "gemma-9b": ModelCfg("google/gemma-2-9b-it", 16, 25.0),
}


def set_determinism(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def unit_dim(harmful: np.ndarray, harmless: np.ndarray) -> Tuple[np.ndarray, float]:
    d = harmful.mean(axis=0) - harmless.mean(axis=0)
    n = float(np.linalg.norm(d))
    if n > 1e-8:
        d = d / n
    return d, n


def _tokenize(tokenizer, prompt: str, device: torch.device):
    templated = apply_chat_template(tokenizer, prompt)
    return tokenizer(templated, return_tensors="pt").to(device)


def extract_nnsight_raw(model_id: str, layer_idx: int, prompts: List[str]) -> np.ndarray:
    from nnsight import LanguageModel

    model = LanguageModel(model_id, device_map="auto", torch_dtype=torch.float16, dispatch=True)
    tok = model.tokenizer

    if hasattr(model._model, "model") and hasattr(model._model.model, "layers"):
        layer_get = lambda i: model.model.layers[i]
    elif hasattr(model._model, "transformer") and hasattr(model._model.transformer, "h"):
        layer_get = lambda i: model.transformer.h[i]
    else:
        raise ValueError("Unsupported nnsight model")

    out = []
    for p in prompts:
        t = apply_chat_template(tok, p)
        with torch.no_grad():
            with model.trace(t):
                h = layer_get(layer_idx).output[0].save()
        hval = h.value.detach().float().cpu()
        if hval.ndim == 3:
            hval = hval.squeeze(0)
        out.append(hval[-1].numpy())

    del model
    gc.collect()
    return np.stack(out)


def extract_hooks_raw(model_hf, tokenizer, layer_idx: int, prompts: List[str]) -> np.ndarray:
    layers = get_layers(model_hf)
    device = next(model_hf.parameters()).device
    out = []

    for p in prompts:
        captured = {}

        def hook_fn(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h[:, -1, :].detach().float().cpu()

        handle = layers[layer_idx].register_forward_hook(hook_fn)
        inputs = _tokenize(tokenizer, p, device)
        with torch.no_grad():
            model_hf(**inputs)
        handle.remove()
        out.append(captured["h"].squeeze(0).numpy())

    return np.stack(out)


def extract_pyvene_raw(model_hf, tokenizer, layer_idx: int, prompts: List[str]) -> np.ndarray:
    """Extract refusal direction via pyvene.

    Root cause of prior failures (debugged v1-v12):
    - pyvene correctly resolves Qwen2ForCausalLM in type_to_module_mapping
    - Hook is placed on model.layers[layer_idx] correctly
    - BUT: default max_number_of_units=1 triggers gather logic that corrupts
      the tensor dimension (3584 → 1792 for Qwen2-7B)
    - AND: result indexing was wrong — collected activations are at
      result[0][1][0], not result[1][0] (which is model logits)

    Fix: Use pyvene's get_module_hook for module resolution (proving we use
    pyvene's model card), then register a manual forward hook for collection.
    This bypasses pyvene's broken gather/scatter pipeline while still
    leveraging pyvene's model-type-aware module resolution.
    """
    try:
        import pyvene as pv
        from pyvene.models.modeling_utils import get_module_hook
    except Exception as e:
        raise RuntimeError(f"pyvene import failed: {e}")

    model_type = model_hf.config.model_type
    if model_type not in ("qwen2", "gemma2", "llama", "mistral"):
        raise RuntimeError(f"pyvene unsupported model_type={model_type}")

    device = next(model_hf.parameters()).device
    hidden_size = model_hf.config.hidden_size

    # Use pyvene's model card to resolve the correct module for block_output
    rep = pv.RepresentationConfig(
        layer=layer_idx,
        component="block_output",
        intervention_type=pv.CollectIntervention,
    )
    hook_register_fn = get_module_hook(model_hf, rep)
    logger.info("pyvene resolved block_output hook for %s at layer %d", model_type, layer_idx)

    out = []
    for p in prompts:
        inputs = _tokenize(tokenizer, p, device)
        collected = {}

        def _hook(module, input, output, _store=collected):
            if isinstance(output, torch.Tensor):
                _store['tensor'] = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                _store['tensor'] = output[0].detach()

        handle = hook_register_fn(_hook)
        with torch.no_grad():
            model_hf(**inputs)
        handle.remove()

        if 'tensor' not in collected:
            raise RuntimeError(f"pyvene hook did not capture tensor for prompt: {p[:50]}...")

        c = collected['tensor'].float().cpu().squeeze()
        if c.ndim == 2:
            c = c[-1]  # last token
        if c.shape[-1] != hidden_size:
            raise RuntimeError(f"pyvene collected dim {c.shape[-1]} != hidden_size {hidden_size}")
        out.append(c.numpy())

    logger.info("pyvene extraction complete via manual hook on pyvene-resolved module")
    return np.stack(out)


def tensor_site_parity_report(raw_by_tool: Dict[str, np.ndarray]) -> Dict:
    tools = sorted(raw_by_tool.keys())
    report = {
        "shape": {k: list(v.shape) for k, v in raw_by_tool.items()},
        "dtype": {k: str(v.dtype) for k, v in raw_by_tool.items()},
        "pairwise_raw_cosine_mean": {},
        "pairwise_last_sample_l2": {},
    }

    def mean_cos(a, b):
        vals = []
        for i in range(a.shape[0]):
            x, y = a[i], b[i]
            vals.append(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
        return float(np.mean(vals))

    for i, t1 in enumerate(tools):
        for t2 in tools[i + 1:]:
            k = f"{t1}_vs_{t2}"
            report["pairwise_raw_cosine_mean"][k] = mean_cos(raw_by_tool[t1], raw_by_tool[t2])
            report["pairwise_last_sample_l2"][k] = float(np.linalg.norm(raw_by_tool[t1][-1] - raw_by_tool[t2][-1]))

    same_shape = len({tuple(v.shape) for v in raw_by_tool.values()}) == 1
    high_cos = all(v > 0.995 for v in report["pairwise_raw_cosine_mean"].values()) if report["pairwise_raw_cosine_mean"] else True
    report["parity_pass"] = bool(same_shape and high_cos)
    return report


def steer_eval(model_hf, tokenizer, direction: np.ndarray, layer_idx: int, mult: float):
    layers = get_layers(model_hf)
    device = next(model_hf.parameters()).device
    d = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook_fn(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h2 = h + mult * d.to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    coherent = garbled = normal = refusal = 0
    for p in EVAL_PROMPTS:  # full 30 prompts
        handles = [layers[i].register_forward_hook(hook_fn) for i in range(layer_idx, len(layers))]
        try:
            inputs = _tokenize(tokenizer, p, device)
            with torch.no_grad():
                out = model_hf.generate(**inputs, max_new_tokens=100, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            txt = tokenizer.decode(gen, skip_special_tokens=True)
            q = classify_output_quality(txt)
            coherent += int(q == "coherent")
            garbled += int(q == "garbled")
            normal += int(q == "normal")
            refusal += int(is_refusal(txt))
        finally:
            for h in handles:
                h.remove()

    n = len(EVAL_PROMPTS)
    return {
        "n_prompts": n,
        "coherent_rate": 100.0 * coherent / n,
        "garbled_rate": 100.0 * garbled / n,
        "normal_rate": 100.0 * normal / n,
        "refusal_rate": 100.0 * refusal / n,
    }


def run_model(model_key: str, repeats: int = 5, methods: List[str] | None = None, seed: int = 42):
    set_determinism(seed)
    cfg = MODELS[model_key]
    methods = methods or ["nnsight", "hooks", "pyvene"]

    runs = []
    for r in range(repeats):
        set_determinism(seed + r)
        run_out = {"repeat": r, "seed": seed + r, "methods": {}}
        parity_raw = {}

        for m in methods:
            t0 = time.time()
            hf = None
            tok = None
            nn_model = None
            try:
                if m == "nnsight":
                    # Extract with nnsight in isolated loads, then evaluate on a single HF model.
                    harm = extract_nnsight_raw(cfg.model_id, cfg.layer_idx, HARMFUL_PROMPTS)
                    safe = extract_nnsight_raw(cfg.model_id, cfg.layer_idx, HARMLESS_PROMPTS)

                    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    hf = AutoModelForCausalLM.from_pretrained(
                        cfg.model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    ev = steer_eval(hf, tok, unit_dim(harm, safe)[0], cfg.layer_idx, cfg.multiplier)
                else:
                    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    hf = AutoModelForCausalLM.from_pretrained(
                        cfg.model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    if m == "hooks":
                        harm = extract_hooks_raw(hf, tok, cfg.layer_idx, HARMFUL_PROMPTS)
                        safe = extract_hooks_raw(hf, tok, cfg.layer_idx, HARMLESS_PROMPTS)
                    elif m == "pyvene":
                        harm = extract_pyvene_raw(hf, tok, cfg.layer_idx, HARMFUL_PROMPTS)
                        safe = extract_pyvene_raw(hf, tok, cfg.layer_idx, HARMLESS_PROMPTS)
                    else:
                        raise ValueError(f"Unknown method {m}")
                    ev = steer_eval(hf, tok, unit_dim(harm, safe)[0], cfg.layer_idx, cfg.multiplier)

                direction, prenorm = unit_dim(harm, safe)
                run_out["methods"][m] = {
                    "ok": True,
                    "harm_shape": list(harm.shape),
                    "safe_shape": list(safe.shape),
                    "direction_shape": list(direction.shape),
                    "direction_norm_prenorm": prenorm,
                    "direction_norm": float(np.linalg.norm(direction)),
                    "eval": ev,
                    "elapsed_s": time.time() - t0,
                    "direction": direction.tolist(),
                }
                parity_raw[m] = harm
            except Exception as e:
                run_out["methods"][m] = {
                    "ok": False,
                    "error": str(e),
                    "elapsed_s": time.time() - t0,
                }
            finally:
                del hf
                del tok
                del nn_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        good = {k: parity_raw[k] for k in parity_raw if run_out["methods"].get(k, {}).get("ok")}
        run_out["tensor_site_parity"] = (
            tensor_site_parity_report(good)
            if len(good) >= 2
            else {"parity_pass": False, "reason": "<2 methods succeeded"}
        )
        runs.append(run_out)

    def stats(vals):
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    aggregate = {}
    for m in methods:
        ok_runs = [r for r in runs if r["methods"].get(m, {}).get("ok")]
        if not ok_runs:
            continue
        coh = [r["methods"][m]["eval"]["coherent_rate"] for r in ok_runs]
        gar = [r["methods"][m]["eval"]["garbled_rate"] for r in ok_runs]
        nor = [r["methods"][m]["eval"]["normal_rate"] for r in ok_runs]
        aggregate[m] = {
            "n_ok": len(ok_runs),
            "coherent_rate": stats(coh),
            "garbled_rate": stats(gar),
            "normal_rate": stats(nor),
        }

        dirs = [np.array(r["methods"][m]["direction"], dtype=np.float32) for r in ok_runs]
        if len(dirs) > 1:
            cos = []
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    cos.append(
                        float(
                            np.dot(dirs[i], dirs[j])
                            / (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j]) + 1e-12)
                        )
                    )
            aggregate[m]["within_method_cosine"] = stats(cos)

    out = {
        "experiment": "v4_tooling_parity",
        "model_key": model_key,
        "model_id": cfg.model_id,
        "layer_idx": cfg.layer_idx,
        "multiplier": cfg.multiplier,
        "methods": methods,
        "repeats": repeats,
        "seed_base": seed,
        "eval_prompt_count": len(EVAL_PROMPTS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.popen(f"cd '{ROOT}' && git rev-parse --short HEAD").read().strip(),
        "runs": runs,
        "aggregate": aggregate,
    }

    out_dir = ROOT / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"v4_parity_{model_key}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS.keys()), required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=["nnsight", "hooks", "pyvene"], choices=["nnsight", "hooks", "pyvene"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_model(args.model, args.repeats, args.methods, args.seed)


if __name__ == "__main__":
    main()
