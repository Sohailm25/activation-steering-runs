#!/usr/bin/env python3
"""
Phase 2 B1: Same-family direction transfer (Qwen 3B → 7B → 14B → 32B).

Extracts DIM refusal direction from source model, applies to target model.
Handles dimension mismatches with geometric-only comparison.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class ModelCfg:
    model_id: str
    layer_idx: int
    hidden_dim: int
    multiplier: float


MODELS: Dict[str, ModelCfg] = {
    "qwen-3b": ModelCfg("Qwen/Qwen2.5-3B-Instruct", 21, 2048, 15.0),
    "qwen-7b": ModelCfg("Qwen/Qwen2.5-7B-Instruct", 16, 3584, 15.0),
    "qwen-14b": ModelCfg("Qwen/Qwen2.5-14B-Instruct", 24, 5120, 15.0),
    "qwen-32b": ModelCfg("Qwen/Qwen2.5-32B-Instruct", 32, 5120, 15.0),
}

GPU_MAP = {
    "qwen-3b": "A10G",
    "qwen-7b": "A10G",
    "qwen-14b": "A100",
    "qwen-32b": "A100-80GB",
}


def get_git_commit() -> str:
    # Try git first, fall back to env var (set at image build time)
    try:
        c = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True
        ).strip()
        if c:
            return c
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT", "unknown")


def extract_direction(model_id: str, layer_idx: int) -> tuple[np.ndarray, float]:
    """Extract DIM direction using nnsight (locked tool)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from infrastructure.v4_tooling_parity import extract_nnsight_raw, unit_dim
    from src.prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

    harm = extract_nnsight_raw(model_id, layer_idx, HARMFUL_PROMPTS)
    safe = extract_nnsight_raw(model_id, layer_idx, HARMLESS_PROMPTS)
    direction, prenorm = unit_dim(harm, safe)
    return direction, prenorm


def eval_steered(model_id: str, direction: np.ndarray, layer_idx: int, multiplier: float) -> dict:
    """Evaluate steering effect on target model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sys
    sys.path.insert(0, str(ROOT))
    from src.prompts import EVAL_PROMPTS
    from src.utils import apply_chat_template, classify_output_quality, is_refusal

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported arch: {type(model).__name__}")

    device = next(model.parameters()).device
    d = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook_fn(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h2 = h + multiplier * d.to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    coherent = garbled = normal = refusal = 0
    for p in EVAL_PROMPTS:
        handles = [layers[i].register_forward_hook(hook_fn) for i in range(layer_idx, len(layers))]
        try:
            templated = apply_chat_template(tok, p)
            inputs = tok(templated, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            txt = tok.decode(gen, skip_special_tokens=True)
            q = classify_output_quality(txt)
            coherent += int(q == "coherent")
            garbled += int(q == "garbled")
            normal += int(q == "normal")
            refusal += int(is_refusal(txt))
        finally:
            for h in handles:
                h.remove()

    n = len(EVAL_PROMPTS)
    del model, tok
    gc.collect()
    import torch as _t
    if _t.cuda.is_available():
        _t.cuda.empty_cache()

    return {
        "n_prompts": n,
        "coherent_rate": 100.0 * coherent / n,
        "garbled_rate": 100.0 * garbled / n,
        "normal_rate": 100.0 * normal / n,
        "refusal_rate": 100.0 * refusal / n,
    }


def run_transfer(
    source_key: str,
    target_key: str,
    geometric_only: bool = False,
    multiplier_override: Optional[float] = None,
) -> dict:
    """Run a single transfer experiment."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.prompts import EVAL_PROMPTS

    src_cfg = MODELS[source_key]
    tgt_cfg = MODELS[target_key]
    effective_multiplier = multiplier_override if multiplier_override is not None else tgt_cfg.multiplier
    is_self = source_key == target_key
    dim_match = src_cfg.hidden_dim == tgt_cfg.hidden_dim

    if is_self:
        mode = "self_control"
    elif geometric_only or not dim_match:
        mode = "geometric_only"
    else:
        mode = "direct_transfer"

    logger.info(f"B1 transfer: {source_key} -> {target_key} (mode={mode})")
    t0 = time.time()

    # Extract source direction
    logger.info(f"Extracting DIM from {source_key} at layer {src_cfg.layer_idx}")
    src_dir, src_prenorm = extract_direction(src_cfg.model_id, src_cfg.layer_idx)
    logger.info(f"Source direction shape={src_dir.shape}, prenorm={src_prenorm:.4f}")

    result = {
        "experiment": "v4_transfer_b1",
        "source": {
            "model_key": source_key,
            "model_id": src_cfg.model_id,
            "layer_idx": src_cfg.layer_idx,
            "hidden_dim": src_cfg.hidden_dim,
        },
        "target": {
            "model_key": target_key,
            "model_id": tgt_cfg.model_id,
            "layer_idx": tgt_cfg.layer_idx,
            "hidden_dim": tgt_cfg.hidden_dim,
        },
        "multiplier": effective_multiplier,
        "mode": mode,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eval_prompt_count": len(EVAL_PROMPTS),
        "extraction": {
            "method": "nnsight",
            "direction_shape": list(src_dir.shape),
            "direction_norm_prenorm": float(src_prenorm),
        },
    }

    # If geometric comparison needed (different dims or geometric_only mode)
    if mode == "geometric_only" or (not is_self and dim_match):
        # Always extract target direction for cosine comparison
        logger.info(f"Extracting DIM from {target_key} at layer {tgt_cfg.layer_idx}")
        tgt_dir, tgt_prenorm = extract_direction(tgt_cfg.model_id, tgt_cfg.layer_idx)
        result["target_extraction"] = {
            "method": "nnsight",
            "direction_shape": list(tgt_dir.shape),
            "direction_norm_prenorm": float(tgt_prenorm),
        }

        if dim_match:
            cos = float(np.dot(src_dir, tgt_dir) / (np.linalg.norm(src_dir) * np.linalg.norm(tgt_dir) + 1e-12))
        else:
            # Can't compute cosine directly with different dims
            cos = None

        result["geometric"] = {
            "cross_cosine": cos,
            "source_norm": float(np.linalg.norm(src_dir)),
            "target_norm": float(np.linalg.norm(tgt_dir)),
            "norm_ratio": float(np.linalg.norm(src_dir) / (np.linalg.norm(tgt_dir) + 1e-12)),
            "dim_match": dim_match,
        }

    # Behavioral eval (self-control or direct transfer with matching dims)
    if mode in ("self_control", "direct_transfer"):
        logger.info(f"Running behavioral eval on {target_key} with source direction")
        ev = eval_steered(tgt_cfg.model_id, src_dir, tgt_cfg.layer_idx, effective_multiplier)
        result["eval"] = ev

    result["elapsed_s"] = time.time() - t0

    # Save artifact
    out_dir = ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "b1_self" if is_self else f"b1_transfer_{source_key}_to_{target_key}"
    if mode == "geometric_only":
        prefix = f"b1_geometric_{source_key}_vs_{target_key}"
    fname = f"{prefix}_{stamp}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {out_path}")

    return result


# ─── Modal wrapper ───────────────────────────────────────────────────────────

try:
    import modal

    APP_NAME = "v4-transfer-b1"
    volume = modal.Volume.from_name("v4-parity-results", create_if_missing=True)
    VOLUME_PATH = "/results"

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.1",
            "transformers>=4.49",
            "accelerate>=1.2",
            "nnsight==0.3.7",
            "bitsandbytes",
            "numpy>=1.26,<2.0",
            "sentencepiece>=0.2",
            "protobuf>=4.25",
        )
        .env({
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/app",
            "HF_HOME": "/results/model_cache",
            "TRANSFORMERS_CACHE": "/results/model_cache",
            "GIT_COMMIT": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip(),
        })
        .add_local_dir(str(ROOT), remote_path="/app")
    )

    SECRETS = [modal.Secret.from_name("hf-secret")]
    app = modal.App(APP_NAME)

    @app.function(image=image, gpu="A10G", timeout=3600, volumes={VOLUME_PATH: volume}, secrets=SECRETS)
    def run_small(source: str, target: str, geometric_only: bool = False, multiplier: float = 0.0):
        result = run_transfer(source, target, geometric_only, multiplier_override=multiplier if multiplier > 0 else None)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        m_tag = f"_m{multiplier}" if multiplier > 0 else ""
        path = f"{VOLUME_PATH}/b1_{source}_to_{target}{m_tag}_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        return result

    @app.function(image=image, gpu="A100", timeout=5400, volumes={VOLUME_PATH: volume}, secrets=SECRETS)
    def run_medium(source: str, target: str, geometric_only: bool = False, multiplier: float = 0.0):
        result = run_transfer(source, target, geometric_only, multiplier_override=multiplier if multiplier > 0 else None)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        m_tag = f"_m{multiplier}" if multiplier > 0 else ""
        path = f"{VOLUME_PATH}/b1_{source}_to_{target}{m_tag}_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        return result

    @app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={VOLUME_PATH: volume}, secrets=SECRETS)
    def run_large(source: str, target: str, geometric_only: bool = False, multiplier: float = 0.0):
        result = run_transfer(source, target, geometric_only, multiplier_override=multiplier if multiplier > 0 else None)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        m_tag = f"_m{multiplier}" if multiplier > 0 else ""
        path = f"{VOLUME_PATH}/b1_{source}_to_{target}{m_tag}_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        volume.commit()
        return result

    @app.local_entrypoint()
    def main(source: str = "qwen-7b", target: str = "qwen-7b", geometric_only: bool = False, multiplier: float = 0.0):
        # Pick GPU tier based on the LARGER model in the pair
        sizes = {"qwen-3b": 0, "qwen-7b": 1, "qwen-14b": 2, "qwen-32b": 3}
        max_size = max(sizes.get(source, 0), sizes.get(target, 0))

        if max_size <= 1:
            out = run_small.remote(source, target, geometric_only, multiplier)
        elif max_size == 2:
            out = run_medium.remote(source, target, geometric_only, multiplier)
        else:
            out = run_large.remote(source, target, geometric_only, multiplier)

        # Save local summary
        local_dir = ROOT / "results" / "phase2"
        local_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = local_dir / f"b1_{source}_to_{target}_{stamp}.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved local summary: {p}")

except ImportError:
    pass  # Modal not available locally
