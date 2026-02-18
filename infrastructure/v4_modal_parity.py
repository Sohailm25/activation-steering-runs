#!/usr/bin/env python3
"""Modal wrapper for Phase-1 tooling parity runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import modal

APP_NAME = "v4-tooling-parity"
ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infrastructure"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("v4-parity-results", create_if_missing=True)
VOLUME_PATH = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.49",
        "accelerate>=1.2",
        "nnsight==0.3.7",
        "pyvene>=0.1.2",
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
    })
    .add_local_dir(str(ROOT), remote_path="/app")
)

SECRETS = [modal.Secret.from_name("hf-secret")]


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
    secrets=SECRETS,
)
def run_qwen(repeats: int = 5):
    from infrastructure.v4_tooling_parity import run_model

    result = run_model("qwen-7b", repeats=repeats)
    path = f"{VOLUME_PATH}/v4_parity_qwen-7b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return {"saved": path, "summary": result.get("aggregate", {})}


@app.function(
    image=image,
    gpu="A100",
    timeout=5400,
    volumes={VOLUME_PATH: volume},
    secrets=SECRETS,
)
def run_gemma(repeats: int = 5):
    from infrastructure.v4_tooling_parity import run_model

    result = run_model("gemma-9b", repeats=repeats)
    path = f"{VOLUME_PATH}/v4_parity_gemma-9b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return {"saved": path, "summary": result.get("aggregate", {})}


@app.local_entrypoint()
def main(model: str = "qwen-7b", repeats: int = 5):
    if model == "qwen-7b":
        out = run_qwen.remote(repeats)
    elif model == "gemma-9b":
        out = run_gemma.remote(repeats)
    else:
        raise ValueError("model must be qwen-7b or gemma-9b")

    local = ROOT / "results" / "phase1"
    local.mkdir(parents=True, exist_ok=True)
    p = local / f"modal_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved local summary: {p}")
