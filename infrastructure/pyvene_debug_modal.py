"""Modal wrapper for pyvene component debug."""
import modal

ROOT_PATH = "/app"
app = modal.App("pyvene-debug")
volume = modal.Volume.from_name("v4-parity-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1", "transformers>=4.49", "accelerate>=1.2",
        "pyvene>=0.1.2", "numpy>=1.26,<2.0", "sentencepiece>=0.2", "protobuf>=4.25",
    )
    .env({"HF_HOME": "/results/model_cache", "TRANSFORMERS_CACHE": "/results/model_cache"})
    .add_local_file(
        "infrastructure/pyvene_debug.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v2.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v2.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v3.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v3.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v4.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v4.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v5.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v5.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v6.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v6.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v7.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v7.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v8.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v8.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v9.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v9.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v10.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v10.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v11.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v11.py",
    )
    .add_local_file(
        "infrastructure/pyvene_debug_v12.py",
        remote_path=f"{ROOT_PATH}/pyvene_debug_v12.py",
    )
)

@app.function(
    image=image, gpu="A10G", timeout=600,
    volumes={"/results": volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def run_debug():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, f"{ROOT_PATH}/pyvene_debug_v12.py"],
        capture_output=True, text=True, timeout=300,
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

@app.local_entrypoint()
def main():
    out = run_debug.remote()
    print("=== STDOUT ===")
    print(out["stdout"])
    if out["stderr"]:
        print("=== STDERR ===")
        print(out["stderr"][-2000:])
    print(f"Return code: {out['returncode']}")
