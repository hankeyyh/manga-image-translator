"""
Modal deployment for Manga Image Translator.

This deployment correctly handles the Master/Worker subprocess architecture:
- Master: FastAPI server (server/main.py) that handles HTTP requests
- Worker: MangaTranslator subprocess (mode/share.py) that does actual ML processing
- Starts worker subprocess within the container
- Manages lifecycle of both master and worker processes
- Properly forwards environment variables and GPU configuration
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

import modal

# Get the parent directory (project root) relative to this file
DEPLOY_DIR = Path(__file__).parent
PROJECT_ROOT = DEPLOY_DIR.parent

# Import configuration
from modal_config import (
    APP_NAME,
    IMAGE_NAME,
    MODEL_VOLUME_NAME,
    RESULT_VOLUME_NAME,
    MODEL_MOUNT_PATH,
    RESULT_MOUNT_PATH,
    APP_ROOT_PATH,
    ENV_SECRET_NAME,
    GPU_CONFIG,
    BASE_IMAGE,
    APT_PACKAGES,
    ENV_VARS,
)

# Create Modal app
app = modal.App(APP_NAME)

# Create or reference persistent volumes
model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True
)
result_volume = modal.Volume.from_name(
    RESULT_VOLUME_NAME,
    create_if_missing=True
)

# Build container image
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.10")
    # Install system packages including build tools (needed for pyhyphen, pydensecrf)
    .apt_install([
        "build-essential",  # gcc, g++, make
        "gcc",
        "g++",
    ] + APT_PACKAGES)
    .env(ENV_VARS)
    # Copy requirements.txt and install Python dependencies
    .add_local_file(str(PROJECT_ROOT / "requirements.txt"), "/app/requirements.txt", copy=True)
    .add_local_file(str(DEPLOY_DIR / "modal_config.py"), "/root/modal_config.py", copy=True)
    .run_commands(
        "cd /app && pip install --no-cache-dir -r requirements.txt",
        gpu="t4",  # Use GPU during build for PyTorch installation
    )
    # Create necessary directories
    .run_commands(
        "mkdir -p /app/models",
        "mkdir -p /app/result",
        "mkdir -p /app/upload-cache",
    )
    # Copy application code (will be mounted on container startup for fast iteration)
    .add_local_dir(str(PROJECT_ROOT / "manga_translator"), "/app/manga_translator")
    .add_local_dir(str(PROJECT_ROOT / "server"), "/app/server")
    .add_local_dir(str(PROJECT_ROOT / "fonts"), "/app/fonts")
    .add_local_file(str(PROJECT_ROOT / "docker_prepare.py"), "/app/docker_prepare.py")
)


@app.function(
    image=image,
    gpu=GPU_CONFIG["gpu"],
    cpu=GPU_CONFIG["cpu"],
    memory=GPU_CONFIG["memory"],
    timeout=GPU_CONFIG["timeout"],
    min_containers=GPU_CONFIG["min_containers"],
    scaledown_window=GPU_CONFIG.get("scaledown_window", 300),
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        RESULT_MOUNT_PATH: result_volume,
    },
    secrets=[
        modal.Secret.from_name(ENV_SECRET_NAME),
    ],
)
@modal.concurrent(max_inputs=GPU_CONFIG["max_inputs"])
@modal.asgi_app()
def web():
    """
    Main ASGI web application endpoint with worker subprocess.

    This function:
    1. Spawns worker subprocesses immediately (before heavy master imports)
    2. Imports the FastAPI app while workers boot in parallel
    3. Waits for workers, then returns the ASGI app

    Worker count is MT_WORKER_COUNT, capped at 2 for single-GPU cold start.

    Architecture:
    - Master (FastAPI): Handles HTTP, queues tasks, returns results
    - Workers (subprocesses): Load models, run detection/OCR/inpainting
    - Communication: HTTP requests with pickle serialization + nonce auth
    """
    import sys
    import subprocess
    import os
    import time
    import atexit
    sys.path.insert(0, "/app")

    # Get nonce from environment (set by Modal secret)
    nonce = os.environ.get('MT_WEB_NONCE')
    if not nonce:
        print("WARNING: MT_WEB_NONCE not set, generating temporary nonce")
        import secrets
        nonce = secrets.token_hex(16)
        os.environ['MT_WEB_NONCE'] = nonce

    worker_host = "127.0.0.1"
    worker_base_port = 5004
    # Extra workers on a single GPU mainly contend on import and slow cold start.
    # Live secret may still have MT_WORKER_COUNT=8; cap here.
    worker_count_cap = 2
    raw_worker_count = os.getenv("MT_WORKER_COUNT", "2")
    try:
        worker_count = int(str(raw_worker_count).strip())
    except (TypeError, ValueError):
        print(f"Invalid MT_WORKER_COUNT={raw_worker_count!r}, falling back to 2")
        worker_count = 2
    if worker_count < 1:
        print(f"MT_WORKER_COUNT={worker_count} < 1, falling back to 2")
        worker_count = 2
    if worker_count > worker_count_cap:
        print(
            f"MT_WORKER_COUNT={worker_count} capped to {worker_count_cap} "
            "to reduce cold start on a single GPU"
        )
        worker_count = worker_count_cap
    worker_ports = [worker_base_port + i for i in range(worker_count)]
    worker_processes = []

    # This function is always scheduled with a GPU; skip torch import.
    use_gpu = True
    print(f"Using GPU: {use_gpu}")

    def build_worker_cmd(port: int):
        return [
            sys.executable,
            '-m', 'manga_translator',
            'shared',
            '--host', worker_host,
            '--port', str(port),
            '--nonce', nonce,
            '--use-gpu',
            '--ignore-errors',
            '--notify-progress-fail',
        ]

    def terminate_workers():
        for proc in worker_processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def cleanup_workers():
        print("Cleaning up worker subprocesses...")
        terminate_workers()
        print("Worker subprocesses terminated")

    print(f"Starting {worker_count} worker subprocess(es)")
    for port in worker_ports:
        worker_cmd = build_worker_cmd(port)
        print(f"Starting worker subprocess: {' '.join(worker_cmd)}")
        proc = subprocess.Popen(
            worker_cmd,
            cwd="/app",
            stdout=None,
            stderr=None,
        )
        worker_processes.append(proc)

    atexit.register(cleanup_workers)

    # Heavy imports overlap with worker process boot.
    from argparse import Namespace
    import httpx
    from server.main import app as fastapi_app, prepare
    from server.instance import ExecutorInstance, executor_instances

    def wait_for_worker(port: int, proc: subprocess.Popen):
        print(f"Waiting for worker HTTP server on port {port}...")
        max_wait_time = 120
        check_interval = 0.5
        started = time.monotonic()
        last_log_at = -999.0
        url = f"http://{worker_host}:{port}/is_locked"
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= max_wait_time:
                break
            if proc.poll() is not None:
                print(f"Worker on port {port} died during startup! Exit code: {proc.returncode}")
                return False
            try:
                response = httpx.get(url, timeout=0.5)
                if response.status_code == 200:
                    print(f"Worker on port {port} is ready! (took {elapsed:.1f}s)")
                    return True
            except Exception as e:
                if elapsed - last_log_at >= 5.0:
                    print(f"Worker on port {port} not ready yet (waited {elapsed:.1f}s): {e}")
                    last_log_at = elapsed
            time.sleep(check_interval)
        print(f"Worker on port {port} failed to start within {max_wait_time}s")
        return False

    for port, proc in zip(worker_ports, worker_processes):
        if not wait_for_worker(port, proc):
            terminate_workers()
            raise RuntimeError(f"Worker subprocess on port {port} failed to start")
        print(f"Worker subprocess started with PID: {proc.pid} port={port}")

    for port in worker_ports:
        executor_instances.register(ExecutorInstance(ip=worker_host, port=port))
        print(f"Registered worker at {worker_host}:{port}")

    args = Namespace(
        host=worker_host,
        port=worker_base_port,
        nonce=nonce,
        start_instance=False,
        use_gpu=use_gpu,
        use_gpu_limited=False,
        ignore_errors=True,
        notify_progress_fail=True,
        verbose=True,
        models_ttl=None,
        pre_dict=None,
        post_dict=None,
    )
    prepare(args)

    @fastapi_app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        import httpx

        workers = []
        async with httpx.AsyncClient() as client:
            for port, proc in zip(worker_ports, worker_processes):
                alive = proc.poll() is None
                healthy = False
                if alive:
                    try:
                        response = await client.get(
                            f"http://{worker_host}:{port}/is_locked",
                            timeout=5.0
                        )
                        healthy = response.status_code == 200
                    except Exception as e:
                        print(f"Worker health check failed on port {port}: {e}")
                workers.append({
                    "pid": proc.pid,
                    "alive": alive,
                    "healthy": healthy,
                    "host": worker_host,
                    "port": port,
                })

        all_healthy = bool(workers) and all(w["alive"] and w["healthy"] for w in workers)
        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "manga-translator",
            "gpu_available": use_gpu,
            "worker_count": len(workers),
            "workers": workers,
        }

    return fastapi_app


@app.function(
    image=image,
    gpu=GPU_CONFIG["gpu"],
    cpu=2.0,
    memory=8192,
    timeout=3600,  # 1 hour for model downloads
    volumes={
        MODEL_MOUNT_PATH: model_volume,
    },
)
def download_models():
    """
    Download all required models to the persistent volume.

    This function should be run once during initial setup to pre-populate
    the model cache. It can also be run periodically to update models.

    Usage:
        modal run deploy.modal_app::download_models
    """
    import subprocess
    import sys

    print("Starting model download...")
    print(f"Model volume mounted at: {MODEL_MOUNT_PATH}")

    # Set environment variables for model paths BEFORE running script
    env = os.environ.copy()
    env["TORCH_HOME"] = MODEL_MOUNT_PATH
    env["HF_HOME"] = f"{MODEL_MOUNT_PATH}/huggingface"
    env["TRANSFORMERS_CACHE"] = f"{MODEL_MOUNT_PATH}/transformers"
    env["XDG_CACHE_HOME"] = f"{MODEL_MOUNT_PATH}/cache"

    print(f"Environment variables set:")
    print(f"  TORCH_HOME={env['TORCH_HOME']}")
    print(f"  HF_HOME={env['HF_HOME']}")
    print(f"  TRANSFORMERS_CACHE={env['TRANSFORMERS_CACHE']}")

    # Run the docker_prepare.py script with environment
    try:
        result = subprocess.run(
            [sys.executable, "/app/docker_prepare.py", "--continue-on-error"],
            cwd="/app",
            env=env,  # Pass environment variables
            check=True,
            capture_output=True,
            text=True,
        )
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # List what was downloaded
        print("\nChecking downloaded files...")
        list_result = subprocess.run(
            ["find", MODEL_MOUNT_PATH, "-type", "f", "-name", "*.onnx", "-o", "-name", "*.pt", "-o", "-name", "*.pth"],
            capture_output=True,
            text=True,
        )
        if list_result.stdout:
            print("Found model files:")
            print(list_result.stdout)
        else:
            print("⚠️ Warning: No model files found!")

        # Commit the volume changes
        print("\nCommitting volume changes...")
        model_volume.commit()

        print("✅ Model download completed successfully!")
        return {"status": "success", "message": "All models downloaded"}

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during model download: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

        # Still commit partial downloads
        model_volume.commit()

        return {
            "status": "partial_failure",
            "message": "Some models may have failed to download",
            "error": str(e)
        }


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    volumes={
        RESULT_MOUNT_PATH: result_volume,
    },
)
def cleanup_old_results(max_age_days: int = 7, max_count: int = 100):
    """
    Clean up old result files from the result volume.

    Args:
        max_age_days: Delete results older than this many days
        max_count: Keep at most this many most recent results

    Usage:
        modal run deploy.modal_app::cleanup_old_results --max-age-days 7
    """
    import shutil
    from datetime import datetime, timedelta

    result_path = Path(RESULT_MOUNT_PATH)

    if not result_path.exists():
        print("Result directory does not exist")
        return {"status": "skipped", "message": "No results to clean"}

    # Get all result directories
    result_dirs = [d for d in result_path.iterdir() if d.is_dir()]
    result_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    deleted_count = 0
    kept_count = 0
    cutoff_time = datetime.now().timestamp() - (max_age_days * 86400)

    for i, result_dir in enumerate(result_dirs):
        # Keep the most recent max_count results
        if i < max_count:
            # But still delete if too old
            if result_dir.stat().st_mtime < cutoff_time:
                print(f"Deleting old result: {result_dir.name}")
                shutil.rmtree(result_dir)
                deleted_count += 1
            else:
                kept_count += 1
        else:
            # Delete anything beyond max_count
            print(f"Deleting excess result: {result_dir.name}")
            shutil.rmtree(result_dir)
            deleted_count += 1

    # Commit volume changes
    result_volume.commit()

    print(f"✅ Cleanup completed: {deleted_count} deleted, {kept_count} kept")
    return {
        "status": "success",
        "deleted": deleted_count,
        "kept": kept_count,
    }


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        RESULT_MOUNT_PATH: result_volume,
    },
)
def list_volumes():
    """
    List contents of both volumes for debugging.

    Usage:
        modal run deploy.modal_app::list_volumes
    """
    import subprocess

    print("=" * 60)
    print("MODEL VOLUME CONTENTS:")
    print("=" * 60)

    # List top-level directories
    print("\nTop-level structure:")
    result = subprocess.run(
        ["ls", "-lah", MODEL_MOUNT_PATH],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Show directory sizes
    print("\nDirectory sizes:")
    result = subprocess.run(
        ["du", "-h", "-d", "2", MODEL_MOUNT_PATH],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Count files by type
    print("\nFile count by type:")
    for ext in [".onnx", ".pt", ".pth", ".bin", ".safetensors"]:
        result = subprocess.run(
            ["find", MODEL_MOUNT_PATH, "-name", f"*{ext}", "-type", "f"],
            capture_output=True,
            text=True,
        )
        count = len([l for l in result.stdout.strip().split("\n") if l])
        print(f"  {ext}: {count} files")

    # List some example files
    print("\nExample model files (first 20):")
    result = subprocess.run(
        ["find", MODEL_MOUNT_PATH, "-type", "f", "-name", "*.onnx", "-o", "-name", "*.pt", "-o", "-name", "*.pth"],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n")[:20]
    for line in lines:
        if line:
            print(f"  {line}")

    print("\n" + "=" * 60)
    print("RESULT VOLUME CONTENTS:")
    print("=" * 60)
    result = subprocess.run(
        ["ls", "-lah", RESULT_MOUNT_PATH],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    return {"status": "success"}


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """
    Local entrypoint for testing Modal functions.

    Usage:
        modal run deploy.modal_app
    """
    print("Manga Image Translator - Modal Deployment")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  modal deploy deploy.modal_app         # Deploy the web service")
    print("  modal run deploy.modal_app::download_models  # Download models")
    print("  modal run deploy.modal_app::cleanup_old_results  # Clean results")
    print("  modal run deploy.modal_app::list_volumes  # List volume contents")
    print("\nFor more information, see deploy/README.md")
