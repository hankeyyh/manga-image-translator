"""
Modal deployment configuration for Manga Image Translator.

This file contains all constants and configuration for Modal deployment.
"""

# Modal App Configuration
APP_NAME = "manga-translator"

# Volume Names
MODEL_VOLUME_NAME = "manga-models"
RESULT_VOLUME_NAME = "manga-results"

# Mount Paths
MODEL_MOUNT_PATH = "/app/models"
RESULT_MOUNT_PATH = "/app/result"

# Secret Names
ENV_SECRET_NAME = "manga-translator-env"  # Contains all env vars including MT_WEB_NONCE

# GPU Configuration
GPU_CONFIG = {
    "gpu": "A10G",  # Options: T4 (cost-effective), A10G, A100
    "cpu": 4.0,
    "memory": 16384,  # 16GB RAM
    "timeout": 120,  # 2 minutes
    "min_containers": 0,  # Set to 1 or more for production
    # "max_containers": 1,  # 临时压测，只限制1台
    "max_inputs": 8,  # Limit concurrency to prevent OOM
    "scaledown_window": 300,  # 5 minutes idle timeout
}

# Base Image Configuration
BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime"

# System Dependencies
APT_PACKAGES = [
    "ffmpeg",
    "libsm6",
    "libxext6",
    "libxrender-dev",
    "libgomp1",
    "libglib2.0-0",
    "curl",
    "wget",
    "git",
]

# Environment Variables for Model Loading
ENV_VARS = {
    "PYTHONPATH": "/app",
    "TORCH_HOME": "/app/models",
    "HF_HOME": "/app/models/huggingface",
    "TRANSFORMERS_CACHE": "/app/models/transformers",
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONUNBUFFERED": "1",
}
