"""
GPU Profile Configuration

Defines optimized batch sizes and parallelism settings for different GPU types.
These profiles are tuned to maximize training speed while avoiding OOM errors.
"""

from typing import Dict, Any, Optional
import os


# GPU Profile definitions
GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "rtx_4090": {
        "name": "NVIDIA RTX 4090",
        "vram_gb": 24,
        "batch_size_1d": 512,  # Daily intervals - 4x increase vs original 128
        "batch_size_1h": 768,  # Hourly intervals - 3x increase vs original 256
        "max_workers": 2,      # Parallel model training (ProcessPoolExecutor)
        "memory_per_model_1d": 6,  # Estimated VRAM usage per model (GB) for daily
        "memory_per_model_1h": 12,  # Estimated VRAM usage per model (GB) for hourly
        "notes": "Optimized for 2 parallel models with larger batch sizes. "
                 "Trades parallelism for faster per-model training.",
    },
    "rtx_5090": {
        "name": "NVIDIA RTX 5090",
        "vram_gb": 32,  # Expected VRAM for RTX 5090
        "batch_size_1d": 1024,  # Daily intervals - 8x increase vs original 128
        "batch_size_1h": 1536,  # Hourly intervals - 6x increase vs original 256
        "max_workers": 2,       # Can potentially use 3 workers with this VRAM
        "memory_per_model_1d": 12,  # Estimated VRAM usage per model (GB)
        "memory_per_model_1h": 16,  # Estimated VRAM usage per model (GB)
        "notes": "Larger VRAM allows bigger batch sizes. "
                 "May support 3 parallel workers - requires testing.",
    },
    "a100_40gb": {
        "name": "NVIDIA A100 40GB",
        "vram_gb": 40,
        "batch_size_1d": 1024,
        "batch_size_1h": 2048,
        "max_workers": 2,
        "memory_per_model_1d": 12,
        "memory_per_model_1h": 20,
        "notes": "Data center GPU optimized for large-scale training. "
                 "Can handle very large batch sizes.",
    },
    "a100_80gb": {
        "name": "NVIDIA A100 80GB",
        "vram_gb": 80,
        "batch_size_1d": 2048,
        "batch_size_1h": 4096,
        "max_workers": 3,
        "memory_per_model_1d": 16,
        "memory_per_model_1h": 26,
        "notes": "Maximum performance profile. Can handle 3-4 parallel models "
                 "with very large batch sizes.",
    },
    "h200": {
        "name": "NVIDIA H200 SXM",
        "vram_gb": 141,
        "batch_size_1d": 2048,  # Daily intervals - same as A100 80GB
        "batch_size_1h": 2560,  # Hourly intervals - 0.625x A100 80GB (ultra conservative)
        "max_workers": 3,       # 3 parallel models to avoid OOM
        "memory_per_model_1d": 25,  # ~25GB per model for daily
        "memory_per_model_1h": 35,  # ~35GB per model for hourly (plenty of headroom)
        "notes": "High performance profile for H200 with 141GB VRAM. "
                 "Ultra conservative batch sizes to avoid OOM with 17-feature models. "
                 "Reduced from 5120->3584->2560 for hourly after repeated OOM errors. "
                 "Still fast training - 10-12x vs CPU, with memory headroom.",
    },
    "b200": {
        "name": "NVIDIA B200",
        "vram_gb": 180,
        "batch_size_1d": 2048,  # Daily intervals - same as H200
        "batch_size_1h": 3200,  # Hourly intervals - 25% increase over H200
        "max_workers": 4,       # 4 parallel models (39GB more VRAM than H200)
        "memory_per_model_1d": 28,  # ~28GB per model for daily
        "memory_per_model_1h": 40,  # ~40GB per model for hourly
        "notes": "High performance profile for B200 with 180GB VRAM (39GB more than H200). "
                 "Conservative batch size increase for hourly (3200 vs H200's 2560). "
                 "Supports 4 parallel workers vs H200's 3. "
                 "Total capacity: 4 workers × 40GB = 160GB used, 20GB buffer.",
    },
}


# GPU name to profile mapping for auto-detection
GPU_NAME_MAPPING = {
    "rtx 4090": "rtx_4090",
    "rtx 5090": "rtx_5090",
    "a100-sxm4-40gb": "a100_40gb",
    "a100-sxm4-80gb": "a100_80gb",
    "a100 80gb": "a100_80gb",
    "a100 40gb": "a100_40gb",
    "h200": "h200",
    "b200": "b200",
}


def detect_gpu_profile() -> Optional[str]:
    """
    Auto-detect GPU profile based on available hardware.

    Returns:
        Profile name if GPU detected, None otherwise
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        # Get GPU name from PyTorch
        gpu_name = torch.cuda.get_device_name(0).lower()

        # Try to match GPU name to profile
        for name_pattern, profile in GPU_NAME_MAPPING.items():
            if name_pattern in gpu_name:
                print(f"🔍 GPU AUTO-DETECT: Detected '{torch.cuda.get_device_name(0)}' -> using profile '{profile}'")
                return profile

        # If no match, return None (will use default)
        print(f"⚠️  GPU AUTO-DETECT: Unknown GPU '{torch.cuda.get_device_name(0)}', using default profile")
        return None

    except Exception as e:
        print(f"⚠️  GPU AUTO-DETECT: Failed to detect GPU ({e}), using default profile")
        return None


def get_gpu_profile(profile_name: str = None) -> Dict[str, Any]:
    """
    Get GPU profile configuration.

    Args:
        profile_name: Profile name (e.g., 'rtx_4090', 'rtx_5090')
                     If None, auto-detects GPU or reads from GPU_PROFILE env var

    Returns:
        Dictionary with GPU profile settings

    Raises:
        ValueError: If profile name is not found
    """
    if profile_name is None:
        # First try environment variable
        profile_name = os.getenv("GPU_PROFILE")

        # If not set, try auto-detection
        if profile_name is None:
            detected_profile = detect_gpu_profile()
            if detected_profile:
                profile_name = detected_profile
            else:
                # Fall back to default
                profile_name = "rtx_4090"
                print(f"ℹ️  Using default GPU profile: {profile_name}")

    profile_name = profile_name.lower()

    if profile_name not in GPU_PROFILES:
        available = ", ".join(GPU_PROFILES.keys())
        raise ValueError(
            f"GPU profile '{profile_name}' not found. "
            f"Available profiles: {available}"
        )

    return GPU_PROFILES[profile_name]


def get_batch_size(interval: str, profile_name: str = None) -> int:
    """
    Get optimal batch size for the given interval and GPU profile.

    Args:
        interval: Data interval ('1d' or '1h')
        profile_name: GPU profile name (uses env var if None)

    Returns:
        Optimal batch size for the interval
    """
    profile = get_gpu_profile(profile_name)

    if interval == '1d':
        return profile['batch_size_1d']
    else:  # '1h' or other short intervals
        return profile['batch_size_1h']


def get_max_workers(profile_name: str = None) -> int:
    """
    Get maximum parallel workers for the GPU profile.

    Args:
        profile_name: GPU profile name (uses env var if None)

    Returns:
        Maximum number of parallel workers
    """
    profile = get_gpu_profile(profile_name)
    return profile['max_workers']


def print_profile_info(profile_name: str = None):
    """Print detailed information about the GPU profile."""
    profile = get_gpu_profile(profile_name)

    print(f"\n{'='*70}")
    print(f"GPU PROFILE: {profile['name']}")
    print(f"{'='*70}")
    print(f"VRAM:                {profile['vram_gb']} GB")
    print(f"Batch Size (1d):     {profile['batch_size_1d']}")
    print(f"Batch Size (1h):     {profile['batch_size_1h']}")
    print(f"Max Workers:         {profile['max_workers']}")
    print(f"Memory/Model (1d):   ~{profile['memory_per_model_1d']} GB")
    print(f"Memory/Model (1h):   ~{profile['memory_per_model_1h']} GB")
    print(f"\nNotes: {profile['notes']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Print all available profiles
    print("\n" + "="*70)
    print("AVAILABLE GPU PROFILES")
    print("="*70)

    for profile_name in GPU_PROFILES.keys():
        print_profile_info(profile_name)
