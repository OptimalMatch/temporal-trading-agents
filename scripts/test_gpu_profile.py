#!/usr/bin/env python3
"""
GPU Profile Testing Utility

Test different GPU profiles to find optimal settings for your hardware.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.gpu_profiles import GPU_PROFILES, get_gpu_profile, print_profile_info
import torch


def check_gpu():
    """Check if GPU is available and print info."""
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        print("   Make sure NVIDIA drivers and CUDA are installed.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\n✅ GPU Detected: {gpu_name}")
    print(f"   Total VRAM: {total_memory:.1f} GB")

    return True


def recommend_profile():
    """Recommend a profile based on detected GPU."""
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_name(0).lower()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\n{'='*70}")
    print("PROFILE RECOMMENDATION")
    print(f"{'='*70}")

    # Match GPU name or VRAM
    if "4090" in gpu_name or "rtx 4090" in gpu_name:
        recommended = "rtx_4090"
    elif "5090" in gpu_name or "rtx 5090" in gpu_name:
        recommended = "rtx_5090"
    elif "a100" in gpu_name:
        if total_vram > 70:
            recommended = "a100_80gb"
        else:
            recommended = "a100_40gb"
    elif total_vram >= 70:
        recommended = "a100_80gb"
    elif total_vram >= 35:
        recommended = "a100_40gb"
    elif total_vram >= 28:
        recommended = "rtx_5090"
    else:
        recommended = "rtx_4090"

    print(f"Recommended profile: {recommended}")
    print(f"\nTo use this profile:")
    print(f"  export GPU_PROFILE={recommended}")
    print(f"  docker compose up -d")

    return recommended


def list_all_profiles():
    """List all available profiles."""
    print(f"\n{'='*70}")
    print("ALL AVAILABLE GPU PROFILES")
    print(f"{'='*70}")

    for profile_name in GPU_PROFILES.keys():
        print_profile_info(profile_name)


def test_profile(profile_name: str):
    """Test a specific profile."""
    print(f"\n{'='*70}")
    print(f"TESTING PROFILE: {profile_name}")
    print(f"{'='*70}")

    try:
        profile = get_gpu_profile(profile_name)
        print_profile_info(profile_name)

        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            required_vram = profile['vram_gb']

            print(f"VRAM Check:")
            print(f"  Required: {required_vram} GB")
            print(f"  Available: {total_vram:.1f} GB")

            if total_vram >= required_vram * 0.95:  # Allow 5% tolerance
                print(f"  ✅ VRAM sufficient for this profile")
            else:
                print(f"  ⚠️  WARNING: May not have enough VRAM!")
                print(f"     Consider using a profile with lower requirements")

        # Estimate training time
        print(f"\nEstimated Performance:")
        print(f"  Hourly training (2.6M samples, 10 epochs):")
        print(f"    - Batch size: {profile['batch_size_1h']}")
        print(f"    - Iterations per epoch: ~{2_600_000 // profile['batch_size_1h']:,}")
        print(f"    - Workers: {profile['max_workers']} parallel models")

        # Calculate rough time estimate
        iterations_per_epoch = 2_600_000 // profile['batch_size_1h']
        total_iterations = iterations_per_epoch * 10  # 10 epochs
        # Assume ~15 iterations/sec on modern GPU
        time_per_model = total_iterations / 15 / 60  # minutes
        time_5_models = time_per_model * 5 / profile['max_workers']  # With parallelism

        print(f"    - Estimated time (5 models): ~{int(time_5_models)} minutes")

    except ValueError as e:
        print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Profile Testing Utility")
    parser.add_argument("--list", action="store_true", help="List all available profiles")
    parser.add_argument("--test", type=str, help="Test a specific profile")
    parser.add_argument("--recommend", action="store_true", help="Get profile recommendation")
    parser.add_argument("--check", action="store_true", help="Check GPU availability")

    args = parser.parse_args()

    if args.list:
        list_all_profiles()
    elif args.test:
        test_profile(args.test)
    elif args.recommend:
        check_gpu()
        recommend_profile()
    elif args.check:
        check_gpu()
    else:
        # Default: check GPU and recommend
        check_gpu()
        recommend_profile()
        print("\nFor more options, run: python scripts/test_gpu_profile.py --help")


if __name__ == "__main__":
    main()
