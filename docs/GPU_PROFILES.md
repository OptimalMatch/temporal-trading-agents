# GPU Profile Configuration

This document describes the GPU profile system for optimizing training performance across different GPU types.

## Overview

The GPU profile system allows you to optimize batch sizes and parallel training workers based on your GPU's VRAM capacity. This ensures maximum training speed while avoiding out-of-memory (OOM) errors.

## Available Profiles

### RTX 4090 (Default)
- **VRAM**: 24 GB
- **Batch Size (1d)**: 512 (4x speedup)
- **Batch Size (1h)**: 768 (3x speedup)
- **Max Workers**: 2 parallel models
- **Memory per Model**: ~6 GB (daily), ~12 GB (hourly)

**Use case**: Consumer/prosumer GPU with excellent price/performance for training.

### RTX 5090 (Upcoming)
- **VRAM**: 32 GB (expected)
- **Batch Size (1d)**: 1024 (8x speedup)
- **Batch Size (1h)**: 1536 (6x speedup)
- **Max Workers**: 2-3 parallel models
- **Memory per Model**: ~12 GB (daily), ~16 GB (hourly)

**Use case**: Next-gen consumer GPU with larger VRAM for bigger batch sizes.

### A100 40GB
- **VRAM**: 40 GB
- **Batch Size (1d)**: 1024
- **Batch Size (1h)**: 2048 (8x speedup)
- **Max Workers**: 2 parallel models
- **Memory per Model**: ~12 GB (daily), ~20 GB (hourly)

**Use case**: Data center GPU, ideal for cloud deployments (RunPod, AWS, GCP).

### A100 80GB
- **VRAM**: 80 GB
- **Batch Size (1d)**: 2048
- **Batch Size (1h)**: 4096 (16x speedup!)
- **Max Workers**: 3 parallel models
- **Memory per Model**: ~16 GB (daily), ~26 GB (hourly)

**Use case**: Maximum performance for production deployments and large-scale training.

## Usage

### Local Deployment

#### Default (RTX 4090)
No configuration needed - RTX 4090 profile is the default:

```bash
docker compose up -d
```

#### Switch to Different Profile
Set the `GPU_PROFILE` environment variable:

```bash
# For RTX 5090
export GPU_PROFILE=rtx_5090
docker compose up -d

# For A100 40GB
export GPU_PROFILE=a100_40gb
docker compose up -d

# For A100 80GB
export GPU_PROFILE=a100_80gb
docker compose up -d
```

Or add to your `.env` file:
```bash
GPU_PROFILE=rtx_5090
```

### RunPod Deployment

RunPod provides on-demand GPU instances. Here's how to deploy with different profiles:

#### 1. RTX 4090 on RunPod
```bash
# In your RunPod instance terminal
git clone <your-repo>
cd temporal-trading-agents

# Set profile for RTX 4090 (or leave default)
export GPU_PROFILE=rtx_4090

# Deploy
docker compose up -d
```

#### 2. RTX 5090 on RunPod (When Available)
```bash
export GPU_PROFILE=rtx_5090
docker compose up -d
```

#### 3. A100 40GB on RunPod
```bash
# RunPod offers A100 instances
export GPU_PROFILE=a100_40gb
docker compose up -d
```

#### 4. A100 80GB on RunPod
```bash
export GPU_PROFILE=a100_80gb
docker compose up -d
```

### Verifying Profile Configuration

After starting the backend, check the logs to verify the profile:

```bash
docker logs temporal-trading-backend | grep GPU
```

You should see:
```
🎮 GPU: Using profile 'NVIDIA RTX 4090' (24GB VRAM)
⚡ API: ProcessPoolExecutor initialized with 2 workers
```

## Creating Custom Profiles

To create a custom profile, edit `backend/gpu_profiles.py`:

```python
GPU_PROFILES["my_custom_gpu"] = {
    "name": "My Custom GPU",
    "vram_gb": 16,
    "batch_size_1d": 256,
    "batch_size_1h": 512,
    "max_workers": 2,
    "memory_per_model_1d": 4,
    "memory_per_model_1h": 8,
    "notes": "Custom configuration for testing",
}
```

Then use it:
```bash
export GPU_PROFILE=my_custom_gpu
docker compose up -d
```

## Performance Comparison

| GPU | Daily Batch | Hourly Batch | Speedup | Workers | Training Time (5 models) |
|-----|-------------|--------------|---------|---------|--------------------------|
| RTX 4090 | 512 | 768 | 3-4x | 2 | ~45 min (hourly) |
| RTX 5090 | 1024 | 1536 | 6-8x | 2-3 | ~30 min (hourly) |
| A100 40GB | 1024 | 2048 | 8x | 2 | ~25 min (hourly) |
| A100 80GB | 2048 | 4096 | 16x | 3 | ~15 min (hourly) |

*Estimates based on 2.6M hourly samples, 10 epochs per model*

## Troubleshooting

### Out of Memory Errors

If you see CUDA OOM errors:

1. Check your actual GPU VRAM:
   ```bash
   nvidia-smi
   ```

2. Use a more conservative profile or create a custom one with smaller batch sizes

3. Reduce `max_workers` in your custom profile

### Slow Training

If training is slower than expected:

1. Verify you're using the correct profile for your GPU
2. Check GPU utilization: `nvidia-smi`
3. Try a profile with larger batch sizes if you have VRAM headroom

### Profile Not Found

If you get "GPU profile not found" error:

1. Check available profiles: `docker exec temporal-trading-backend python -m backend.gpu_profiles`
2. Verify GPU_PROFILE spelling (use lowercase: `rtx_4090`, not `RTX_4090`)

## Migration Guide: RTX 4090 → RTX 5090

When migrating to RTX 5090:

```bash
# Stop current services
docker compose down

# Update environment
export GPU_PROFILE=rtx_5090

# Rebuild and restart
docker compose build --no-cache backend
docker compose up -d

# Verify
docker logs temporal-trading-backend | grep GPU
```

Expected improvement:
- **2x larger batches** (768 → 1536 for hourly)
- **Potential for 3 workers** instead of 2
- **~33% faster training** overall

## Future Profiles

Planned additions:
- RTX 6000 Ada
- H100 (80GB/94GB)
- L40S
- Consumer GPUs (RTX 4080, 4070 Ti)

To request a new profile, open an issue with your GPU specs and desired batch sizes.
