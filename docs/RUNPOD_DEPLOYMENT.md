# RunPod Deployment Guide

This guide explains how to deploy Temporal Trading Agents on RunPod with different GPU profiles.

## Prerequisites

- RunPod account (https://runpod.io)
- Docker template or PyTorch template
- Access to your repository (GitHub, GitLab, etc.)

## Quick Start

### 1. Create RunPod Instance

Choose your GPU based on budget and performance needs:

| GPU | VRAM | Cost/hr | Recommended Profile | Speedup |
|-----|------|---------|---------------------|---------|
| RTX 4090 | 24 GB | ~$0.44 | `rtx_4090` | 3-4x |
| RTX 5090 | 32 GB | ~$0.60* | `rtx_5090` | 6-8x |
| A100 40GB | 40 GB | ~$1.14 | `a100_40gb` | 8x |
| A100 80GB | 80 GB | ~$1.89 | `a100_80gb` | 16x |

*Estimated pricing when available

### 2. Initial Setup

SSH into your RunPod instance:

```bash
# Install Docker if not present
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/<your-username>/temporal-trading-agents.git
cd temporal-trading-agents
```

### 3. Configure Environment

Create `.env` file with your API keys:

```bash
cat > .env << 'EOF'
# API Keys
POLYGON_API_KEY=your_polygon_key_here
MASSIVE_ACCESS_KEY_ID=your_massive_key_here
MASSIVE_SECRET_ACCESS_KEY=your_massive_secret_here
MASSIVE_S3_ENDPOINT=your_s3_endpoint_here
MASSIVE_S3_BUCKET=your_bucket_here
MASSIVE_API_KEY=your_massive_api_key_here

# GPU Profile - Set based on your RunPod GPU
GPU_PROFILE=rtx_4090
EOF
```

### 4. Set GPU Profile

#### For RTX 4090
```bash
export GPU_PROFILE=rtx_4090
```

#### For RTX 5090 (when available)
```bash
export GPU_PROFILE=rtx_5090
```

#### For A100 40GB
```bash
export GPU_PROFILE=a100_40gb
```

#### For A100 80GB
```bash
export GPU_PROFILE=a100_80gb
```

Or add it to your `.env` file:
```bash
echo "GPU_PROFILE=rtx_5090" >> .env
```

### 5. Deploy

```bash
# Build and start services
docker compose up -d

# Check logs
docker logs temporal-trading-backend

# Verify GPU profile
docker logs temporal-trading-backend | grep GPU
```

You should see:
```
🎮 GPU: Using profile 'NVIDIA RTX 5090' (32GB VRAM)
⚡ API: ProcessPoolExecutor initialized with 2 workers
```

### 6. Test Installation

```bash
# Check GPU detection
docker exec temporal-trading-backend python scripts/test_gpu_profile.py --check

# Get profile recommendation
docker exec temporal-trading-backend python scripts/test_gpu_profile.py --recommend

# Test your profile
docker exec temporal-trading-backend python scripts/test_gpu_profile.py --test rtx_5090
```

### 7. Access Services

RunPod provides public URLs. Configure port forwarding:

- **Backend API**: Port 10750
- **Frontend Dashboard**: Port 10752
- **MongoDB**: Port 10751 (internal only)

Use RunPod's TCP proxy or set up your own reverse proxy.

## Profile-Specific Setup

### RTX 4090 Setup
```bash
# Standard setup (already default)
export GPU_PROFILE=rtx_4090
docker compose up -d
```

**Performance:**
- Training time: ~45 min for 5-model hourly ensemble
- Batch size: 768 (hourly), 512 (daily)
- 2 parallel models

### RTX 5090 Setup
```bash
# Upgraded performance
export GPU_PROFILE=rtx_5090
docker compose up -d
```

**Performance:**
- Training time: ~30 min for 5-model hourly ensemble
- Batch size: 1536 (hourly), 1024 (daily)
- 2-3 parallel models (test with your data)

### A100 40GB Setup
```bash
# Cloud/data center GPU
export GPU_PROFILE=a100_40gb
docker compose up -d
```

**Performance:**
- Training time: ~25 min for 5-model hourly ensemble
- Batch size: 2048 (hourly), 1024 (daily)
- 2 parallel models

### A100 80GB Setup
```bash
# Maximum performance
export GPU_PROFILE=a100_80gb
docker compose up -d
```

**Performance:**
- Training time: ~15 min for 5-model hourly ensemble
- Batch size: 4096 (hourly), 2048 (daily)
- 3 parallel models

## Monitoring and Troubleshooting

### Monitor GPU Usage

```bash
# Inside container
docker exec temporal-trading-backend nvidia-smi

# Continuous monitoring
docker exec temporal-trading-backend watch -n 1 nvidia-smi
```

### Check Training Progress

```bash
# View backend logs
docker logs -f temporal-trading-backend

# Check for errors
docker logs temporal-trading-backend 2>&1 | grep -i error

# View GPU memory usage
docker logs temporal-trading-backend 2>&1 | grep -i memory
```

### Common Issues

#### Out of Memory (OOM)

If you see CUDA OOM errors:

```bash
# Switch to a more conservative profile
export GPU_PROFILE=rtx_4090  # Even if you have RTX 5090
docker compose up -d

# Or create custom profile with smaller batches
```

#### Slow Training

```bash
# Verify correct profile is active
docker logs temporal-trading-backend | grep GPU

# Check GPU utilization
docker exec temporal-trading-backend nvidia-smi

# Should see 95-100% GPU utilization during training
```

#### Wrong Profile Loaded

```bash
# Clear environment and restart
unset GPU_PROFILE
export GPU_PROFILE=rtx_5090
docker compose down
docker compose up -d
```

## Cost Optimization

### On-Demand vs Spot

**Spot Instances** (50-70% cheaper, can be interrupted):
```bash
# Use for:
- Development/testing
- Non-critical backtests
- Experimentation

# Training can resume from cache if interrupted
```

**On-Demand** (stable, guaranteed):
```bash
# Use for:
- Production paper trading
- Critical analysis
- Important backtests
```

### Auto-Shutdown

Add to your RunPod startup script:
```bash
#!/bin/bash
# Auto-shutdown after 2 hours of idle
timeout 7200 docker compose up
```

### Storage Optimization

Use RunPod's network storage for model cache:
```bash
# Mount RunPod network volume
docker compose down
# Edit docker-compose.yml to use RunPod volume path
docker compose up -d
```

## Migrating Between GPU Types

### RTX 4090 → RTX 5090

```bash
# Stop services
docker compose down

# Update profile
export GPU_PROFILE=rtx_5090

# Optional: Clear model cache to retrain with larger batches
docker volume rm temporal-trading-agents_model_cache

# Restart
docker compose up -d
```

### Testing Before Migration

```bash
# Test new profile without changing
docker exec temporal-trading-backend python scripts/test_gpu_profile.py --test rtx_5090

# Check VRAM requirements vs available
```

## Advanced Configuration

### Custom Profiles

Create custom profile for specific RunPod GPU:

```python
# In backend/gpu_profiles.py
GPU_PROFILES["runpod_custom"] = {
    "name": "RunPod Custom",
    "vram_gb": 24,
    "batch_size_1d": 512,
    "batch_size_1h": 1024,  # Between rtx_4090 and rtx_5090
    "max_workers": 2,
    "memory_per_model_1d": 6,
    "memory_per_model_1h": 14,
    "notes": "Custom profile for RunPod testing",
}
```

Then use it:
```bash
export GPU_PROFILE=runpod_custom
docker compose up -d
```

### Multiple Instances

Run multiple deployments on same RunPod instance:

```bash
# Instance 1: RTX 5090 profile
cd ~/temporal-trading-1
export GPU_PROFILE=rtx_5090
docker compose -p trading1 up -d

# Instance 2: RTX 4090 profile (same GPU, conservative settings)
cd ~/temporal-trading-2
export GPU_PROFILE=rtx_4090
docker compose -p trading2 up -d
```

## Support

- Documentation: `docs/GPU_PROFILES.md`
- Test utility: `scripts/test_gpu_profile.py`
- Issues: https://github.com/<your-repo>/issues

## Next Steps

1. ✅ Deploy on RunPod with appropriate GPU
2. ✅ Configure GPU profile
3. ✅ Run test analysis
4. ✅ Monitor performance
5. ✅ Optimize based on results
