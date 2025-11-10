# RunPod Deployment Guide

This guide explains how to deploy Temporal Trading Agents on RunPod with different GPU profiles.

RunPod instances come with CUDA and PyTorch pre-installed, so we run the backend directly without Docker.

## Prerequisites

- RunPod account (https://runpod.io)
- PyTorch template (recommended) or Ubuntu + CUDA template
- Access to your repository (GitHub, GitLab, etc.)

## Quick Start (Automated)

### 1. Create RunPod Instance

Choose your GPU based on budget and performance needs:

| GPU | VRAM | Cost/hr | Recommended Profile | Speedup |
|-----|------|---------|---------------------|---------|
| RTX 4090 | 24 GB | ~$0.44 | `rtx_4090` | 3-4x |
| RTX 5090 | 32 GB | ~$0.60* | `rtx_5090` | 6-8x |
| A100 40GB | 40 GB | ~$1.14 | `a100_40gb` | 8x |
| A100 80GB | 80 GB | ~$1.89 | `a100_80gb` | 16x |

*Estimated pricing when available

**Recommended Template**: PyTorch 2.0+ (includes CUDA, Python, and common ML libraries)

### 2. One-Command Deployment

SSH into your RunPod instance and run:

```bash
# Clone repository
git clone https://github.com/<your-username>/temporal-trading-agents.git
cd temporal-trading-agents

# Run automated deployment
chmod +x scripts/runpod_deploy.sh
./scripts/runpod_deploy.sh
```

The deployment script will:
- ✅ Auto-detect your GPU and recommend optimal profile
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Setup MongoDB (local or Atlas)
- ✅ Configure environment variables
- ✅ Create systemd service for auto-start
- ✅ Optionally start the backend

**That's it!** The script handles everything automatically.

### 3. Managing the Backend

After deployment, use the helper script:

```bash
# Start the backend
./scripts/runpod_start.sh start

# Check status
./scripts/runpod_start.sh status

# View logs
./scripts/runpod_start.sh logs

# Restart
./scripts/runpod_start.sh restart

# Stop
./scripts/runpod_start.sh stop
```

Or use systemd (if installed by deployment script):

```bash
# Start
sudo systemctl start temporal-trading

# Status
sudo systemctl status temporal-trading

# Logs
sudo journalctl -u temporal-trading -f

# Enable auto-start on reboot
sudo systemctl enable temporal-trading
```

### 4. Configure API Keys

Edit the `.env` file created by the deployment script:

```bash
nano .env
```

Add your API keys:
```bash
POLYGON_API_KEY=your_actual_polygon_key
MASSIVE_ACCESS_KEY_ID=your_actual_key
MASSIVE_SECRET_ACCESS_KEY=your_actual_secret
# ... etc
```

Then restart:
```bash
./scripts/runpod_start.sh restart
```

### 5. Access the Backend

RunPod provides public URLs. Find your instance's IP:

```bash
curl ifconfig.me
```

Access the API:
- **Health Check**: `http://<your-ip>:8000/health`
- **API Docs**: `http://<your-ip>:8000/docs`
- **Backend API**: `http://<your-ip>:8000`

**Security Note**: For production, use RunPod's TCP proxy or set up nginx with SSL.

### 6. Verify Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Check GPU detection
python scripts/test_gpu_profile.py --check

# Get profile recommendation
python scripts/test_gpu_profile.py --recommend

# Test your profile
python scripts/test_gpu_profile.py --test rtx_5090

# Check backend health
curl http://localhost:8000/health
```

## Manual Setup (Advanced)

If you prefer manual setup or the automated script didn't work:

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

### 2. Setup MongoDB

**Option A: MongoDB Atlas (Recommended)**
```bash
# Get connection string from https://cloud.mongodb.com
export MONGODB_URL="mongodb+srv://username:password@cluster.mongodb.net/temporal_trading"
```

**Option B: Local MongoDB**
```bash
# Install MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org
sudo systemctl start mongod

export MONGODB_URL="mongodb://localhost:27017"
```

### 3. Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
GPU_PROFILE=rtx_5090
MONGODB_URL=mongodb://localhost:27017
POLYGON_API_KEY=your_key_here
# ... add other keys
EOF

# Load environment
export $(grep -v '^#' .env | xargs)
```

### 4. Start Backend

```bash
# Activate virtual environment
source venv/bin/activate

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Profile-Specific Configuration

The deployment script auto-detects your GPU, but you can manually override:

### RTX 4090
```bash
export GPU_PROFILE=rtx_4090
./scripts/runpod_start.sh restart
```

**Performance:**
- Training time: ~45 min for 5-model hourly ensemble
- Batch size: 768 (hourly), 512 (daily)
- 2 parallel models
- Memory: ~24GB VRAM usage

### RTX 5090
```bash
export GPU_PROFILE=rtx_5090
./scripts/runpod_start.sh restart
```

**Performance:**
- Training time: ~30 min for 5-model hourly ensemble
- Batch size: 1536 (hourly), 1024 (daily)
- 2-3 parallel models
- Memory: ~32GB VRAM usage

### A100 40GB
```bash
export GPU_PROFILE=a100_40gb
./scripts/runpod_start.sh restart
```

**Performance:**
- Training time: ~25 min for 5-model hourly ensemble
- Batch size: 2048 (hourly), 1024 (daily)
- 2 parallel models
- Memory: ~40GB VRAM usage

### A100 80GB
```bash
export GPU_PROFILE=a100_80gb
./scripts/runpod_start.sh restart
```

**Performance:**
- Training time: ~15 min for 5-model hourly ensemble
- Batch size: 4096 (hourly), 2048 (daily)
- 3 parallel models
- Memory: ~80GB VRAM usage

## Monitoring and Troubleshooting

### Monitor GPU Usage

```bash
# Check GPU status
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# GPU utilization with details
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
```

### Check Training Progress

```bash
# View logs (if using systemd)
sudo journalctl -u temporal-trading -f

# View logs (if using runpod_start.sh)
./scripts/runpod_start.sh logs

# Or check direct log file
tail -f /tmp/temporal-trading.log

# Check for errors
grep -i error /tmp/temporal-trading.log
```

### Common Issues

#### Out of Memory (OOM)

If you see CUDA OOM errors:

```bash
# Switch to a more conservative profile
export GPU_PROFILE=rtx_4090  # Even if you have RTX 5090
echo "GPU_PROFILE=rtx_4090" >> .env
./scripts/runpod_start.sh restart

# Or create custom profile with smaller batches in backend/gpu_profiles.py
```

#### Slow Training

```bash
# Check backend status
./scripts/runpod_start.sh status

# Verify correct profile is active
grep "GPU:" /tmp/temporal-trading.log | tail -1

# Check GPU utilization
nvidia-smi
# Should see 95-100% GPU utilization during training
```

#### Wrong Profile Loaded

```bash
# Update .env file
nano .env
# Change GPU_PROFILE=rtx_5090

# Restart backend
./scripts/runpod_start.sh restart
```

#### Backend Won't Start

```bash
# Check Python environment
source venv/bin/activate
python --version
pip list | grep torch

# Check for port conflicts
lsof -i :8000

# Try manual start to see errors
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
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

Create an auto-shutdown script:
```bash
#!/bin/bash
# Auto-shutdown after 2 hours of idle
# Check if backend is running every 5 minutes
# Shutdown if no activity for 2 hours

IDLE_TIME=0
MAX_IDLE=7200  # 2 hours

while true; do
    if pgrep -f "uvicorn backend.main:app" > /dev/null; then
        # Check if any training is happening (GPU activity)
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [ "$GPU_UTIL" -gt 10 ]; then
            IDLE_TIME=0
        else
            IDLE_TIME=$((IDLE_TIME + 300))
        fi
    fi

    if [ "$IDLE_TIME" -ge "$MAX_IDLE" ]; then
        echo "Idle for 2 hours, shutting down..."
        sudo shutdown -h now
    fi

    sleep 300
done
```

Save as `/root/auto_shutdown.sh` and run in background:
```bash
chmod +x /root/auto_shutdown.sh
nohup /root/auto_shutdown.sh &
```

## Migrating Between GPU Types

### RTX 4090 → RTX 5090

When upgrading to a RunPod instance with RTX 5090:

```bash
# Stop backend
./scripts/runpod_start.sh stop

# Update profile in .env
sed -i 's/GPU_PROFILE=rtx_4090/GPU_PROFILE=rtx_5090/' .env

# Optional: Clear model cache to retrain with larger batches
rm -rf /workspace/model_cache/*

# Restart with new profile
./scripts/runpod_start.sh start

# Verify new profile loaded
./scripts/runpod_start.sh status
```

### Testing Before Migration

```bash
# Test new profile without changing active config
source venv/bin/activate
python scripts/test_gpu_profile.py --test rtx_5090

# Check VRAM requirements vs available
nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

## Advanced Configuration

### Custom Profiles

Create custom profile for specific RunPod GPU:

```python
# Edit backend/gpu_profiles.py
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
echo "GPU_PROFILE=runpod_custom" >> .env
./scripts/runpod_start.sh restart
```

### Running on Different Ports

To run multiple instances on the same RunPod instance:

```bash
# Instance 1 on port 8000
cd ~/temporal-trading-1
export PORT=8000
export GPU_PROFILE=rtx_5090
./scripts/runpod_start.sh start

# Instance 2 on port 8001
cd ~/temporal-trading-2
export PORT=8001
export GPU_PROFILE=rtx_4090
./scripts/runpod_start.sh start
```

### Persistent Storage

Use RunPod's network volumes for model cache:

```bash
# Mount RunPod volume to /workspace
# Then update .env to point to network volume
echo "MODEL_CACHE_DIR=/runpod-volume/model_cache" >> .env

# Create directory on network volume
mkdir -p /runpod-volume/model_cache

# Restart backend
./scripts/runpod_start.sh restart
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
