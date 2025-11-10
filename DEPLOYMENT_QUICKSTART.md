# Deployment Quick Start Guide

Choose your deployment method based on your environment.

## Local Deployment (Docker)

For local development with Docker and GPU:

```bash
# Clone repository
git clone https://github.com/<your-username>/temporal-trading-agents.git
cd temporal-trading-agents

# Set GPU profile (optional, default: rtx_4090)
export GPU_PROFILE=rtx_4090

# Start all services
docker compose up -d

# Check status
docker logs temporal-trading-backend | grep GPU
```

**See**: [docker-compose.yml](docker-compose.yml) for configuration

## RunPod Deployment (All-in-One)

For RunPod instances with CUDA/PyTorch pre-installed:

```bash
# SSH into your RunPod instance
# Clone repository
git clone https://github.com/<your-username>/temporal-trading-agents.git
cd temporal-trading-agents

# One-command automated all-in-one deployment
./scripts/runpod_deploy.sh

# Automatically installs and configures:
# ✅ MongoDB 7.0 (local database)
# ✅ Backend API (FastAPI + PyTorch)
# ✅ Frontend Dashboard (nginx)
# ✅ Auto-detects GPU and selects optimal profile
# ✅ Creates systemd services for auto-start
```

**See**: [docs/RUNPOD_DEPLOYMENT.md](docs/RUNPOD_DEPLOYMENT.md) for details

## GPU Profiles

The system automatically optimizes batch sizes and parallelism based on GPU:

| Profile | GPU | VRAM | Batch (1h) | Workers | Speed |
|---------|-----|------|------------|---------|-------|
| `rtx_4090` | RTX 4090 | 24GB | 768 | 2 | 3-4x |
| `rtx_5090` | RTX 5090 | 32GB | 1536 | 2 | 6-8x |
| `a100_40gb` | A100 | 40GB | 2048 | 2 | 8x |
| `a100_80gb` | A100 | 80GB | 4096 | 3 | 16x |

**See**: [docs/GPU_PROFILES.md](docs/GPU_PROFILES.md) for all profiles

## Quick Commands

### Local (Docker)
```bash
# Start
docker compose up -d

# Stop
docker compose down

# Logs
docker logs -f temporal-trading-backend

# GPU status
docker exec temporal-trading-backend nvidia-smi
```

### RunPod (Direct)
```bash
# Start
./scripts/runpod_start.sh start

# Stop
./scripts/runpod_start.sh stop

# Status
./scripts/runpod_start.sh status

# Logs
./scripts/runpod_start.sh logs

# GPU status
nvidia-smi
```

## Switching GPU Profiles

### Local (Docker)
```bash
export GPU_PROFILE=rtx_5090
docker compose down
docker compose up -d
```

### RunPod (Direct)
```bash
# Update .env
echo "GPU_PROFILE=rtx_5090" >> .env

# Restart
./scripts/runpod_start.sh restart
```

## Accessing the Services

### Local (Docker)
- **Backend API**: `http://localhost:10750/`
- **API Docs**: `http://localhost:10750/docs`
- **Health Check**: `http://localhost:10750/health`
- **Frontend**: `http://localhost:10752/`

### RunPod (All-in-One)
- **Frontend**: `http://<instance-ip>/` (port 80)
- **API** (proxied): `http://<instance-ip>/api/`
- **API Docs**: `http://<instance-ip>/docs`
- **Health Check**: `http://<instance-ip>/health`

Everything runs on port 80 with nginx routing.

## Testing GPU Profile

```bash
# Local (Docker)
docker exec temporal-trading-backend python scripts/test_gpu_profile.py --recommend

# RunPod (Direct)
source venv/bin/activate
python scripts/test_gpu_profile.py --recommend
```

## Next Steps

1. ✅ Deploy using method above
2. ✅ Configure API keys in `.env`
3. ✅ Verify GPU profile is correct
4. ✅ Run a test analysis
5. ✅ Monitor GPU usage during training

## Documentation

- [GPU Profiles](docs/GPU_PROFILES.md) - All available GPU configurations
- [RunPod Deployment](docs/RUNPOD_DEPLOYMENT.md) - Detailed RunPod guide
- [Docker Compose](docker-compose.yml) - Local deployment config

## Support

- **Test utility**: `scripts/test_gpu_profile.py`
- **Deployment script**: `scripts/runpod_deploy.sh`
- **Management script**: `scripts/runpod_start.sh`
