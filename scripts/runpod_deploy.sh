#!/bin/bash
set -e

#############################################################################
# RunPod Deployment Script for Temporal Trading Agents
#############################################################################
# This script deploys the backend directly on a RunPod instance without Docker.
# RunPod instances come with CUDA/PyTorch pre-installed.
#
# Usage:
#   1. SSH into your RunPod instance
#   2. git clone <your-repo>
#   3. cd temporal-trading-agents
#   4. ./scripts/runpod_deploy.sh
#
# Environment variables (set before running):
#   GPU_PROFILE - GPU profile to use (default: auto-detect)
#   MONGODB_URL - MongoDB connection string (default: cloud MongoDB)
#   POLYGON_API_KEY - Polygon.io API key
#   PORT - Backend port (default: 8000)
#############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RunPod Deployment - Temporal Trading${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if we're on RunPod
if [ ! -f /etc/runpod-release ]; then
    echo -e "${YELLOW}⚠️  Warning: Not detected as RunPod instance${NC}"
    echo -e "${YELLOW}   This script is optimized for RunPod but will try to run anyway${NC}\n"
fi

# Detect GPU and recommend profile
echo -e "${GREEN}🔍 Detecting GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo -e "   GPU: ${GPU_NAME}"
    echo -e "   VRAM: ${GPU_MEMORY} MB (~$((GPU_MEMORY / 1024)) GB)\n"

    # Auto-detect profile if not set
    if [ -z "$GPU_PROFILE" ]; then
        if [[ "$GPU_NAME" =~ "4090" ]]; then
            export GPU_PROFILE="rtx_4090"
        elif [[ "$GPU_NAME" =~ "5090" ]]; then
            export GPU_PROFILE="rtx_5090"
        elif [[ "$GPU_NAME" =~ "A100" ]] && [ $((GPU_MEMORY / 1024)) -gt 70 ]; then
            export GPU_PROFILE="a100_80gb"
        elif [[ "$GPU_NAME" =~ "A100" ]]; then
            export GPU_PROFILE="a100_40gb"
        else
            # Default based on VRAM
            if [ $((GPU_MEMORY / 1024)) -gt 70 ]; then
                export GPU_PROFILE="a100_80gb"
            elif [ $((GPU_MEMORY / 1024)) -gt 35 ]; then
                export GPU_PROFILE="a100_40gb"
            elif [ $((GPU_MEMORY / 1024)) -gt 28 ]; then
                export GPU_PROFILE="rtx_5090"
            else
                export GPU_PROFILE="rtx_4090"
            fi
        fi
        echo -e "${GREEN}✅ Auto-detected profile: ${GPU_PROFILE}${NC}\n"
    else
        echo -e "${GREEN}✅ Using configured profile: ${GPU_PROFILE}${NC}\n"
    fi
else
    echo -e "${RED}❌ No GPU detected!${NC}"
    echo -e "${RED}   nvidia-smi not found. Make sure you're on a GPU instance.${NC}\n"
    exit 1
fi

# Check Python version
echo -e "${GREEN}🐍 Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}\n"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "   ${PYTHON_VERSION}\n"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${GREEN}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}\n"
fi

# Activate virtual environment
echo -e "${GREEN}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}📦 Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "\n${GREEN}📦 Installing dependencies...${NC}"
if [ -f "backend/requirements.txt" ]; then
    pip install -r backend/requirements.txt
    echo -e "${GREEN}✅ Backend dependencies installed${NC}\n"
else
    echo -e "${RED}❌ backend/requirements.txt not found!${NC}\n"
    exit 1
fi

# Setup MongoDB
echo -e "${GREEN}💾 Configuring MongoDB...${NC}"
if [ -z "$MONGODB_URL" ]; then
    echo -e "${YELLOW}⚠️  No MONGODB_URL set. You have two options:${NC}"
    echo -e "${YELLOW}   1. Use MongoDB Atlas (recommended for RunPod)${NC}"
    echo -e "${YELLOW}   2. Install MongoDB locally on this instance${NC}\n"

    read -p "Use MongoDB Atlas? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n${BLUE}MongoDB Atlas Setup:${NC}"
        echo -e "1. Go to https://cloud.mongodb.com"
        echo -e "2. Create a free cluster"
        echo -e "3. Get your connection string"
        echo -e "4. Set it as: export MONGODB_URL='mongodb+srv://...'\n"
        read -p "Enter MongoDB Atlas URL (or press Enter to skip): " ATLAS_URL
        if [ ! -z "$ATLAS_URL" ]; then
            export MONGODB_URL="$ATLAS_URL"
            echo -e "${GREEN}✅ MongoDB URL configured${NC}\n"
        else
            echo -e "${YELLOW}⚠️  Skipping MongoDB setup. You'll need to configure it manually.${NC}\n"
            export MONGODB_URL="mongodb://localhost:27017"
        fi
    else
        echo -e "\n${GREEN}Installing MongoDB locally...${NC}"
        # Install MongoDB
        wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
        echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
        sudo apt-get update
        sudo apt-get install -y mongodb-org

        # Start MongoDB
        sudo systemctl start mongod
        sudo systemctl enable mongod

        export MONGODB_URL="mongodb://localhost:27017"
        echo -e "${GREEN}✅ MongoDB installed and running${NC}\n"
    fi
else
    echo -e "${GREEN}✅ Using configured MongoDB: ${MONGODB_URL}${NC}\n"
fi

# Create necessary directories
echo -e "${GREEN}📁 Creating directories...${NC}"
mkdir -p /workspace/model_cache
mkdir -p /workspace/torch_compile_cache/triton
mkdir -p /workspace/torch_compile_cache/inductor
mkdir -p /tmp/crypto_data_cache
echo -e "${GREEN}✅ Directories created${NC}\n"

# Set environment variables
echo -e "${GREEN}🔧 Configuring environment...${NC}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_CACHE_DIR=/workspace/model_cache
export TRITON_CACHE_DIR=/workspace/torch_compile_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_compile_cache/inductor

# Set default port
PORT=${PORT:-8000}

# Display configuration
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "GPU Profile:      ${GREEN}${GPU_PROFILE}${NC}"
echo -e "MongoDB URL:      ${GREEN}${MONGODB_URL}${NC}"
echo -e "Backend Port:     ${GREEN}${PORT}${NC}"
echo -e "Model Cache:      ${GREEN}${MODEL_CACHE_DIR}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Test GPU profile
echo -e "${GREEN}🧪 Testing GPU profile...${NC}"
python3 scripts/test_gpu_profile.py --recommend

# Check if API keys are set
echo -e "\n${YELLOW}📋 Checking API keys...${NC}"
MISSING_KEYS=()
if [ -z "$POLYGON_API_KEY" ]; then
    MISSING_KEYS+=("POLYGON_API_KEY")
fi
if [ -z "$MASSIVE_ACCESS_KEY_ID" ]; then
    MISSING_KEYS+=("MASSIVE_ACCESS_KEY_ID (optional)")
fi

if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing API keys:${NC}"
    for key in "${MISSING_KEYS[@]}"; do
        echo -e "   - ${key}"
    done
    echo -e "\n${YELLOW}You can set them now or later in .env file${NC}\n"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${GREEN}📝 Creating .env file...${NC}"
    cat > .env << EOF
# MongoDB
MONGODB_URL=${MONGODB_URL}

# GPU Configuration
GPU_PROFILE=${GPU_PROFILE}

# API Keys (fill these in)
POLYGON_API_KEY=${POLYGON_API_KEY:-your_polygon_key_here}
MASSIVE_ACCESS_KEY_ID=${MASSIVE_ACCESS_KEY_ID:-your_massive_key_here}
MASSIVE_SECRET_ACCESS_KEY=${MASSIVE_SECRET_ACCESS_KEY:-your_massive_secret_here}
MASSIVE_S3_ENDPOINT=${MASSIVE_S3_ENDPOINT:-your_s3_endpoint_here}
MASSIVE_S3_BUCKET=${MASSIVE_S3_BUCKET:-your_bucket_here}
MASSIVE_API_KEY=${MASSIVE_API_KEY:-your_massive_api_key_here}

# PyTorch
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache directories
MODEL_CACHE_DIR=/workspace/model_cache
TRITON_CACHE_DIR=/workspace/torch_compile_cache/triton
TORCHINDUCTOR_CACHE_DIR=/workspace/torch_compile_cache/inductor

# Backend
PYTHONUNBUFFERED=1
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
    echo -e "${YELLOW}⚠️  Edit .env file to add your API keys${NC}\n"
fi

# Load .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create systemd service for auto-start
echo -e "${GREEN}🚀 Creating systemd service...${NC}"
INSTALL_DIR=$(pwd)
cat > /tmp/temporal-trading.service << EOF
[Unit]
Description=Temporal Trading Agents Backend
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${INSTALL_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="GPU_PROFILE=${GPU_PROFILE}"
Environment="MONGODB_URL=${MONGODB_URL}"
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/temporal-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
echo -e "${GREEN}✅ Systemd service created${NC}\n"

# Ask if user wants to start now
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "You can now:"
echo -e "  ${GREEN}1. Start the backend:${NC}"
echo -e "     sudo systemctl start temporal-trading"
echo -e ""
echo -e "  ${GREEN}2. Enable auto-start on boot:${NC}"
echo -e "     sudo systemctl enable temporal-trading"
echo -e ""
echo -e "  ${GREEN}3. Check status:${NC}"
echo -e "     sudo systemctl status temporal-trading"
echo -e ""
echo -e "  ${GREEN}4. View logs:${NC}"
echo -e "     sudo journalctl -u temporal-trading -f"
echo -e ""
echo -e "  ${GREEN}5. Run manually (for testing):${NC}"
echo -e "     source venv/bin/activate"
echo -e "     uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"
echo -e ""

read -p "Start the backend now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}🚀 Starting backend...${NC}"
    sudo systemctl start temporal-trading
    sleep 3
    sudo systemctl status temporal-trading --no-pager

    echo -e "\n${GREEN}✅ Backend started!${NC}"
    echo -e "   Access at: ${GREEN}http://$(hostname -I | awk '{print $1}'):${PORT}${NC}"
    echo -e "   Health check: ${GREEN}http://$(hostname -I | awk '{print $1}'):${PORT}/health${NC}\n"
    echo -e "View logs: ${BLUE}sudo journalctl -u temporal-trading -f${NC}\n"
else
    echo -e "\n${YELLOW}Backend not started. Start it manually when ready.${NC}\n"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"
