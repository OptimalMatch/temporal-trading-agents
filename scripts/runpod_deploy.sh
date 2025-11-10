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
    # Check if PyTorch is already installed (common on RunPod)
    if python3 -c "import torch" 2>/dev/null; then
        echo -e "${YELLOW}   PyTorch already installed, skipping torch/torchvision/torchaudio${NC}"
        # Install everything except PyTorch packages
        grep -v "^torch" backend/requirements.txt > /tmp/requirements_no_torch.txt
        pip install -r /tmp/requirements_no_torch.txt
        rm /tmp/requirements_no_torch.txt
    else
        # Install everything including PyTorch
        pip install -r backend/requirements.txt
    fi
    echo -e "${GREEN}✅ Backend dependencies installed${NC}\n"
else
    echo -e "${RED}❌ backend/requirements.txt not found!${NC}\n"
    exit 1
fi

# Setup MongoDB (All-in-One: Install locally by default)
echo -e "${GREEN}💾 Installing MongoDB...${NC}"
if [ -z "$MONGODB_URL" ]; then
    echo -e "${GREEN}Installing MongoDB 7.0 locally...${NC}"

    # Check if MongoDB is already installed
    if command -v mongod &> /dev/null; then
        echo -e "${YELLOW}   MongoDB already installed, skipping...${NC}"
    else
        # Install MongoDB
        wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add - 2>/dev/null
        echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
        sudo apt-get update -qq
        sudo apt-get install -y mongodb-org
    fi

    # Start MongoDB (handle both systemd and non-systemd environments)
    if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
        # SystemD available (traditional Linux)
        sudo systemctl start mongod
        sudo systemctl enable mongod
        sleep 2
        if sudo systemctl is-active --quiet mongod; then
            echo -e "${GREEN}✅ MongoDB installed and running${NC}"
        else
            echo -e "${YELLOW}⚠️  MongoDB service not running, attempting to start...${NC}"
            sudo systemctl restart mongod
            sleep 2
        fi
    else
        # No systemd (Docker container / RunPod)
        echo -e "${YELLOW}   systemd not available, starting MongoDB manually...${NC}"

        # Create MongoDB data directory
        sudo mkdir -p /data/db
        sudo chown -R mongodb:mongodb /data/db

        # Start MongoDB in background
        sudo -u mongodb mongod --fork --logpath /var/log/mongodb/mongod.log --dbpath /data/db

        sleep 2

        # Check if MongoDB is running
        if pgrep -x mongod > /dev/null; then
            echo -e "${GREEN}✅ MongoDB started in background${NC}"
        else
            echo -e "${RED}❌ Failed to start MongoDB${NC}"
            echo -e "${RED}   Check logs: tail /var/log/mongodb/mongod.log${NC}"
        fi
    fi

    export MONGODB_URL="mongodb://localhost:27017"
    echo -e "${GREEN}   URL: ${MONGODB_URL}${NC}\n"
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

# Install and configure Frontend
echo -e "\n${GREEN}🎨 Installing Frontend...${NC}"

# Install nginx if not present
if ! command -v nginx &> /dev/null; then
    echo -e "${GREEN}Installing nginx...${NC}"
    sudo apt-get update -qq
    sudo apt-get install -y nginx
    echo -e "${GREEN}✅ nginx installed${NC}"
else
    echo -e "${YELLOW}   nginx already installed${NC}"
fi

# Build frontend (if needed) or copy files
if [ -d "frontend" ]; then
    echo -e "${GREEN}Deploying frontend...${NC}"

    # Check if frontend needs to be built
    if [ -f "frontend/package.json" ]; then
        echo -e "${GREEN}Building frontend...${NC}"

        # Install Node.js if not present
        if ! command -v node &> /dev/null; then
            echo -e "${GREEN}Installing Node.js...${NC}"
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi

        # Build frontend
        cd frontend
        npm install
        npm run build
        cd ..

        # Copy build output to nginx
        sudo rm -rf /var/www/html/temporal-trading
        sudo mkdir -p /var/www/html/temporal-trading
        sudo cp -r frontend/dist/* /var/www/html/temporal-trading/ 2>/dev/null || \
        sudo cp -r frontend/build/* /var/www/html/temporal-trading/ 2>/dev/null || \
        sudo cp -r frontend/* /var/www/html/temporal-trading/

        echo -e "${GREEN}✅ Frontend built and deployed${NC}"
    else
        # Just copy static files
        sudo rm -rf /var/www/html/temporal-trading
        sudo mkdir -p /var/www/html/temporal-trading
        sudo cp -r frontend/* /var/www/html/temporal-trading/
        echo -e "${GREEN}✅ Frontend deployed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  frontend/ directory not found, skipping frontend deployment${NC}"
fi

# Configure nginx
echo -e "${GREEN}Configuring nginx...${NC}"
FRONTEND_PORT=${FRONTEND_PORT:-80}

sudo tee /etc/nginx/sites-available/temporal-trading > /dev/null << 'NGINX_EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    root /var/www/html/temporal-trading;
    index index.html;

    server_name _;

    # Frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Backend API proxy
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support for backend
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # Health check
    location /health {
        proxy_pass http://localhost:8000/health;
    }

    # API docs
    location /docs {
        proxy_pass http://localhost:8000/docs;
    }
}
NGINX_EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/temporal-trading /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test and start nginx
if sudo nginx -t 2>/dev/null; then
    # Start nginx (handle both systemd and non-systemd)
    if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
        sudo systemctl restart nginx
        sudo systemctl enable nginx
    else
        # Kill existing nginx and start fresh
        sudo pkill nginx || true
        sudo nginx
    fi
    echo -e "${GREEN}✅ nginx configured and running${NC}"
else
    echo -e "${YELLOW}⚠️  nginx configuration test failed, check /etc/nginx/sites-available/temporal-trading${NC}"
fi

echo ""

# Create systemd service or startup script
INSTALL_DIR=$(pwd)

if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
    # SystemD available - create service
    echo -e "${GREEN}🚀 Creating systemd service for backend...${NC}"
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
else
    # No systemd - will use runpod_start.sh script
    echo -e "${YELLOW}⚠️  systemd not available (Docker environment)${NC}"
    echo -e "${YELLOW}   Use ./scripts/runpod_start.sh to manage services${NC}\n"
fi

# Get instance IP
INSTANCE_IP=$(hostname -I | awk '{print $1}')

# Ask if user wants to start now
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}All-in-One Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}Services Installed:${NC}"
echo -e "  ✅ MongoDB 7.0"
echo -e "  ✅ Backend API (FastAPI + PyTorch)"
echo -e "  ✅ Frontend Dashboard (nginx)"
echo -e ""

# Show appropriate management commands based on environment
if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
    # SystemD environment
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
else
    # Docker/RunPod environment
    echo -e "You can now:"
    echo -e "  ${GREEN}1. Start all services:${NC}"
    echo -e "     ./scripts/runpod_start.sh start"
    echo -e ""
    echo -e "  ${GREEN}2. Check status:${NC}"
    echo -e "     ./scripts/runpod_start.sh status"
    echo -e ""
    echo -e "  ${GREEN}3. View logs:${NC}"
    echo -e "     ./scripts/runpod_start.sh logs"
    echo -e ""
    echo -e "  ${GREEN}4. Stop services:${NC}"
    echo -e "     ./scripts/runpod_start.sh stop"
    echo -e ""
fi

read -p "Start the backend now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}🚀 Starting all services...${NC}"

    if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
        # SystemD environment
        sudo systemctl start mongod
        sudo systemctl start nginx
        sudo systemctl start temporal-trading
        sleep 3

        echo -e "\n${GREEN}✅ Services Status:${NC}"

        if sudo systemctl is-active --quiet mongod; then
            echo -e "   ✅ MongoDB: ${GREEN}Running${NC}"
        else
            echo -e "   ❌ MongoDB: ${RED}Not running${NC}"
        fi

        if sudo systemctl is-active --quiet nginx; then
            echo -e "   ✅ nginx: ${GREEN}Running${NC}"
        else
            echo -e "   ❌ nginx: ${RED}Not running${NC}"
        fi

        if sudo systemctl is-active --quiet temporal-trading; then
            echo -e "   ✅ Backend: ${GREEN}Running${NC}"
        else
            echo -e "   ❌ Backend: ${RED}Not running${NC}"
        fi
    else
        # Docker/RunPod environment - use runpod_start.sh
        ./scripts/runpod_start.sh start
    fi

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Access Your Application${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Frontend:${NC}     http://${INSTANCE_IP}/"
    echo -e "${GREEN}Backend API:${NC}  http://${INSTANCE_IP}/api/"
    echo -e "${GREEN}API Docs:${NC}     http://${INSTANCE_IP}/docs"
    echo -e "${GREEN}Health:${NC}       http://${INSTANCE_IP}/health"
    echo -e "${GREEN}MongoDB:${NC}      mongodb://localhost:27017"
    echo -e "${BLUE}========================================${NC}\n"

    if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
        echo -e "View logs: ${BLUE}sudo journalctl -u temporal-trading -f${NC}\n"
    else
        echo -e "View logs: ${BLUE}./scripts/runpod_start.sh logs${NC}\n"
    fi
else
    if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
        echo -e "\n${YELLOW}Services not started. Start them manually when ready:${NC}"
        echo -e "  sudo systemctl start mongod"
        echo -e "  sudo systemctl start nginx"
        echo -e "  sudo systemctl start temporal-trading\n"
    else
        echo -e "\n${YELLOW}Services not started. Start them manually when ready:${NC}"
        echo -e "  ./scripts/runpod_start.sh start\n"
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ All-in-One Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""
echo -e "Next steps:"
echo -e "  1. Edit .env to add your API keys"
echo -e "  2. Access frontend at http://${INSTANCE_IP}/"
echo -e "  3. Start analyzing crypto markets!"
echo -e ""
