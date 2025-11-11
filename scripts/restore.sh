#!/bin/bash

# Temporal Trading Agents - Restore Script
# Usage: ./restore.sh <path-to-backup.tar.gz>

set -e  # Exit on error

if [ -z "$1" ]; then
    echo "❌ Error: No backup file specified"
    echo ""
    echo "Usage: ./restore.sh <path-to-backup.tar.gz>"
    echo ""
    echo "Example:"
    echo "  ./restore.sh ~/temporal-backups/temporal-complete-20251110-155455.tar.gz"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "🔄 TEMPORAL TRADING AGENTS - RESTORE"
echo "======================================"
echo ""
echo "Backup file: $BACKUP_FILE"
echo ""

# Create restore directory
RESTORE_DIR=~/restore
mkdir -p $RESTORE_DIR

echo "📦 Extracting backup archive..."
tar -xzf "$BACKUP_FILE" -C $RESTORE_DIR
echo "  ✓ Backup extracted to $RESTORE_DIR"
echo ""

# Restore model cache
if [ -d "$RESTORE_DIR/model_cache" ]; then
    echo "💾 Restoring model cache..."

    # Check if Docker is being used with backend container
    if command -v docker &> /dev/null && docker ps --format '{{.Names}}' | grep -q 'temporal-trading-backend'; then
        echo "  ℹ Detected Docker backend container - restoring to Docker volume"

        # Count files to restore
        FILE_COUNT=$(find "$RESTORE_DIR/model_cache" -type f | wc -l)
        echo "  Copying $FILE_COUNT files to container..."

        # Copy directly to container
        docker cp "$RESTORE_DIR/model_cache/." temporal-trading-backend:/app/model_cache/

        # Verify
        CONTAINER_SIZE=$(docker exec temporal-trading-backend du -sh /app/model_cache | cut -f1)
        CONTAINER_FILES=$(docker exec temporal-trading-backend sh -c 'find /app/model_cache -type f | wc -l')
        echo "  ✓ Model cache restored to Docker volume ($CONTAINER_SIZE, $CONTAINER_FILES files)"
    else
        # Running on host without Docker or container not running
        if [ -d "/workspace" ]; then
            # Running in a container with /workspace mount
            MODEL_CACHE_TARGET="/workspace/model_cache"
        else
            # Running on host - use local directory
            MODEL_CACHE_TARGET="$(pwd)/model_cache"
        fi

        echo "  Target: $MODEL_CACHE_TARGET"

        # Create backup of existing cache if it exists
        if [ -d "$MODEL_CACHE_TARGET" ]; then
            BACKUP_SUFFIX=$(date +%Y%m%d-%H%M%S)
            echo "  ⚠ Existing model_cache found, backing up to model_cache.backup-$BACKUP_SUFFIX"
            mv "$MODEL_CACHE_TARGET" "${MODEL_CACHE_TARGET}.backup-${BACKUP_SUFFIX}"
        fi

        mkdir -p "$(dirname $MODEL_CACHE_TARGET)"
        cp -r "$RESTORE_DIR/model_cache" "$MODEL_CACHE_TARGET"
        echo "  ✓ Model cache restored ($(du -sh $MODEL_CACHE_TARGET | cut -f1))"
    fi
else
    echo "  ⚠ No model_cache found in backup"
fi
echo ""

# Restore .env file
if [ -f "$RESTORE_DIR/.env" ]; then
    echo "🔐 Restoring .env file..."

    ENV_TARGET="$(pwd)/.env"

    # Backup existing .env if it exists
    if [ -f "$ENV_TARGET" ]; then
        BACKUP_SUFFIX=$(date +%Y%m%d-%H%M%S)
        echo "  ⚠ Existing .env found, backing up to .env.backup-$BACKUP_SUFFIX"
        cp "$ENV_TARGET" "${ENV_TARGET}.backup-${BACKUP_SUFFIX}"
    fi

    cp "$RESTORE_DIR/.env" "$ENV_TARGET"
    echo "  ✓ .env file restored"
    echo ""
    echo "  ⚠ IMPORTANT: Review the .env file and update any host-specific settings"
    echo "    - API keys may need to be updated"
    echo "    - File paths may need adjustment"
    echo "    - GPU profile may differ on this machine"
else
    echo "  ⚠ No .env file found in backup"
fi
echo ""

# Restore MongoDB data
if [ -d "$RESTORE_DIR/mongodb-json" ]; then
    echo "🗄️  Restoring MongoDB data..."

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "  ⚠ Virtual environment not found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -q motor pymongo
    else
        source venv/bin/activate
    fi

    # Auto-detect MongoDB URL
    if [ -z "$MONGODB_URL" ]; then
        # Check if MongoDB is running in Docker on custom port
        if command -v docker &> /dev/null && docker ps --format '{{.Ports}}' | grep -q '10751->27017'; then
            MONGODB_URL="mongodb://localhost:10751"
            echo "  ℹ Detected MongoDB in Docker on port 10751"
        else
            MONGODB_URL="mongodb://localhost:27017"
        fi
    fi

    export MONGODB_URL
    echo "  Using MongoDB URL: $MONGODB_URL"

    # Run the restore script
    cd "$(dirname "$0")/.."
    python scripts/restoreToMongo.py "$RESTORE_DIR/mongodb-json"

    deactivate
else
    echo "  ⚠ No mongodb-json directory found in backup"
fi
echo ""

echo "✅ RESTORE COMPLETE!"
echo ""
echo "📋 Summary:"
echo "  - Model cache: $([ -d "$MODEL_CACHE_TARGET" ] && echo "✓ Restored" || echo "⚠ Not found in backup")"
echo "  - .env file: $([ -f "$(pwd)/.env" ] && echo "✓ Restored" || echo "⚠ Not found in backup")"
echo "  - MongoDB data: $([ -d "$RESTORE_DIR/mongodb-json" ] && echo "✓ Restored" || echo "⚠ Not found in backup")"
echo ""
echo "🔄 Next steps:"
echo "  1. Review and update .env file for this machine"
echo "  2. Rebuild Docker containers if using Docker:"
echo "     docker compose build --no-cache backend frontend"
echo "     docker compose up -d"
echo "  3. Verify MongoDB data in the application"
echo ""
echo "🧹 Cleanup:"
echo "  To remove temporary restore files:"
echo "  rm -rf $RESTORE_DIR"
echo ""
