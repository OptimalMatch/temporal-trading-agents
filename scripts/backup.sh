#!/bin/bash
source venv/bin/activate
cd ~/temporal-trading-agents
python extractFromMongo.py
BACKUP_NAME="temporal-complete-$(date +%Y%m%d-%H%M%S)"
echo ""
echo "📦 Creating backup archive..."
tar -czf ~/backups/${BACKUP_NAME}.tar.gz \
    -C /workspace model_cache \
    -C ~/backups mongodb-json \
    -C ~/temporal-trading-agents .env

#rm -rf ~/backups/mongodb-json
echo ""
echo "✅ BACKUP COMPLETE!"
ls -lh ~/backups/${BACKUP_NAME}.tar.gz
echo ""
echo "📥 Download with:"
echo "  scp -P YOUR_PORT root@YOUR_HOST:~/backups/${BACKUP_NAME}.tar.gz ~/temporal-backups/"
