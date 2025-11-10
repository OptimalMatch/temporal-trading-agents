import asyncio
import sys
import os

# Set the MongoDB URL
os.environ['MONGODB_URL'] = 'mongodb://localhost:27017'

sys.path.insert(0, '/root/temporal-trading-agents')
from backend.database import Database
import json
from datetime import datetime

async def export_all():
    db = Database()
    await db.connect()

    backup_dir = os.path.expanduser('~/backups/mongodb-json')
    os.makedirs(backup_dir, exist_ok=True)

    # Consensus results
    results = []
    async for doc in db.db.consensus_results.find({}):
        doc['_id'] = str(doc['_id'])
        for key, val in doc.items():
            if isinstance(val, datetime):
                doc[key] = val.isoformat()
        results.append(doc)
    with open(f'{backup_dir}/consensus_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ {len(results)} consensus results")

    # Strategy analyses
    analyses = []
    async for doc in db.db.strategy_analyses.find({}):
        doc['_id'] = str(doc['_id'])
        for key, val in doc.items():
            if isinstance(val, datetime):
                doc[key] = val.isoformat()
        analyses.append(doc)
    with open(f'{backup_dir}/strategy_analyses.json', 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"  ✓ {len(analyses)} strategy analyses")

    # HuggingFace configs
    configs = []
    async for doc in db.db.huggingface_configs.find({}):
        doc['_id'] = str(doc['_id'])
        for key, val in doc.items():
            if isinstance(val, datetime):
                doc[key] = val.isoformat()
        configs.append(doc)
    with open(f'{backup_dir}/huggingface_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"  ✓ {len(configs)} HuggingFace configs")

    await db.disconnect()
    print("  ✅ MongoDB data exported to ~/backups/mongodb-json/")

asyncio.run(export_all())
