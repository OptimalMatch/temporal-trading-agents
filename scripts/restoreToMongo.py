import asyncio
import sys
import os
import json
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient

async def import_all(backup_dir, mongodb_url):
    """Import MongoDB data from JSON backup files."""

    # Connect to MongoDB
    client = AsyncIOMotorClient(mongodb_url)
    db = client.temporal_trading

    backup_dir = os.path.expanduser(backup_dir)

    if not os.path.exists(backup_dir):
        print(f"❌ Backup directory not found: {backup_dir}")
        client.close()
        return

    # Restore consensus results
    consensus_file = f'{backup_dir}/consensus_results.json'
    if os.path.exists(consensus_file):
        with open(consensus_file, 'r') as f:
            results = json.load(f)

        # Convert ISO datetime strings back to datetime objects
        for doc in results:
            for key, val in doc.items():
                if isinstance(val, str) and 'T' in val and (val.endswith('Z') or '+' in val or val.count(':') >= 2):
                    try:
                        doc[key] = datetime.fromisoformat(val.replace('Z', '+00:00'))
                    except:
                        pass  # Keep as string if not a valid datetime

            # Remove _id so MongoDB can generate new ones
            if '_id' in doc:
                del doc['_id']

        if results:
            # Clear existing data (optional - comment out if you want to preserve existing data)
            # await db.consensus_results.delete_many({})
            await db.consensus_results.insert_many(results)
            print(f"  ✓ Restored {len(results)} consensus results")
        else:
            print(f"  ⚠ No consensus results to restore")
    else:
        print(f"  ⚠ Consensus results file not found: {consensus_file}")

    # Restore strategy analyses
    analyses_file = f'{backup_dir}/strategy_analyses.json'
    if os.path.exists(analyses_file):
        with open(analyses_file, 'r') as f:
            analyses = json.load(f)

        for doc in analyses:
            for key, val in doc.items():
                if isinstance(val, str) and 'T' in val and (val.endswith('Z') or '+' in val or val.count(':') >= 2):
                    try:
                        doc[key] = datetime.fromisoformat(val.replace('Z', '+00:00'))
                    except:
                        pass

            if '_id' in doc:
                del doc['_id']

        if analyses:
            # await db.strategy_analyses.delete_many({})
            await db.strategy_analyses.insert_many(analyses)
            print(f"  ✓ Restored {len(analyses)} strategy analyses")
        else:
            print(f"  ⚠ No strategy analyses to restore")
    else:
        print(f"  ⚠ Strategy analyses file not found: {analyses_file}")

    # Restore HuggingFace configs
    configs_file = f'{backup_dir}/huggingface_configs.json'
    if os.path.exists(configs_file):
        with open(configs_file, 'r') as f:
            configs = json.load(f)

        for doc in configs:
            for key, val in doc.items():
                if isinstance(val, str) and 'T' in val and (val.endswith('Z') or '+' in val or val.count(':') >= 2):
                    try:
                        doc[key] = datetime.fromisoformat(val.replace('Z', '+00:00'))
                    except:
                        pass

            if '_id' in doc:
                del doc['_id']

        if configs:
            # await db.huggingface_configs.delete_many({})
            await db.huggingface_configs.insert_many(configs)
            print(f"  ✓ Restored {len(configs)} HuggingFace configs")
        else:
            print(f"  ⚠ No HuggingFace configs to restore")
    else:
        print(f"  ⚠ HuggingFace configs file not found: {configs_file}")

    client.close()
    print("\n  ✅ MongoDB data restored successfully!")

if __name__ == "__main__":
    # Get MongoDB URL from environment or use default
    mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')

    # Get backup directory from command line or use default
    if len(sys.argv) < 2:
        backup_dir = os.path.expanduser('~/restore/mongodb-json')
        print(f"Using default backup directory: {backup_dir}")
    else:
        backup_dir = sys.argv[1]

    asyncio.run(import_all(backup_dir, mongodb_url))
