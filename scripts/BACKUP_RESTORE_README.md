# Backup and Restore Guide

This guide explains how to backup and restore your Temporal Trading Agents installation, including trained models, MongoDB data, and configuration.

## What Gets Backed Up

The backup includes:
1. **Model Cache** (`model_cache/`) - All trained forecasting models
2. **MongoDB Data** - JSON exports of:
   - Consensus results
   - Strategy analyses
   - HuggingFace configurations
3. **Environment Variables** (`.env`) - API keys and configuration

## Backup Process

### On the Source Machine (e.g., RunPod H200)

1. **Navigate to your installation:**
   ```bash
   cd ~/temporal-trading-agents
   ```

2. **Run the backup script:**
   ```bash
   ./scripts/backup.sh
   ```

   This will:
   - Export MongoDB data to JSON files
   - Create a compressed tar.gz archive containing:
     - Model cache
     - MongoDB JSON exports
     - .env file
   - Save it to `~/backups/temporal-complete-YYYYMMDD-HHMMSS.tar.gz`

3. **Download the backup to your local machine:**
   ```bash
   # From your local machine
   scp -P PORT root@HOST:~/backups/temporal-complete-*.tar.gz ~/temporal-backups/

   # Example (as shown in backup.sh output):
   scp -P 14516 -i ~/.ssh/id_ed25519 root@103.196.86.35:~/backups/temporal-complete-20251110-155455.tar.gz ~/temporal-backups/
   ```

## Restore Process

### On the Target Machine (e.g., New Server)

1. **Upload the backup to the target machine:**
   ```bash
   # From your local machine
   scp ~/temporal-backups/temporal-complete-*.tar.gz user@new-host:~/
   ```

2. **Clone the repository (if not already done):**
   ```bash
   git clone https://github.com/yourusername/temporal-trading-agents.git
   cd temporal-trading-agents
   ```

3. **Run the restore script:**
   ```bash
   ./scripts/restore.sh ~/temporal-complete-20251110-155455.tar.gz
   ```

   This will:
   - Extract the backup archive to `~/restore/`
   - Restore the model cache to the appropriate location
   - Restore the .env file (backing up any existing one)
   - Restore MongoDB data from JSON exports

4. **Review and update .env file:**
   ```bash
   nano .env
   ```

   Update any machine-specific settings:
   - Verify API keys are correct
   - Update GPU profile if different hardware: `GPU_PROFILE=rtx_4090` or `GPU_PROFILE=h200_sxm`
   - Check file paths are appropriate for the new machine

5. **Start the application:**

   **If using Docker:**
   ```bash
   # Ensure MongoDB is running
   docker compose up -d mongodb

   # Wait for MongoDB to be healthy
   docker compose ps

   # Start backend and frontend
   docker compose up -d backend frontend
   ```

   **If running directly:**
   ```bash
   # Start MongoDB (if not already running)
   sudo systemctl start mongodb

   # Activate virtual environment
   source venv/bin/activate

   # Start backend
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

6. **Verify the restore:**
   - Check that models are available in the model cache
   - Verify MongoDB data through the API or UI
   - Test HuggingFace configurations if applicable

7. **Cleanup (optional):**
   ```bash
   rm -rf ~/restore
   ```

## Files Reference

### Backup Scripts
- `scripts/extractFromMongo.py` - Exports MongoDB collections to JSON
- `scripts/backup.sh` - Creates compressed backup archive

### Restore Scripts
- `scripts/restoreToMongo.py` - Imports JSON data back into MongoDB
- `scripts/restore.sh` - Main restore script (extracts and restores everything)

## Troubleshooting

### MongoDB Connection Issues

If restore fails to connect to MongoDB:

1. **Check MongoDB is running:**
   ```bash
   docker compose ps mongodb  # If using Docker
   # or
   sudo systemctl status mongodb  # If running directly
   ```

2. **Verify MongoDB URL:**
   ```bash
   echo $MONGODB_URL
   # Should be: mongodb://localhost:27017
   # or for Docker: mongodb://mongodb:27017
   ```

3. **Test connection manually:**
   ```bash
   mongosh mongodb://localhost:27017
   ```

### Model Cache Restore Issues

If models aren't loading after restore:

1. **Check model cache location:**
   ```bash
   ls -la model_cache/  # Host installation
   # or
   docker exec temporal-trading-backend ls -la /workspace/model_cache/  # Docker
   ```

2. **Verify permissions:**
   ```bash
   chmod -R 755 model_cache/
   ```

3. **Check disk space:**
   ```bash
   df -h
   ```

### .env File Issues

If environment variables aren't being picked up:

1. **Verify .env file exists:**
   ```bash
   ls -la .env
   ```

2. **Check file format (no BOM, Unix line endings):**
   ```bash
   file .env
   # Should show: ASCII text
   ```

3. **Reload environment:**
   ```bash
   # For Docker
   docker compose down
   docker compose up -d

   # For direct installation
   source .env
   ```

## Security Notes

- **Never commit .env files to git** - They contain sensitive API keys
- **Secure backup files** - They contain your API keys and configuration
- **Use secure transfer methods** - SCP with SSH keys, not plain FTP
- **Review restored .env** - Ensure no credentials are exposed
- **Rotate API keys** - If backup security is uncertain, rotate all API keys

## Migration Checklist

- [ ] Backup created on source machine
- [ ] Backup downloaded to local machine
- [ ] Backup uploaded to target machine
- [ ] Repository cloned on target machine
- [ ] Restore script executed successfully
- [ ] .env file reviewed and updated for new machine
- [ ] MongoDB connection verified
- [ ] Model cache verified (check file count and size)
- [ ] Docker containers rebuilt (if using Docker)
- [ ] Application started and accessible
- [ ] Test analysis run to verify models work
- [ ] HuggingFace configurations tested (if applicable)
- [ ] Cleanup: temporary restore files removed
- [ ] Cleanup: old backups archived or deleted

## Automated Backups (Optional)

To set up automated daily backups:

```bash
# Edit crontab
crontab -e

# Add line for daily backup at 2 AM
0 2 * * * cd ~/temporal-trading-agents && ./scripts/backup.sh >> ~/backups/backup.log 2>&1
```

To set up backup rotation (keep only last 7 days):

```bash
# Add to crontab after backup
0 3 * * * find ~/backups -name "temporal-complete-*.tar.gz" -mtime +7 -delete
```
