"""
MongoDB-backed data synchronization manager.
Manages background download jobs with database persistence and progress tracking.
"""
import sys
from pathlib import Path
import threading
import time
from typing import Optional, Dict, Callable
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.models import DataSyncJob, SyncJobStatus, TickerWatchlist, DataInventory
from strategies.data_cache import get_cache


class DataSyncManager:
    """Manages market data synchronization with MongoDB persistence"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.cache = get_cache()
        self.running_jobs: Dict[str, threading.Thread] = {}  # job_id -> thread
        self.max_concurrent = 3  # Maximum concurrent downloads

    async def initialize(self):
        """Initialize collections and indexes"""
        # Create indexes for faster queries
        await self.db.data_sync_jobs.create_index("job_id", unique=True)
        await self.db.data_sync_jobs.create_index([("symbol", 1), ("status", 1)])
        await self.db.ticker_watchlist.create_index("symbol", unique=True)
        await self.db.data_inventory.create_index([("symbol", 1), ("period", 1), ("interval", 1)], unique=True)

    # ==================== Sync Job Management ====================

    async def create_sync_job(self, symbol: str, period: Optional[str] = None, interval: str = '1d') -> DataSyncJob:
        """
        Create a new sync job or return existing running job.

        Args:
            symbol: Trading symbol
            period: Data period (auto-detect if None)
            interval: Data interval

        Returns:
            DataSyncJob model
        """
        # Auto-detect period
        if period is None:
            is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
            period = '2y' if is_crypto else '5y'

        # Check for existing running/paused job
        existing = await self.db.data_sync_jobs.find_one({
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "status": {"$in": [SyncJobStatus.RUNNING.value, SyncJobStatus.PAUSED.value, SyncJobStatus.PENDING.value]}
        })

        if existing:
            return DataSyncJob(**existing)

        # Check if at max concurrency
        running_count = await self.db.data_sync_jobs.count_documents({"status": SyncJobStatus.RUNNING.value})
        status = SyncJobStatus.PENDING if running_count >= self.max_concurrent else SyncJobStatus.PENDING

        # Create new job
        job = DataSyncJob(
            symbol=symbol,
            period=period,
            interval=interval,
            status=status
        )

        await self.db.data_sync_jobs.insert_one(job.dict())
        return job

    async def start_sync_job(self, job_id: str) -> bool:
        """
        Start a sync job in background.

        Args:
            job_id: Job ID to start

        Returns:
            True if started, False otherwise
        """
        # Get job from database
        job_doc = await self.db.data_sync_jobs.find_one({"job_id": job_id})
        if not job_doc:
            return False

        job = DataSyncJob(**job_doc)

        # Check if already running
        if job.status == SyncJobStatus.RUNNING:
            return True

        # Check concurrency limit
        running_count = await self.db.data_sync_jobs.count_documents({"status": SyncJobStatus.RUNNING.value})
        if running_count >= self.max_concurrent:
            return False

        # Update status to running
        await self.db.data_sync_jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": SyncJobStatus.RUNNING.value,
                "started_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }}
        )

        # Start download in background thread
        thread = threading.Thread(target=self._run_sync_job, args=(job_id,))
        thread.daemon = True
        thread.start()

        self.running_jobs[job_id] = thread

        return True

    def _run_sync_job(self, job_id: str):
        """
        Run the sync job in background thread (synchronous).

        Args:
            job_id: Job ID to run
        """
        import asyncio

        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_sync_job_async(job_id))
        finally:
            loop.close()

    async def _run_sync_job_async(self, job_id: str):
        """
        Run the sync job asynchronously.

        Args:
            job_id: Job ID to run
        """
        from strategies.massive_s3_data_source import get_massive_s3_source

        # Get job from database
        job_doc = await self.db.data_sync_jobs.find_one({"job_id": job_id})
        if not job_doc:
            return

        job = DataSyncJob(**job_doc)

        try:
            # Create progress callback
            async def progress_callback(completed: int, total: int, elapsed: float, skipped: int = 0):
                progress_percent = (completed / total * 100) if total > 0 else 0
                eta = (elapsed / completed * (total - completed)) if completed > 0 else 0

                await self.db.data_sync_jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "progress_percent": progress_percent,
                        "total_files": total,
                        "completed_files": completed,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                        "updated_at": datetime.utcnow()
                    }}
                )

            # Get S3 data source
            s3_source = get_massive_s3_source()

            # Fetch data with progress callback
            # Note: This is synchronous, we need to run in executor
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                data = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    s3_source.fetch_data,
                    job.symbol,
                    job.period,
                    job.interval
                )

            # Update job as completed
            await self.db.data_sync_jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": SyncJobStatus.COMPLETED.value,
                    "progress_percent": 100.0,
                    "completed_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }}
            )

            # Update inventory
            await self._update_inventory(job.symbol, job.period, job.interval, data)

            # Clear progress marker
            self.cache.clear_progress(job.symbol, job.period, job.interval)

        except Exception as e:
            # Update job as failed
            await self.db.data_sync_jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": SyncJobStatus.FAILED.value,
                    "error_message": str(e),
                    "completed_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }}
            )

        finally:
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            # Start next pending job if any
            await self._start_next_pending_job()

    async def _start_next_pending_job(self):
        """Start the next pending job if under concurrency limit"""
        running_count = await self.db.data_sync_jobs.count_documents({"status": SyncJobStatus.RUNNING.value})

        if running_count < self.max_concurrent:
            # Find next pending job
            pending_job = await self.db.data_sync_jobs.find_one(
                {"status": SyncJobStatus.PENDING.value},
                sort=[("created_at", 1)]
            )

            if pending_job:
                await self.start_sync_job(pending_job["job_id"])

    async def _update_inventory(self, symbol: str, period: str, interval: str, data):
        """Update data inventory after successful download"""
        if data is None or data.empty:
            return

        inventory = DataInventory(
            symbol=symbol,
            period=period,
            interval=interval,
            total_days=len(data),
            date_range_start=data.index.min().to_pydatetime() if len(data) > 0 else None,
            date_range_end=data.index.max().to_pydatetime() if len(data) > 0 else None,
            last_updated_at=datetime.utcnow(),
            is_complete=True,  # Assume complete for now
            missing_dates=[]
        )

        # Get file size from cache
        cache_key = self.cache._get_cache_key(symbol, period, interval)
        cache_path = self.cache._get_cache_path(cache_key)
        if cache_path.exists():
            inventory.file_size_bytes = cache_path.stat().st_size

        # Upsert inventory
        await self.db.data_inventory.update_one(
            {"symbol": symbol, "period": period, "interval": interval},
            {"$set": inventory.dict()},
            upsert=True
        )

    # ==================== Job Control ====================

    async def pause_job(self, job_id: str) -> bool:
        """Pause a running job"""
        result = await self.db.data_sync_jobs.update_one(
            {"job_id": job_id, "status": SyncJobStatus.RUNNING.value},
            {"$set": {
                "pause_requested": True,
                "updated_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0

    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        job_doc = await self.db.data_sync_jobs.find_one({"job_id": job_id})
        if not job_doc or job_doc["status"] != SyncJobStatus.PAUSED.value:
            return False

        # Update status and clear pause flag
        await self.db.data_sync_jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": SyncJobStatus.PENDING.value,
                "pause_requested": False,
                "updated_at": datetime.utcnow()
            }}
        )

        # Try to start it
        return await self.start_sync_job(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        result = await self.db.data_sync_jobs.update_one(
            {"job_id": job_id, "status": {"$in": [SyncJobStatus.RUNNING.value, SyncJobStatus.PAUSED.value, SyncJobStatus.PENDING.value]}},
            {"$set": {
                "cancel_requested": True,
                "status": SyncJobStatus.CANCELLED.value,
                "completed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }}
        )

        # Clear progress marker
        if result.modified_count > 0:
            job_doc = await self.db.data_sync_jobs.find_one({"job_id": job_id})
            if job_doc:
                self.cache.clear_progress(job_doc["symbol"], job_doc["period"], job_doc["interval"])

        return result.modified_count > 0

    # ==================== Watchlist Management ====================

    async def add_to_watchlist(self, symbol: str, period: Optional[str] = None, interval: str = '1d',
                              auto_sync: bool = True, priority: int = 0, tags: list = None) -> TickerWatchlist:
        """Add ticker to watchlist"""
        if period is None:
            is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
            period = '2y' if is_crypto else '5y'

        watchlist_item = TickerWatchlist(
            symbol=symbol,
            period=period,
            interval=interval,
            auto_sync=auto_sync,
            priority=priority,
            tags=tags or []
        )

        await self.db.ticker_watchlist.update_one(
            {"symbol": symbol},
            {"$set": watchlist_item.dict()},
            upsert=True
        )

        return watchlist_item

    async def get_watchlist(self, enabled_only: bool = True) -> list:
        """Get watchlist tickers"""
        query = {"enabled": True} if enabled_only else {}
        cursor = self.db.ticker_watchlist.find(query).sort("priority", -1)
        return [TickerWatchlist(**doc) async for doc in cursor]

    async def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove ticker from watchlist"""
        result = await self.db.ticker_watchlist.delete_one({"symbol": symbol})
        return result.deleted_count > 0

    # ==================== Inventory Management ====================

    async def get_inventory(self, symbol: Optional[str] = None) -> list:
        """Get data inventory"""
        query = {"symbol": symbol} if symbol else {}
        cursor = self.db.data_inventory.find(query)
        return [DataInventory(**doc) async for doc in cursor]


# Global instance
_sync_manager: Optional[DataSyncManager] = None


async def get_sync_manager(db: AsyncIOMotorDatabase) -> DataSyncManager:
    """Get or create the global sync manager instance"""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = DataSyncManager(db)
        await _sync_manager.initialize()
    return _sync_manager
