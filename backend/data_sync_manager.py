"""
MongoDB-backed data synchronization manager.
Manages background download jobs with database persistence and progress tracking.
"""
import sys
from pathlib import Path
import threading
import time
import os
from typing import Optional, Dict, Callable
from datetime import datetime, timedelta, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
import pymongo  # Synchronous MongoDB client for background threads

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

        # Store MongoDB URL for creating sync clients in background threads
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://mongodb:27017")
        self.db_name = "temporal_trading"

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
                "started_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
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

        # Create synchronous MongoDB client for this background thread
        # Motor (async) won't work in background threads, so we use pymongo (sync)
        sync_client = pymongo.MongoClient(self.mongodb_url)
        sync_db = sync_client[self.db_name]

        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_sync_job_async(job_id, loop, sync_db))
        finally:
            loop.close()
            sync_client.close()

    async def _run_sync_job_async(self, job_id: str, loop, sync_db):
        """
        Run the sync job asynchronously.

        Args:
            job_id: Job ID to run
            loop: Event loop for this thread
            sync_db: Synchronous MongoDB database for progress updates
        """
        from strategies.massive_s3_data_source import get_massive_s3_source
        from strategies.polygon_data_source import fetch_crypto_data_polygon
        import pandas as pd

        # Get job from database using sync client (Motor won't work in background threads)
        job_doc = sync_db.data_sync_jobs.find_one({"job_id": job_id})
        if not job_doc:
            return

        job = DataSyncJob(**job_doc)

        def fetch_hybrid_data(symbol: str, period: str, interval: str, progress_callback=None):
            """
            Hybrid data fetching: S3 for historical bulk + REST API for recent data gap.
            """
            print(f"🔄 HYBRID MODE: Fetching {symbol} ({period}, {interval})")

            # Step 1: Fetch bulk historical data from S3
            print(f"📦 Fetching historical data from S3...")
            s3_data = s3_source.fetch_data(symbol, period, interval, progress_callback)

            if s3_data is None or s3_data.empty:
                print(f"⚠️  No S3 data found, trying REST API only...")
                return fetch_crypto_data_polygon(symbol, period, interval)

            # Step 2: Check for gap between latest S3 data and today
            latest_s3_date = s3_data.index.max()
            today = pd.Timestamp.now(tz='UTC').floor('D')  # Floor to midnight UTC
            gap_days = (today - latest_s3_date).days

            print(f"📊 Latest S3 data: {latest_s3_date.strftime('%Y-%m-%d')}")
            print(f"📅 Today: {today.strftime('%Y-%m-%d')}")
            print(f"⏳ Gap: {gap_days} days")

            # Step 3: If gap > 1 day, fetch recent data from REST API
            if gap_days > 1:
                try:
                    print(f"📡 Fetching recent {gap_days} days from REST API...")

                    # Calculate period for REST API (just the gap)
                    gap_period = f"{gap_days}d" if gap_days < 30 else f"{gap_days//30}mo"

                    rest_data = fetch_crypto_data_polygon(symbol, gap_period, interval)

                    if rest_data is not None and not rest_data.empty:
                        # Merge S3 + REST API data
                        print(f"🔀 Merging S3 data ({len(s3_data)} rows) + REST API data ({len(rest_data)} rows)")

                        # Combine and remove duplicates (keep REST API data for overlaps)
                        combined = pd.concat([s3_data, rest_data])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        combined = combined.sort_index()

                        print(f"✅ Hybrid dataset: {len(combined)} total rows ({s3_data.index.min().strftime('%Y-%m-%d')} to {combined.index.max().strftime('%Y-%m-%d')})")
                        return combined
                    else:
                        print(f"⚠️  REST API returned no data, using S3 data only")
                        return s3_data

                except Exception as e:
                    print(f"⚠️  REST API fetch failed: {e}, using S3 data only")
                    return s3_data
            else:
                print(f"✅ S3 data is up-to-date (gap <= 1 day)")
                return s3_data

        def fetch_hybrid_data_range(symbol: str, start_date, end_date, interval: str, progress_callback=None):
            """
            Hybrid data fetching for specific date range: S3 + REST API if needed.
            """
            print(f"🔄 HYBRID RANGE: Fetching {symbol} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, {interval})")

            # Fetch from S3 for the range
            s3_data = s3_source.fetch_data_range(symbol, start_date, end_date, interval, progress_callback)

            if s3_data is None or s3_data.empty:
                # No S3 data, try REST API for entire range
                print(f"⚠️  No S3 data for range, trying REST API...")
                days_in_range = (end_date - start_date).days
                gap_period = f"{days_in_range}d" if days_in_range < 30 else f"{days_in_range//30}mo"
                return fetch_crypto_data_polygon(symbol, gap_period, interval)

            # Check if S3 data covers the full range
            latest_s3_date = s3_data.index.max()
            if latest_s3_date < pd.Timestamp(end_date, tz='UTC'):
                # Gap exists, fill with REST API
                gap_days = (pd.Timestamp(end_date, tz='UTC') - latest_s3_date).days
                if gap_days > 1:
                    try:
                        print(f"📡 Filling {gap_days} day gap with REST API...")
                        gap_period = f"{gap_days}d"
                        rest_data = fetch_crypto_data_polygon(symbol, gap_period, interval)

                        if rest_data is not None and not rest_data.empty:
                            combined = pd.concat([s3_data, rest_data])
                            combined = combined[~combined.index.duplicated(keep='last')]
                            combined = combined.sort_index()
                            print(f"✅ Hybrid range dataset: {len(combined)} rows")
                            return combined
                    except Exception as e:
                        print(f"⚠️  REST API fetch failed: {e}, using S3 data only")

            return s3_data

        try:
            # Get S3 data source
            s3_source = get_massive_s3_source()

            # Create synchronous progress callback that uses sync_db
            # This works because sync_db is a synchronous pymongo client
            def sync_progress_callback(completed: int, total: int, elapsed: float, skipped: int = 0):
                """Synchronous progress callback using pymongo (not Motor)"""
                progress_percent = (completed / total * 100) if total > 0 else 0
                eta = (elapsed / completed * (total - completed)) if completed > 0 else 0

                sync_db.data_sync_jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "progress_percent": progress_percent,
                        "total_files": total,
                        "completed_files": completed,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                        "updated_at": datetime.now(timezone.utc)
                    }}
                )

            # Fetch data with progress callback
            # Note: This is synchronous, we need to run in executor
            from concurrent.futures import ThreadPoolExecutor
            from datetime import datetime as dt

            # Check if this is a delta job to fetch only missing data
            is_delta = job_doc.get('is_delta_job', False)
            delta_ranges = job_doc.get('delta_ranges', [])

            if is_delta and delta_ranges:
                # Fetch only the delta ranges (missing data)
                print(f"🎯 Delta job: fetching only missing data for {job.symbol}")
                import pandas as pd
                delta_dataframes = []

                # Aggregate progress across all delta ranges
                cumulative_files_completed = 0
                cumulative_files_total = 0
                current_range_files_total = 0
                start_time = time.time()

                # Create aggregated progress callback
                def aggregated_progress_callback(completed: int, total: int, elapsed: float, skipped: int = 0):
                    """Aggregate progress across all delta ranges"""
                    nonlocal cumulative_files_completed, cumulative_files_total, current_range_files_total

                    # Track the total for current range (gets updated as S3 lists files)
                    current_range_files_total = total

                    # Calculate overall progress
                    overall_completed = cumulative_files_completed + completed
                    overall_total = cumulative_files_total + current_range_files_total
                    overall_elapsed = time.time() - start_time
                    progress_percent = (overall_completed / overall_total * 100) if overall_total > 0 else 0
                    eta = (overall_elapsed / overall_completed * (overall_total - overall_completed)) if overall_completed > 0 else 0

                    sync_db.data_sync_jobs.update_one(
                        {"job_id": job_id},
                        {"$set": {
                            "progress_percent": progress_percent,
                            "total_files": overall_total,
                            "completed_files": overall_completed,
                            "elapsed_seconds": overall_elapsed,
                            "eta_seconds": eta,
                            "updated_at": datetime.now(timezone.utc)
                        }}
                    )

                with ThreadPoolExecutor(max_workers=1) as executor:
                    for idx, delta_range in enumerate(delta_ranges):
                        range_type = delta_range['type']
                        start_date = dt.fromisoformat(delta_range['start'])
                        end_date = dt.fromisoformat(delta_range['end'])

                        print(f"  📥 Fetching {range_type} range {idx+1}/{len(delta_ranges)}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

                        # Reset current range tracking
                        current_range_files_total = 0

                        delta_data = await loop.run_in_executor(
                            executor,
                            lambda s=start_date, e=end_date: fetch_hybrid_data_range(
                                job.symbol, s, e, job.interval, aggregated_progress_callback
                            )
                        )

                        if delta_data is not None and not delta_data.empty:
                            delta_dataframes.append(delta_data)

                        # After range completes, add its total to cumulative
                        cumulative_files_completed += current_range_files_total
                        cumulative_files_total += current_range_files_total

                # Combine all delta dataframes
                if delta_dataframes:
                    data = pd.concat(delta_dataframes).sort_index()
                    print(f"✅ Fetched {len(data)} delta data points from {cumulative_files_total} files (saved ~{job_doc.get('old_period', 'unknown')} of redundant downloads)")
                else:
                    data = None
            else:
                # Normal job: fetch entire period using hybrid mode (S3 + REST API)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    data = await loop.run_in_executor(
                        executor,
                        lambda: fetch_hybrid_data(job.symbol, job.period, job.interval, sync_progress_callback)
                    )

            # Save to cache (merge if delta job, replace if normal job)
            if data is not None and not data.empty:
                if is_delta:
                    print(f"🔀 Merging delta data with existing cache for {job.symbol}")
                    self.cache.merge_and_set(data, job.symbol, job.period, job.interval)
                    # Reload the full merged dataset from cache for inventory update
                    data = self.cache.get(job.symbol, job.period, job.interval)
                    print(f"📊 Reloaded merged dataset: {len(data) if data is not None else 0} rows")
                else:
                    self.cache.set(data, job.symbol, job.period, job.interval)

            # Update job as completed (using sync client)
            sync_db.data_sync_jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": SyncJobStatus.COMPLETED.value,
                    "progress_percent": 100.0,
                    "completed_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }}
            )

            # Update inventory (using sync client)
            # For delta jobs, also pass the old period to clean up duplicate entries
            old_period = job_doc.get('old_period')
            self._update_inventory_sync(sync_db, job.symbol, job.period, job.interval, data, old_period)

            # Clear progress marker
            self.cache.clear_progress(job.symbol, job.period, job.interval)

            # Trigger strategy analysis if requested
            if job_doc.get('trigger_analysis_on_complete', False):
                print(f"🎯 Triggering consensus analysis for {job.symbol} after successful sync")
                await self._trigger_consensus_analysis(job.symbol, loop)

                # Update last_auto_analysis_at timestamp
                sync_db.data_inventory.update_one(
                    {"symbol": job.symbol, "interval": job.interval},
                    {"$set": {
                        "last_auto_analysis_at": datetime.now(timezone.utc)
                    }}
                )

        except Exception as e:
            # Update job as failed (using sync client)
            sync_db.data_sync_jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": SyncJobStatus.FAILED.value,
                    "error_message": str(e),
                    "completed_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }}
            )

        finally:
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            # Note: We don't start the next pending job here because we're in a background thread
            # The main API will handle starting pending jobs when new jobs are created/queried

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

    def _update_inventory_sync(self, sync_db, symbol: str, period: str, interval: str, data, old_period: str = None):
        """Update data inventory after successful download (synchronous version for background threads)"""
        # If no new data but this is a delta job, try to use existing cache data
        if (data is None or data.empty) and old_period:
            print(f"ℹ️  No new data found for delta sync, checking existing cache...")
            # Try to load existing cache data using the OLD period
            existing_data = self.cache.get(symbol, old_period, interval)
            if existing_data is not None and not existing_data.empty:
                print(f"✓ Found existing cache with {len(existing_data)} rows, updating inventory with new period")
                # Update the cache with the new period
                self.cache.set(existing_data, symbol, period, interval)
                data = existing_data
            else:
                print(f"⚠️  No existing cache data found, skipping inventory update")
                return
        elif data is None or data.empty:
            return

        inventory = DataInventory(
            symbol=symbol,
            period=period,
            interval=interval,
            total_days=len(data),
            date_range_start=data.index.min().to_pydatetime() if len(data) > 0 else None,
            date_range_end=data.index.max().to_pydatetime() if len(data) > 0 else None,
            last_updated_at=datetime.now(timezone.utc),
            is_complete=True,  # Assume complete for now
            missing_dates=[]
        )

        # Get file size from cache
        cache_key = self.cache._get_cache_key(symbol, period, interval)
        cache_path = self.cache._get_cache_path(cache_key)
        if cache_path.exists():
            inventory.file_size_bytes = cache_path.stat().st_size

        # For delta jobs, delete the old inventory entry with the old period
        if old_period and old_period != period:
            print(f"🗑️  Deleting old inventory entry for {symbol} ({old_period})")
            sync_db.data_inventory.delete_one({
                "symbol": symbol,
                "period": old_period,
                "interval": interval
            })

        # Upsert inventory (using sync client)
        sync_db.data_inventory.update_one(
            {"symbol": symbol, "period": period, "interval": interval},
            {"$set": inventory.dict()},
            upsert=True
        )

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
            last_updated_at=datetime.now(timezone.utc),
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
                "updated_at": datetime.now(timezone.utc)
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
                "updated_at": datetime.now(timezone.utc)
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
                "completed_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
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

    # ==================== Strategy Analysis Triggering ====================

    async def _trigger_consensus_analysis(self, symbol: str, loop):
        """
        Trigger consensus analysis for a symbol after data sync completes.
        Uses the analysis queue to prevent concurrent GPU usage.
        This is called from background threads, so we need to handle it carefully.

        Args:
            symbol: Trading symbol
            loop: Event loop for async execution
        """
        import requests
        import os

        # Get the backend URL from environment or use default
        backend_url = os.getenv("BACKEND_URL", "http://backend:8000")

        # Enqueue the analysis instead of running it directly
        # Make HTTP request to enqueue consensus analysis
        # We use requests (sync) because we're in a background thread
        try:
            response = requests.post(
                f"{backend_url}/api/v1/analyze/consensus/enqueue",
                json={"symbol": symbol},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Consensus analysis enqueued for {symbol}: job_id={data.get('job_id')}, queue_position={data.get('queue_position')}")
            else:
                print(f"⚠️  Failed to enqueue consensus analysis for {symbol}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error enqueuing consensus analysis for {symbol}: {e}")

    # ==================== Auto-Scheduling Management ====================

    async def enable_auto_schedule(self, symbol: str, interval: str, frequency: str) -> Optional[datetime]:
        """
        Enable automatic delta sync + analysis scheduling for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval
            frequency: Schedule frequency ('daily', '12h', '6h')

        Returns:
            Next scheduled run time
        """
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        # Import scheduler
        from backend.scheduler import get_scheduler
        from backend.database import Database

        # Get database instance - we need it for the scheduler
        temp_db = Database()
        await temp_db.connect()
        scheduler = get_scheduler(temp_db)

        # Determine trigger based on frequency
        trigger = None
        if frequency == "daily":
            # Run at 9 AM UTC every day
            trigger = CronTrigger(hour=9, minute=0)
        elif frequency == "12h":
            # Run every 12 hours
            trigger = IntervalTrigger(hours=12)
        elif frequency == "6h":
            # Run every 6 hours
            trigger = IntervalTrigger(hours=6)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Create unique job ID for this symbol+interval combination
        job_id = f"auto_sync_{symbol}_{interval}"

        # Schedule the job
        job = scheduler.scheduler.add_job(
            self._run_scheduled_delta_sync,
            trigger=trigger,
            args=[symbol, interval],
            id=job_id,
            replace_existing=True,
            name=f"Auto Delta Sync: {symbol} ({interval})"
        )

        next_run = job.next_run_time

        # Update inventory with schedule settings
        await self.db.data_inventory.update_one(
            {"symbol": symbol, "interval": interval},
            {"$set": {
                "auto_schedule_enabled": True,
                "schedule_frequency": frequency,
                "next_scheduled_sync": next_run,
                "scheduler_job_id": job_id
            }}
        )

        print(f"✅ Auto-schedule enabled for {symbol} ({interval}): {frequency}, next run: {next_run}")

        return next_run

    async def disable_auto_schedule(self, symbol: str, interval: str):
        """
        Disable automatic delta sync + analysis scheduling for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval
        """
        from backend.scheduler import get_scheduler
        from backend.database import Database

        # Get database instance
        temp_db = Database()
        await temp_db.connect()
        scheduler = get_scheduler(temp_db)

        # Get inventory to find job ID
        inventory_doc = await self.db.data_inventory.find_one({
            "symbol": symbol,
            "interval": interval
        })

        if inventory_doc and inventory_doc.get("scheduler_job_id"):
            job_id = inventory_doc["scheduler_job_id"]
            try:
                scheduler.scheduler.remove_job(job_id)
                print(f"✅ Removed scheduled job {job_id}")
            except Exception as e:
                print(f"⚠️  Failed to remove job {job_id}: {e}")

        # Update inventory
        await self.db.data_inventory.update_one(
            {"symbol": symbol, "interval": interval},
            {"$set": {
                "auto_schedule_enabled": False,
                "next_scheduled_sync": None,
                "scheduler_job_id": None
            }}
        )

        print(f"✅ Auto-schedule disabled for {symbol} ({interval})")

    async def _run_scheduled_delta_sync(self, symbol: str, interval: str):
        """
        Run scheduled delta sync + analysis.
        This is called by APScheduler.

        Args:
            symbol: Trading symbol
            interval: Data interval
        """
        import asyncio

        print(f"🔄 Running scheduled delta sync for {symbol} ({interval})")

        # Get current inventory
        inventory_doc = await self.db.data_inventory.find_one({
            "symbol": symbol,
            "interval": interval
        })

        if not inventory_doc:
            print(f"⚠️  No inventory found for {symbol} ({interval}), skipping")
            return

        inventory = DataInventory(**inventory_doc)

        # Determine new period (extend by the schedule frequency)
        # For simplicity, we'll always try to get "today" as the end date
        from strategies.massive_s3_data_source import get_massive_s3_source
        from datetime import datetime as dt

        s3_source = get_massive_s3_source()

        # Get current period and extend it to today
        current_period = inventory.period

        # Use the same period but it will fetch delta to today
        # The extend endpoint will calculate what's missing
        new_period = current_period  # Keep same period, just extend to current date

        # Make HTTP request to schedule delta sync endpoint
        import requests
        import os

        backend_url = os.getenv("BACKEND_URL", "http://backend:8000")

        try:
            response = requests.post(
                f"{backend_url}/api/v1/inventory/{symbol}/schedule-delta-sync",
                params={
                    "new_period": new_period,
                    "interval": interval,
                    "trigger_analysis": True
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Scheduled delta sync started for {symbol}: {data.get('job_id')}")

                # Update last run times
                await self.db.data_inventory.update_one(
                    {"symbol": symbol, "interval": interval},
                    {"$set": {
                        "last_auto_sync_at": dt.now(timezone.utc)
                    }}
                )
            else:
                print(f"⚠️  Failed to start delta sync for {symbol}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error running scheduled delta sync for {symbol}: {e}")

    # ==================== Inventory Management ====================

    async def get_inventory(self, symbol: Optional[str] = None) -> list:
        """Get data inventory"""
        query = {"symbol": symbol} if symbol else {}
        cursor = self.db.data_inventory.find(query)
        return [DataInventory(**doc) async for doc in cursor]

    async def restore_auto_schedules(self):
        """
        Restore auto-schedule jobs from database on startup.
        This re-registers all enabled auto-schedules with APScheduler.
        """
        from backend.scheduler import get_scheduler
        from backend.database import Database

        print("🔄 AUTO-SCHEDULE: Restoring auto-schedules from database...")

        # Get database instance
        temp_db = Database()
        await temp_db.connect()
        scheduler = get_scheduler(temp_db)

        # Find all inventory items with auto-schedule enabled
        enabled_schedules = await self.db.data_inventory.find({
            "auto_schedule_enabled": True
        }).to_list(length=None)

        restored_count = 0
        for item_doc in enabled_schedules:
            try:
                item = DataInventory(**item_doc)

                # Re-enable the auto-schedule (this will re-register with APScheduler)
                next_run = await self.enable_auto_schedule(
                    item.symbol,
                    item.interval,
                    item.schedule_frequency
                )

                restored_count += 1
                print(f"  ✅ Restored auto-schedule: {item.symbol} ({item.interval}) - {item.schedule_frequency}")

            except Exception as e:
                print(f"  ❌ Failed to restore auto-schedule for {item_doc.get('symbol')}: {e}")

        print(f"🔄 AUTO-SCHEDULE: Restored {restored_count} auto-schedule(s)")
        return restored_count


# Global instance
_sync_manager: Optional[DataSyncManager] = None


async def get_sync_manager(db: AsyncIOMotorDatabase) -> DataSyncManager:
    """Get or create the global sync manager instance"""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = DataSyncManager(db)
        await _sync_manager.initialize()
    return _sync_manager
