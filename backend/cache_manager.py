"""
Cache manager for market data preloading with progress tracking.
Manages background download jobs and tracks their progress.
"""
import sys
from pathlib import Path
import asyncio
from typing import Dict, Optional
from datetime import datetime
import threading

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from strategies.data_cache import get_cache
from strategies.cached_data_fetch import fetch_crypto_data_cached


class PreloadJob:
    """Represents a single preload job with progress tracking"""

    def __init__(self, symbol: str, period: str, interval: str = '1d'):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.status = 'pending'  # pending, running, completed, failed, cancelled
        self.progress = 0  # 0-100
        self.total_files = 0
        self.completed_files = 0
        self.elapsed_seconds = 0
        self.eta_seconds = 0
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.cancelled = False

    def to_dict(self):
        """Convert job to dictionary"""
        return {
            'symbol': self.symbol,
            'period': self.period,
            'interval': self.interval,
            'status': self.status,
            'progress': self.progress,
            'total_files': self.total_files,
            'completed_files': self.completed_files,
            'elapsed_seconds': self.elapsed_seconds,
            'eta_seconds': self.eta_seconds,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class CacheManager:
    """Manages market data cache and preload jobs"""

    def __init__(self):
        self.jobs: Dict[str, PreloadJob] = {}  # key: symbol_period_interval
        self.cache = get_cache()

    def _get_job_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate unique job key"""
        return f"{symbol}_{period}_{interval}"

    def list_cached_data(self):
        """List all cached data files"""
        import os
        from pathlib import Path

        cache_dir = Path("/tmp/crypto_data_cache")
        cached_files = []

        if not cache_dir.exists():
            return cached_files

        for file_path in cache_dir.glob("*.pkl"):
            # Parse filename: SYMBOL_PERIOD_INTERVAL.pkl
            filename = file_path.stem
            parts = filename.split('_')

            if len(parts) >= 3:
                symbol = parts[0]
                period = parts[1]
                interval = parts[2]

                stat = file_path.stat()
                cached_files.append({
                    'symbol': symbol,
                    'period': period,
                    'interval': interval,
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        # Also check for progress markers
        for job_key, job in self.jobs.items():
            if job.status == 'running':
                # Check if there's a progress marker
                progress = self.cache.get_progress(job.symbol, job.period, job.interval)
                if progress:
                    # Find matching file entry
                    for cached_file in cached_files:
                        if (cached_file['symbol'] == job.symbol and
                            cached_file['period'] == job.period and
                            cached_file['interval'] == job.interval):
                            cached_file['downloading'] = True
                            cached_file['download_progress'] = job.progress
                            break

        return cached_files

    def get_job_status(self, symbol: str, period: str = None, interval: str = '1d'):
        """Get status of a preload job"""
        # Auto-detect period if not provided
        if period is None:
            is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
            period = '2y' if is_crypto else '5y'

        job_key = self._get_job_key(symbol, period, interval)
        job = self.jobs.get(job_key)

        if job:
            return job.to_dict()

        # Check if data is already cached
        cached_data = self.cache.get(symbol, period, interval)
        if cached_data is not None:
            return {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'status': 'completed',
                'progress': 100,
                'error': None,
            }

        return None

    def start_preload(self, symbol: str, period: str = None, interval: str = '1d'):
        """Start a preload job in background"""
        # Auto-detect period if not provided
        if period is None:
            is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
            period = '2y' if is_crypto else '5y'

        job_key = self._get_job_key(symbol, period, interval)

        # Check if job already exists and is running
        if job_key in self.jobs:
            job = self.jobs[job_key]
            if job.status == 'running':
                return job.to_dict()

        # Create new job
        job = PreloadJob(symbol, period, interval)
        self.jobs[job_key] = job

        # Start download in background thread
        thread = threading.Thread(target=self._run_preload, args=(job,))
        thread.daemon = True
        thread.start()

        return job.to_dict()

    def _run_preload(self, job: PreloadJob):
        """Run the preload job (runs in background thread)"""
        try:
            job.status = 'running'
            job.started_at = datetime.now()

            # Note: fetch_crypto_data_cached will handle the actual download
            # We need to hook into the progress callback somehow
            # For now, we'll just run it and mark as complete

            data = fetch_crypto_data_cached(job.symbol, job.period, job.interval)

            if data is not None and not data.empty:
                job.status = 'completed'
                job.progress = 100
                job.completed_at = datetime.now()
            else:
                job.status = 'failed'
                job.error = 'No data returned'

        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.completed_at = datetime.now()

    def cancel_preload(self, symbol: str, period: str = None, interval: str = '1d'):
        """Cancel a running preload job"""
        # Auto-detect period if not provided
        if period is None:
            is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
            period = '2y' if is_crypto else '5y'

        job_key = self._get_job_key(symbol, period, interval)
        job = self.jobs.get(job_key)

        if job and job.status == 'running':
            job.cancelled = True
            job.status = 'cancelled'
            job.completed_at = datetime.now()

            # Clear progress marker
            self.cache.clear_progress(symbol, period, interval)

            return True

        return False

    def delete_cached_data(self, symbol: str, period: str = None, interval: str = '1d'):
        """Delete cached data for a symbol"""
        # Auto-detect period if not provided
        if period is None:
            is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
            period = '2y' if is_crypto else '5y'

        # Cancel any running job first
        self.cancel_preload(symbol, period, interval)

        # Delete the cache file
        cache_key = self.cache._get_cache_key(symbol, period, interval)
        cache_path = self.cache._get_cache_path(cache_key)

        if cache_path.exists():
            cache_path.unlink()
            return True

        return False


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
