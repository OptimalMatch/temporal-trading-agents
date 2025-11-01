"""
Simple file-based cache for market data to avoid excessive API calls.
"""
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class DataCache:
    """File-based cache for market data."""

    def __init__(self, cache_dir: str = "/tmp/crypto_data_cache", ttl_hours: int = 24):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cached data in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate cache key for given parameters."""
        return f"{symbol}_{period}_{interval}.pkl"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path for cache file."""
        return self.cache_dir / cache_key

    def get(self, symbol: str, period: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not expired.

        Args:
            symbol: Trading symbol
            period: Data period
            interval: Data interval

        Returns:
            DataFrame if cache hit and not expired, None otherwise
        """
        cache_key = self._get_cache_key(symbol, period, interval)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            # Check if cache is expired
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age > self.ttl:
                print(f"💾 Cache expired for {symbol} (age: {cache_age})")
                return None

            # Load cached data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            print(f"💾 Cache hit for {symbol} (age: {cache_age})")
            return data

        except Exception as e:
            print(f"⚠️  Error loading cache for {symbol}: {e}")
            return None

    def set(self, data: pd.DataFrame, symbol: str, period: str, interval: str = '1d'):
        """
        Store data in cache.

        Args:
            data: DataFrame to cache
            symbol: Trading symbol
            period: Data period
            interval: Data interval
        """
        if data is None or data.empty:
            return

        cache_key = self._get_cache_key(symbol, period, interval)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Cached {len(data)} rows for {symbol}")
        except Exception as e:
            print(f"⚠️  Error caching data for {symbol}: {e}")

    def clear(self, symbol: Optional[str] = None):
        """
        Clear cache for specific symbol or all symbols.

        Args:
            symbol: If provided, only clear cache for this symbol
        """
        try:
            if symbol:
                # Clear specific symbol
                for cache_file in self.cache_dir.glob(f"{symbol}_*.pkl"):
                    cache_file.unlink()
                # Also clear progress marker
                progress_file = self.cache_dir / f".progress_{symbol}"
                if progress_file.exists():
                    progress_file.unlink()
                print(f"💾 Cleared cache for {symbol}")
            else:
                # Clear all
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                for progress_file in self.cache_dir.glob(".progress_*"):
                    progress_file.unlink()
                print("💾 Cleared all cache")
        except Exception as e:
            print(f"⚠️  Error clearing cache: {e}")

    def get_progress(self, symbol: str, period: str, interval: str = '1d') -> Optional[set]:
        """
        Get progress marker (set of processed file keys) for resuming downloads.

        Args:
            symbol: Trading symbol
            period: Data period
            interval: Data interval

        Returns:
            Set of processed file keys, or None if no progress marker exists
        """
        progress_key = f".progress_{symbol}_{period}_{interval}"
        progress_path = self.cache_dir / progress_key

        if not progress_path.exists():
            return None

        try:
            with open(progress_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Error loading progress marker: {e}")
            return None

    def save_progress(self, processed_keys: set, symbol: str, period: str, interval: str = '1d'):
        """
        Save progress marker for resuming downloads.

        Args:
            processed_keys: Set of successfully processed file keys
            symbol: Trading symbol
            period: Data period
            interval: Data interval
        """
        progress_key = f".progress_{symbol}_{period}_{interval}"
        progress_path = self.cache_dir / progress_key

        try:
            with open(progress_path, 'wb') as f:
                pickle.dump(processed_keys, f)
        except Exception as e:
            print(f"⚠️  Error saving progress marker: {e}")

    def clear_progress(self, symbol: str, period: str, interval: str = '1d'):
        """
        Clear progress marker after successful completion.

        Args:
            symbol: Trading symbol
            period: Data period
            interval: Data interval
        """
        progress_key = f".progress_{symbol}_{period}_{interval}"
        progress_path = self.cache_dir / progress_key

        if progress_path.exists():
            try:
                progress_path.unlink()
            except Exception as e:
                print(f"⚠️  Error clearing progress marker: {e}")


# Global cache instance
_cache = None


def get_cache() -> DataCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache
