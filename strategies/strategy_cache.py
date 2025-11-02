"""
Strategy Result Caching System

Provides in-memory LRU caching for strategy analysis results to avoid recomputation.
Particularly valuable during:
- Backtesting (same stats used across multiple bars)
- Parameter optimization (running multiple backtests with same model)
- Real-time analysis (avoiding redundant calculations)

Uses thread-safe caching with automatic eviction based on LRU policy.
"""
import hashlib
import json
import functools
import threading
from typing import Dict, Any, Callable
from collections import OrderedDict


class StrategyCache:
    """
    Thread-safe LRU cache for strategy analysis results.

    Caches strategy function outputs based on hashed inputs to avoid
    expensive recomputation when the same inputs are encountered.
    """

    def __init__(self, maxsize: int = 1000):
        """
        Initialize strategy cache.

        Args:
            maxsize: Maximum number of cached results (default 1000)
        """
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _hash_dict(self, d: Dict) -> str:
        """
        Create a stable hash of a dictionary.

        Args:
            d: Dictionary to hash

        Returns:
            Hex digest string
        """
        # Convert dict to sorted JSON string for stable hashing
        json_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _make_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_parts = []

        # Hash each argument
        for arg in args:
            if isinstance(arg, dict):
                key_parts.append(self._hash_dict(arg))
            elif isinstance(arg, (int, float, str, bool)):
                key_parts.append(str(arg))
            elif arg is None:
                key_parts.append("None")
            else:
                # For complex objects, try to convert to dict or use repr
                try:
                    if hasattr(arg, '__dict__'):
                        key_parts.append(self._hash_dict(vars(arg)))
                    else:
                        key_parts.append(repr(arg))
                except:
                    # If all else fails, use id (not ideal but safe)
                    key_parts.append(f"obj_{id(arg)}")

        # Add kwargs
        if kwargs:
            key_parts.append(self._hash_dict(kwargs))

        return ":".join(key_parts)

    def get(self, key: str) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return None

    def set(self, key: str, value: Any):
        """
        Store value in cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = value

                # Evict oldest if at capacity
                if len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)  # Remove oldest (FIFO)

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
            }


# Global cache instance
_strategy_cache = StrategyCache(maxsize=1000)


def get_strategy_cache() -> StrategyCache:
    """Get the global strategy cache instance."""
    return _strategy_cache


def cached_strategy(func: Callable) -> Callable:
    """
    Decorator to cache strategy function results.

    Automatically caches the return value based on function arguments.
    Thread-safe and uses LRU eviction policy.

    Usage:
        @cached_strategy
        def analyze_my_strategy(stats: Dict, current_price: float) -> Dict:
            # expensive computation
            return result

    Args:
        func: Strategy function to cache

    Returns:
        Wrapped function with caching
    """
    cache = get_strategy_cache()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key from arguments
        cache_key = f"{func.__name__}:{cache._make_cache_key(*args, **kwargs)}"

        # Try to get from cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Cache miss - compute result
        result = func(*args, **kwargs)

        # Store in cache
        cache.set(cache_key, result)

        return result

    # Attach cache stats method to wrapper
    wrapper.cache_stats = lambda: cache.stats()
    wrapper.cache_clear = lambda: cache.clear()

    return wrapper


def print_cache_stats():
    """Print current cache statistics."""
    stats = get_strategy_cache().stats()
    print(f"\n📊 Strategy Cache Statistics:")
    print(f"   Size: {stats['size']}/{stats['maxsize']}")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate_pct']}%")
