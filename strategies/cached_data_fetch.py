"""
Cached wrapper for temporal-forecasting data fetching.
Patches the library's fetch_crypto_data function to use caching.
"""
import time
from typing import Optional
from strategies.data_cache import get_cache


# Store original function
_original_fetch_crypto_data = None
_yfinance_session_configured = False


def fetch_crypto_data_cached(symbol: str, period: str = '2y', interval: str = '1d'):
    """
    Cached wrapper for fetch_crypto_data that uses Polygon.io and caching.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD')
        period: Data period (e.g., '2y', '1y', '6mo')
        interval: Data interval (e.g., '1d', '1h')

    Returns:
        DataFrame with OHLCV data
    """
    cache = get_cache()

    # Try to get from cache first
    cached_data = cache.get(symbol, period, interval=interval)
    if cached_data is not None:
        return cached_data

    # Not in cache or expired, fetch fresh data from Massive.com REST API
    print(f"💹 Cache miss - fetching fresh data for {symbol} via Massive.com REST API")

    # Add delay before fetching to avoid rate limiting
    time.sleep(0.5)  # 0.5 second delay before each fetch

    try:
        # Try S3 flat files first for more historical data
        s3_data = None
        try:
            from strategies.massive_s3_data_source import fetch_crypto_data_massive_s3
            s3_data = fetch_crypto_data_massive_s3(symbol, period=period, interval=interval)
            if s3_data is not None and not s3_data.empty:
                print(f"✓ Got {len(s3_data)} rows from S3 flat files for {symbol}")
        except Exception as s3_error:
            print(f"⚠️  S3 fetch failed ({s3_error}), will use REST API only")

        # Check if S3 data is stale and backfill with REST API
        from datetime import datetime, timedelta
        import pandas as pd

        rest_data = None
        if s3_data is not None and not s3_data.empty:
            # Check how old the last data point is
            last_s3_time = s3_data.index[-1]
            if not hasattr(last_s3_time, 'to_pydatetime'):
                last_s3_time = pd.Timestamp(last_s3_time)

            now = datetime.now()
            age_hours = (now - last_s3_time.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600

            # If data is more than 2 hours old, backfill with REST API
            if age_hours > 2:
                print(f"⏰ S3 data is {age_hours:.1f} hours old, backfilling recent data from REST API")
                try:
                    from strategies.polygon_data_source import fetch_crypto_data_polygon
                    # Fetch from S3 end date to now
                    rest_start = last_s3_time.to_pydatetime()
                    rest_end = now
                    rest_data = fetch_crypto_data_polygon(symbol, interval=interval,
                                                          start_date=rest_start, end_date=rest_end)
                    if rest_data is not None and not rest_data.empty:
                        print(f"✓ Got {len(rest_data)} recent rows from REST API")
                except Exception as rest_error:
                    print(f"⚠️  REST API backfill failed: {rest_error}")
        else:
            # No S3 data, use REST API only
            print(f"📡 Fetching all data from REST API")
            from strategies.polygon_data_source import fetch_crypto_data_polygon
            rest_data = fetch_crypto_data_polygon(symbol, period=period, interval=interval)

        # Merge S3 and REST data
        if s3_data is not None and not s3_data.empty and rest_data is not None and not rest_data.empty:
            # Combine and deduplicate
            data = pd.concat([s3_data, rest_data])
            data = data[~data.index.duplicated(keep='last')]  # Keep newer data on overlap
            data = data.sort_index()
            print(f"✓ Merged S3 and REST data: {len(data)} total rows")
        elif s3_data is not None and not s3_data.empty:
            data = s3_data
        elif rest_data is not None and not rest_data.empty:
            data = rest_data
        else:
            print(f"❌ No data available from either S3 or REST API")
            data = None

        # Cache the result
        if data is not None and not data.empty:
            cache.set(data, symbol, period, interval=interval)
            # Clear progress marker since download is complete
            cache.clear_progress(symbol, period, interval)
            print(f"💾 Cached {len(data)} rows for {symbol}")
        else:
            print(f"⚠️  Received empty data for {symbol}, not caching")

        return data

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        raise


def configure_yfinance_session():
    """
    Configure yfinance with proper User-Agent headers and session settings
    to avoid rate limiting and blocking.
    """
    global _yfinance_session_configured

    if _yfinance_session_configured:
        return

    try:
        import yfinance as yf
        import requests

        # Configure requests session with realistic browser headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

        # Patch yfinance at multiple levels to ensure headers are used
        try:
            # Method 1: Patch the base headers
            import yfinance.scrapers.quote
            yfinance.scrapers.quote._BASE_HEADERS_ = headers
        except:
            pass

        try:
            # Method 2: Patch the utils module
            import yfinance.utils
            yfinance.utils.user_agent_headers = headers
        except:
            pass

        try:
            # Method 3: Create and set a default session
            session = requests.Session()
            session.headers.update(headers)

            # Store it in yfinance's cache module if available
            import yfinance.cache
            yfinance.cache._CACHE_SESSION = session
        except:
            pass

        _yfinance_session_configured = True
        print("✓ Configured yfinance with User-Agent headers (multiple patches applied)")

    except Exception as e:
        print(f"⚠️  Error configuring yfinance session: {e}")


def patch_temporal_data_fetch():
    """
    Monkey-patch the temporal library's fetch_crypto_data function
    to use Massive.com S3 flat files with caching instead of yfinance.
    """
    global _original_fetch_crypto_data

    try:
        from temporal import data_sources

        # Store original function
        if hasattr(data_sources, 'fetch_crypto_data'):
            _original_fetch_crypto_data = data_sources.fetch_crypto_data

            # Replace with Massive.com S3 flat files cached version
            data_sources.fetch_crypto_data = fetch_crypto_data_cached

            print("✓ Patched temporal.data_sources.fetch_crypto_data to use Massive.com REST API with caching")
        else:
            print("⚠️  temporal.data_sources.fetch_crypto_data not found, skipping patch")

    except ImportError as e:
        print(f"⚠️  Could not import temporal.data_sources: {e}")
    except Exception as e:
        print(f"⚠️  Error patching temporal data fetch: {e}")


def unpatch_temporal_data_fetch():
    """Restore the original fetch_crypto_data function."""
    global _original_fetch_crypto_data

    try:
        from temporal import data_sources

        if _original_fetch_crypto_data is not None:
            data_sources.fetch_crypto_data = _original_fetch_crypto_data
            print("✓ Restored original temporal.data_sources.fetch_crypto_data")

    except Exception as e:
        print(f"⚠️  Error unpatching temporal data fetch: {e}")
