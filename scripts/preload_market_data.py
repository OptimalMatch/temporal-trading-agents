#!/usr/bin/env python3
"""
Pre-download and cache market data from Massive.com S3 flat files.
This avoids slow downloads during analysis by preparing the cache ahead of time.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategies.cached_data_fetch import fetch_crypto_data_cached
import argparse


def preload_symbol(symbol: str, period: str = 'auto', interval: str = '1d'):
    """
    Pre-download and cache data for a specific symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD', 'AAPL')
        period: Time period ('2y', '5y', or 'auto' to detect based on asset type)
        interval: Data interval (default '1d')
    """
    # Auto-detect period based on asset type
    if period == 'auto':
        is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
        period = '2y' if is_crypto else '5y'
        print(f"🔍 Auto-detected: {symbol} is a {'crypto' if is_crypto else 'stock'}, using {period} period")

    print(f"\n{'='*70}")
    print(f"PRE-LOADING DATA: {symbol}")
    print(f"  Period: {period} | Interval: {interval}")
    print(f"{'='*70}\n")

    try:
        df = fetch_crypto_data_cached(symbol, period=period, interval=interval)

        if df is not None and not df.empty:
            print(f"\n✅ SUCCESS: Cached {len(df)} days of {symbol} data")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            return True
        else:
            print(f"\n❌ FAILED: No data retrieved for {symbol}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def preload_batch(symbols: list, period: str = 'auto', interval: str = '1d'):
    """
    Pre-download and cache data for multiple symbols.

    Args:
        symbols: List of trading symbols
        period: Time period for all symbols
        interval: Data interval
    """
    print(f"\n{'='*70}")
    print(f"BATCH PRE-LOAD: {len(symbols)} symbols")
    print(f"{'='*70}\n")

    results = []
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        success = preload_symbol(symbol, period=period, interval=interval)
        results.append((symbol, success))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    successful = [s for s, success in results if success]
    failed = [s for s, success in results if not success]

    print(f"\n✅ Successful: {len(successful)}/{len(symbols)}")
    if successful:
        for s in successful:
            print(f"   - {s}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(symbols)}")
        for s in failed:
            print(f"   - {s}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pre-download and cache market data from Massive.com S3 flat files'
    )
    parser.add_argument(
        'symbols',
        nargs='+',
        help='Trading symbols to download (e.g., BTC-USD AAPL TSLA)'
    )
    parser.add_argument(
        '--period',
        default='auto',
        help='Time period: 2y, 5y, or auto (default: auto - detects based on asset type)'
    )
    parser.add_argument(
        '--interval',
        default='1d',
        help='Data interval (default: 1d)'
    )

    args = parser.parse_args()

    if len(args.symbols) == 1:
        preload_symbol(args.symbols[0], period=args.period, interval=args.interval)
    else:
        preload_batch(args.symbols, period=args.period, interval=args.interval)
