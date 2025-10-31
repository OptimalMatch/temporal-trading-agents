"""
Polygon.io / Massive.com data source for crypto and stock data.
Uses the REST API to fetch OHLCV data.
"""
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional


def fetch_crypto_data_polygon(symbol: str, period: str = '2y', interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Fetch cryptocurrency data from Polygon.io/Massive.com REST API.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD', 'ETH-USD')
        period: Data period (e.g., '2y', '1y', '6mo', '3mo', '1mo')
        interval: Data interval (e.g., '1d', '1h', '5m')

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    # Convert symbol format (BTC-USD -> X:BTCUSD)
    if '-' in symbol:
        base, quote = symbol.split('-')
        polygon_symbol = f"X:{base}{quote}"
    else:
        polygon_symbol = symbol

    # Calculate date range
    end_date = datetime.now()
    period_map = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825,
        'max': 3650
    }
    days = period_map.get(period, 730)
    start_date = end_date - timedelta(days=days)

    # Convert interval to Polygon format
    interval_map = {
        '1m': ('minute', 1),
        '5m': ('minute', 5),
        '15m': ('minute', 15),
        '30m': ('minute', 30),
        '1h': ('hour', 1),
        '1d': ('day', 1),
        '1wk': ('week', 1),
        '1mo': ('month', 1)
    }
    timespan, multiplier = interval_map.get(interval, ('day', 1))

    # Build API URL
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }

    try:
        print(f"📡 Fetching {symbol} from Polygon.io ({from_date} to {to_date})")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Accept both OK and DELAYED status (DELAYED means delayed data which is fine for historical analysis)
        if data.get('status') not in ['OK', 'DELAYED']:
            print(f"⚠️  Polygon API returned status: {data.get('status')}, {data.get('message', '')}")
            return None

        results = data.get('results', [])
        if not results:
            print(f"⚠️  No data returned for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Rename columns to match yfinance format
        df.rename(columns={
            't': 'Date',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }, inplace=True)

        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)

        # Select and order columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"✓ Fetched {len(df)} rows for {symbol}")

        return df

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"⚠️  Rate limit exceeded for Polygon API")
        else:
            print(f"❌ HTTP error fetching {symbol}: {e}")
        raise

    except Exception as e:
        print(f"❌ Error fetching {symbol} from Polygon: {e}")
        raise
