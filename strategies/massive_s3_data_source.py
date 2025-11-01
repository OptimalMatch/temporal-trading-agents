"""
Massive.com (formerly Polygon.io) S3 flat files data source.
Fetches market data from S3-hosted compressed CSV files.
"""
import os
import gzip
import io
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class MassiveS3DataSource:
    """Fetches market data from Massive.com S3 flat files."""

    def __init__(self,
                 access_key_id: Optional[str] = None,
                 secret_access_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None,
                 bucket_name: Optional[str] = None):
        """
        Initialize Massive.com S3 data source.

        Args:
            access_key_id: AWS access key ID. If not provided, reads from MASSIVE_ACCESS_KEY_ID env var.
            secret_access_key: AWS secret access key. If not provided, reads from MASSIVE_SECRET_ACCESS_KEY env var.
            endpoint_url: S3 endpoint URL. If not provided, reads from MASSIVE_S3_ENDPOINT env var.
            bucket_name: S3 bucket name. If not provided, reads from MASSIVE_S3_BUCKET env var.
        """
        self.access_key_id = access_key_id or os.getenv('MASSIVE_ACCESS_KEY_ID')
        self.secret_access_key = secret_access_key or os.getenv('MASSIVE_SECRET_ACCESS_KEY')
        self.endpoint_url = endpoint_url or os.getenv('MASSIVE_S3_ENDPOINT', 'https://files.massive.com')
        self.bucket_name = bucket_name or os.getenv('MASSIVE_S3_BUCKET', 'flatfiles')

        if not self.access_key_id or not self.secret_access_key:
            raise ValueError("Massive.com S3 credentials not provided. Set MASSIVE_ACCESS_KEY_ID and MASSIVE_SECRET_ACCESS_KEY environment variables.")

        # Initialize boto3 session
        self.session = boto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

        # Create S3 client
        self.s3 = self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            config=Config(signature_version='s3v4'),
        )

    def _convert_period_to_dates(self, period: str) -> tuple:
        """
        Convert yfinance-style period to start/end dates.

        Args:
            period: Period string like '1y', '2y', '6mo', '1d'

        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        end_date = datetime.now()

        if period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=years * 365)
        elif period.endswith('mo'):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=months * 30)
        elif period.endswith('d'):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        elif period.endswith('w'):
            weeks = int(period[:-1])
            start_date = end_date - timedelta(weeks=weeks)
        else:
            # Default to 1 year
            start_date = end_date - timedelta(days=365)

        return start_date, end_date

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        return '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol

    def _get_ticker_symbol(self, symbol: str) -> str:
        """
        Convert yfinance symbol to Massive.com format.

        Args:
            symbol: Symbol like 'BTC-USD' (crypto) or 'AAPL' (stock)

        Returns:
            Massive.com compatible symbol
            - Crypto: 'X:BTC-USD'
            - Stocks: 'AAPL' (unchanged)
        """
        if self._is_crypto(symbol):
            # Crypto: BTC-USD -> X:BTC-USD
            return f'X:{symbol}'
        else:
            # US Stock: AAPL -> AAPL (keep as-is)
            return symbol

    def _list_available_files(self, prefix: str, start_date: datetime, end_date: datetime) -> list:
        """
        List available S3 files for the given prefix and date range with metadata.

        Args:
            prefix: S3 prefix (e.g., 'global_crypto/day_aggs_v1')
            start_date: Start date
            end_date: End date

        Returns:
            List of tuples (key, size) for S3 objects
        """
        files = []

        try:
            paginator = self.s3.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    # Extract date from filename (format: YYYY-MM-DD.csv.gz)
                    try:
                        # Example: global_crypto/day_aggs_v1/2024/01/2024-01-15.csv.gz
                        filename = key.split('/')[-1]
                        date_str = filename.replace('.csv.gz', '')
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')

                        if start_date <= file_date <= end_date:
                            files.append((key, size))
                    except:
                        continue

        except Exception as e:
            print(f"⚠️  Error listing S3 files: {e}")

        return sorted(files, key=lambda x: x[0])

    def _download_and_parse_file(self, key: str, expected_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Download and parse a single S3 CSV file with size verification.

        Args:
            key: S3 object key
            expected_size: Expected file size in bytes for verification

        Returns:
            DataFrame with parsed data
        """
        try:
            # Download file
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            content_bytes = response['Body'].read()

            # Verify file size if provided
            if expected_size is not None:
                actual_size = len(content_bytes)
                if actual_size != expected_size:
                    print(f"\n⚠️  Size mismatch for {key}: expected {expected_size} bytes, got {actual_size} bytes")
                    return None

            # Decompress gzip
            with gzip.GzipFile(fileobj=io.BytesIO(content_bytes)) as gzipfile:
                content = gzipfile.read()

            # Parse CSV
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

            return df

        except Exception as e:
            print(f"⚠️  Error downloading/parsing {key}: {e}")
            return None

    def _download_files_parallel(self, file_metadata: list, ticker: str,
                                 max_workers: int = 20,
                                 progress_callback: Optional[Callable] = None,
                                 processed_cache: Optional[set] = None,
                                 symbol: str = None,
                                 period: str = None,
                                 interval: str = None) -> list:
        """
        Download and parse multiple S3 files in parallel with progress tracking and resume support.

        Args:
            file_metadata: List of tuples (key, size) for S3 objects
            ticker: Ticker symbol to filter for
            max_workers: Maximum number of parallel downloads
            progress_callback: Optional callback function(completed, total, elapsed_time, skipped)
            processed_cache: Optional set of already-processed keys for resume support
            symbol: Symbol for progress marker saving
            period: Period for progress marker saving
            interval: Interval for progress marker saving

        Returns:
            List of filtered DataFrames
        """
        all_data = []
        completed = 0
        skipped = 0
        start_time = time.time()

        # Filter out already-processed files if resuming
        if processed_cache:
            files_to_process = [(key, size) for key, size in file_metadata if key not in processed_cache]
            skipped = len(file_metadata) - len(files_to_process)
            if skipped > 0:
                print(f"🔄 Resuming: Skipping {skipped} already-processed files")
        else:
            files_to_process = file_metadata

        if not files_to_process:
            return all_data

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks with size verification
            future_to_metadata = {
                executor.submit(self._download_and_parse_file, key, size): (key, size)
                for key, size in files_to_process
            }

            # Process completed downloads
            for future in as_completed(future_to_metadata):
                completed += 1
                elapsed = time.time() - start_time
                key, size = future_to_metadata[future]

                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        # Filter for our ticker
                        if 'ticker' in df.columns:
                            ticker_data = df[df['ticker'] == ticker]
                            if not ticker_data.empty:
                                all_data.append(ticker_data)
                                # Mark as successfully processed
                                if processed_cache is not None:
                                    processed_cache.add(key)
                except Exception as e:
                    print(f"\n⚠️  Error processing {key}: {e}")

                # Call progress callback
                if progress_callback:
                    progress_callback(completed, len(files_to_process), elapsed, skipped)

                # Save progress marker every 100 files
                if processed_cache is not None and symbol and completed % 100 == 0:
                    from strategies.data_cache import get_cache
                    cache = get_cache()
                    cache.save_progress(processed_cache, symbol, period, interval)

        return all_data

    def fetch_data(self, symbol: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch OHLCV data from Massive.com S3 flat files.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD' for crypto, 'AAPL' for stocks)
            period: Time period (e.g., '1y', '2y', '5y', '6mo')
            interval: Data interval ('1d' for day aggregates, '1h' for minute/hour aggregates)

        Returns:
            DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)
        """
        ticker = self._get_ticker_symbol(symbol)
        start_date, end_date = self._convert_period_to_dates(period)
        is_crypto = self._is_crypto(symbol)

        # Determine S3 prefix based on asset type and interval
        if is_crypto:
            if interval == '1d':
                prefix = 'global_crypto/day_aggs_v1'
            elif interval == '1h':
                prefix = 'global_crypto/minute_aggs_v1'  # We'll filter to hourly later
            elif interval == '1m':
                prefix = 'global_crypto/minute_aggs_v1'
            else:
                prefix = 'global_crypto/day_aggs_v1'
        else:
            # US Stocks
            if interval == '1d':
                prefix = 'us_stocks_sip/day_aggs_v1'
            elif interval == '1h':
                prefix = 'us_stocks_sip/minute_aggs_v1'  # We'll filter to hourly later
            elif interval == '1m':
                prefix = 'us_stocks_sip/minute_aggs_v1'
            else:
                prefix = 'us_stocks_sip/day_aggs_v1'

        print(f"📊 Fetching {symbol} from Massive.com S3 ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

        # List available files
        files = self._list_available_files(prefix, start_date, end_date)

        if not files:
            print(f"⚠️  No files found for {symbol} in date range")
            return pd.DataFrame()

        print(f"📁 Found {len(files)} files to process")

        # Try to get progress marker for resume support
        from strategies.data_cache import get_cache
        cache = get_cache()
        processed_cache = cache.get_progress(symbol, period, interval) or set()

        # Download and parse all files in parallel
        all_data = self._download_files_parallel(
            files, ticker,
            max_workers=20,  # Parallel downloads
            progress_callback=lambda completed, total, elapsed, skipped:
                print(f"\r⏳ Progress: {completed}/{total} files ({completed/total*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"ETA: {(elapsed/completed*(total-completed)) if completed > 0 else 0:.1f}s " +
                      (f"| Resumed: {skipped} skipped" if skipped > 0 else ""),
                      end='', flush=True) if completed % 10 == 0 or completed == total else None,
            processed_cache=processed_cache,
            symbol=symbol,
            period=period,
            interval=interval
        )
        print()  # New line after progress

        # Save final progress marker
        if processed_cache:
            cache.save_progress(processed_cache, symbol, period, interval)

        if not all_data:
            print(f"⚠️  No data found for ticker {ticker}")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert to yfinance-compatible format
        # Expected columns from Massive: ticker, volume, open, close, high, low, window_start, transactions
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'window_start': 'Date'
        }

        # Rename columns
        combined_df = combined_df.rename(columns=column_mapping)

        # Convert window_start (nanoseconds timestamp) to datetime
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], unit='ns')
            combined_df = combined_df.set_index('Date')

        # Select only required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        combined_df = combined_df[[col for col in required_cols if col in combined_df.columns]]

        # Sort by date
        combined_df = combined_df.sort_index()

        print(f"✓ Fetched {len(combined_df)} rows for {symbol} from S3 flat files")

        return combined_df


# Global instance
_massive_s3_source = None


def get_massive_s3_source(access_key_id: Optional[str] = None,
                          secret_access_key: Optional[str] = None) -> MassiveS3DataSource:
    """Get or create the global Massive S3 data source instance."""
    global _massive_s3_source
    if _massive_s3_source is None:
        _massive_s3_source = MassiveS3DataSource(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key
        )
    return _massive_s3_source


def fetch_crypto_data_massive_s3(symbol: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch crypto data using Massive.com S3 flat files (yfinance-compatible interface).

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD')
        period: Time period (e.g., '1y', '2y')
        interval: Data interval (e.g., '1d')

    Returns:
        DataFrame with OHLCV data
    """
    source = get_massive_s3_source()
    return source.fetch_data(symbol, period, interval)
