"""
Massive.com (formerly Polygon.io) S3 flat files data source.
Fetches market data from S3-hosted compressed CSV files.
"""
import os
import gzip
import io
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import boto3
from botocore.config import Config


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

    def _get_ticker_symbol(self, symbol: str) -> str:
        """
        Convert yfinance symbol to Massive.com format.

        Args:
            symbol: Symbol like 'BTC-USD', 'ETH-USD'

        Returns:
            Massive.com compatible symbol (e.g., 'BTC', 'ETH')
        """
        # Convert crypto symbols
        if '-USD' in symbol:
            # BTC-USD -> BTC
            return symbol.replace('-USD', '')
        else:
            return symbol

    def _list_available_files(self, prefix: str, start_date: datetime, end_date: datetime) -> list:
        """
        List available S3 files for the given prefix and date range.

        Args:
            prefix: S3 prefix (e.g., 'global_crypto/day_aggs_v1')
            start_date: Start date
            end_date: End date

        Returns:
            List of S3 object keys
        """
        files = []

        try:
            paginator = self.s3.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract date from filename (format: YYYY-MM-DD.csv.gz)
                    try:
                        # Example: global_crypto/day_aggs_v1/2024/01/2024-01-15.csv.gz
                        filename = key.split('/')[-1]
                        date_str = filename.replace('.csv.gz', '')
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')

                        if start_date <= file_date <= end_date:
                            files.append(key)
                    except:
                        continue

        except Exception as e:
            print(f"⚠️  Error listing S3 files: {e}")

        return sorted(files)

    def _download_and_parse_file(self, key: str) -> Optional[pd.DataFrame]:
        """
        Download and parse a single S3 CSV file.

        Args:
            key: S3 object key

        Returns:
            DataFrame with parsed data
        """
        try:
            # Download file
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)

            # Decompress gzip
            with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gzipfile:
                content = gzipfile.read()

            # Parse CSV
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

            return df

        except Exception as e:
            print(f"⚠️  Error downloading/parsing {key}: {e}")
            return None

    def fetch_data(self, symbol: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch OHLCV data from Massive.com S3 flat files.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Time period (e.g., '1y', '2y', '6mo')
            interval: Data interval ('1d' for day aggregates, '1h' for minute/hour aggregates)

        Returns:
            DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)
        """
        ticker = self._get_ticker_symbol(symbol)
        start_date, end_date = self._convert_period_to_dates(period)

        # Determine S3 prefix based on interval
        if interval == '1d':
            prefix = 'global_crypto/day_aggs_v1'
        elif interval == '1h':
            prefix = 'global_crypto/minute_aggs_v1'  # We'll filter to hourly later
        elif interval == '1m':
            prefix = 'global_crypto/minute_aggs_v1'
        else:
            prefix = 'global_crypto/day_aggs_v1'

        print(f"📊 Fetching {symbol} from Massive.com S3 ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

        # List available files
        files = self._list_available_files(prefix, start_date, end_date)

        if not files:
            print(f"⚠️  No files found for {symbol} in date range")
            return pd.DataFrame()

        print(f"📁 Found {len(files)} files to process")

        # Download and parse all files
        all_data = []
        for file_key in files:
            df = self._download_and_parse_file(file_key)
            if df is not None and not df.empty:
                # Filter for our ticker
                if 'ticker' in df.columns:
                    ticker_data = df[df['ticker'] == ticker]
                    if not ticker_data.empty:
                        all_data.append(ticker_data)

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
