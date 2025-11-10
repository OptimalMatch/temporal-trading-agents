"""
Model Cache for Temporal Trading Agents

Provides persistent caching of trained PyTorch models to avoid retraining from scratch.
Supports incremental learning (fine-tuning) when new data becomes available.
"""

import os
import torch
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json
import tempfile
import shutil

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ModelCache:
    """
    Manages persistent storage and retrieval of trained PyTorch models.

    Features:
    - Save/load trained models with metadata
    - Time-based cache invalidation
    - Data-driven cache invalidation
    - Support for incremental learning (fine-tuning)
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the model cache.

        Args:
            cache_dir: Directory to store cached models. If None, uses MODEL_CACHE_DIR env var or default.
        """
        import os
        if cache_dir is None:
            cache_dir = os.environ.get('MODEL_CACHE_DIR', '/app/model_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, symbol: str, interval: str, lookback: int,
                       focus: str, forecast_horizon: int) -> str:
        """
        Generate a unique cache key for a model configuration.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            interval: Data interval ('1h', '1d')
            lookback: Lookback window size
            focus: Model focus ('momentum', 'balanced', 'mean_reversion')
            forecast_horizon: Forecast horizon

        Returns:
            Cache key string
        """
        key_parts = f"{symbol}_{interval}_{lookback}_{focus}_{forecast_horizon}"
        # Use hash to keep filename reasonable length
        key_hash = hashlib.md5(key_parts.encode()).hexdigest()[:12]
        return f"{symbol}_{interval}_{focus}_{key_hash}"

    def _get_model_path(self, cache_key: str) -> Path:
        """Get the file path for a cached model."""
        return self.cache_dir / f"{cache_key}_model.pt"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get the file path for model metadata."""
        return self.cache_dir / f"{cache_key}_meta.json"

    def _get_scaler_path(self, cache_key: str) -> Path:
        """Get the file path for the scaler."""
        return self.cache_dir / f"{cache_key}_scaler.pkl"

    def exists(self, symbol: str, interval: str, lookback: int,
               focus: str, forecast_horizon: int) -> bool:
        """
        Check if a cached model exists for the given configuration.

        Returns:
            True if model exists in cache, False otherwise
        """
        cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)
        model_path = self._get_model_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        return model_path.exists() and metadata_path.exists()

    def get_metadata(self, symbol: str, interval: str, lookback: int,
                    focus: str, forecast_horizon: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a cached model.

        Returns:
            Metadata dictionary or None if not found
        """
        if not self.exists(symbol, interval, lookback, focus, forecast_horizon):
            return None

        cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)
        metadata_path = self._get_metadata_path(cache_key)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Convert timestamp strings back to datetime
        if 'training_timestamp' in metadata:
            metadata['training_timestamp'] = datetime.fromisoformat(metadata['training_timestamp'])
        if 'data_end_timestamp' in metadata:
            metadata['data_end_timestamp'] = datetime.fromisoformat(metadata['data_end_timestamp'])

        return metadata

    def is_stale(self, symbol: str, interval: str, lookback: int,
                 focus: str, forecast_horizon: int,
                 max_age_hours: float = 6.0,
                 latest_data_timestamp: Optional[datetime] = None) -> bool:
        """
        Check if a cached model is stale and needs updating.

        Args:
            symbol: Trading symbol
            interval: Data interval
            lookback: Lookback window
            focus: Model focus
            forecast_horizon: Forecast horizon
            max_age_hours: Maximum age before model is considered stale
            latest_data_timestamp: Latest available data timestamp (for data-driven invalidation)

        Returns:
            True if model is stale, False if still fresh
        """
        metadata = self.get_metadata(symbol, interval, lookback, focus, forecast_horizon)
        if metadata is None:
            return True  # No cache = stale

        # Time-based staleness check
        training_time = metadata['training_timestamp']
        age_hours = (datetime.now() - training_time).total_seconds() / 3600

        if age_hours > max_age_hours:
            return True  # Too old

        # Data-driven staleness check (if latest data timestamp provided)
        if latest_data_timestamp and 'data_end_timestamp' in metadata:
            model_data_end = metadata['data_end_timestamp']

            # If there's significantly new data, model is stale
            # "Significant" = more than 2 hours of new data for hourly, 1 day for daily
            threshold_hours = 2 if interval == '1h' else 24
            new_data_hours = (latest_data_timestamp - model_data_end).total_seconds() / 3600

            if new_data_hours > threshold_hours:
                return True  # New data available

        return False  # Still fresh

    def save(self, model, scaler, symbol: str, interval: str, lookback: int,
             focus: str, forecast_horizon: int, feature_columns: list,
             data_end_timestamp: datetime, training_epochs: int,
             best_val_loss: float, data_start_timestamp: Optional[datetime] = None,
             training_samples: Optional[int] = None) -> str:
        """
        Save a trained model and its metadata to cache.

        Args:
            model: Trained PyTorch model
            scaler: Fitted StandardScaler
            symbol: Trading symbol
            interval: Data interval
            lookback: Lookback window
            focus: Model focus
            forecast_horizon: Forecast horizon
            feature_columns: List of feature column names
            data_end_timestamp: Timestamp of last data point used in training
            training_epochs: Number of epochs trained
            best_val_loss: Best validation loss achieved
            data_start_timestamp: Timestamp of first data point used in training (optional)
            training_samples: Number of training samples (optional)

        Returns:
            Cache key for the saved model
        """
        cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)

        # Save model state dict
        model_path = self._get_model_path(cache_key)
        torch.save(model.state_dict(), model_path)

        # Save scaler
        scaler_path = self._get_scaler_path(cache_key)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save metadata
        metadata = {
            'cache_key': cache_key,
            'symbol': symbol,
            'interval': interval,
            'lookback': lookback,
            'focus': focus,
            'forecast_horizon': forecast_horizon,
            'feature_columns': feature_columns,
            'training_timestamp': datetime.now().isoformat(),
            'data_end_timestamp': data_end_timestamp.isoformat(),
            'training_epochs': training_epochs,
            'best_val_loss': best_val_loss,
            'model_architecture': {
                'd_model': model.d_model if hasattr(model, 'd_model') else None,
                'num_encoder_layers': model.num_encoder_layers if hasattr(model, 'num_encoder_layers') else None,
                'num_decoder_layers': model.num_decoder_layers if hasattr(model, 'num_decoder_layers') else None,
            }
        }

        # Add optional dataset metadata
        if data_start_timestamp is not None:
            metadata['data_start_timestamp'] = data_start_timestamp.isoformat()
        if training_samples is not None:
            metadata['training_samples'] = training_samples

        metadata_path = self._get_metadata_path(cache_key)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved model to cache: {cache_key}")
        return cache_key

    def load(self, model_template, symbol: str, interval: str, lookback: int,
             focus: str, forecast_horizon: int) -> Optional[Tuple[Any, Any, Dict]]:
        """
        Load a cached model, scaler, and metadata.

        Args:
            model_template: An instance of the model class (for architecture)
            symbol: Trading symbol
            interval: Data interval
            lookback: Lookback window
            focus: Model focus
            forecast_horizon: Forecast horizon

        Returns:
            Tuple of (model, scaler, metadata) or None if not found
        """
        if not self.exists(symbol, interval, lookback, focus, forecast_horizon):
            return None

        cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)

        # Load metadata
        metadata = self.get_metadata(symbol, interval, lookback, focus, forecast_horizon)

        # Load model state dict
        model_path = self._get_model_path(cache_key)
        model_template.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Load scaler
        scaler_path = self._get_scaler_path(cache_key)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print(f"📂 Loaded model from cache: {cache_key} (age: {self._get_age_hours(metadata):.1f}h)")

        return model_template, scaler, metadata

    def _get_age_hours(self, metadata: Dict) -> float:
        """Get the age of a cached model in hours."""
        training_time = metadata['training_timestamp']
        return (datetime.now() - training_time).total_seconds() / 3600

    def clear(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        """
        Clear cached models.

        Args:
            symbol: If specified, only clear models for this symbol
            interval: If specified, only clear models for this interval
        """
        pattern = "*"
        if symbol and interval:
            pattern = f"{symbol}_{interval}_*"
        elif symbol:
            pattern = f"{symbol}_*"
        elif interval:
            pattern = f"*_{interval}_*"

        count = 0
        for file_path in self.cache_dir.glob(pattern):
            file_path.unlink()
            count += 1

        print(f"🗑️  Cleared {count} cached model files")

    def export_to_huggingface(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        focus: str,
        forecast_horizon: int,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        path_in_repo: Optional[str] = None,
        ensemble_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export a cached model to HuggingFace Hub.

        Args:
            symbol: Trading symbol
            interval: Data interval
            lookback: Lookback window
            focus: Model focus
            forecast_horizon: Forecast horizon
            repo_id: HuggingFace repository ID (e.g., "username/model-name")
            token: HuggingFace API token (or set HUGGING_FACE_HUB_TOKEN env var)
            private: Whether to create a private repository
            commit_message: Custom commit message
            path_in_repo: Path within repository (e.g., "momentum_lookback30/")
            ensemble_info: Optional dict with ensemble metadata:
                - total_members: Total number of models in ensemble
                - member_index: Index of this model (1-based)
                - all_members: List of dicts with 'focus', 'lookback', 'path' for each member

        Returns:
            URL to the uploaded model on HuggingFace Hub

        Raises:
            ImportError: If huggingface_hub is not installed
            ValueError: If model doesn't exist in cache
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface-hub"
            )

        if not self.exists(symbol, interval, lookback, focus, forecast_horizon):
            raise ValueError(
                f"Model not found in cache for {symbol} {interval} "
                f"(lookback={lookback}, focus={focus}, horizon={forecast_horizon})"
            )

        cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)
        metadata = self.get_metadata(symbol, interval, lookback, focus, forecast_horizon)

        # Create temporary directory for HF upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy model file
            model_src = self._get_model_path(cache_key)
            model_dst = temp_path / "pytorch_model.bin"
            shutil.copy(model_src, model_dst)

            # Copy scaler
            scaler_src = self._get_scaler_path(cache_key)
            if scaler_src.exists():
                scaler_dst = temp_path / "scaler.pkl"
                shutil.copy(scaler_src, scaler_dst)

            # Save metadata as config.json (HF standard)
            config_path = temp_path / "config.json"
            with open(config_path, 'w') as f:
                # Convert datetime objects to strings for JSON
                metadata_copy = metadata.copy()
                if 'training_timestamp' in metadata_copy and isinstance(metadata_copy['training_timestamp'], datetime):
                    metadata_copy['training_timestamp'] = metadata_copy['training_timestamp'].isoformat()
                if 'data_end_timestamp' in metadata_copy and isinstance(metadata_copy['data_end_timestamp'], datetime):
                    metadata_copy['data_end_timestamp'] = metadata_copy['data_end_timestamp'].isoformat()
                json.dump(metadata_copy, f, indent=2)

            # Create README
            readme_path = temp_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(self._generate_model_card(symbol, interval, lookback, focus, forecast_horizon, metadata, ensemble_info))

            # Upload to HuggingFace Hub
            try:
                # Test connectivity before attempting upload
                import socket
                try:
                    socket.gethostbyname("huggingface.co")
                    print(f"🌐 Network check: Successfully resolved huggingface.co")
                except Exception as dns_error:
                    print(f"⚠️  Network check: Failed to resolve huggingface.co - {str(dns_error)}")
                    raise RuntimeError(
                        f"Network connectivity issue: Cannot resolve huggingface.co. "
                        f"RunPod instance may not have internet access or DNS is misconfigured. "
                        f"Error: {str(dns_error)}"
                    ) from dns_error

                api = HfApi(token=token)

                if not commit_message:
                    commit_message = (
                        f"Upload {symbol} {focus} model "
                        f"(interval={interval}, lookback={lookback}, horizon={forecast_horizon})"
                    )

                # Check if repo exists, create if needed
                try:
                    api.repo_info(repo_id=repo_id, repo_type="model", token=token)
                except Exception:
                    # Repo doesn't exist, create it
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        private=private,
                        exist_ok=True,
                        token=token
                    )

                url = api.upload_folder(
                    folder_path=str(temp_path),
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_message,
                    create_pr=False,
                    path_in_repo=path_in_repo,
                )

                repo_url = f"https://huggingface.co/{repo_id}"
                if path_in_repo:
                    repo_url += f"/tree/main/{path_in_repo}"

                print(f"🤗 Exported model to HuggingFace: {repo_id}")
                print(f"   Path: {path_in_repo or '(root)'}")
                print(f"   URL: {repo_url}")

                return repo_url

            except Exception as e:
                # Provide detailed error information
                error_msg = f"Failed to upload to HuggingFace Hub"
                if hasattr(e, 'response'):
                    # HTTP error from HF API
                    error_msg += f" (HTTP {e.response.status_code})"
                    if hasattr(e.response, 'text'):
                        error_msg += f": {e.response.text}"
                else:
                    error_msg += f": {str(e)}"

                # Check common issues
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    error_msg += "\n💡 Hint: Check that HUGGING_FACE_HUB_TOKEN is valid and has write access"
                elif "403" in str(e) or "forbidden" in str(e).lower():
                    error_msg += f"\n💡 Hint: Token may not have permission to write to repo '{repo_id}'"
                elif "404" in str(e):
                    error_msg += f"\n💡 Hint: Repository '{repo_id}' may not exist or token lacks access"

                raise RuntimeError(error_msg) from e

    def import_from_huggingface(
        self,
        repo_id: str,
        symbol: str,
        interval: str,
        lookback: int,
        focus: str,
        forecast_horizon: int,
        token: Optional[str] = None,
        force: bool = False
    ) -> Tuple[Path, Path, Dict]:
        """
        Import a model from HuggingFace Hub into the local cache.

        Args:
            repo_id: HuggingFace repository ID (e.g., "username/model-name")
            symbol: Trading symbol (for cache key generation)
            interval: Data interval
            lookback: Lookback window
            focus: Model focus
            forecast_horizon: Forecast horizon
            token: HuggingFace API token (for private repos)
            force: Force re-download even if model exists in cache

        Returns:
            Tuple of (model_path, scaler_path, metadata)

        Raises:
            ImportError: If huggingface_hub is not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface-hub"
            )

        # Check if already in cache (unless force=True)
        if not force and self.exists(symbol, interval, lookback, focus, forecast_horizon):
            print(f"📂 Model already in cache, loading from cache (use force=True to re-download)")
            cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)
            return (
                self._get_model_path(cache_key),
                self._get_scaler_path(cache_key),
                self.get_metadata(symbol, interval, lookback, focus, forecast_horizon)
            )

        cache_key = self._get_cache_key(symbol, interval, lookback, focus, forecast_horizon)

        print(f"🤗 Downloading model from HuggingFace: {repo_id}")

        # Download model file
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            token=token,
            repo_type="model"
        )

        # Download config/metadata
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            token=token,
            repo_type="model"
        )

        # Download scaler (optional)
        scaler_file = None
        try:
            scaler_file = hf_hub_download(
                repo_id=repo_id,
                filename="scaler.pkl",
                token=token,
                repo_type="model"
            )
        except Exception:
            print("⚠️  No scaler found in HuggingFace repo")

        # Copy to local cache
        model_dst = self._get_model_path(cache_key)
        shutil.copy(model_file, model_dst)

        if scaler_file:
            scaler_dst = self._get_scaler_path(cache_key)
            shutil.copy(scaler_file, scaler_dst)

        metadata_dst = self._get_metadata_path(cache_key)
        shutil.copy(config_file, metadata_dst)

        # Load metadata
        with open(config_file, 'r') as f:
            metadata = json.load(f)

        # Convert timestamp strings back to datetime
        if 'training_timestamp' in metadata:
            metadata['training_timestamp'] = datetime.fromisoformat(metadata['training_timestamp'])
        if 'data_end_timestamp' in metadata:
            metadata['data_end_timestamp'] = datetime.fromisoformat(metadata['data_end_timestamp'])

        print(f"✅ Imported model to cache: {cache_key}")

        return model_dst, self._get_scaler_path(cache_key) if scaler_file else None, metadata

    def _generate_model_card(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        focus: str,
        forecast_horizon: int,
        metadata: Dict[str, Any],
        ensemble_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a model card (README) for HuggingFace."""
        training_time = metadata.get('training_timestamp', 'Unknown')
        if isinstance(training_time, datetime):
            training_time = training_time.strftime('%Y-%m-%d %H:%M UTC')

        val_loss = metadata.get('best_val_loss', 'N/A')
        epochs = metadata.get('training_epochs', 'N/A')

        # Format dataset date range
        data_start = metadata.get('data_start_timestamp', 'Unknown')
        data_end = metadata.get('data_end_timestamp', 'Unknown')
        if isinstance(data_start, datetime):
            data_start = data_start.strftime('%Y-%m-%d')
        if isinstance(data_end, datetime):
            data_end = data_end.strftime('%Y-%m-%d')

        dataset_range = f"{data_start} to {data_end}" if data_start != 'Unknown' and data_end != 'Unknown' else 'N/A'
        training_samples = metadata.get('training_samples', 'N/A')

        # Build ensemble context section if provided
        ensemble_section = ""
        title_suffix = f"{focus}"

        if ensemble_info:
            total = ensemble_info.get('total_members', 0)
            index = ensemble_info.get('member_index', 0)
            all_members = ensemble_info.get('all_members', [])

            title_suffix = f"Ensemble Member {index}/{total}: {focus} (lookback={lookback})"

            # Build member list
            member_list = []
            for i, member in enumerate(all_members, 1):
                member_focus = member.get('focus', 'unknown')
                member_lookback = member.get('lookback', 0)
                member_path = member.get('path', '')
                if i == index:
                    member_list.append(f"{i}. **{member_focus}** (lookback={member_lookback}) - *This model*")
                else:
                    member_list.append(f"{i}. [{member_focus} (lookback={member_lookback})]({member_path})")

            ensemble_section = f"""
## Ensemble Context

This model is **member {index} of {total}** in a consensus ensemble for {symbol} ({interval}).

The ensemble uses {total} models with different lookback periods and focus strategies to generate diverse forecasts. These forecasts are then aggregated using multiple consensus strategies (gradient, confidence, timeframe, volatility, mean reversion, acceleration, swing, risk-adjusted) to produce robust trading signals.

### All Ensemble Members

{chr(10).join(member_list)}

### How the Ensemble Works

1. Each member model generates independent forecasts using its specific lookback window and focus
2. Multiple consensus strategies analyze the ensemble's forecasts from different perspectives
3. Each strategy produces action recommendations (BUY/SELL/HOLD) with confidence scores
4. Final consensus aggregates all strategy recommendations into a unified trading signal

"""

        return f"""---
tags:
- time-series
- forecasting
- temporal
- trading
- {symbol.lower()}
- ensemble
library_name: temporal-forecasting
---

# Temporal Trading Model: {symbol} ({title_suffix})

This is a pre-trained Temporal transformer model for time series forecasting of {symbol}.{' This model is part of a consensus ensemble.' if ensemble_info else ''}
{ensemble_section}
## Model Details

- **Symbol**: {symbol}
- **Interval**: {interval}
- **Lookback Window**: {lookback} periods
- **Forecast Horizon**: {forecast_horizon} periods
- **Focus**: {focus}
- **Training Date**: {training_time}
- **Training Epochs**: {epochs}
- **Best Validation Loss**: {val_loss}

## Training Dataset

- **Date Range**: {dataset_range}
- **Training Samples**: {training_samples}
- **Lookback Period**: {lookback} {interval} intervals

## Model Architecture

- **Model Type**: Temporal Transformer
- **d_model**: {metadata.get('model_architecture', {}).get('d_model', 'N/A')}
- **Encoder Layers**: {metadata.get('model_architecture', {}).get('num_encoder_layers', 'N/A')}
- **Decoder Layers**: {metadata.get('model_architecture', {}).get('num_decoder_layers', 'N/A')}

## Features

Input features used: {len(metadata.get('feature_columns', []))} features

## Usage

```python
from strategies.model_cache import get_model_cache

# Import from HuggingFace
cache = get_model_cache()
model_path, scaler_path, metadata = cache.import_from_huggingface(
    repo_id="YOUR_REPO_ID",
    symbol="{symbol}",
    interval="{interval}",
    lookback={lookback},
    focus="{focus}",
    forecast_horizon={forecast_horizon}
)
```

## License

GPL-3.0-or-later

## Citation

```
@software{{temporal_trading_model,
  title = {{Temporal Trading Model: {symbol} ({focus})}},
  author = {{Unidatum Integrated Products LLC}},
  year = {{2025}},
  url = {{https://github.com/OptimalMatch/temporal-trading-agents}}
}}
```
"""

    def _generate_ensemble_overview_card(
        self,
        symbol: str,
        interval: str,
        forecast_horizon: int,
        repo_id: str,
        all_members: List[Dict[str, Any]],
        consensus_id: Optional[str] = None
    ) -> str:
        """Generate an ensemble overview card (root README) for HuggingFace."""
        from datetime import datetime

        # Build member list
        member_list = []
        for member in all_members:
            focus = member.get('focus', 'unknown')
            lookback = member.get('lookback', 0)
            path = member.get('path', '').lstrip('../')  # Remove ../ prefix for display
            member_list.append(f"- [{focus} (lookback={lookback})](./tree/main/{path})")

        return f"""---
tags:
- time-series
- forecasting
- temporal
- trading
- ensemble
- consensus
- {symbol.lower()}
library_name: temporal-forecasting
---

# {symbol} Consensus Ensemble ({interval})

This repository contains a **consensus ensemble** of {len(all_members)} temporal transformer models for {symbol} price forecasting.

## Overview

Instead of relying on a single model, this ensemble combines multiple models with different characteristics to produce robust, diverse forecasts. Each model specializes in different market patterns:

- **Momentum models**: Capture trending behavior with shorter lookback windows
- **Balanced models**: General-purpose forecasting with medium lookback windows
- **Mean reversion models**: Detect overbought/oversold conditions with longer windows

## Ensemble Members

{chr(10).join(member_list)}

## How It Works

### 1. Independent Forecasting
Each model generates independent {forecast_horizon}-period forecasts using its specific lookback window and focus strategy.

### 2. Consensus Analysis
Eight different consensus strategies analyze the ensemble's forecasts:

- **Gradient Strategy**: Analyzes forecast momentum and direction changes
- **Confidence Strategy**: Weighs predictions by model confidence scores
- **Timeframe Strategy**: Balances short-term vs long-term predictions
- **Volatility Strategy**: Adjusts for market volatility conditions
- **Mean Reversion Strategy**: Identifies reversal opportunities
- **Acceleration Strategy**: Detects accelerating price movements
- **Swing Strategy**: Targets swing trading opportunities
- **Risk-Adjusted Strategy**: Optimizes risk-reward ratios

### 3. Unified Signal
All strategy recommendations are aggregated into a final consensus signal with:
- **Action**: BUY, SELL, or HOLD
- **Confidence Score**: 0.0 to 1.0
- **Expected Return**: Basis points (bps)

## Usage

### Import Individual Models

```python
from strategies.model_cache import get_model_cache

cache = get_model_cache()

# Import a specific ensemble member
model_path, scaler_path, metadata = cache.import_from_huggingface(
    repo_id="{repo_id}",
    symbol="{symbol}",
    interval="{interval}",
    lookback=30,  # Choose appropriate lookback
    focus="momentum",  # Choose appropriate focus
    forecast_horizon={forecast_horizon},
    path_in_repo="momentum_lookback30"  # Specify subdirectory
)
```

### Run Consensus Analysis

```python
from backend.main import run_consensus_analysis_worker
from backend.database import Database

async def run_analysis():
    db = Database()
    await db.connect()

    result = await run_consensus_analysis_worker(
        symbol="{symbol}",
        horizons=[{forecast_horizon}],
        interval="{interval}",
        analysis_id="my-analysis",
        db=db
    )

    print(f"Consensus: {{result['consensus']}}")
    print(f"Action: {{result['action']}}")
    print(f"Confidence: {{result['confidence']}}")
```

## Model Details

- **Symbol**: {symbol}
- **Interval**: {interval}
- **Forecast Horizon**: {forecast_horizon} periods
- **Ensemble Size**: {len(all_members)} models
- **Architecture**: Temporal Transformer
- **Training Framework**: PyTorch
- **Export Date**: {datetime.now().strftime('%Y-%m-%d')}

## License

GPL-3.0-or-later

## Citation

```
@software{{temporal_trading_ensemble,
  title = {{{symbol} Consensus Ensemble}},
  author = {{Unidatum Integrated Products LLC}},
  year = {{2025}},
  url = {{https://github.com/OptimalMatch/temporal-trading-agents}}
}}
```

## Links

- [GitHub Repository](https://github.com/OptimalMatch/temporal-trading-agents)
- [Documentation](https://github.com/OptimalMatch/temporal-trading-agents/blob/main/README.md)
"""

    def export_ensemble_overview_to_huggingface(
        self,
        symbol: str,
        interval: str,
        forecast_horizon: int,
        repo_id: str,
        all_members: List[Dict[str, Any]],
        token: Optional[str] = None,
        consensus_id: Optional[str] = None
    ) -> str:
        """
        Export an ensemble overview README to the root of a HuggingFace repository.

        Args:
            symbol: Trading symbol
            interval: Data interval
            forecast_horizon: Forecast horizon
            repo_id: HuggingFace repository ID
            all_members: List of ensemble members with focus, lookback, path
            token: HuggingFace API token
            consensus_id: Optional consensus analysis ID

        Returns:
            URL to the repository
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface-hub"
            )

        # Generate the overview content
        overview_content = self._generate_ensemble_overview_card(
            symbol=symbol,
            interval=interval,
            forecast_horizon=forecast_horizon,
            repo_id=repo_id,
            all_members=all_members,
            consensus_id=consensus_id
        )

        # Create temporary file for upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            readme_path = temp_path / "README.md"

            with open(readme_path, 'w') as f:
                f.write(overview_content)

            # Upload to HuggingFace Hub root
            api = HfApi(token=token)

            url = api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Update ensemble overview for {symbol} ({interval})"
            )

            repo_url = f"https://huggingface.co/{repo_id}"
            print(f"🤗 Uploaded ensemble overview to: {repo_url}")

            return repo_url

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model cache.

        Returns:
            Dictionary with cache statistics
        """
        model_files = list(self.cache_dir.glob("*_model.pt"))

        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*"))

        return {
            'cache_dir': str(self.cache_dir),
            'num_models': len(model_files),
            'total_size_mb': total_size / (1024 * 1024),
            'models': [f.stem.replace('_model', '') for f in model_files]
        }


# Global cache instance
_model_cache = None

def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache
