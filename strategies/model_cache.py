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
from typing import Optional, Dict, Any, Tuple
import json


class ModelCache:
    """
    Manages persistent storage and retrieval of trained PyTorch models.

    Features:
    - Save/load trained models with metadata
    - Time-based cache invalidation
    - Data-driven cache invalidation
    - Support for incremental learning (fine-tuning)
    """

    def __init__(self, cache_dir: str = "/tmp/model_cache"):
        """
        Initialize the model cache.

        Args:
            cache_dir: Directory to store cached models
        """
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
             best_val_loss: float) -> str:
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
