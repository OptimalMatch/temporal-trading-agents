"""
HuggingFace Model Sharing Example

This example demonstrates how to export and import pre-trained models
to/from HuggingFace Hub for easy sharing and collaboration.

Usage:
    1. Train and cache models using the regular training pipeline
    2. Export models to HuggingFace Hub
    3. Import models from HuggingFace Hub (can skip training on new machines)
"""

import os
from strategies.model_cache import get_model_cache

def export_example():
    """
    Example: Export a cached model to HuggingFace Hub.

    Prerequisites:
    - Model must exist in cache (train it first)
    - HuggingFace account with API token
    - Set environment variable: HUGGING_FACE_HUB_TOKEN
    """

    # Get the model cache
    cache = get_model_cache()

    # Model configuration
    symbol = "BTC-EUR"
    interval = "1h"
    lookback = 45
    focus = "momentum"
    forecast_horizon = 24

    # Your HuggingFace repository ID (format: username/repo-name)
    repo_id = "username/btc-eur-momentum-1h"

    # Get HuggingFace token from environment
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print("⚠️  Please set HUGGING_FACE_HUB_TOKEN environment variable")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return

    # Check if model exists in cache
    if not cache.exists(symbol, interval, lookback, focus, forecast_horizon):
        print(f"❌ Model not found in cache. Train it first!")
        print(f"   Symbol: {symbol}, Interval: {interval}")
        print(f"   Lookback: {lookback}, Focus: {focus}, Horizon: {forecast_horizon}")
        return

    try:
        # Export to HuggingFace
        url = cache.export_to_huggingface(
            symbol=symbol,
            interval=interval,
            lookback=lookback,
            focus=focus,
            forecast_horizon=forecast_horizon,
            repo_id=repo_id,
            token=token,
            private=False,  # Set to True for private models
            commit_message=f"Add {symbol} {focus} model for {interval} trading"
        )

        print(f"✅ Successfully exported model!")
        print(f"   View at: {url}")

    except Exception as e:
        print(f"❌ Export failed: {e}")


def import_example():
    """
    Example: Import a model from HuggingFace Hub into local cache.

    This allows you to skip training and use pre-trained models.
    """

    # Get the model cache
    cache = get_model_cache()

    # Model configuration (must match the exported model)
    symbol = "BTC-EUR"
    interval = "1h"
    lookback = 45
    focus = "momentum"
    forecast_horizon = 24

    # HuggingFace repository to import from
    repo_id = "username/btc-eur-momentum-1h"

    # Token (optional, only needed for private repos)
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    try:
        # Import from HuggingFace
        model_path, scaler_path, metadata = cache.import_from_huggingface(
            repo_id=repo_id,
            symbol=symbol,
            interval=interval,
            lookback=lookback,
            focus=focus,
            forecast_horizon=forecast_horizon,
            token=token,
            force=False  # Set to True to re-download if already cached
        )

        print(f"✅ Successfully imported model!")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Training date: {metadata.get('training_timestamp')}")
        print(f"   Best val loss: {metadata.get('best_val_loss'):.6f}")

        # Now you can use this model in your trading strategies!
        # The imported model is now in the local cache and can be loaded
        # using the standard cache.load() method

    except Exception as e:
        print(f"❌ Import failed: {e}")


def batch_export_example():
    """
    Example: Export multiple models at once.

    Useful after training multiple ensemble models.
    """

    cache = get_model_cache()
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print("⚠️  Please set HUGGING_FACE_HUB_TOKEN environment variable")
        return

    # Configuration for ensemble models
    symbol = "BTC-EUR"
    interval = "1h"
    forecast_horizon = 24

    # Multiple model configurations
    configs = [
        {'lookback': 30, 'focus': 'momentum', 'name': 'short-momentum'},
        {'lookback': 45, 'focus': 'momentum', 'name': 'mid-momentum'},
        {'lookback': 60, 'focus': 'balanced', 'name': 'balanced'},
        {'lookback': 60, 'focus': 'mean_reversion', 'name': 'mean-reversion'},
    ]

    for config in configs:
        lookback = config['lookback']
        focus = config['focus']
        model_name = config['name']

        # Skip if not in cache
        if not cache.exists(symbol, interval, lookback, focus, forecast_horizon):
            print(f"⚠️  Skipping {model_name}: not found in cache")
            continue

        # Generate repo ID
        repo_id = f"username/{symbol.lower()}-{model_name}-{interval}"

        try:
            url = cache.export_to_huggingface(
                symbol=symbol,
                interval=interval,
                lookback=lookback,
                focus=focus,
                forecast_horizon=forecast_horizon,
                repo_id=repo_id,
                token=token,
                private=False
            )
            print(f"✅ Exported {model_name}: {url}")

        except Exception as e:
            print(f"❌ Failed to export {model_name}: {e}")


def list_cached_models():
    """
    List all models currently in the local cache.
    """

    cache = get_model_cache()
    stats = cache.get_cache_stats()

    print(f"\n📊 Cache Statistics")
    print(f"   Directory: {stats['cache_dir']}")
    print(f"   Models: {stats['num_models']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"\n📋 Cached models:")

    for model_key in stats['models']:
        print(f"   - {model_key}")


if __name__ == "__main__":
    print("="*70)
    print("HuggingFace Model Sharing Examples")
    print("="*70)

    # List currently cached models
    list_cached_models()

    print("\n" + "="*70)
    print("Available operations:")
    print("="*70)
    print("1. export_example() - Export a single model to HuggingFace")
    print("2. import_example() - Import a model from HuggingFace")
    print("3. batch_export_example() - Export multiple models")
    print("4. list_cached_models() - Show cached models")
    print()
    print("To use, import this module and call the functions:")
    print("  from examples.huggingface_model_sharing import export_example")
    print("  export_example()")
    print("="*70)
