# HuggingFace Model Sharing Integration

This document describes how to export and import pre-trained Temporal models to/from HuggingFace Hub.

## Overview

The HuggingFace integration allows you to:
- **Export** trained models from local cache to HuggingFace Hub for sharing
- **Import** pre-trained models from HuggingFace Hub to skip training
- **Collaborate** by sharing models with team members or the community
- **Version control** your models with HuggingFace's built-in versioning

## Prerequisites

### 1. Install Dependencies

The required packages are already in `backend/requirements.txt`:
```bash
pip install transformers huggingface-hub
```

### 2. HuggingFace Account & Token

1. Create a free account at [https://huggingface.co/](https://huggingface.co/)
2. Generate an API token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set the environment variable:
```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

Or in Docker:
```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
```

## Exporting Models

### Basic Export

```python
from strategies.model_cache import get_model_cache

cache = get_model_cache()

# Export a trained model
url = cache.export_to_huggingface(
    symbol="BTC-EUR",
    interval="1h",
    lookback=45,
    focus="momentum",
    forecast_horizon=24,
    repo_id="your-username/btc-eur-momentum-1h",
    token=None,  # Uses HUGGING_FACE_HUB_TOKEN env var
    private=False  # Public repository
)

print(f"Model exported to: {url}")
```

### What Gets Exported

The export includes:
1. **pytorch_model.bin** - Model weights
2. **scaler.pkl** - Fitted StandardScaler for feature normalization
3. **config.json** - Model metadata (training date, parameters, validation loss, etc.)
4. **README.md** - Auto-generated model card with usage instructions

### Batch Export (Ensemble Models)

```python
from strategies.model_cache import get_model_cache

cache = get_model_cache()
token = "your_hf_token"

symbol = "BTC-EUR"
interval = "1h"
forecast_horizon = 24

configs = [
    {'lookback': 30, 'focus': 'momentum', 'name': 'short-momentum'},
    {'lookback': 45, 'focus': 'momentum', 'name': 'mid-momentum'},
    {'lookback': 60, 'focus': 'balanced', 'name': 'balanced'},
    {'lookback': 60, 'focus': 'mean_reversion', 'name': 'mean-reversion'},
]

for config in configs:
    repo_id = f"your-username/{symbol.lower()}-{config['name']}-{interval}"

    cache.export_to_huggingface(
        symbol=symbol,
        interval=interval,
        lookback=config['lookback'],
        focus=config['focus'],
        forecast_horizon=forecast_horizon,
        repo_id=repo_id,
        token=token
    )
```

## Importing Models

### Basic Import

```python
from strategies.model_cache import get_model_cache

cache = get_model_cache()

# Import a pre-trained model
model_path, scaler_path, metadata = cache.import_from_huggingface(
    repo_id="your-username/btc-eur-momentum-1h",
    symbol="BTC-EUR",
    interval="1h",
    lookback=45,
    focus="momentum",
    forecast_horizon=24,
    token=None,  # Only needed for private repos
    force=False  # Set True to re-download
)

print(f"Imported model trained on: {metadata['training_timestamp']}")
print(f"Validation loss: {metadata['best_val_loss']:.6f}")
```

### Using Imported Models

After importing, the model is in your local cache and can be loaded normally:

```python
from temporal import Temporal
import torch

# Create model instance with same architecture
model = Temporal(
    input_dim=len(metadata['feature_columns']),
    d_model=metadata['model_architecture']['d_model'],
    num_encoder_layers=metadata['model_architecture']['num_encoder_layers'],
    num_decoder_layers=metadata['model_architecture']['num_decoder_layers'],
    # ... other params
)

# Load the cached model
model.load_state_dict(torch.load(model_path))
model.eval()

# Load scaler
import pickle
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
```

## Integration with Training Pipeline

The model cache automatically saves models after training. To enable HuggingFace export in your workflow:

```python
# 1. Train model (automatically cached)
from examples.crypto_ensemble_forecast import train_ensemble_model

model_info = train_ensemble_model(
    symbol="BTC-EUR",
    period="2y",
    lookback=45,
    forecast_horizon=24,
    epochs=15,
    focus="momentum",
    model_name="Momentum Model",
    interval="1h"
)

# 2. Export to HuggingFace
from strategies.model_cache import get_model_cache

cache = get_model_cache()
cache.export_to_huggingface(
    symbol="BTC-EUR",
    interval="1h",
    lookback=45,
    focus="momentum",
    forecast_horizon=24,
    repo_id="your-username/btc-eur-momentum-1h"
)
```

## Use Cases

### 1. Sharing Models with Team

Export trained models to private HuggingFace repos for team collaboration:

```python
cache.export_to_huggingface(
    # ... model params ...
    repo_id="your-org/private-btc-model",
    private=True,  # Private repository
    token="your_token"
)
```

Team members can then import:

```python
cache.import_from_huggingface(
    repo_id="your-org/private-btc-model",
    # ... model params ...
    token="team_member_token"
)
```

### 2. Continuous Training & Versioning

Update models as new data becomes available:

```python
# Train updated model
train_model_with_latest_data()

# Export with version in commit message
cache.export_to_huggingface(
    # ... params ...
    commit_message="Update model with data through 2025-01-15"
)
```

### 3. Production Deployment

Skip training on production servers:

```python
# In production environment
cache.import_from_huggingface(
    repo_id="your-org/production-btc-model",
    # ... params ...
)

# Model is now ready to use for live trading
```

### 4. Research & Experimentation

Share research models publicly:

```python
cache.export_to_huggingface(
    # ... params ...
    repo_id="your-username/research-btc-model",
    private=False,  # Public for community
    commit_message="Experimental momentum model with custom features"
)
```

## Model Card

Each exported model includes an auto-generated model card (README.md) with:
- Model details (symbol, interval, lookback, horizon, focus)
- Training metadata (date, epochs, validation loss)
- Model architecture parameters
- Usage instructions
- Citation information

Example model card: https://huggingface.co/your-username/btc-eur-momentum-1h

## Security Considerations

1. **API Tokens**: Never commit tokens to git. Use environment variables.
2. **Private Models**: Set `private=True` for proprietary models
3. **Model Review**: Review model cards before making repositories public
4. **Access Control**: Use HuggingFace organization repos for team access control

## Troubleshooting

### "huggingface_hub not installed"

```bash
pip install huggingface-hub transformers
```

### "Model not found in cache"

Train the model first before exporting:
```python
# Check what's in cache
cache.get_cache_stats()

# Train the model if needed
train_ensemble_model(...)
```

### Authentication Errors

Ensure your token is set correctly:
```bash
echo $HUGGING_FACE_HUB_TOKEN
```

### Import Fails with "Repository not found"

- Check the repo_id is correct
- For private repos, ensure token has access
- Verify the repository exists on HuggingFace

## Examples

See `examples/huggingface_model_sharing.py` for complete working examples.

## API Reference

### `export_to_huggingface()`

```python
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
    commit_message: Optional[str] = None
) -> str
```

### `import_from_huggingface()`

```python
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
) -> Tuple[Path, Path, Dict]
```

## License

Models exported to HuggingFace inherit the GPL-3.0-or-later license from the temporal-forecasting package.
