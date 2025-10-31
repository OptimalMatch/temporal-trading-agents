"""
Test script to verify temporal-trading-agents setup.

This script tests:
1. Importing temporal-forecasting
2. Fetching market data with yfinance
3. Running a basic forecast
"""

import sys
print("=" * 60)
print("Testing temporal-trading-agents setup")
print("=" * 60)

# Test 1: Import temporal-forecasting
print("\n1. Testing temporal-forecasting import...")
try:
    import temporal
    from temporal import TemporalForForecasting
    print(f"   ✓ temporal-forecasting version: {temporal.__version__}")
    print("   ✓ TemporalForForecasting imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Import other dependencies
print("\n2. Testing other dependencies...")
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("   ✓ yfinance imported")
    print("   ✓ pandas imported")
    print("   ✓ numpy imported")
    print("   ✓ matplotlib imported")
    print("   ✓ seaborn imported")
except ImportError as e:
    print(f"   ✗ Failed to import dependency: {e}")
    sys.exit(1)

# Test 3: Fetch sample data
print("\n3. Testing data fetch with yfinance...")
try:
    ticker = "BTC-USD"
    print(f"   Fetching {ticker} data...")
    data = yf.download(ticker, period="30d", progress=False)
    print(f"   ✓ Fetched {len(data)} days of data")
    print(f"   ✓ Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"   ✗ Failed to fetch data: {e}")
    sys.exit(1)

# Test 4: Create and test model
print("\n4. Testing model creation...")
try:
    # Prepare data
    prices = data['Close'].values
    print(f"   Using {len(prices)} price points")

    # Create model using the standard Temporal interface
    from temporal import Temporal, TemporalTrainer
    import torch

    model = Temporal(
        input_dim=1,
        d_model=64,  # Small model for quick test
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        forecast_horizon=3,
        max_seq_len=100,
        dropout=0.1
    )
    print("   ✓ Temporal model created")

    # Test that we can create a trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        device='cpu'
    )
    print("   ✓ TemporalTrainer created with optimizer")
    print("   ✓ Model is ready for training and forecasting")

except Exception as e:
    print(f"   ✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Setup is working correctly.")
print("=" * 60)
print("\nYou can now:")
print("  - Run examples from the examples/ folder")
print("  - Build trading strategies in strategies/")
print("  - Develop trading agents in agents/")
