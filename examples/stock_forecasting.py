"""
Stock Price Forecasting Example

Demonstrates how to:
1. Fetch real stock price data from Yahoo Finance
2. Prepare data for the Temporal model
3. Train a model to forecast stock prices
4. Evaluate and visualize results
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer
from temporal.data_sources import (
    fetch_stock_data,
    prepare_for_temporal,
    add_technical_indicators,
    split_train_val_test
)


def main():
    print("=" * 70)
    print("STOCK PRICE FORECASTING WITH TEMPORAL")
    print("=" * 70)

    # Configuration
    TICKER = 'AAPL'  # Apple stock
    PERIOD = '2y'    # 2 years of data
    LOOKBACK = 60    # Use 60 days to predict
    FORECAST_HORIZON = 5  # Predict 5 days ahead
    BATCH_SIZE = 32
    EPOCHS = 50

    # Step 1: Fetch stock data
    print(f"\n1. Fetching {TICKER} stock data...")
    print(f"   Period: {PERIOD}")
    print(f"   Fetching from Yahoo Finance...")

    df = fetch_stock_data(TICKER, period=PERIOD, interval='1d')
    print(f"   ✓ Fetched {len(df)} days of data")
    print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"\n   Sample data:")
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head())

    # Step 2: Prepare data
    print(f"\n2. Preparing data for training...")

    # Option A: Use just closing price (univariate)
    print("   Using closing price only (univariate)")
    data = prepare_for_temporal(df, feature_columns='Close')

    # Option B: Use multiple features (multivariate) - uncomment to use
    # print("   Using multiple features (multivariate)")
    # df_with_indicators = add_technical_indicators(df)
    # df_with_indicators = df_with_indicators.dropna()  # Remove NaN from indicators
    # data = prepare_for_temporal(
    #     df_with_indicators,
    #     feature_columns=['Close', 'Volume', 'SMA_7', 'RSI_14']
    # )

    print(f"   Data shape: {data.shape}")
    print(f"   Features: {data.shape[1]}")
    print(f"   Time steps: {data.shape[0]}")

    # Step 3: Normalize data
    print(f"\n3. Normalizing data...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    print(f"   ✓ Applied StandardScaler")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Std: {scaler.scale_}")

    # Step 4: Split data
    print(f"\n4. Splitting data...")
    train_data, val_data, test_data = split_train_val_test(
        data_normalized,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")

    # Step 5: Create datasets
    print(f"\n5. Creating datasets...")
    print(f"   Lookback: {LOOKBACK} days")
    print(f"   Forecast horizon: {FORECAST_HORIZON} days")

    train_dataset = TimeSeriesDataset(
        train_data,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        stride=1
    )
    val_dataset = TimeSeriesDataset(
        val_data,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        stride=1
    )
    test_dataset = TimeSeriesDataset(
        test_data,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        stride=1
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   Train sequences: {len(train_dataset)}")
    print(f"   Val sequences:   {len(val_dataset)}")
    print(f"   Test sequences:  {len(test_dataset)}")

    # Step 6: Create model
    print(f"\n6. Creating Temporal model...")
    model = Temporal(
        input_dim=data.shape[1],
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        d_ff=1024,
        forecast_horizon=FORECAST_HORIZON,
        dropout=0.1
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 7: Train model
    print(f"\n7. Training model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_clip=1.0
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        early_stopping_patience=10,
        save_path=f"{TICKER}_best_model.pt"
    )

    print(f"\n   Training completed!")
    print(f"   Best validation loss: {min(history['val_losses']):.6f}")

    # Step 8: Evaluate on test set
    print(f"\n8. Evaluating on test set...")
    test_predictions = []
    test_actuals = []

    model.eval()
    with torch.no_grad():
        for src, decoder_input, target_output in test_loader:
            src = src.to(trainer.device)
            forecast = model.forecast(src)
            test_predictions.append(forecast.cpu().numpy())
            test_actuals.append(target_output.numpy())

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_actuals = np.concatenate(test_actuals, axis=0)

    # Denormalize for evaluation
    test_pred_original = scaler.inverse_transform(
        test_predictions.reshape(-1, data.shape[1])
    ).reshape(test_predictions.shape)

    test_actual_original = scaler.inverse_transform(
        test_actuals.reshape(-1, data.shape[1])
    ).reshape(test_actuals.shape)

    # Calculate metrics
    mse = np.mean((test_pred_original - test_actual_original) ** 2)
    mae = np.mean(np.abs(test_pred_original - test_actual_original))
    rmse = np.sqrt(mse)

    # Calculate MAPE (avoiding division by zero)
    mask = test_actual_original != 0
    mape = np.mean(np.abs((test_actual_original[mask] - test_pred_original[mask]) /
                          test_actual_original[mask])) * 100

    print(f"\n   Test Set Metrics:")
    print(f"   MSE:  ${mse:.2f}")
    print(f"   MAE:  ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")

    # Step 9: Visualize results
    print(f"\n9. Creating visualizations...")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss', alpha=0.7)
    plt.plot(history['val_losses'], label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot sample predictions
    plt.subplot(1, 2, 2)
    sample_idx = 0
    historical_days = 30  # Show last 30 days of history

    # Get one test sample
    src, decoder_input, target_output = test_dataset[sample_idx]

    # Denormalize
    src_original = scaler.inverse_transform(src.numpy())
    pred_sample = test_pred_original[sample_idx, :, 0]
    actual_sample = test_actual_original[sample_idx, :, 0]

    # Plot
    hist_x = np.arange(-historical_days, 0)
    hist_y = src_original[-historical_days:, 0]
    pred_x = np.arange(0, FORECAST_HORIZON)

    plt.plot(hist_x, hist_y, 'b-', label='Historical', linewidth=2)
    plt.plot(pred_x, actual_sample, 'g-', label='Actual Future', linewidth=2)
    plt.plot(pred_x, pred_sample, 'r--', label='Predicted', linewidth=2)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    plt.xlabel('Days')
    plt.ylabel(f'{TICKER} Stock Price ($)')
    plt.title(f'{TICKER} {FORECAST_HORIZON}-Day Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{TICKER}_forecast_results.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to '{TICKER}_forecast_results.png'")

    # Step 10: Make future prediction
    print(f"\n10. Making future prediction...")
    latest_data = data_normalized[-LOOKBACK:]
    latest_tensor = torch.FloatTensor(latest_data).unsqueeze(0).to(trainer.device)

    with torch.no_grad():
        future_forecast = model.forecast(latest_tensor)

    future_forecast_original = scaler.inverse_transform(
        future_forecast.cpu().numpy().reshape(-1, data.shape[1])
    ).reshape(future_forecast.shape)

    print(f"\n   Next {FORECAST_HORIZON}-day forecast for {TICKER}:")
    for i in range(FORECAST_HORIZON):
        print(f"   Day {i+1}: ${future_forecast_original[0, i, 0]:.2f}")

    print("\n" + "=" * 70)
    print("FORECASTING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel saved: {TICKER}_best_model.pt")
    print(f"Visualization: {TICKER}_forecast_results.png")
    print(f"\nYou can now:")
    print(f"  1. Load the saved model for future predictions")
    print(f"  2. Try different tickers by changing TICKER variable")
    print(f"  3. Experiment with different lookback windows")
    print(f"  4. Add technical indicators for multivariate forecasting")


if __name__ == "__main__":
    main()
