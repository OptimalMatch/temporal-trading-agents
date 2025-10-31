"""
Improved Cryptocurrency Forecasting with Confidence Intervals

This version adds:
- Monte Carlo dropout for uncertainty estimation
- Confidence intervals on predictions
- Multiple prediction scenarios
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
    fetch_crypto_data,
    prepare_for_temporal,
    split_train_val_test
)


def add_technical_indicators(df):
    """Add technical indicators to dataframe."""
    df = df.copy()

    # Returns and log returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Volatility
    df['Volatility_7'] = df['Returns'].rolling(window=7).std()
    df['Volatility_21'] = df['Returns'].rolling(window=21).std()

    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Price momentum
    df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
    df['Momentum_21'] = df['Close'] - df['Close'].shift(21)

    # Volume indicators
    df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']

    # Price range
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

    # Drop NaN values
    df = df.dropna()

    return df


def predict_with_uncertainty(model, input_tensor, scaler, feature_columns, n_samples=50):
    """
    Make predictions with uncertainty estimation using Monte Carlo Dropout.

    Args:
        model: Trained Temporal model
        input_tensor: Input data tensor
        scaler: Fitted scaler
        feature_columns: List of feature column names
        n_samples: Number of Monte Carlo samples

    Returns:
        mean_forecast, std_forecast, all_forecasts
    """
    model.train()  # Enable dropout for MC sampling

    all_forecasts = []

    for _ in range(n_samples):
        with torch.no_grad():
            forecast = model.forecast(input_tensor)
            # Extract Close price (first feature)
            forecast_full = np.zeros((len(forecast[0]), len(feature_columns)))
            forecast_full[:, 0] = forecast.cpu().numpy()[0, :, 0]
            forecast_original = scaler.inverse_transform(forecast_full)[:, 0]
            all_forecasts.append(forecast_original)

    all_forecasts = np.array(all_forecasts)
    mean_forecast = np.mean(all_forecasts, axis=0)
    std_forecast = np.std(all_forecasts, axis=0)

    model.eval()  # Back to eval mode

    return mean_forecast, std_forecast, all_forecasts


def main():
    print("=" * 70)
    print("IMPROVED BITCOIN FORECASTING WITH UNCERTAINTY")
    print("=" * 70)

    # Train model
    print("\n1. Training model...")
    from crypto_forecasting import train_crypto_model

    model_info = train_crypto_model(
        symbol='BTC-USD',
        period='2y',
        lookback=60,
        forecast_horizon=7,
        epochs=50,
        use_all_features=True
    )

    model = model_info['model']
    scaler = model_info['scaler']
    trainer = model_info['trainer']
    feature_columns = model_info['feature_columns']

    # Prepare latest data
    print("\n2. Preparing forecast with uncertainty...")
    df_latest = fetch_crypto_data('BTC-USD', period='3mo')
    df_latest = add_technical_indicators(df_latest)
    data_latest = prepare_for_temporal(df_latest, feature_columns)
    data_latest_norm = scaler.transform(data_latest)

    latest = data_latest_norm[-60:]
    latest_tensor = torch.FloatTensor(latest).unsqueeze(0).to(trainer.device)

    # Get predictions with uncertainty
    mean_forecast, std_forecast, all_forecasts = predict_with_uncertainty(
        model, latest_tensor, scaler, feature_columns, n_samples=100
    )

    current_price = df_latest['Close'].iloc[-1]

    # Print results
    print(f"\n   Current BTC Price: ${current_price:,.2f}")
    print(f"\n   7-Day Forecast with 95% Confidence Intervals:")
    print("   " + "-" * 60)

    for i in range(len(mean_forecast)):
        lower_95 = mean_forecast[i] - 1.96 * std_forecast[i]
        upper_95 = mean_forecast[i] + 1.96 * std_forecast[i]
        change = ((mean_forecast[i] - current_price) / current_price) * 100
        uncertainty = (std_forecast[i] / mean_forecast[i]) * 100

        direction = "📈" if change > 0 else "📉"
        print(f"   Day {i+1}: ${mean_forecast[i]:,.2f} ({direction} {change:+.2f}%)")
        print(f"          95% CI: [${lower_95:,.2f}, ${upper_95:,.2f}]")
        print(f"          Uncertainty: ±{uncertainty:.1f}%")

    # Calculate scenario probabilities
    final_prices = all_forecasts[:, -1]
    prob_above_current = np.mean(final_prices > current_price) * 100
    prob_below_current = np.mean(final_prices < current_price) * 100

    print(f"\n   Scenario Analysis (Day 7):")
    print(f"   Probability price > ${current_price:,.0f}: {prob_above_current:.1f}%")
    print(f"   Probability price < ${current_price:,.0f}: {prob_below_current:.1f}%")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Historical prices
    hist_days = 60
    hist_prices = df_latest['Close'].iloc[-hist_days:].values
    hist_x = np.arange(-hist_days, 0)

    axes[0].plot(hist_x, hist_prices, 'b-', linewidth=2)
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('Bitcoin Price (Last 60 Days)')
    axes[0].grid(True, alpha=0.3)

    # Forecast with confidence intervals
    forecast_x = np.arange(0, 7)
    axes[1].plot(np.arange(-30, 0), df_latest['Close'].iloc[-30:].values,
                 'b-', label='Historical', linewidth=2)
    axes[1].plot(forecast_x, mean_forecast, 'r-', label='Mean Forecast',
                 linewidth=2, marker='o')
    axes[1].fill_between(forecast_x,
                         mean_forecast - 1.96*std_forecast,
                         mean_forecast + 1.96*std_forecast,
                         alpha=0.3, color='red', label='95% CI')
    axes[1].axvline(x=0, color='black', linestyle=':', alpha=0.5)
    axes[1].axhline(y=current_price, color='green', linestyle='--',
                    alpha=0.5, label=f'Current: ${current_price:,.0f}')
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('7-Day Forecast with Uncertainty')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Distribution of Day 7 predictions
    axes[2].hist(final_prices, bins=30, alpha=0.7, edgecolor='black')
    axes[2].axvline(current_price, color='green', linestyle='--',
                    linewidth=2, label=f'Current: ${current_price:,.0f}')
    axes[2].axvline(mean_forecast[-1], color='red', linestyle='-',
                    linewidth=2, label=f'Mean: ${mean_forecast[-1]:,.0f}')
    axes[2].set_xlabel('Price ($)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Day 7 Predictions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bitcoin_forecast_improved.png', dpi=150, bbox_inches='tight')
    print(f"\n   ✓ Saved visualization to 'bitcoin_forecast_improved.png'")

    print("\n" + "=" * 70)
    print("IMPROVED FORECASTING COMPLETED!")
    print("=" * 70)
    print("\nKey Insights:")
    print("- The confidence intervals show the range of likely outcomes")
    print("- High uncertainty reflects crypto's volatile nature")
    print("- Consider the probability distribution, not just the mean forecast")


if __name__ == "__main__":
    main()
