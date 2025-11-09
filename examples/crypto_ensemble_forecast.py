"""
Ensemble Cryptocurrency Forecasting

Uses multiple models with different perspectives to provide balanced predictions:
- Short-term momentum model (30-day lookback)
- Medium-term balanced model (60-day lookback)
- Long-term trend model (90-day lookback)
- Momentum-focused features
- Mean-reversion focused features

Aggregates predictions to show realistic scenarios.
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


def add_technical_indicators(df, focus='balanced'):
    """
    Add technical indicators with different focus.

    Args:
        df: DataFrame with OHLCV data
        focus: 'momentum', 'mean_reversion', or 'balanced'
    """
    df = df.copy()

    # Base indicators (always included)
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Volatility
    df['Volatility_7'] = df['Returns'].rolling(window=7).std()
    df['Volatility_21'] = df['Returns'].rolling(window=21).std()

    # Price position relative to MAs (mean reversion signals)
    df['Price_to_MA7'] = (df['Close'] - df['MA_7']) / df['MA_7']
    df['Price_to_MA21'] = (df['Close'] - df['MA_21']) / df['MA_21']
    df['Price_to_MA50'] = (df['Close'] - df['MA_50']) / df['MA_50']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Deviation'] = df['RSI'] - 50  # Deviation from neutral

    if focus == 'momentum':
        # Momentum-focused features
        df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
        df['Momentum_14'] = df['Close'] - df['Close'].shift(14)
        df['Momentum_21'] = df['Close'] - df['Close'].shift(21)
        df['Price_Acceleration'] = df['Returns'].diff()

        feature_columns = ['Close', 'Open', 'High', 'Low', 'Volume',
                          'Returns', 'MA_7', 'MA_21', 'Volatility_7',
                          'RSI', 'Momentum_7', 'Momentum_14', 'Momentum_21',
                          'Price_Acceleration']

    elif focus == 'mean_reversion':
        # Mean reversion focused features
        df['BB_Upper'] = df['MA_21'] + 2 * df['Volatility_21'] * df['Close']
        df['BB_Lower'] = df['MA_21'] - 2 * df['Volatility_21'] * df['Close']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # How stretched is price from mean?
        df['Price_Stretch'] = (df['Close'] - df['MA_21']).abs() / df['Volatility_21'] / df['Close']

        feature_columns = ['Close', 'Open', 'High', 'Low', 'Volume',
                          'Returns', 'MA_7', 'MA_21', 'MA_50',
                          'Price_to_MA7', 'Price_to_MA21', 'Price_to_MA50',
                          'RSI', 'RSI_Deviation', 'BB_Position', 'Price_Stretch']

    else:  # balanced
        df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
        df['Momentum_21'] = df['Close'] - df['Close'].shift(21)
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

        feature_columns = ['Close', 'Open', 'High', 'Low', 'Volume',
                          'Returns', 'Log_Returns', 'MA_7', 'MA_21', 'MA_50',
                          'Volatility_7', 'Volatility_21', 'RSI',
                          'Momentum_7', 'Momentum_21', 'Volume_Ratio', 'Price_Range']

    df = df.dropna()
    return df, feature_columns


def train_ensemble_model(symbol, period, lookback, forecast_horizon, epochs, focus, model_name, interval='1d',
                        use_cache=True, max_cache_age_hours=6.0, fine_tune_epochs=3):
    """
    Train a single model for the ensemble with optional caching and fine-tuning.

    Args:
        symbol: Trading symbol
        period: Historical data period
        lookback: Lookback window size
        forecast_horizon: Forecast horizon
        epochs: Number of training epochs for fresh training
        focus: Model focus type
        model_name: Display name
        interval: Data interval
        use_cache: Whether to use model caching (default: True)
        max_cache_age_hours: Maximum cache age before retraining (default: 6.0)
        fine_tune_epochs: Number of epochs for fine-tuning cached models (default: 3)
    """
    from strategies.model_cache import get_model_cache
    from datetime import datetime

    interval_label = "hours" if interval == '1h' else "days"
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"  Lookback: {lookback} {interval_label} | Focus: {focus}")
    print(f"{'='*70}")

    # Fetch data
    df = fetch_crypto_data(symbol, period=period, interval=interval)
    print(f"✓ Fetched {len(df)} data points ({interval} interval)")

    # Get latest data timestamp for cache staleness check
    latest_data_timestamp = df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.fromisoformat(str(df.index[-1]))

    # Add indicators
    df, feature_columns = add_technical_indicators(df, focus=focus)
    data = prepare_for_temporal(df, feature_columns=feature_columns)
    print(f"✓ Prepared {data.shape[0]} samples with {data.shape[1]} features")

    # Normalize
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Split
    train_data, val_data, test_data = split_train_val_test(data_normalized)

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, lookback, forecast_horizon)
    val_dataset = TimeSeriesDataset(val_data, lookback, forecast_horizon)

    # Adaptive batch size based on interval and dataset size
    # Daily data: fewer samples (~1774 for 5y) → batch size (128)
    # Hourly data: MASSIVE samples (2.6M+!) → very small batch size (256)
    # Ultra-conservative to handle huge hourly datasets with 4+ parallel processes
    if interval == '1d':
        batch_size = 128  # Daily intervals
    else:
        batch_size = 256  # Hourly intervals - reduced due to massive dataset size

    # Ensure batch size doesn't exceed dataset size
    max_batch_size = min(batch_size, len(train_dataset) // 2)  # At least 2 batches
    batch_size = max(32, max_batch_size)  # Minimum batch size of 32

    # Optimized DataLoader settings for faster training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # Parallel data loading - increased for better CPU utilization
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Match training batch size
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Create model template - optimized for RTX 4090 (24GB VRAM)
    model = Temporal(
        input_dim=data.shape[1],
        d_model=512,  # Increased from 256 for better model capacity
        num_encoder_layers=6,  # Increased from 4 for deeper network
        num_decoder_layers=6,  # Increased from 4 for deeper network
        num_heads=8,
        d_ff=2048,  # Increased from 1024 for wider feedforward network
        forecast_horizon=forecast_horizon,
        dropout=0.1
    )

    # Check cache if enabled
    model_cache = get_model_cache()
    cached_result = None
    use_cached = False

    if use_cache:
        is_stale = model_cache.is_stale(
            symbol, interval, lookback, focus, forecast_horizon,
            max_age_hours=max_cache_age_hours,
            latest_data_timestamp=latest_data_timestamp
        )

        if not is_stale:
            # Try to load cached model
            cached_result = model_cache.load(model, symbol, interval, lookback, focus, forecast_horizon)

            if cached_result is not None:
                model, cached_scaler, metadata = cached_result
                use_cached = True

                # Check if feature columns match
                if metadata['feature_columns'] == feature_columns:
                    scaler = cached_scaler
                    print(f"🔄 Using cached model (age: {(datetime.now() - metadata['training_timestamp']).total_seconds() / 3600:.1f}h)")
                else:
                    print(f"⚠️  Feature mismatch - retraining from scratch")
                    use_cached = False

    # Train or fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_clip=1.0,
        use_amp=True  # Enable mixed precision training for 1.5-2x speedup
    )

    if use_cached:
        # Fine-tune with reduced epochs and lower learning rate
        print(f"🔧 Fine-tuning cached model ({fine_tune_epochs} epochs)...")
        optimizer.param_groups[0]['lr'] = 1e-5  # Lower learning rate for fine-tuning

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=fine_tune_epochs,
            early_stopping_patience=2,
            save_path=None
        )
        training_type = "fine-tuned"
        total_epochs = metadata.get('training_epochs', 0) + fine_tune_epochs
    else:
        # Full training from scratch
        print(f"🏋️  Training from scratch ({epochs} epochs)...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            early_stopping_patience=5,
            save_path=None
        )
        training_type = "fresh"
        total_epochs = epochs

    best_val_loss = min(history['val_losses'])
    print(f"✓ Best validation loss: {best_val_loss:.6f} ({training_type})")

    # Clear GPU cache to free memory for next model (helps with parallel training)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save to cache if enabled
    if use_cache:
        model_cache.save(
            model=model,
            scaler=scaler,
            symbol=symbol,
            interval=interval,
            lookback=lookback,
            focus=focus,
            forecast_horizon=forecast_horizon,
            feature_columns=feature_columns,
            data_end_timestamp=latest_data_timestamp,
            training_epochs=total_epochs,
            best_val_loss=best_val_loss
        )

    return {
        'model': model,
        'scaler': scaler,
        'trainer': trainer,
        'feature_columns': feature_columns,
        'lookback': lookback,
        'focus': focus,
        'name': model_name,
        'df_original': df,
        'cached': use_cached,
        'training_type': training_type
    }


def make_ensemble_predictions(ensemble_models, symbol, forecast_horizon=7, interval='1d'):
    """Make predictions using all models in the ensemble."""
    print(f"\n{'='*70}")
    print("GENERATING ENSEMBLE PREDICTIONS")
    print(f"{'='*70}")

    # Auto-detect available data period from cache for this specific interval
    from strategies.data_cache import get_cache
    cache = get_cache()

    # For daily intervals, prefer shorter periods to avoid learning from stale historical data
    # For hourly intervals, longer periods are fine (5y hourly is more recent than 5y daily)
    # Crypto markets evolve quickly - old price levels aren't predictive
    if interval == '1d':
        preferred_periods = ['2y', '1y', '6mo', '5y']  # Prefer 2y for daily
    else:  # '1h' or other short intervals
        preferred_periods = ['5y', '2y', '1y', '6mo']  # Prefer 5y for hourly

    available_period = None
    for test_period in preferred_periods:
        cached_data = cache.get(symbol, test_period, interval=interval)
        if cached_data is not None and not cached_data.empty:
            available_period = test_period
            break

    # Default based on interval if nothing found
    period = available_period or ('2y' if interval == '1d' else '5y')

    is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
    asset_type = 'crypto' if is_crypto else 'stock'

    if available_period:
        print(f"📊 Using {period} of available cached data for {asset_type} predictions (interval: {interval})")
    else:
        print(f"📊 Requesting {period} of historical data for {asset_type} predictions (interval: {interval})")

    all_predictions = []
    prediction_details = []

    for model_info in ensemble_models:
        # Fetch latest data - use period to match training data and enable saving full history
        df_latest = fetch_crypto_data(symbol, period=period, interval=interval)
        df_latest, _ = add_technical_indicators(df_latest, focus=model_info['focus'])

        # Prepare features
        data_latest = prepare_for_temporal(df_latest, model_info['feature_columns'])
        data_latest_norm = model_info['scaler'].transform(data_latest)

        # Get input window
        lookback = model_info['lookback']
        latest = data_latest_norm[-lookback:]
        latest_tensor = torch.FloatTensor(latest).unsqueeze(0).to(model_info['trainer'].device)

        # Predict
        model_info['model'].eval()
        with torch.no_grad():
            forecast = model_info['model'].forecast(latest_tensor)

        # Extract Close price (first feature)
        forecast_full = np.zeros((forecast_horizon, len(model_info['feature_columns'])))
        forecast_full[:, 0] = forecast.cpu().numpy()[0, :, 0]
        forecast_prices = model_info['scaler'].inverse_transform(forecast_full)[:, 0]

        all_predictions.append(forecast_prices)

        # Store details
        current_price = df_latest['Close'].iloc[-1]
        final_change = ((forecast_prices[-1] - current_price) / current_price) * 100

        prediction_details.append({
            'name': model_info['name'],
            'prices': forecast_prices,
            'final_change': final_change,
            'focus': model_info['focus']
        })

        print(f"✓ {model_info['name']}: Day 7 = ${forecast_prices[-1]:,.2f} ({final_change:+.1f}%)")

    all_predictions = np.array(all_predictions)

    # Calculate ensemble statistics
    ensemble_stats = {
        'mean': np.mean(all_predictions, axis=0),
        'median': np.median(all_predictions, axis=0),
        'std': np.std(all_predictions, axis=0),
        'min': np.min(all_predictions, axis=0),
        'max': np.max(all_predictions, axis=0),
        'q25': np.percentile(all_predictions, 25, axis=0),
        'q75': np.percentile(all_predictions, 75, axis=0),
        'all_predictions': all_predictions,
        'details': prediction_details
    }

    return ensemble_stats, df_latest


def visualize_ensemble(ensemble_stats, df_latest, symbol='BTC-USD'):
    """Create comprehensive visualization of ensemble predictions."""
    current_price = df_latest['Close'].iloc[-1]
    forecast_days = len(ensemble_stats['mean'])

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Historical prices with ensemble forecast
    ax1 = fig.add_subplot(gs[0, :])
    hist_days = 60
    hist_prices = df_latest['Close'].iloc[-hist_days:].values
    hist_x = np.arange(-hist_days, 0)
    forecast_x = np.arange(0, forecast_days)

    ax1.plot(hist_x, hist_prices, 'b-', linewidth=2, label='Historical', alpha=0.8)
    ax1.plot(forecast_x, ensemble_stats['median'], 'r-', linewidth=3,
             label='Ensemble Median', marker='o', markersize=8)
    ax1.fill_between(forecast_x, ensemble_stats['q25'], ensemble_stats['q75'],
                     alpha=0.3, color='orange', label='25th-75th Percentile (Likely Range)')
    ax1.fill_between(forecast_x, ensemble_stats['min'], ensemble_stats['max'],
                     alpha=0.15, color='red', label='Min-Max (Possible Range)')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axhline(y=current_price, color='green', linestyle=':', alpha=0.7,
                linewidth=2, label=f'Current: ${current_price:,.0f}')
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{symbol} Ensemble Forecast - Multiple Model Perspectives', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. All individual model predictions
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(np.arange(-30, 0), df_latest['Close'].iloc[-30:].values,
             'b-', linewidth=2, label='Historical', alpha=0.6)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ensemble_stats['details'])))
    for i, detail in enumerate(ensemble_stats['details']):
        ax2.plot(forecast_x, detail['prices'], '--', linewidth=2,
                color=colors[i], alpha=0.7, marker='o', markersize=4,
                label=f"{detail['name']}: {detail['final_change']:+.1f}%")
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=current_price, color='green', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Days', fontsize=10)
    ax2.set_ylabel('Price ($)', fontsize=10)
    ax2.set_title('Individual Model Forecasts', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Day 7 prediction distribution
    ax3 = fig.add_subplot(gs[1, 1])
    day7_predictions = ensemble_stats['all_predictions'][:, -1]
    ax3.hist(day7_predictions, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.axvline(current_price, color='green', linestyle='--', linewidth=2, label=f'Current: ${current_price:,.0f}')
    ax3.axvline(ensemble_stats['median'][-1], color='red', linestyle='-', linewidth=3,
               label=f'Median: ${ensemble_stats["median"][-1]:,.0f}')
    ax3.axvline(ensemble_stats['mean'][-1], color='orange', linestyle=':', linewidth=2,
               label=f'Mean: ${ensemble_stats["mean"][-1]:,.0f}')
    ax3.set_xlabel('Day 7 Price ($)', fontsize=10)
    ax3.set_ylabel('Number of Models', fontsize=10)
    ax3.set_title('Day 7 Prediction Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Prediction uncertainty over time
    ax4 = fig.add_subplot(gs[1, 2])
    uncertainty_pct = (ensemble_stats['std'] / ensemble_stats['median']) * 100
    ax4.plot(forecast_x, uncertainty_pct, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.fill_between(forecast_x, 0, uncertainty_pct, alpha=0.3, color='purple')
    ax4.set_xlabel('Days Ahead', fontsize=10)
    ax4.set_ylabel('Uncertainty (%)', fontsize=10)
    ax4.set_title('Forecast Uncertainty Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Scenario analysis table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    scenarios = [
        ['Scenario', 'Day 1', 'Day 3', 'Day 7', 'Change from Current'],
        ['Best Case (Max)',
         f'${ensemble_stats["max"][0]:,.0f}',
         f'${ensemble_stats["max"][2]:,.0f}',
         f'${ensemble_stats["max"][-1]:,.0f}',
         f'{((ensemble_stats["max"][-1]-current_price)/current_price*100):+.1f}%'],
        ['Optimistic (75th %ile)',
         f'${ensemble_stats["q75"][0]:,.0f}',
         f'${ensemble_stats["q75"][2]:,.0f}',
         f'${ensemble_stats["q75"][-1]:,.0f}',
         f'{((ensemble_stats["q75"][-1]-current_price)/current_price*100):+.1f}%'],
        ['Most Likely (Median)',
         f'${ensemble_stats["median"][0]:,.0f}',
         f'${ensemble_stats["median"][2]:,.0f}',
         f'${ensemble_stats["median"][-1]:,.0f}',
         f'{((ensemble_stats["median"][-1]-current_price)/current_price*100):+.1f}%'],
        ['Pessimistic (25th %ile)',
         f'${ensemble_stats["q25"][0]:,.0f}',
         f'${ensemble_stats["q25"][2]:,.0f}',
         f'${ensemble_stats["q25"][-1]:,.0f}',
         f'{((ensemble_stats["q25"][-1]-current_price)/current_price*100):+.1f}%'],
        ['Worst Case (Min)',
         f'${ensemble_stats["min"][0]:,.0f}',
         f'${ensemble_stats["min"][2]:,.0f}',
         f'${ensemble_stats["min"][-1]:,.0f}',
         f'{((ensemble_stats["min"][-1]-current_price)/current_price*100):+.1f}%'],
    ]

    table = ax5.table(cellText=scenarios, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows
    colors_rows = ['#90EE90', '#B0E57C', '#FFFFCC', '#FFB366', '#FF6B6B']
    for i, color in enumerate(colors_rows, start=1):
        for j in range(5):
            table[(i, j)].set_facecolor(color)

    ax5.set_title('Scenario Analysis Summary', fontsize=14, fontweight='bold', pad=20)

    plt.savefig('bitcoin_ensemble_forecast.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to 'bitcoin_ensemble_forecast.png'")


def main():
    print("="*70)
    print("ENSEMBLE CRYPTOCURRENCY FORECASTING")
    print("Multiple Models, Multiple Perspectives, Realistic Predictions")
    print("="*70)

    symbol = 'BTC-USD'
    forecast_horizon = 7

    # Define ensemble models (optimized epochs for faster training)
    ensemble_configs = [
        {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Short-term Momentum'},
        {'lookback': 60, 'focus': 'balanced', 'epochs': 15, 'name': 'Medium-term Balanced'},
        {'lookback': 90, 'focus': 'balanced', 'epochs': 15, 'name': 'Long-term Trend'},
        {'lookback': 60, 'focus': 'mean_reversion', 'epochs': 15, 'name': 'Mean Reversion'},
        {'lookback': 45, 'focus': 'momentum', 'epochs': 15, 'name': 'Mid-term Momentum'},
    ]

    # Train ensemble
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE MODELS")
    print("="*70)

    ensemble_models = []
    for config in ensemble_configs:
        model_info = train_ensemble_model(
            symbol=symbol,
            period='2y',
            lookback=config['lookback'],
            forecast_horizon=forecast_horizon,
            epochs=config['epochs'],
            focus=config['focus'],
            model_name=config['name']
        )
        ensemble_models.append(model_info)

    # Make predictions
    ensemble_stats, df_latest = make_ensemble_predictions(ensemble_models, symbol, forecast_horizon)

    # Print summary
    current_price = df_latest['Close'].iloc[-1]
    print(f"\n{'='*70}")
    print("ENSEMBLE FORECAST SUMMARY")
    print(f"{'='*70}")
    print(f"\nCurrent {symbol} Price: ${current_price:,.2f}")
    print(f"\n7-Day Forecast Scenarios:")
    print(f"  Best Case (Max):         ${ensemble_stats['max'][-1]:,.2f} ({((ensemble_stats['max'][-1]-current_price)/current_price*100):+.1f}%)")
    print(f"  Optimistic (75th %ile):  ${ensemble_stats['q75'][-1]:,.2f} ({((ensemble_stats['q75'][-1]-current_price)/current_price*100):+.1f}%)")
    print(f"  Most Likely (Median):    ${ensemble_stats['median'][-1]:,.2f} ({((ensemble_stats['median'][-1]-current_price)/current_price*100):+.1f}%)")
    print(f"  Pessimistic (25th %ile): ${ensemble_stats['q25'][-1]:,.2f} ({((ensemble_stats['q25'][-1]-current_price)/current_price*100):+.1f}%)")
    print(f"  Worst Case (Min):        ${ensemble_stats['min'][-1]:,.2f} ({((ensemble_stats['min'][-1]-current_price)/current_price*100):+.1f}%)")

    prob_above = np.mean(ensemble_stats['all_predictions'][:, -1] > current_price) * 100
    prob_below = np.mean(ensemble_stats['all_predictions'][:, -1] < current_price) * 100

    print(f"\nModel Consensus:")
    print(f"  {int(prob_above)}% of models predict price will be ABOVE ${current_price:,.0f}")
    print(f"  {int(prob_below)}% of models predict price will be BELOW ${current_price:,.0f}")

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATION")
    print(f"{'='*70}")
    visualize_ensemble(ensemble_stats, df_latest, symbol)

    print(f"\n{'='*70}")
    print("ENSEMBLE FORECASTING COMPLETE!")
    print(f"{'='*70}")
    print("\nKey Insights:")
    print("✓ Multiple models provide different perspectives")
    print("✓ Median forecast is more reliable than any single model")
    print("✓ Range between 25th-75th percentile shows likely outcomes")
    print("✓ Use this information to make informed decisions, not guarantees")


if __name__ == "__main__":
    main()
