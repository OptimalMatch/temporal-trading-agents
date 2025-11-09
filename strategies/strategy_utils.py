"""
Strategy Utilities

Common utilities and helper functions for trading strategies.
Provides shared functionality for training models, analyzing forecasts,
and calculating risk metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sys
import importlib.util


def load_ensemble_module(module_path: str = "../examples/crypto_ensemble_forecast.py"):
    """Load the ensemble forecasting module dynamically."""
    # Patch temporal data fetch to use Polygon.io before loading ensemble
    from strategies.cached_data_fetch import patch_temporal_data_fetch
    patch_temporal_data_fetch()

    spec = importlib.util.spec_from_file_location("ensemble", module_path)
    ensemble = importlib.util.module_from_spec(spec)
    sys.modules["ensemble"] = ensemble
    spec.loader.exec_module(ensemble)
    return ensemble


def train_ensemble(symbol: str, forecast_horizon: int, configs: List[Dict],
                   name: str, ensemble_module, interval: str = '1d',
                   use_cache: bool = True, max_cache_age_hours: float = 6.0,
                   fine_tune_epochs: int = 3) -> Tuple[Dict, pd.DataFrame]:
    """
    Train an ensemble of models for a specific forecast horizon.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD')
        forecast_horizon: Number of periods to forecast
        configs: List of model configurations
        name: Display name for this ensemble
        ensemble_module: The loaded ensemble module
        interval: Data interval ('1d' for daily, '1h' for hourly)
        use_cache: Whether to use model caching (default: True)
        max_cache_age_hours: Maximum cache age before retraining (default: 6.0)
        fine_tune_epochs: Number of epochs for fine-tuning cached models (default: 3)

    Returns:
        Tuple of (ensemble_stats, latest_dataframe)
    """
    interval_label = "hours" if interval == '1h' else "days"
    print(f"\n{'='*70}")
    print(f"TRAINING {name} ENSEMBLE ({forecast_horizon}-{interval_label[:-1]} forecast at {interval} interval)")
    print(f"{'='*70}")

    if use_cache:
        print(f"💾 Model caching enabled (max age: {max_cache_age_hours}h, fine-tune: {fine_tune_epochs} epochs)")
    else:
        print(f"🔄 Model caching disabled - training from scratch")

    # Determine period based on asset type
    # Stocks have 5 years of historical data, crypto has 2 years
    is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
    period = '2y' if is_crypto else '5y'
    print(f"📊 Using {period} of historical data for {'crypto' if is_crypto else 'stock'} analysis (interval: {interval})")

    ensemble_models = []
    for config in configs:
        model_info = ensemble_module.train_ensemble_model(
            symbol=symbol,
            period=period,
            lookback=config['lookback'],
            forecast_horizon=forecast_horizon,
            epochs=config['epochs'],
            focus=config['focus'],
            model_name=config['name'],
            interval=interval,
            use_cache=use_cache,
            max_cache_age_hours=max_cache_age_hours,
            fine_tune_epochs=fine_tune_epochs
        )
        ensemble_models.append(model_info)

    # Make predictions
    ensemble_stats, df_latest = ensemble_module.make_ensemble_predictions(
        ensemble_models, symbol, forecast_horizon, interval=interval
    )

    return ensemble_stats, df_latest


def calculate_forecast_metrics(stats: Dict, current_price: float,
                               horizon_days: int) -> Dict:
    """
    Calculate key metrics from forecast statistics.

    Args:
        stats: Forecast statistics dictionary
        current_price: Current asset price
        horizon_days: Forecast horizon in days

    Returns:
        Dictionary of calculated metrics
    """
    median = stats['median'][-1]
    q25 = stats['q25'][-1]
    q75 = stats['q75'][-1]
    min_price = stats['min'][-1]
    max_price = stats['max'][-1]

    # Calculate percentage changes
    median_change_pct = ((median - current_price) / current_price) * 100
    q25_change_pct = ((q25 - current_price) / current_price) * 100
    q75_change_pct = ((q75 - current_price) / current_price) * 100

    # Calculate volatility/uncertainty
    forecast_range = q75 - q25
    forecast_range_pct = (forecast_range / median) * 100

    # Model confidence
    prob_above = np.mean(stats['all_predictions'][:, -1] > current_price) * 100

    # Risk metrics
    worst_case_loss = ((min_price - current_price) / current_price) * 100
    best_case_gain = ((max_price - current_price) / current_price) * 100

    return {
        'horizon_days': horizon_days,
        'median': median,
        'q25': q25,
        'q75': q75,
        'min': min_price,
        'max': max_price,
        'median_change_pct': median_change_pct,
        'q25_change_pct': q25_change_pct,
        'q75_change_pct': q75_change_pct,
        'forecast_range': forecast_range,
        'forecast_range_pct': forecast_range_pct,
        'prob_above': prob_above,
        'worst_case_loss': worst_case_loss,
        'best_case_gain': best_case_gain,
    }


def calculate_position_size(forecast_range_pct: float, base_size: float = 1.0,
                           min_size: float = 0.1, max_size: float = 2.0) -> float:
    """
    Calculate position size based on forecast uncertainty.

    Args:
        forecast_range_pct: Forecast range as percentage of median price
        base_size: Base position size (default 1.0 = 100%)
        min_size: Minimum position size multiplier
        max_size: Maximum position size multiplier

    Returns:
        Position size multiplier
    """
    # Lower uncertainty = larger position
    # Normalize forecast_range_pct (typical range: 2-20%)
    if forecast_range_pct < 5:
        multiplier = max_size  # High confidence
    elif forecast_range_pct < 10:
        multiplier = base_size + 0.5  # Above average confidence
    elif forecast_range_pct < 15:
        multiplier = base_size  # Average confidence
    elif forecast_range_pct < 20:
        multiplier = base_size - 0.3  # Below average confidence
    else:
        multiplier = min_size  # Low confidence

    return np.clip(multiplier, min_size, max_size)


def calculate_stop_loss(entry_price: float, worst_case_price: float,
                       cushion_pct: float = 2.0) -> float:
    """
    Calculate stop loss price with cushion below worst case forecast.

    Args:
        entry_price: Expected entry price
        worst_case_price: Worst case forecast price
        cushion_pct: Additional cushion below worst case (default 2%)

    Returns:
        Stop loss price
    """
    worst_case_loss_pct = ((worst_case_price - entry_price) / entry_price) * 100
    stop_loss_pct = worst_case_loss_pct - cushion_pct
    stop_loss = entry_price * (1 + stop_loss_pct / 100)
    return stop_loss


def calculate_risk_reward_ratio(entry_price: float, target_price: float,
                               stop_loss: float) -> float:
    """
    Calculate risk/reward ratio.

    Args:
        entry_price: Entry price
        target_price: Target exit price
        stop_loss: Stop loss price

    Returns:
        Risk/reward ratio
    """
    potential_reward = target_price - entry_price
    potential_risk = entry_price - stop_loss

    if potential_risk <= 0:
        return float('inf')

    return potential_reward / potential_risk


def analyze_forecast_gradient(stats: Dict, threshold: float = 0.5) -> Dict:
    """
    Analyze the shape/gradient of the forecast curve.

    Args:
        stats: Forecast statistics dictionary
        threshold: Threshold for detecting significant changes (% per day)

    Returns:
        Dictionary with gradient analysis
    """
    median = stats['median']
    days = len(median)

    # Calculate daily returns
    daily_returns = np.diff(median) / median[:-1] * 100

    # Calculate gradients (rate of change)
    gradients = []
    for i in range(len(daily_returns) - 1):
        grad = daily_returns[i+1] - daily_returns[i]
        gradients.append(grad)

    # Identify curve shape
    first_half_avg = np.mean(daily_returns[:len(daily_returns)//2])
    second_half_avg = np.mean(daily_returns[len(daily_returns)//2:])

    # Determine shape pattern
    if first_half_avg > threshold and second_half_avg < 0:
        shape = "INVERTED_U"  # Rise then fall
        description = "Quick rise followed by decline"
    elif first_half_avg < -threshold and second_half_avg > threshold:
        shape = "U_SHAPED"  # Dip then recovery
        description = "Dip followed by recovery"
    elif first_half_avg > threshold and second_half_avg > first_half_avg * 0.7:
        shape = "STEEP_RISE"  # Steep sustained rise
        description = "Steep sustained upward trend"
    elif first_half_avg > 0 and second_half_avg > 0 and second_half_avg < first_half_avg * 0.5:
        shape = "GRADUAL_RISE"  # Gradual rise that slows
        description = "Gradual rise with deceleration"
    elif first_half_avg < 0 and second_half_avg < 0:
        shape = "DECLINE"  # Sustained decline
        description = "Sustained downward trend"
    else:
        shape = "FLAT"  # Relatively flat
        description = "Relatively flat or uncertain"

    # Find peak/trough days
    peak_day = np.argmax(median)
    trough_day = np.argmin(median)

    # Volatility of daily returns
    return_volatility = np.std(daily_returns)

    return {
        'shape': shape,
        'description': description,
        'daily_returns': daily_returns,
        'first_half_avg': first_half_avg,
        'second_half_avg': second_half_avg,
        'peak_day': peak_day,
        'peak_value': median[peak_day],
        'trough_day': trough_day,
        'trough_value': median[trough_day],
        'return_volatility': return_volatility,
    }


def calculate_ensemble_confidence(stats: Dict, current_price: float) -> Dict:
    """
    Calculate confidence metrics from ensemble predictions.

    Args:
        stats: Forecast statistics with all_predictions
        current_price: Current asset price

    Returns:
        Dictionary with confidence metrics
    """
    all_predictions = stats['all_predictions']
    final_predictions = all_predictions[:, -1]

    # Percentage of models predicting above/below current
    pct_above = np.mean(final_predictions > current_price) * 100
    pct_below = 100 - pct_above

    # Agreement strength
    agreement = max(pct_above, pct_below)

    if agreement >= 80:
        confidence_level = "HIGH"
    elif agreement >= 65:
        confidence_level = "MEDIUM"
    elif agreement >= 55:
        confidence_level = "LOW"
    else:
        confidence_level = "VERY_LOW"

    # Standard deviation of final predictions
    prediction_std = np.std(final_predictions)
    prediction_cv = (prediction_std / np.mean(final_predictions)) * 100

    return {
        'pct_above': pct_above,
        'pct_below': pct_below,
        'agreement': agreement,
        'confidence_level': confidence_level,
        'prediction_std': prediction_std,
        'prediction_cv': prediction_cv,
    }


def format_strategy_output(strategy_name: str, signal: str,
                          metrics: Dict, details: Dict) -> str:
    """
    Format strategy output in a consistent way.

    Args:
        strategy_name: Name of the strategy
        signal: Trading signal (BUY, SELL, HOLD, WAIT)
        metrics: Key metrics dictionary
        details: Additional details dictionary

    Returns:
        Formatted string output
    """
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"STRATEGY: {strategy_name}")
    output.append(f"{'='*70}")
    output.append(f"\nSIGNAL: {signal}")
    output.append(f"\n{'='*70}")

    if metrics:
        output.append("\nKey Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                output.append(f"  {key}: {value:.2f}")
            else:
                output.append(f"  {key}: {value}")

    if details:
        output.append("\nDetails:")
        for key, value in details.items():
            if isinstance(value, float):
                output.append(f"  {key}: {value:.2f}")
            else:
                output.append(f"  {key}: {value}")

    output.append(f"\n{'='*70}")

    return "\n".join(output)


def get_default_ensemble_configs(horizon: int) -> List[Dict]:
    """
    Get default ensemble configurations for a given horizon.

    Args:
        horizon: Forecast horizon in days

    Returns:
        List of configuration dictionaries
    """
    if horizon <= 7:
        return [
            {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Short-term Momentum'},
            {'lookback': 60, 'focus': 'balanced', 'epochs': 15, 'name': 'Medium-term Balanced'},
            {'lookback': 90, 'focus': 'balanced', 'epochs': 15, 'name': 'Long-term Trend'},
            {'lookback': 60, 'focus': 'mean_reversion', 'epochs': 15, 'name': 'Mean Reversion'},
            {'lookback': 45, 'focus': 'momentum', 'epochs': 15, 'name': 'Mid-term Momentum'},
        ]
    elif horizon <= 14:
        return [
            {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Short-term Momentum'},
            {'lookback': 45, 'focus': 'balanced', 'epochs': 15, 'name': 'Medium-term Balanced'},
            {'lookback': 60, 'focus': 'balanced', 'epochs': 15, 'name': 'Long-term Trend'},
            {'lookback': 45, 'focus': 'mean_reversion', 'epochs': 15, 'name': 'Mean Reversion'},
            {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Mid-term Momentum'},
        ]
    else:  # 21-30 days
        return [
            {'lookback': 45, 'focus': 'momentum', 'epochs': 15, 'name': 'Short-term Momentum'},
            {'lookback': 60, 'focus': 'balanced', 'epochs': 15, 'name': 'Medium-term Balanced'},
            {'lookback': 90, 'focus': 'balanced', 'epochs': 15, 'name': 'Long-term Trend'},
            {'lookback': 60, 'focus': 'mean_reversion', 'epochs': 15, 'name': 'Mean Reversion'},
            {'lookback': 45, 'focus': 'momentum', 'epochs': 15, 'name': 'Mid-term Momentum'},
        ]
