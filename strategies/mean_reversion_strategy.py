"""
Mean Reversion with Forecast Confirmation Strategy

Identifies overbought/oversold conditions and trades mean reversion ONLY when
the forecast confirms the expected return toward the mean.

Strategy Logic:
- Calculate moving averages (20-day, 50-day SMA)
- Identify significant deviations from mean (>5%, >10%, >15%)
- Forecast must show convergence back toward the mean
- Position size scales with deviation magnitude
- Exit when price returns to mean or forecast changes

Signals:
- OVERSOLD_REVERT: Price well below MA, forecast shows recovery
- OVERBOUGHT_REVERT: Price well above MA, forecast shows decline (short)
- FALSE_SIGNAL: Deviation present but forecast doesn't support reversion
- NO_SIGNAL: Price near mean, no opportunity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_utils import (
    load_ensemble_module,
    train_ensemble,
    calculate_forecast_metrics,
    calculate_stop_loss,
    calculate_risk_reward_ratio,
    get_default_ensemble_configs,
)
from strategies.strategy_cache import cached_strategy


def calculate_mean_reversion_metrics(df: pd.DataFrame, current_price: float) -> Dict:
    """
    Calculate mean reversion indicators.

    Args:
        df: DataFrame with price history
        current_price: Current asset price

    Returns:
        Dictionary with mean reversion metrics
    """
    # Handle both uppercase and lowercase column names
    close_col = 'Close' if 'Close' in df.columns else 'close'

    # Calculate moving averages
    sma_20 = df[close_col].rolling(window=20).mean().iloc[-1]
    sma_50 = df[close_col].rolling(window=50).mean().iloc[-1]

    # Calculate deviations
    deviation_20_pct = ((current_price - sma_20) / sma_20) * 100
    deviation_50_pct = ((current_price - sma_50) / sma_50) * 100

    # Calculate volatility (standard deviation)
    std_20 = df[close_col].rolling(window=20).std().iloc[-1]
    z_score_20 = (current_price - sma_20) / std_20 if std_20 > 0 else 0

    # Determine condition
    if abs(z_score_20) > 2:
        condition = "EXTREME"
    elif abs(z_score_20) > 1.5:
        condition = "SIGNIFICANT"
    elif abs(z_score_20) > 1:
        condition = "MODERATE"
    else:
        condition = "NORMAL"

    # Determine direction
    if z_score_20 < -1:
        direction = "OVERSOLD"
    elif z_score_20 > 1:
        direction = "OVERBOUGHT"
    else:
        direction = "NEUTRAL"

    return {
        'sma_20': sma_20,
        'sma_50': sma_50,
        'deviation_20_pct': deviation_20_pct,
        'deviation_50_pct': deviation_50_pct,
        'std_20': std_20,
        'z_score_20': z_score_20,
        'condition': condition,
        'direction': direction,
    }


def analyze_mean_reversion_strategy(stats_14day: Dict, df: pd.DataFrame,
                                    current_price: float) -> Dict:
    """
    Analyze mean reversion opportunity with forecast confirmation.

    Args:
        stats_14day: 14-day forecast statistics
        df: Price history DataFrame
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Calculate mean reversion metrics
    mr_metrics = calculate_mean_reversion_metrics(df, current_price)

    # Get forecast metrics
    forecast_metrics = calculate_forecast_metrics(stats_14day, current_price, 14)

    # Calculate target reversion price (mean)
    target_mean = mr_metrics['sma_20']
    forecast_median = forecast_metrics['median']

    # Check if forecast supports mean reversion
    # For oversold: forecast should be above current and moving toward mean
    # For overbought: forecast should be below current and moving toward mean

    deviation = mr_metrics['deviation_20_pct']
    z_score = mr_metrics['z_score_20']

    # Calculate how much forecast converges toward mean
    current_deviation_pct = ((current_price - target_mean) / target_mean) * 100
    forecast_deviation_pct = ((forecast_median - target_mean) / target_mean) * 100
    convergence = current_deviation_pct - forecast_deviation_pct  # Positive = moving toward mean

    # Determine signal
    if mr_metrics['direction'] == "OVERSOLD" and convergence > 2:
        # Price below mean, forecast shows recovery toward mean
        signal = "OVERSOLD_REVERT"
        entry_price = current_price
        target_price = min(target_mean, forecast_median)  # Exit at mean or forecast, whichever comes first

        # Position size based on deviation magnitude
        if mr_metrics['condition'] == "EXTREME":
            position_size = 1.25  # 125%
        elif mr_metrics['condition'] == "SIGNIFICANT":
            position_size = 1.0  # 100%
        else:
            position_size = 0.75  # 75%

    elif mr_metrics['direction'] == "OVERBOUGHT" and convergence < -2:
        # Price above mean, forecast shows decline toward mean
        signal = "OVERBOUGHT_REVERT"
        entry_price = current_price
        target_price = max(target_mean, forecast_median)  # Short position

        # Position size based on deviation magnitude
        if mr_metrics['condition'] == "EXTREME":
            position_size = 1.25
        elif mr_metrics['condition'] == "SIGNIFICANT":
            position_size = 1.0
        else:
            position_size = 0.75

    elif mr_metrics['direction'] in ["OVERSOLD", "OVERBOUGHT"]:
        # Deviation present but forecast doesn't support reversion
        signal = "FALSE_SIGNAL"
        entry_price = current_price
        target_price = forecast_median
        position_size = 0.0

    else:
        # Price near mean, no opportunity
        signal = "NO_SIGNAL"
        entry_price = current_price
        target_price = forecast_median
        position_size = 0.0

    # Calculate expected gain
    if signal in ["OVERSOLD_REVERT", "OVERBOUGHT_REVERT"]:
        expected_gain_pct = ((target_price - entry_price) / entry_price) * 100
    else:
        expected_gain_pct = 0.0

    # Risk management
    if signal == "OVERSOLD_REVERT":
        stop_loss = entry_price * 0.95  # 5% stop loss
    elif signal == "OVERBOUGHT_REVERT":
        stop_loss = entry_price * 1.05  # 5% stop loss above (short)
    else:
        stop_loss = entry_price * 0.95

    risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss) if position_size > 0 else None

    strategy_data = {
        'signal': signal,
        'entry_price': entry_price,
        'target_price': target_price,
        'target_mean': target_mean,
        'stop_loss': stop_loss,
        'expected_gain_pct': expected_gain_pct,
        'risk_reward_ratio': risk_reward,
        'position_size': position_size,
        'position_size_pct': position_size * 100,
        'current_price': current_price,
        'mr_metrics': mr_metrics,
        'forecast_metrics': forecast_metrics,
        'convergence': convergence,
    }

    return strategy_data


def print_mean_reversion_strategy(strategy_data: Dict):
    """Print formatted output for mean reversion strategy."""
    print(f"\n{'='*70}")
    print("MEAN REVERSION WITH FORECAST CONFIRMATION STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Mean reversion metrics
    mr = strategy_data['mr_metrics']
    print(f"\n{'='*70}")
    print("MEAN REVERSION ANALYSIS")
    print(f"{'='*70}")
    print(f"\n20-Day SMA: ${mr['sma_20']:,.2f}")
    print(f"50-Day SMA: ${mr['sma_50']:,.2f}")
    print(f"\nDeviation from 20-SMA: {mr['deviation_20_pct']:+.2f}%")
    print(f"Deviation from 50-SMA: ${mr['deviation_50_pct']:+.2f}%")
    print(f"\nZ-Score (20-day): {mr['z_score_20']:.2f}")
    print(f"Condition: {mr['condition']}")
    print(f"Direction: {mr['direction']}")

    # Forecast confirmation
    print(f"\n{'='*70}")
    print("FORECAST CONFIRMATION")
    print(f"{'='*70}")
    print(f"\nForecast Median (14-day): ${strategy_data['forecast_metrics']['median']:,.2f}")
    print(f"Expected Change: {strategy_data['forecast_metrics']['median_change_pct']:+.2f}%")
    print(f"\nConvergence to Mean: {strategy_data['convergence']:+.2f}%")
    if strategy_data['convergence'] > 2:
        print("  ✓ Forecast confirms reversion UP toward mean")
    elif strategy_data['convergence'] < -2:
        print("  ✓ Forecast confirms reversion DOWN toward mean")
    else:
        print("  ✗ Forecast does NOT confirm mean reversion")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"{'='*70}")

    signal = strategy_data['signal']

    if signal == "OVERSOLD_REVERT":
        print(f"\n📊 Trading Plan - BUY THE DIP:")
        print(f"\n  Price is {mr['condition']} OVERSOLD")
        print(f"  Current: ${strategy_data['current_price']:,.2f}")
        print(f"  Deviation: {mr['deviation_20_pct']:+.2f}% below 20-SMA")
        print(f"  Z-Score: {mr['z_score_20']:.2f} (below -1 = oversold)")

        print(f"\n  Forecast confirms upward reversion:")
        print(f"    - Forecast predicts: {strategy_data['forecast_metrics']['median_change_pct']:+.2f}%")
        print(f"    - Convergence to mean: {strategy_data['convergence']:+.2f}%")

        print(f"\n  1. BUY NOW at ${strategy_data['entry_price']:,.2f}")
        print(f"     Position: {strategy_data['position_size_pct']:.0f}%")

        print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
        print(f"     Exit at mean: ${strategy_data['target_mean']:,.2f}")
        print(f"     Expected gain: {strategy_data['expected_gain_pct']:+.2f}%")

        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        if strategy_data['risk_reward_ratio']:
            print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        print(f"\n  💡 Strategy: Price deviated too far below mean")
        print(f"     Forecast predicts return to equilibrium")
        print(f"     Classic mean reversion opportunity")

    elif signal == "OVERBOUGHT_REVERT":
        print(f"\n📊 Trading Plan - SHORT THE SPIKE:")
        print(f"\n  Price is {mr['condition']} OVERBOUGHT")
        print(f"  Current: ${strategy_data['current_price']:,.2f}")
        print(f"  Deviation: {mr['deviation_20_pct']:+.2f}% above 20-SMA")
        print(f"  Z-Score: {mr['z_score_20']:.2f} (above +1 = overbought)")

        print(f"\n  Forecast confirms downward reversion:")
        print(f"    - Forecast predicts: {strategy_data['forecast_metrics']['median_change_pct']:+.2f}%")
        print(f"    - Convergence to mean: {strategy_data['convergence']:+.2f}%")

        print(f"\n  1. SHORT NOW at ${strategy_data['entry_price']:,.2f}")
        print(f"     Position: {strategy_data['position_size_pct']:.0f}%")

        print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
        print(f"     Exit at mean: ${strategy_data['target_mean']:,.2f}")
        print(f"     Expected gain: {abs(strategy_data['expected_gain_pct']):+.2f}%")

        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        if strategy_data['risk_reward_ratio']:
            print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        print(f"\n  💡 Strategy: Price deviated too far above mean")
        print(f"     Forecast predicts return to equilibrium")
        print(f"     Mean reversion short opportunity")

    elif signal == "FALSE_SIGNAL":
        print(f"\n⚠️  FALSE SIGNAL - DO NOT TRADE")
        print(f"\n  Price shows deviation from mean:")
        print(f"    Direction: {mr['direction']}")
        print(f"    Deviation: {mr['deviation_20_pct']:+.2f}%")
        print(f"    Z-Score: {mr['z_score_20']:.2f}")

        print(f"\n  BUT forecast does NOT confirm reversion:")
        print(f"    Forecast: {strategy_data['forecast_metrics']['median_change_pct']:+.2f}%")
        print(f"    Convergence: {strategy_data['convergence']:+.2f}%")

        print(f"\n  💡 Recommendation: Wait for forecast to align")
        print(f"     Price may continue in current direction")
        print(f"     Mean reversion not yet confirmed")

    else:  # NO_SIGNAL
        print(f"\n  Price is near mean - no reversion opportunity")
        print(f"    20-SMA: ${mr['sma_20']:,.2f}")
        print(f"    Current: ${strategy_data['current_price']:,.2f}")
        print(f"    Deviation: {mr['deviation_20_pct']:+.2f}%")
        print(f"    Z-Score: {mr['z_score_20']:.2f}")
        print(f"\n  💡 Recommendation: Wait for significant deviation")

    # Risk analysis
    print(f"\n{'='*70}")
    print("RISK ANALYSIS")
    print(f"{'='*70}")
    fm = strategy_data['forecast_metrics']
    print(f"\nForecast Range (Q25-Q75): {fm['forecast_range_pct']:.1f}%")
    print(f"Worst Case: ${fm['min']:,.2f} ({fm['worst_case_loss']:+.2f}%)")
    print(f"Best Case: ${fm['max']:,.2f} ({fm['best_case_gain']:+.2f}%)")
    print(f"Probability Above Current: {fm['prob_above']:.1f}%")

    print(f"\n{'='*70}")


def visualize_mean_reversion_strategy(strategy_data: Dict, stats: Dict, df: pd.DataFrame,
                                      save_path: str = 'mean_reversion_strategy.png'):
    """Create visualization for mean reversion strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    mr = strategy_data['mr_metrics']

    # Handle both uppercase and lowercase column names
    close_col = 'Close' if 'Close' in df.columns else 'close'

    # Plot 1: Price history with moving averages
    ax1 = axes[0, 0]

    # Last 60 days of price data
    recent_data = df.tail(60)
    ax1.plot(recent_data.index, recent_data[close_col], 'b-', linewidth=2, label='Price', alpha=0.7)

    # Calculate and plot SMAs
    sma_20_series = recent_data[close_col].rolling(window=20).mean()
    sma_50_series = recent_data[close_col].rolling(window=50).mean()

    ax1.plot(recent_data.index, sma_20_series, 'g-', linewidth=2, label='20-Day SMA', alpha=0.7)
    ax1.plot(recent_data.index, sma_50_series, 'r-', linewidth=2, label='50-Day SMA', alpha=0.7)

    # Mark current price
    ax1.axhline(current_price, color='black', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.5)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Price vs Moving Averages - {mr["direction"]} ({mr["condition"]})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Deviation histogram
    ax2 = axes[0, 1]

    deviations = ((recent_data[close_col] - sma_20_series) / sma_20_series * 100).dropna()
    ax2.hist(deviations, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(mr['deviation_20_pct'], color='red', linestyle='--', linewidth=2,
                label=f'Current: {mr["deviation_20_pct"]:+.1f}%')
    ax2.axvline(0, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Mean')

    # Mark zones
    ax2.axvspan(-15, -10, alpha=0.2, color='green', label='Oversold')
    ax2.axvspan(10, 15, alpha=0.2, color='red', label='Overbought')

    ax2.set_xlabel('Deviation from 20-SMA (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Historical Deviation Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Z-Score evolution
    ax3 = axes[1, 0]

    sma = recent_data[close_col].rolling(window=20).mean()
    std = recent_data[close_col].rolling(window=20).std()
    z_scores = ((recent_data[close_col] - sma) / std).dropna()

    ax3.plot(z_scores.index, z_scores, 'b-', linewidth=2, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Extreme (+/-2)')
    ax3.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(1, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Significant (+/-1)')
    ax3.axhline(-1, color='orange', linestyle='--', linewidth=1, alpha=0.3)

    # Mark current z-score
    ax3.scatter([z_scores.index[-1]], [mr['z_score_20']], s=200, color='red',
               zorder=5, edgecolors='black', linewidths=2)

    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Z-Score', fontsize=12)
    ax3.set_title(f'Z-Score Evolution - Current: {mr["z_score_20"]:.2f}',
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Strategy summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Mean Reversion Analysis:\n"
    strategy_text += f"  Direction: {mr['direction']}\n"
    strategy_text += f"  Condition: {mr['condition']}\n"
    strategy_text += f"  Deviation: {mr['deviation_20_pct']:+.2f}%\n"
    strategy_text += f"  Z-Score: {mr['z_score_20']:.2f}\n\n"

    strategy_text += f"Forecast Confirmation:\n"
    strategy_text += f"  14-day Change: {strategy_data['forecast_metrics']['median_change_pct']:+.2f}%\n"
    strategy_text += f"  Convergence: {strategy_data['convergence']:+.2f}%\n\n"

    if strategy_data['position_size'] > 0:
        strategy_text += f"Trade Setup:\n"
        strategy_text += f"  Entry: ${strategy_data['entry_price']:,.2f}\n"
        strategy_text += f"  Target: ${strategy_data['target_price']:,.2f}\n"
        strategy_text += f"  Expected: {strategy_data['expected_gain_pct']:+.2f}%\n"
        strategy_text += f"  Position: {strategy_data['position_size_pct']:.0f}%\n"

    ax4.text(0.1, 0.9, strategy_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Mean Reversion with Forecast Confirmation Strategy."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"MEAN REVERSION STRATEGY ANALYSIS - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    # Handle both uppercase and lowercase column names
    close_col = 'Close' if 'Close' in df_14day.columns else 'close'
    current_price = df_14day[close_col].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_mean_reversion_strategy(stats_14day, df_14day, current_price)

    # Print results
    print_mean_reversion_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_mean_reversion_strategy(strategy_data, stats_14day, df_14day)

    print(f"\n{'='*70}")
    print("MEAN REVERSION STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
