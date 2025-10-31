"""
Forecast Gradient Strategy

Analyzes the SHAPE of the forecast curve rather than just endpoints.
Identifies patterns like U-shaped (dip then recovery), inverted-U (peak then decline),
steep rises, and gradual trends to optimize entry and exit timing.

Strategy Logic:
- U_SHAPED: Buy at predicted trough, sell at recovery
- INVERTED_U: Buy now, sell at predicted peak
- STEEP_RISE: Buy now, ride momentum
- GRADUAL_RISE: Buy now, hold longer
- DECLINE: Stay out or short
- FLAT: No clear opportunity
"""

import numpy as np
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
    analyze_forecast_gradient,
    calculate_stop_loss,
    calculate_risk_reward_ratio,
    format_strategy_output,
    get_default_ensemble_configs,
)


def analyze_gradient_strategy(stats_14day: Dict, current_price: float) -> Dict:
    """
    Analyze trading opportunity based on forecast gradient/shape.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Analyze the forecast curve shape
    gradient_analysis = analyze_forecast_gradient(stats_14day)

    # Get forecast metrics
    metrics = calculate_forecast_metrics(stats_14day, current_price, 14)

    shape = gradient_analysis['shape']
    median = stats_14day['median']

    # Strategy decision based on curve shape
    if shape == "U_SHAPED":
        # Buy at the dip, sell at recovery
        trough_day = gradient_analysis['trough_day']
        trough_price = gradient_analysis['trough_value']
        target_price = median[-1]

        entry_price = trough_price
        dip_pct = ((trough_price - current_price) / current_price) * 100
        recovery_pct = ((target_price - trough_price) / trough_price) * 100

        signal = "WAIT_THEN_BUY"
        stop_loss = calculate_stop_loss(entry_price, metrics['min'], cushion_pct=3.0)
        risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)

        strategy_data = {
            'signal': signal,
            'shape': shape,
            'description': gradient_analysis['description'],
            'entry_price': entry_price,
            'entry_day': trough_day,
            'target_price': target_price,
            'target_day': 14,
            'stop_loss': stop_loss,
            'expected_dip_pct': dip_pct,
            'expected_gain_pct': recovery_pct,
            'risk_reward_ratio': risk_reward,
            'current_price': current_price,
        }

    elif shape == "INVERTED_U":
        # Buy now, sell at peak before decline
        peak_day = gradient_analysis['peak_day']
        peak_price = gradient_analysis['peak_value']

        entry_price = current_price
        gain_to_peak = ((peak_price - current_price) / current_price) * 100
        decline_after = ((median[-1] - peak_price) / peak_price) * 100

        signal = "BUY_NOW_EARLY_EXIT"
        stop_loss = current_price * 0.95  # 5% stop loss
        risk_reward = calculate_risk_reward_ratio(entry_price, peak_price, stop_loss)

        strategy_data = {
            'signal': signal,
            'shape': shape,
            'description': gradient_analysis['description'],
            'entry_price': entry_price,
            'entry_day': 0,
            'target_price': peak_price,
            'target_day': peak_day,
            'stop_loss': stop_loss,
            'expected_gain_pct': gain_to_peak,
            'decline_after_peak_pct': decline_after,
            'risk_reward_ratio': risk_reward,
            'current_price': current_price,
            'warning': 'Exit before peak to avoid decline',
        }

    elif shape == "STEEP_RISE":
        # Strong momentum play
        entry_price = current_price
        target_price = median[-1]
        gain_pct = ((target_price - current_price) / current_price) * 100

        signal = "BUY_NOW_MOMENTUM"
        stop_loss = current_price * 0.95  # 5% stop loss
        risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)

        strategy_data = {
            'signal': signal,
            'shape': shape,
            'description': gradient_analysis['description'],
            'entry_price': entry_price,
            'entry_day': 0,
            'target_price': target_price,
            'target_day': 14,
            'stop_loss': stop_loss,
            'expected_gain_pct': gain_pct,
            'risk_reward_ratio': risk_reward,
            'current_price': current_price,
            'note': 'Strong momentum - ride the trend',
        }

    elif shape == "GRADUAL_RISE":
        # Steady gains - hold longer
        entry_price = current_price
        target_price = median[-1]
        gain_pct = ((target_price - current_price) / current_price) * 100

        signal = "BUY_AND_HOLD"
        stop_loss = current_price * 0.93  # Wider stop for longer hold
        risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)

        strategy_data = {
            'signal': signal,
            'shape': shape,
            'description': gradient_analysis['description'],
            'entry_price': entry_price,
            'entry_day': 0,
            'target_price': target_price,
            'target_day': 14,
            'stop_loss': stop_loss,
            'expected_gain_pct': gain_pct,
            'risk_reward_ratio': risk_reward,
            'current_price': current_price,
            'note': 'Gradual rise - hold full period',
        }

    elif shape == "DECLINE":
        # Stay out or consider shorting
        decline_pct = ((median[-1] - current_price) / current_price) * 100

        signal = "STAY_OUT"

        strategy_data = {
            'signal': signal,
            'shape': shape,
            'description': gradient_analysis['description'],
            'expected_decline_pct': decline_pct,
            'current_price': current_price,
            'recommendation': 'Wait for better opportunity or consider shorting',
        }

    else:  # FLAT or unclear
        signal = "NO_CLEAR_SIGNAL"

        strategy_data = {
            'signal': signal,
            'shape': shape,
            'description': gradient_analysis['description'],
            'current_price': current_price,
            'recommendation': 'No clear pattern - wait for better setup',
        }

    # Add gradient analysis to strategy data
    strategy_data['gradient_analysis'] = gradient_analysis
    strategy_data['metrics'] = metrics

    return strategy_data


def print_gradient_strategy(strategy_data: Dict):
    """Print formatted output for gradient strategy."""
    print(f"\n{'='*70}")
    print("FORECAST GRADIENT STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")
    print(f"\nForecast Shape: {strategy_data['shape']}")
    print(f"Description: {strategy_data['description']}")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"{'='*70}")

    signal = strategy_data['signal']

    if signal == "WAIT_THEN_BUY":
        print(f"\n📊 Trading Plan:")
        print(f"  1. WAIT for dip around Day {strategy_data['entry_day']}")
        print(f"     Expected dip: {strategy_data['expected_dip_pct']:+.2f}%")
        print(f"     Entry target: ${strategy_data['entry_price']:,.2f}")
        print(f"\n  2. BUY at trough")
        print(f"\n  3. SELL at recovery around Day {strategy_data['target_day']}")
        print(f"     Target: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected gain from trough: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"\n  4. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

    elif signal == "BUY_NOW_EARLY_EXIT":
        print(f"\n📊 Trading Plan:")
        print(f"  1. BUY NOW at ${strategy_data['entry_price']:,.2f}")
        print(f"\n  2. SELL BEFORE PEAK around Day {strategy_data['target_day']}")
        print(f"     Peak forecast: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected gain to peak: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"\n  ⚠️  WARNING: Forecast shows decline after Day {strategy_data['target_day']}")
        print(f"     Decline after peak: {strategy_data['decline_after_peak_pct']:+.2f}%")
        print(f"     Exit early to capture gains!")
        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

    elif signal in ["BUY_NOW_MOMENTUM", "BUY_AND_HOLD"]:
        print(f"\n📊 Trading Plan:")
        print(f"  1. BUY NOW at ${strategy_data['entry_price']:,.2f}")
        print(f"\n  2. HOLD until Day {strategy_data['target_day']}")
        print(f"     Target: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected gain: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        if 'note' in strategy_data:
            print(f"\n  💡 {strategy_data['note']}")

    elif signal == "STAY_OUT":
        print(f"\n⚠️  Forecast shows decline: {strategy_data['expected_decline_pct']:+.2f}%")
        print(f"  Recommendation: {strategy_data['recommendation']}")

    else:  # NO_CLEAR_SIGNAL
        print(f"\n  Recommendation: {strategy_data['recommendation']}")

    # Risk Analysis
    if 'metrics' in strategy_data:
        metrics = strategy_data['metrics']
        print(f"\n{'='*70}")
        print("RISK ANALYSIS")
        print(f"{'='*70}")
        print(f"\nWorst Case: ${metrics['min']:,.2f} ({metrics['worst_case_loss']:+.2f}%)")
        print(f"Best Case: ${metrics['max']:,.2f} ({metrics['best_case_gain']:+.2f}%)")
        print(f"Forecast Range: {metrics['forecast_range_pct']:.1f}%")

    print(f"\n{'='*70}")


def visualize_gradient_strategy(strategy_data: Dict, save_path: str = 'gradient_strategy.png'):
    """
    Create visualization showing the gradient strategy.

    Args:
        strategy_data: Strategy analysis data
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    gradient = strategy_data['gradient_analysis']
    metrics = strategy_data['metrics']

    median = gradient['daily_returns']  # Using the gradient analysis median
    days = np.arange(0, 14)

    # Get the full median forecast from metrics
    # We need to reconstruct it or get it from somewhere
    # For now, let's work with what we have

    # Plot 1: Forecast curve with entry/exit points
    ax1 = axes[0, 0]

    # We need the actual forecast values - let's add them to strategy_data
    # For now, use a workaround
    ax1.axhline(current_price, color='green', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.7)

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Forecast Shape: {strategy_data["shape"]}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 0.5, f'{strategy_data["description"]}',
             ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    # Plot 2: Daily returns
    ax2 = axes[0, 1]
    days_ret = np.arange(1, len(gradient['daily_returns']) + 1)
    colors = ['green' if r > 0 else 'red' for r in gradient['daily_returns']]
    ax2.bar(days_ret, gradient['daily_returns'], color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.set_title('Expected Daily Returns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight first vs second half
    mid = len(gradient['daily_returns']) // 2
    ax2.axvline(mid, color='blue', linestyle='--', linewidth=2, alpha=0.5,
                label=f'1st half avg: {gradient["first_half_avg"]:.2f}%\n2nd half avg: {gradient["second_half_avg"]:.2f}%')
    ax2.legend(fontsize=9)

    # Plot 3: Signal and strategy
    ax3 = axes[1, 0]
    ax3.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Shape: {strategy_data['shape']}\n"
    strategy_text += f"{strategy_data['description']}\n\n"

    if 'entry_price' in strategy_data:
        strategy_text += f"Entry: ${strategy_data['entry_price']:,.2f} (Day {strategy_data['entry_day']})\n"
        strategy_text += f"Target: ${strategy_data['target_price']:,.2f} (Day {strategy_data['target_day']})\n"
        if 'stop_loss' in strategy_data:
            strategy_text += f"Stop Loss: ${strategy_data['stop_loss']:,.2f}\n"
        if 'expected_gain_pct' in strategy_data:
            strategy_text += f"Expected Gain: {strategy_data['expected_gain_pct']:+.2f}%\n"
        if 'risk_reward_ratio' in strategy_data:
            strategy_text += f"Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}\n"

    ax3.text(0.1, 0.9, strategy_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Risk metrics
    ax4 = axes[1, 1]

    risk_labels = ['Worst Case', 'Q25', 'Median', 'Q75', 'Best Case']
    risk_values = [
        metrics['worst_case_loss'],
        metrics['q25_change_pct'],
        metrics['median_change_pct'],
        metrics['q75_change_pct'],
        metrics['best_case_gain'],
    ]
    colors = ['red' if v < 0 else 'green' for v in risk_values]

    bars = ax4.barh(risk_labels, risk_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.axvline(0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Return (%)', fontsize=12)
    ax4.set_title('Risk/Reward Scenarios', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, value in zip(bars, risk_values):
        width = bar.get_width()
        ax4.text(width + (0.5 if width > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{value:+.1f}%', ha='left' if width > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Forecast Gradient Strategy analysis."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"FORECAST GRADIENT STRATEGY ANALYSIS - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    current_price = df_14day['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_gradient_strategy(stats_14day, current_price)

    # Add the full forecast to strategy data for visualization
    strategy_data['forecast_median'] = stats_14day['median']
    strategy_data['forecast_q25'] = stats_14day['q25']
    strategy_data['forecast_q75'] = stats_14day['q75']

    # Print results
    print_gradient_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_gradient_strategy(strategy_data)

    print(f"\n{'='*70}")
    print("FORECAST GRADIENT STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
