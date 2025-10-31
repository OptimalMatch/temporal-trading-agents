"""
Volatility-Based Position Sizing Strategy

Uses forecast uncertainty (Q25-Q75 range) and prediction variance to dynamically
adjust position sizes. Lower volatility/uncertainty = higher confidence = larger positions.

Strategy Logic:
- VERY LOW volatility (<5% range): 150-200% position (high confidence)
- LOW volatility (5-10% range): 100-150% position (good confidence)
- MEDIUM volatility (10-15% range): 50-100% position (average confidence)
- HIGH volatility (15-20% range): 25-50% position (low confidence)
- VERY HIGH volatility (>20% range): 0-25% position (very low confidence)

Combines volatility sizing with Kelly Criterion for optimal position allocation.
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
    calculate_ensemble_confidence,
    calculate_position_size,
    calculate_stop_loss,
    calculate_risk_reward_ratio,
    format_strategy_output,
    get_default_ensemble_configs,
)


def calculate_kelly_criterion(win_prob: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.

    Kelly % = W - [(1 - W) / R]
    Where:
        W = Win probability
        R = Average Win / Average Loss

    Args:
        win_prob: Probability of winning trade (0-1)
        avg_win: Average win amount (%)
        avg_loss: Average loss amount (%)

    Returns:
        Optimal position size as fraction (0-1)
    """
    if avg_loss == 0 or win_prob == 0:
        return 0.0

    win_loss_ratio = avg_win / abs(avg_loss)
    kelly = win_prob - ((1 - win_prob) / win_loss_ratio)

    # Use fractional Kelly (25% of full Kelly) for safety
    kelly_fraction = 0.25
    return max(0, min(1, kelly * kelly_fraction))


def analyze_volatility_position_sizing(stats_14day: Dict, current_price: float) -> Dict:
    """
    Analyze optimal position sizing based on forecast volatility.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Get forecast metrics
    metrics = calculate_forecast_metrics(stats_14day, current_price, 14)

    # Get confidence metrics
    confidence = calculate_ensemble_confidence(stats_14day, current_price)

    # Calculate volatility-based position size
    forecast_range_pct = metrics['forecast_range_pct']

    # Determine volatility level and base position size
    if forecast_range_pct < 5:
        volatility_level = "VERY_LOW"
        base_position = 1.75  # 175%
    elif forecast_range_pct < 10:
        volatility_level = "LOW"
        base_position = 1.25  # 125%
    elif forecast_range_pct < 15:
        volatility_level = "MEDIUM"
        base_position = 0.75  # 75%
    elif forecast_range_pct < 20:
        volatility_level = "HIGH"
        base_position = 0.375  # 37.5%
    else:
        volatility_level = "VERY_HIGH"
        base_position = 0.125  # 12.5%

    # Calculate Kelly Criterion position size
    win_prob = confidence['pct_above'] / 100 if metrics['median_change_pct'] > 0 else confidence['pct_below'] / 100
    avg_win = abs(metrics['q75_change_pct']) if metrics['median_change_pct'] > 0 else abs(metrics['q25_change_pct'])
    avg_loss = abs(metrics['q25_change_pct']) if metrics['median_change_pct'] > 0 else abs(metrics['q75_change_pct'])

    kelly_position = calculate_kelly_criterion(win_prob, avg_win, avg_loss)

    # Final position size: average of volatility-based and Kelly
    final_position = (base_position + kelly_position) / 2

    # Cap position size
    final_position = min(final_position, 2.0)  # Max 200%
    final_position = max(final_position, 0.0)  # Min 0%

    # Determine signal based on direction and position size
    bullish = metrics['median_change_pct'] > 0

    if final_position >= 1.0 and bullish:
        signal = "STRONG_BUY"
    elif final_position >= 0.5 and bullish:
        signal = "BUY"
    elif final_position >= 0.25 and bullish:
        signal = "SMALL_BUY"
    elif final_position < 0.25 and bullish:
        signal = "NO_TRADE_LOW_CONFIDENCE"
    elif not bullish:
        signal = "STAY_OUT_BEARISH"
    else:
        signal = "NO_TRADE"

    # Calculate trade parameters
    entry_price = current_price
    target_price = metrics['median']
    expected_gain_pct = metrics['median_change_pct']

    stop_loss = calculate_stop_loss(entry_price, metrics['q25'], cushion_pct=3.0)
    risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)

    # Calculate risk amount as percentage of position
    risk_per_position = abs((stop_loss - entry_price) / entry_price) * 100

    strategy_data = {
        'signal': signal,
        'volatility_level': volatility_level,
        'forecast_range_pct': forecast_range_pct,
        'base_position': base_position,
        'kelly_position': kelly_position,
        'final_position': final_position,
        'position_size_pct': final_position * 100,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'expected_gain_pct': expected_gain_pct,
        'risk_reward_ratio': risk_reward,
        'risk_per_position': risk_per_position,
        'win_prob': win_prob * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'current_price': current_price,
        'metrics': metrics,
        'confidence': confidence,
    }

    return strategy_data


def print_volatility_strategy(strategy_data: Dict):
    """Print formatted output for volatility-based position sizing strategy."""
    print(f"\n{'='*70}")
    print("VOLATILITY-BASED POSITION SIZING STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Volatility analysis
    print(f"\n{'='*70}")
    print("VOLATILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"\nForecast Range (Q25-Q75): {strategy_data['forecast_range_pct']:.2f}%")
    print(f"Volatility Level: {strategy_data['volatility_level']}")

    metrics = strategy_data['metrics']
    print(f"\nForecast Uncertainty:")
    print(f"  Q25: ${metrics['q25']:,.2f} ({metrics['q25_change_pct']:+.2f}%)")
    print(f"  Median: ${metrics['median']:,.2f} ({metrics['median_change_pct']:+.2f}%)")
    print(f"  Q75: ${metrics['q75']:,.2f} ({metrics['q75_change_pct']:+.2f}%)")

    # Position sizing calculation
    print(f"\n{'='*70}")
    print("POSITION SIZING CALCULATION")
    print(f"{'='*70}")

    print(f"\nVolatility-Based Size: {strategy_data['base_position']*100:.1f}%")
    print(f"  Based on {strategy_data['forecast_range_pct']:.1f}% forecast range")

    print(f"\nKelly Criterion Size: {strategy_data['kelly_position']*100:.1f}%")
    print(f"  Win Probability: {strategy_data['win_prob']:.1f}%")
    print(f"  Avg Win: {strategy_data['avg_win']:.2f}%")
    print(f"  Avg Loss: {strategy_data['avg_loss']:.2f}%")

    print(f"\nFinal Position Size: {strategy_data['position_size_pct']:.1f}%")
    print(f"  (Average of volatility-based and Kelly)")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"{'='*70}")

    signal = strategy_data['signal']

    if signal in ["STRONG_BUY", "BUY", "SMALL_BUY"]:
        print(f"\n📊 Trading Plan:")
        print(f"\n  Position Size: {strategy_data['position_size_pct']:.1f}%")
        print(f"  Volatility: {strategy_data['volatility_level']}")

        print(f"\n  1. BUY at ${strategy_data['entry_price']:,.2f}")
        print(f"     Allocate {strategy_data['position_size_pct']:.1f}% of standard position size")

        print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected gain: {strategy_data['expected_gain_pct']:+.2f}%")

        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        print(f"     Risk per position: {strategy_data['risk_per_position']:.2f}%")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        # Calculate total risk
        total_risk_pct = strategy_data['risk_per_position'] * strategy_data['final_position']
        print(f"\n  4. TOTAL RISK: {total_risk_pct:.2f}% of capital")

        # Sizing interpretation
        print(f"\n  💡 Position Sizing Rationale:")
        if strategy_data['volatility_level'] in ["VERY_LOW", "LOW"]:
            print(f"     Low forecast uncertainty = high confidence = larger position")
            print(f"     Tight forecast range ({strategy_data['forecast_range_pct']:.1f}%) suggests reliable prediction")
        elif strategy_data['volatility_level'] == "MEDIUM":
            print(f"     Medium forecast uncertainty = moderate position")
            print(f"     Standard position sizing appropriate")
        else:
            print(f"     High forecast uncertainty = low confidence = smaller position")
            print(f"     Wide forecast range ({strategy_data['forecast_range_pct']:.1f}%) suggests higher risk")

        # Kelly interpretation
        if strategy_data['kelly_position'] > 0.5:
            print(f"     Kelly Criterion suggests favorable risk/reward")
        elif strategy_data['kelly_position'] > 0.25:
            print(f"     Kelly Criterion suggests moderate opportunity")
        else:
            print(f"     Kelly Criterion suggests conservative sizing")

    elif signal == "NO_TRADE_LOW_CONFIDENCE":
        print(f"\n⚠️  Position size too small to trade")
        print(f"  Calculated position: {strategy_data['position_size_pct']:.1f}%")
        print(f"  Volatility: {strategy_data['volatility_level']}")
        print(f"  Forecast range: {strategy_data['forecast_range_pct']:.1f}%")
        print(f"\n  Recommendation: Wait for lower volatility / tighter forecast range")

    elif signal == "STAY_OUT_BEARISH":
        print(f"\n⚠️  Bearish forecast")
        print(f"  Expected decline: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"  Target: ${strategy_data['target_price']:,.2f}")
        print(f"\n  Recommendation: Stay out or consider shorting")

    else:  # NO_TRADE
        print(f"\n  No clear opportunity at current volatility levels")
        print(f"  Recommendation: Wait for better setup")

    # Risk breakdown
    print(f"\n{'='*70}")
    print("RISK BREAKDOWN")
    print(f"{'='*70}")
    print(f"\nBest Case: ${metrics['max']:,.2f} ({metrics['best_case_gain']:+.2f}%)")
    print(f"Q75 Case: ${metrics['q75']:,.2f} ({metrics['q75_change_pct']:+.2f}%)")
    print(f"Median: ${metrics['median']:,.2f} ({metrics['median_change_pct']:+.2f}%)")
    print(f"Q25 Case: ${metrics['q25']:,.2f} ({metrics['q25_change_pct']:+.2f}%)")
    print(f"Worst Case: ${metrics['min']:,.2f} ({metrics['worst_case_loss']:+.2f}%)")

    print(f"\n{'='*70}")


def visualize_volatility_strategy(strategy_data: Dict, stats: Dict,
                                  save_path: str = 'volatility_strategy.png'):
    """
    Create visualization showing the volatility-based position sizing strategy.

    Args:
        strategy_data: Strategy analysis data
        stats: Forecast statistics
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    metrics = strategy_data['metrics']

    # Plot 1: Forecast with volatility bands
    ax1 = axes[0, 0]
    days = np.arange(0, 14)

    ax1.axhline(current_price, color='black', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.7)

    ax1.plot(days, stats['median'], 'b-', linewidth=3, marker='o',
             markersize=6, label='Median Forecast', alpha=0.8)

    # Show multiple volatility bands
    ax1.fill_between(days, stats['q25'], stats['q75'],
                     alpha=0.4, color='yellow', label=f'Q25-Q75 ({strategy_data["forecast_range_pct"]:.1f}%)')
    ax1.fill_between(days, stats['q25'], stats['median'],
                     alpha=0.2, color='green', label='Downside Q25-Median')
    ax1.fill_between(days, stats['median'], stats['q75'],
                     alpha=0.2, color='red', label='Upside Median-Q75')

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Forecast Volatility: {strategy_data["volatility_level"]} ({strategy_data["forecast_range_pct"]:.1f}%)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Position sizing comparison
    ax2 = axes[0, 1]

    sizing_methods = ['Volatility\nBased', 'Kelly\nCriterion', 'Final\nPosition']
    sizing_values = [
        strategy_data['base_position'] * 100,
        strategy_data['kelly_position'] * 100,
        strategy_data['position_size_pct']
    ]
    colors = ['blue', 'green', 'orange']

    bars = ax2.bar(sizing_methods, sizing_values, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)

    ax2.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5,
                label='Standard Size (100%)')
    ax2.set_ylabel('Position Size (%)', fontsize=12)
    ax2.set_title('Position Sizing Methods Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, value in zip(bars, sizing_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 3,
                f'{value:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Plot 3: Volatility vs Position Size relationship
    ax3 = axes[1, 0]

    volatility_ranges = [2.5, 7.5, 12.5, 17.5, 25]
    position_sizes = [175, 125, 75, 37.5, 12.5]
    vol_labels = ['VERY LOW\n<5%', 'LOW\n5-10%', 'MEDIUM\n10-15%', 'HIGH\n15-20%', 'VERY HIGH\n>20%']

    colors_vol = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']

    bars = ax3.bar(vol_labels, position_sizes, color=colors_vol, alpha=0.7,
                   edgecolor='black', linewidth=2)

    # Highlight current level
    vol_level_idx = {'VERY_LOW': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'VERY_HIGH': 4}
    current_idx = vol_level_idx.get(strategy_data['volatility_level'], 2)
    bars[current_idx].set_edgecolor('blue')
    bars[current_idx].set_linewidth(4)

    ax3.axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Base Position Size (%)', fontsize=12)
    ax3.set_xlabel('Volatility Level', fontsize=12)
    ax3.set_title(f'Volatility vs Position Size\nCurrent: {strategy_data["volatility_level"]}',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, size in zip(bars, position_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 3,
                f'{size:.0f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 4: Risk/Reward with position sizing
    ax4 = axes[1, 1]

    scenarios = ['Worst', 'Q25', 'Median', 'Q75', 'Best']
    returns = [
        metrics['worst_case_loss'],
        metrics['q25_change_pct'],
        metrics['median_change_pct'],
        metrics['q75_change_pct'],
        metrics['best_case_gain']
    ]

    # Adjust returns by position size
    adjusted_returns = [r * strategy_data['final_position'] for r in returns]
    colors_ret = ['darkred' if r < 0 else 'darkgreen' if r > 5 else 'lightgreen' if r > 0 else 'gray'
                  for r in adjusted_returns]

    bars = ax4.barh(scenarios, adjusted_returns, color=colors_ret, alpha=0.7,
                    edgecolor='black', linewidth=2)

    ax4.axvline(0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Position-Adjusted Return (%)', fontsize=12)
    ax4.set_title(f'Risk/Reward with {strategy_data["position_size_pct"]:.0f}% Position',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, ret in zip(bars, adjusted_returns):
        width = bar.get_width()
        ax4.text(width + (0.3 if width > 0 else -0.3), bar.get_y() + bar.get_height()/2,
                f'{ret:+.1f}%', ha='left' if width > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Volatility-Based Position Sizing Strategy analysis."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"VOLATILITY-BASED POSITION SIZING STRATEGY - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    current_price = df_14day['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_volatility_position_sizing(stats_14day, current_price)

    # Print results
    print_volatility_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_volatility_strategy(strategy_data, stats_14day)

    print(f"\n{'='*70}")
    print("VOLATILITY-BASED POSITION SIZING STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
