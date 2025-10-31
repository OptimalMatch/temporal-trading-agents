"""
Compare All 8 Trading Strategies

Comprehensive comparison tool that runs all 8 advanced trading strategies:
1. Buy-the-Dip (original)
2. Forecast Gradient
3. Confidence-Weighted
4. Multi-Timeframe
5. Volatility Position Sizing
6. Mean Reversion
7. Acceleration/Deceleration
8. Swing Trading
9. Risk-Adjusted

Provides consensus analysis and recommended action across all strategies.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_utils import (
    load_ensemble_module, train_ensemble, get_default_ensemble_configs
)
from strategies.forecast_gradient_strategy import analyze_gradient_strategy
from strategies.confidence_weighted_strategy import analyze_confidence_weighted_strategy
from strategies.multi_timeframe_strategy import train_multiple_timeframes, analyze_multi_timeframe_strategy
from strategies.volatility_position_sizing import analyze_volatility_position_sizing
from strategies.mean_reversion_strategy import analyze_mean_reversion_strategy
from strategies.acceleration_strategy import analyze_acceleration_strategy
from strategies.swing_trading_strategy import analyze_swing_trading_strategy
from strategies.risk_adjusted_strategy import analyze_risk_adjusted_strategy


def run_all_strategies(symbol: str, ensemble_module):
    """
    Run all 8 strategies and collect results.

    Args:
        symbol: Trading symbol
        ensemble_module: Loaded ensemble module

    Returns:
        Dictionary with all strategy results
    """
    results = {}

    # Train 14-day ensemble (used by most strategies)
    configs_14day = get_default_ensemble_configs(14)
    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble_module)
    current_price = df_14day['Close'].iloc[-1]

    print(f"\n{'='*70}")
    print("RUNNING ALL 8 STRATEGIES...")
    print(f"{'='*70}")

    # Strategy 1: Forecast Gradient
    print("\n1/8 Forecast Gradient Strategy...")
    results['gradient'] = analyze_gradient_strategy(stats_14day, current_price)
    results['gradient']['forecast_median'] = stats_14day['median']

    # Strategy 2: Confidence-Weighted
    print("2/8 Confidence-Weighted Strategy...")
    results['confidence'] = analyze_confidence_weighted_strategy(stats_14day, current_price)

    # Strategy 3: Multi-Timeframe
    print("3/8 Multi-Timeframe Strategy...")
    horizons = [3, 7, 14, 21]
    timeframe_data = train_multiple_timeframes(symbol, ensemble_module, horizons)
    results['timeframe'] = analyze_multi_timeframe_strategy(timeframe_data, current_price)

    # Strategy 4: Volatility Position Sizing
    print("4/8 Volatility Position Sizing...")
    results['volatility'] = analyze_volatility_position_sizing(stats_14day, current_price)

    # Strategy 5: Mean Reversion
    print("5/8 Mean Reversion Strategy...")
    results['mean_reversion'] = analyze_mean_reversion_strategy(stats_14day, df_14day, current_price)

    # Strategy 6: Acceleration/Deceleration
    print("6/8 Acceleration/Deceleration...")
    results['acceleration'] = analyze_acceleration_strategy(stats_14day, current_price)

    # Strategy 7: Swing Trading
    print("7/8 Swing Trading Strategy...")
    results['swing'] = analyze_swing_trading_strategy(stats_14day, current_price)

    # Strategy 8: Risk-Adjusted
    print("8/8 Risk-Adjusted Strategy...")
    results['risk_adjusted'] = analyze_risk_adjusted_strategy(stats_14day, current_price)

    results['stats_14day'] = stats_14day
    results['current_price'] = current_price

    return results


def analyze_strategy_consensus(results):
    """Analyze consensus across all strategies."""
    current_price = results['current_price']

    strategies = {
        'Forecast Gradient': results['gradient'],
        'Confidence-Weighted': results['confidence'],
        'Multi-Timeframe': results['timeframe'],
        'Volatility Sizing': results['volatility'],
        'Mean Reversion': results['mean_reversion'],
        'Acceleration': results['acceleration'],
        'Swing Trading': results['swing'],
        'Risk-Adjusted': results['risk_adjusted'],
    }

    # Categorize signals
    bullish_keywords = ['BUY', 'BULLISH', 'MOMENTUM', 'REVERT', 'REVERSAL', 'EXCELLENT', 'GOOD']
    bearish_keywords = ['SELL', 'BEARISH', 'OUT', 'STAY', 'EXIT', 'POOR', 'FALSE']

    bullish_strategies = []
    bearish_strategies = []
    neutral_strategies = []

    for name, data in strategies.items():
        signal = data['signal']

        if any(keyword in signal for keyword in bullish_keywords) and 'POOR' not in signal and 'FALSE' not in signal:
            bullish_strategies.append(name)
        elif any(keyword in signal for keyword in bearish_keywords) or 'NO' in signal:
            bearish_strategies.append(name)
        else:
            neutral_strategies.append(name)

    # Calculate average position sizes for bullish strategies
    bullish_positions = []
    for name in bullish_strategies:
        data = strategies[name]
        if 'position_size_pct' in data and data['position_size_pct'] > 0:
            bullish_positions.append(data['position_size_pct'])

    avg_position = np.mean(bullish_positions) if bullish_positions else 0

    # Determine consensus
    total = len(strategies)
    bullish_count = len(bullish_strategies)
    bearish_count = len(bearish_strategies)

    if bullish_count >= 6:
        consensus = "STRONG BUY CONSENSUS"
        strength = "VERY HIGH"
    elif bullish_count >= 5:
        consensus = "BUY CONSENSUS"
        strength = "HIGH"
    elif bullish_count >= 4:
        consensus = "MODERATE BUY"
        strength = "MODERATE"
    elif bearish_count >= 5:
        consensus = "SELL/AVOID CONSENSUS"
        strength = "HIGH"
    elif bearish_count >= 4:
        consensus = "MODERATE SELL/AVOID"
        strength = "MODERATE"
    else:
        consensus = "MIXED SIGNALS"
        strength = "LOW"

    return {
        'consensus': consensus,
        'strength': strength,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': len(neutral_strategies),
        'total_count': total,
        'bullish_strategies': bullish_strategies,
        'bearish_strategies': bearish_strategies,
        'neutral_strategies': neutral_strategies,
        'avg_position': avg_position,
        'strategies': strategies,
    }


def print_comprehensive_summary(results, consensus):
    """Print comprehensive summary of all strategies."""
    print(f"\n{'='*70}")
    print("COMPREHENSIVE STRATEGY COMPARISON - ALL 8 STRATEGIES")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${results['current_price']:,.2f}")

    # Strategy signals table
    print(f"\n{'='*70}")
    print("STRATEGY SIGNALS")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<30} {'Signal':<35} {'Position':<10}")
    print(f"{'-'*75}")

    for name, data in consensus['strategies'].items():
        signal = data['signal']
        position = f"{data['position_size_pct']:.0f}%" if 'position_size_pct' in data else "N/A"
        print(f"{name:<30} {signal:<35} {position:<10}")

    # Consensus analysis
    print(f"\n{'='*70}")
    print("CONSENSUS ANALYSIS")
    print(f"{'='*70}")

    print(f"\nStrategy Breakdown:")
    print(f"  Bullish: {consensus['bullish_count']}/{consensus['total_count']}")
    for name in consensus['bullish_strategies']:
        print(f"    - {name}")

    print(f"\n  Bearish/Neutral: {consensus['bearish_count'] + consensus['neutral_count']}/{consensus['total_count']}")
    for name in consensus['bearish_strategies']:
        print(f"    - {name}")
    for name in consensus['neutral_strategies']:
        print(f"    - {name}")

    print(f"\nConsensus: {consensus['consensus']}")
    print(f"Strength: {consensus['strength']}")

    # Recommended action
    print(f"\n{'='*70}")
    print("RECOMMENDED ACTION")
    print(f"{'='*70}")

    if consensus['bullish_count'] >= 5:
        print(f"\n✓ STRONG BUY SIGNAL")
        print(f"  {consensus['bullish_count']} out of {consensus['total_count']} strategies recommend buying")
        print(f"  Average position size: {consensus['avg_position']:.0f}%")
        print(f"  Entry: ${results['current_price']:,.2f}")

        # Get targets from different strategies
        targets = []
        for name in consensus['bullish_strategies']:
            data = consensus['strategies'][name]
            if 'target_price' in data:
                targets.append(data['target_price'])

        if targets:
            avg_target = np.mean(targets)
            print(f"  Average target: ${avg_target:,.2f}")
            print(f"  Expected gain: {((avg_target - results['current_price']) / results['current_price'] * 100):+.2f}%")

    elif consensus['bullish_count'] >= 4:
        print(f"\n→ MODERATE BUY SIGNAL")
        print(f"  {consensus['bullish_count']} out of {consensus['total_count']} strategies recommend buying")
        print(f"  Consider smaller position or wait for stronger consensus")
        print(f"  Suggested position: {consensus['avg_position'] * 0.75:.0f}% (reduced)")

    elif consensus['bearish_count'] >= 4:
        print(f"\n✗ AVOID OR SELL")
        print(f"  {consensus['bearish_count']} out of {consensus['total_count']} strategies suggest avoiding")
        print(f"  Recommendation: Stay out or wait for better setup")

    else:
        print(f"\n⚠ MIXED SIGNALS - NO CLEAR CONSENSUS")
        print(f"  Bullish: {consensus['bullish_count']}")
        print(f"  Bearish: {consensus['bearish_count']}")
        print(f"  Neutral: {consensus['neutral_count']}")
        print(f"  Recommendation: Wait for clearer alignment")

    print(f"\n{'='*70}")


def create_mega_visualization(results, consensus, save_path='comprehensive_strategy_comparison.png'):
    """Create comprehensive visualization of all strategies."""
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    current_price = results['current_price']

    # Main forecast plot (larger, spans 2 columns)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    days = np.arange(0, 14)
    stats = results['stats_14day']

    ax_main.axhline(current_price, color='black', linestyle='--', linewidth=2,
                    label=f'Current: ${current_price:,.0f}', alpha=0.7)

    ax_main.plot(days, stats['median'], 'b-', linewidth=3, marker='o',
                markersize=6, label='14-Day Median Forecast', alpha=0.8)
    ax_main.fill_between(days, stats['q25'], stats['q75'],
                        alpha=0.3, color='blue', label='Q25-Q75 Range')

    ax_main.set_xlabel('Days', fontsize=12)
    ax_main.set_ylabel('Price ($)', fontsize=12)
    ax_main.set_title('14-Day Forecast with Strategy Consensus', fontsize=16, fontweight='bold')
    ax_main.legend(fontsize=10)
    ax_main.grid(True, alpha=0.3)

    # Consensus pie chart
    ax_pie = fig.add_subplot(gs[0, 2])
    labels = [f'Bullish\n({consensus["bullish_count"]})',
              f'Bearish\n({consensus["bearish_count"]})',
              f'Neutral\n({consensus["neutral_count"]})']
    sizes = [consensus['bullish_count'], consensus['bearish_count'], consensus['neutral_count']]
    colors = ['green', 'red', 'gray']

    ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
              startangle=90, textprops={'fontsize': 11})
    ax_pie.set_title('Strategy Consensus', fontsize=13, fontweight='bold')

    # Position sizing comparison
    ax_pos = fig.add_subplot(gs[0, 3])

    strategies_with_pos = [(name, data['position_size_pct'])
                          for name, data in consensus['strategies'].items()
                          if 'position_size_pct' in data and data['position_size_pct'] > 0]

    if strategies_with_pos:
        names = [n.split()[0] for n, _ in strategies_with_pos]  # Shorten names
        positions = [p for _, p in strategies_with_pos]

        ax_pos.barh(names, positions, color='green', alpha=0.7, edgecolor='black', linewidth=1)
        ax_pos.axvline(100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Standard (100%)')
        ax_pos.set_xlabel('Position Size (%)', fontsize=11)
        ax_pos.set_title('Position Sizing by Strategy', fontsize=13, fontweight='bold')
        ax_pos.legend(fontsize=9)
        ax_pos.grid(True, alpha=0.3, axis='x')

    # Signal summary table
    ax_table = fig.add_subplot(gs[1, 2:4])
    ax_table.axis('off')

    table_text = "STRATEGY SIGNALS\n" + "="*40 + "\n\n"
    for name, data in consensus['strategies'].items():
        signal = data['signal']
        # Truncate long signals
        if len(signal) > 25:
            signal = signal[:22] + "..."
        table_text += f"{name[:20]:<20} {signal}\n"

    ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Strategy-specific insights (bottom row)
    insights = [
        ('Gradient', results['gradient'], gs[2, 0]),
        ('Confidence', results['confidence'], gs[2, 1]),
        ('Mean Rev', results['mean_reversion'], gs[2, 2]),
        ('Risk Adj', results['risk_adjusted'], gs[2, 3]),
    ]

    for short_name, data, grid_spec in insights:
        ax = fig.add_subplot(grid_spec)
        ax.axis('off')

        insight_text = f"{short_name.upper()}\n{'='*20}\n\n"

        if short_name == 'Gradient':
            insight_text += f"Shape: {data.get('shape', 'N/A')}\n"
            insight_text += f"Signal: {data['signal'][:20]}\n"
        elif short_name == 'Confidence':
            conf = data.get('confidence_metrics', {})
            insight_text += f"Agreement: {conf.get('agreement', 0):.0f}%\n"
            insight_text += f"Level: {data.get('confidence_level', 'N/A')}\n"
        elif short_name == 'Mean Rev':
            mr = data.get('mr_metrics', {})
            insight_text += f"Direction: {mr.get('direction', 'N/A')}\n"
            insight_text += f"Z-Score: {mr.get('z_score_20', 0):.2f}\n"
        elif short_name == 'Risk Adj':
            rm = data.get('risk_metrics', {})
            insight_text += f"Score: {rm.get('risk_adjusted_score', 0):.3f}\n"
            insight_text += f"Sharpe: {rm.get('sharpe_ratio', 0):.2f}\n"

        ax.text(0.1, 0.9, insight_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comprehensive visualization saved to '{save_path}'")


def main():
    """Run comprehensive comparison of all 8 strategies."""
    symbol = 'BTC-USD'

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE 8-STRATEGY ANALYSIS - {symbol}")
    print(f"{'='*70}")

    # Load ensemble module
    ensemble = load_ensemble_module("crypto_ensemble_forecast.py")

    # Run all strategies
    results = run_all_strategies(symbol, ensemble)

    # Analyze consensus
    consensus = analyze_strategy_consensus(results)

    # Print comprehensive summary
    print_comprehensive_summary(results, consensus)

    # Create mega visualization
    print(f"\n{'='*70}")
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print(f"{'='*70}")
    create_mega_visualization(results, consensus)

    print(f"\n{'='*70}")
    print("COMPREHENSIVE 8-STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\n⚠️  DISCLAIMER:")
    print("This analysis is for educational purposes only.")
    print("Cryptocurrency trading carries substantial risk.")
    print("Always do your own research and never invest more than you can afford to lose.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
