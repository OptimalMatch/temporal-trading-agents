"""
Compare All Trading Strategies

Runs all 4 advanced trading strategies on the same asset and compares their
recommendations. This allows you to see how different analytical approaches
lead to different trading decisions.

Strategies:
1. Forecast Gradient Strategy - Analyzes curve shape
2. Confidence-Weighted Strategy - Uses model agreement
3. Multi-Timeframe Strategy - Compares multiple horizons
4. Volatility Position Sizing - Adjusts size based on uncertainty
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_utils import load_ensemble_module, train_ensemble, get_default_ensemble_configs
from strategies.forecast_gradient_strategy import analyze_gradient_strategy, print_gradient_strategy
from strategies.confidence_weighted_strategy import analyze_confidence_weighted_strategy, print_confidence_strategy
from strategies.multi_timeframe_strategy import train_multiple_timeframes, analyze_multi_timeframe_strategy, print_multi_timeframe_strategy
from strategies.volatility_position_sizing import analyze_volatility_position_sizing, print_volatility_strategy
import matplotlib.pyplot as plt
import numpy as np


def compare_strategies_summary(gradient_data, confidence_data, timeframe_data_result, volatility_data, current_price):
    """Create a comparison summary of all strategies."""
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\nCurrent Price: ${current_price:,.2f}")
    print(f"\n{'='*70}")

    # Create comparison table
    print(f"\n{'Strategy':<30} {'Signal':<25} {'Position Size':<15}")
    print(f"{'-'*70}")

    strategies = [
        ("Forecast Gradient", gradient_data['signal'], gradient_data.get('entry_price', 'N/A')),
        ("Confidence-Weighted", confidence_data['signal'], f"{confidence_data['position_size_pct']:.0f}%"),
        ("Multi-Timeframe", timeframe_data_result['signal'], f"{timeframe_data_result['position_size_pct']:.0f}%"),
        ("Volatility Position Sizing", volatility_data['signal'], f"{volatility_data['position_size_pct']:.1f}%"),
    ]

    for name, signal, position in strategies:
        print(f"{name:<30} {signal:<25} {str(position):<15}")

    print(f"\n{'='*70}")
    print("STRATEGY CONSENSUS")
    print(f"{'='*70}")

    # Analyze consensus
    buy_signals = sum(1 for _, signal, _ in strategies
                     if any(x in signal for x in ['BUY', 'BULLISH', 'MOMENTUM']))
    sell_signals = sum(1 for _, signal, _ in strategies
                      if any(x in signal for x in ['SELL', 'BEARISH', 'OUT', 'STAY']))
    neutral_signals = len(strategies) - buy_signals - sell_signals

    print(f"\nBullish signals: {buy_signals}/{len(strategies)}")
    print(f"Bearish signals: {sell_signals}/{len(strategies)}")
    print(f"Neutral signals: {neutral_signals}/{len(strategies)}")

    if buy_signals >= 3:
        consensus = "STRONG BUY CONSENSUS"
    elif buy_signals >= 2:
        consensus = "MODERATE BUY CONSENSUS"
    elif sell_signals >= 3:
        consensus = "STRONG SELL/AVOID CONSENSUS"
    elif sell_signals >= 2:
        consensus = "MODERATE SELL/AVOID CONSENSUS"
    else:
        consensus = "NO CLEAR CONSENSUS"

    print(f"\nConsensus: {consensus}")

    # Detailed insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")

    print(f"\n1. Forecast Shape: {gradient_data['shape']}")
    print(f"   {gradient_data['description']}")

    print(f"\n2. Ensemble Confidence: {confidence_data['confidence_level']}")
    print(f"   {confidence_data['confidence_metrics']['agreement']:.1f}% model agreement")

    print(f"\n3. Timeframe Alignment: {timeframe_data_result['alignment']}")
    print(f"   {timeframe_data_result['bullish_count']}/{timeframe_data_result['total_count']} timeframes bullish")

    print(f"\n4. Forecast Volatility: {volatility_data['volatility_level']}")
    print(f"   {volatility_data['forecast_range_pct']:.1f}% forecast range")

    print(f"\n{'='*70}")
    print("RECOMMENDED ACTION")
    print(f"{'='*70}")

    # Determine recommended action based on consensus
    if buy_signals >= 3:
        # Get average position size from strategies that recommend buying
        position_sizes = []
        if 'position_size_pct' in confidence_data and 'BUY' in confidence_data['signal']:
            position_sizes.append(confidence_data['position_size_pct'])
        if 'position_size_pct' in timeframe_data_result and 'BUY' in timeframe_data_result['signal']:
            position_sizes.append(timeframe_data_result['position_size_pct'])
        if 'position_size_pct' in volatility_data and 'BUY' in volatility_data['signal']:
            position_sizes.append(volatility_data['position_size_pct'])

        avg_position = np.mean(position_sizes) if position_sizes else 100

        print(f"\n✓ Strong consensus to BUY")
        print(f"  Recommended position size: {avg_position:.0f}% of standard")
        print(f"  Entry: ${current_price:,.2f}")

        # Get targets from different strategies
        targets = []
        if gradient_data.get('target_price'):
            targets.append(('Gradient', gradient_data['target_day'], gradient_data['target_price']))
        if confidence_data.get('target_price'):
            targets.append(('Confidence', 14, confidence_data['target_price']))
        if timeframe_data_result.get('target_price'):
            targets.append(('Multi-TF', timeframe_data_result['primary_horizon'], timeframe_data_result['target_price']))
        if volatility_data.get('target_price'):
            targets.append(('Volatility', 14, volatility_data['target_price']))

        if targets:
            avg_target = np.mean([t[2] for t in targets])
            print(f"  Average target: ${avg_target:,.2f}")
            print(f"  Expected gain: {((avg_target - current_price) / current_price * 100):+.2f}%")

    elif buy_signals >= 2:
        print(f"\n→ Moderate buy signal")
        print(f"  Consider smaller position or wait for stronger consensus")
        print(f"  {buy_signals} out of 4 strategies recommend buying")

    elif sell_signals >= 2:
        print(f"\n✗ Bearish consensus")
        print(f"  {sell_signals} out of 4 strategies recommend avoiding/selling")
        print(f"  Recommendation: Stay out or wait for better setup")

    else:
        print(f"\n⚠ Mixed signals")
        print(f"  No clear consensus among strategies")
        print(f"  Recommendation: Wait for better alignment")

    print(f"\n{'='*70}")


def create_comparison_visualization(gradient_data, confidence_data, timeframe_data_result,
                                   volatility_data, stats_14day, current_price):
    """Create a comprehensive visualization comparing all strategies."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main forecast plot
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    days = np.arange(0, 14)

    ax_main.axhline(current_price, color='black', linestyle='--', linewidth=2,
                    label=f'Current: ${current_price:,.0f}', alpha=0.7)

    ax_main.plot(days, stats_14day['median'], 'b-', linewidth=3, marker='o',
                markersize=6, label='14-Day Median Forecast', alpha=0.8)
    ax_main.fill_between(days, stats_14day['q25'], stats_14day['q75'],
                        alpha=0.3, color='blue', label='Q25-Q75 Range')

    ax_main.set_xlabel('Days', fontsize=12)
    ax_main.set_ylabel('Price ($)', fontsize=12)
    ax_main.set_title('Combined Forecast Overview', fontsize=16, fontweight='bold')
    ax_main.legend(fontsize=10)
    ax_main.grid(True, alpha=0.3)

    # Signal comparison
    ax_signals = fig.add_subplot(gs[0, 2])
    ax_signals.axis('off')

    signals_text = "STRATEGY SIGNALS\n" + "="*30 + "\n\n"
    signals_text += f"Gradient:\n  {gradient_data['signal']}\n\n"
    signals_text += f"Confidence:\n  {confidence_data['signal']}\n\n"
    signals_text += f"Multi-TF:\n  {timeframe_data_result['signal']}\n\n"
    signals_text += f"Volatility:\n  {volatility_data['signal']}\n"

    ax_signals.text(0.05, 0.95, signals_text, transform=ax_signals.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Position sizing comparison
    ax_position = fig.add_subplot(gs[1, 2])

    strategies = ['Confidence', 'Multi-TF', 'Volatility']
    positions = [
        confidence_data['position_size_pct'],
        timeframe_data_result['position_size_pct'],
        volatility_data['position_size_pct']
    ]

    colors = ['green' if p >= 75 else 'yellow' if p >= 40 else 'orange' for p in positions]
    bars = ax_position.barh(strategies, positions, color=colors, alpha=0.7,
                            edgecolor='black', linewidth=2)

    ax_position.axvline(100, color='red', linestyle='--', linewidth=1,
                       alpha=0.5, label='Standard (100%)')
    ax_position.set_xlabel('Position Size (%)', fontsize=11)
    ax_position.set_title('Position Sizing Comparison', fontsize=12, fontweight='bold')
    ax_position.legend(fontsize=9)
    ax_position.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, pos in zip(bars, positions):
        width = bar.get_width()
        ax_position.text(width + 2, bar.get_y() + bar.get_height()/2,
                        f'{pos:.0f}%', ha='left', va='center',
                        fontsize=10, fontweight='bold')

    # Key metrics comparison
    ax_metrics = fig.add_subplot(gs[2, 0])

    metrics_labels = ['Shape', 'Confidence', 'Alignment', 'Volatility']
    metrics_values = [
        gradient_data['shape'],
        confidence_data['confidence_level'],
        timeframe_data_result['alignment'],
        volatility_data['volatility_level']
    ]

    metrics_text = "KEY METRICS\n" + "="*40 + "\n\n"
    for label, value in zip(metrics_labels, metrics_values):
        metrics_text += f"{label}: {value}\n"

    ax_metrics.axis('off')
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Consensus gauge
    ax_consensus = fig.add_subplot(gs[2, 1])

    signals = [gradient_data['signal'], confidence_data['signal'],
              timeframe_data_result['signal'], volatility_data['signal']]
    buy_count = sum(1 for s in signals if any(x in s for x in ['BUY', 'BULLISH', 'MOMENTUM']))
    sell_count = sum(1 for s in signals if any(x in s for x in ['SELL', 'BEARISH', 'OUT', 'STAY']))
    neutral_count = 4 - buy_count - sell_count

    consensus_data = [buy_count, neutral_count, sell_count]
    consensus_labels = [f'Bullish\n({buy_count})', f'Neutral\n({neutral_count})', f'Bearish\n({sell_count})']
    consensus_colors = ['green', 'yellow', 'red']

    ax_consensus.pie(consensus_data, labels=consensus_labels, colors=consensus_colors,
                    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
    ax_consensus.set_title('Strategy Consensus', fontsize=12, fontweight='bold')

    # Risk/Reward summary
    ax_risk = fig.add_subplot(gs[2, 2])
    ax_risk.axis('off')

    risk_text = "RISK/REWARD SUMMARY\n" + "="*30 + "\n\n"

    if 'expected_gain_pct' in confidence_data:
        risk_text += f"Expected Gain:\n  {confidence_data['expected_gain_pct']:+.2f}%\n\n"

    if 'risk_reward_ratio' in confidence_data and confidence_data['risk_reward_ratio']:
        risk_text += f"Risk/Reward:\n  1:{confidence_data['risk_reward_ratio']:.2f}\n\n"

    metrics = confidence_data['metrics']
    risk_text += f"Worst Case:\n  {metrics['worst_case_loss']:+.2f}%\n\n"
    risk_text += f"Best Case:\n  {metrics['best_case_gain']:+.2f}%"

    ax_risk.text(0.05, 0.95, risk_text, transform=ax_risk.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to 'strategy_comparison.png'")


def main():
    """Run all strategies and compare results."""
    symbol = 'BTC-USD'

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE STRATEGY COMPARISON - {symbol}")
    print(f"{'='*70}")
    print("\nTraining models and running all 4 advanced strategies...")

    # Load ensemble module
    ensemble = load_ensemble_module("crypto_ensemble_forecast.py")

    # Train 14-day ensemble (used by most strategies)
    configs_14day = get_default_ensemble_configs(14)
    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)
    current_price = df_14day['Close'].iloc[-1]

    # Strategy 1: Forecast Gradient
    print(f"\n{'='*70}")
    print("RUNNING STRATEGY 1: FORECAST GRADIENT")
    print(f"{'='*70}")
    gradient_data = analyze_gradient_strategy(stats_14day, current_price)
    gradient_data['forecast_median'] = stats_14day['median']
    print_gradient_strategy(gradient_data)

    # Strategy 2: Confidence-Weighted
    print(f"\n{'='*70}")
    print("RUNNING STRATEGY 2: CONFIDENCE-WEIGHTED")
    print(f"{'='*70}")
    confidence_data = analyze_confidence_weighted_strategy(stats_14day, current_price)
    print_confidence_strategy(confidence_data)

    # Strategy 3: Multi-Timeframe
    print(f"\n{'='*70}")
    print("RUNNING STRATEGY 3: MULTI-TIMEFRAME")
    print(f"{'='*70}")
    horizons = [3, 7, 14, 21]
    timeframe_data = train_multiple_timeframes(symbol, ensemble, horizons)
    timeframe_data_result = analyze_multi_timeframe_strategy(timeframe_data, current_price)
    print_multi_timeframe_strategy(timeframe_data_result)

    # Strategy 4: Volatility Position Sizing
    print(f"\n{'='*70}")
    print("RUNNING STRATEGY 4: VOLATILITY POSITION SIZING")
    print(f"{'='*70}")
    volatility_data = analyze_volatility_position_sizing(stats_14day, current_price)
    print_volatility_strategy(volatility_data)

    # Compare all strategies
    compare_strategies_summary(gradient_data, confidence_data, timeframe_data_result,
                              volatility_data, current_price)

    # Create comprehensive visualization
    print(f"\n{'='*70}")
    print("CREATING COMPREHENSIVE COMPARISON VISUALIZATION")
    print(f"{'='*70}")
    create_comparison_visualization(gradient_data, confidence_data, timeframe_data_result,
                                   volatility_data, stats_14day, current_price)

    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print("\n⚠️  DISCLAIMER:")
    print("This analysis is for educational purposes only.")
    print("Cryptocurrency trading carries substantial risk.")
    print("Always do your own research and never invest more than you can afford to lose.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
