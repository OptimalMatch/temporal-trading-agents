"""
Confidence-Weighted Ensemble Strategy

Uses model agreement and confidence levels to determine position sizing and trade signals.
Higher ensemble confidence = larger positions. Lower confidence = smaller positions or no trade.

Strategy Logic:
- HIGH confidence (80%+ agreement): Full position size
- MEDIUM confidence (65-80% agreement): 50% position size
- LOW confidence (55-65% agreement): 25% position size
- VERY LOW confidence (<55% agreement): No trade

Direction determined by whether models predict above or below current price.
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
    calculate_stop_loss,
    calculate_risk_reward_ratio,
    format_strategy_output,
    get_default_ensemble_configs,
)


def analyze_confidence_weighted_strategy(stats_14day: Dict, current_price: float) -> Dict:
    """
    Analyze trading opportunity based on ensemble confidence.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Calculate confidence metrics
    confidence = calculate_ensemble_confidence(stats_14day, current_price)

    # Get forecast metrics
    metrics = calculate_forecast_metrics(stats_14day, current_price, 14)

    # Determine direction and position size
    bullish = confidence['pct_above'] > confidence['pct_below']
    direction = "BULLISH" if bullish else "BEARISH"

    # Position sizing based on confidence
    confidence_level = confidence['confidence_level']

    if confidence_level == "HIGH":
        position_size = 1.0  # Full position
        signal = f"{'BUY' if bullish else 'SELL'}_HIGH_CONFIDENCE"
    elif confidence_level == "MEDIUM":
        position_size = 0.5  # Half position
        signal = f"{'BUY' if bullish else 'SELL'}_MEDIUM_CONFIDENCE"
    elif confidence_level == "LOW":
        position_size = 0.25  # Quarter position
        signal = f"{'BUY' if bullish else 'SELL'}_LOW_CONFIDENCE"
    else:  # VERY_LOW
        position_size = 0.0  # No trade
        signal = "NO_TRADE"

    # Calculate trade parameters
    entry_price = current_price
    target_price = metrics['median']

    if bullish:
        expected_gain_pct = metrics['median_change_pct']
        stop_loss = calculate_stop_loss(entry_price, metrics['min'], cushion_pct=2.0)
    else:
        # For bearish, we're shorting or staying out
        expected_gain_pct = -metrics['median_change_pct']  # Gain from short
        stop_loss = entry_price * 1.05  # 5% above for short

    risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss) if bullish else None

    strategy_data = {
        'signal': signal,
        'direction': direction,
        'confidence_level': confidence_level,
        'confidence_metrics': confidence,
        'position_size': position_size,
        'position_size_pct': position_size * 100,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'expected_gain_pct': expected_gain_pct,
        'risk_reward_ratio': risk_reward,
        'current_price': current_price,
        'metrics': metrics,
    }

    return strategy_data


def print_confidence_strategy(strategy_data: Dict):
    """Print formatted output for confidence-weighted strategy."""
    print(f"\n{'='*70}")
    print("CONFIDENCE-WEIGHTED ENSEMBLE STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Confidence metrics
    conf = strategy_data['confidence_metrics']
    print(f"\n{'='*70}")
    print("ENSEMBLE CONFIDENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nModels predicting ABOVE current price: {conf['pct_above']:.1f}%")
    print(f"Models predicting BELOW current price: {conf['pct_below']:.1f}%")
    print(f"\nAgreement Level: {conf['agreement']:.1f}%")
    print(f"Confidence Level: {strategy_data['confidence_level']}")
    print(f"Prediction Std Dev: ${conf['prediction_std']:,.2f}")
    print(f"Coefficient of Variation: {conf['prediction_cv']:.2f}%")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"DIRECTION: {strategy_data['direction']}")
    print(f"{'='*70}")

    signal = strategy_data['signal']

    if signal != "NO_TRADE":
        print(f"\n📊 Trading Plan:")
        print(f"\n  Position Size: {strategy_data['position_size_pct']:.0f}% of normal")
        print(f"  Rationale: {strategy_data['confidence_level']} confidence")

        if strategy_data['direction'] == "BULLISH":
            print(f"\n  1. BUY at ${strategy_data['entry_price']:,.2f}")
            print(f"     Position: {strategy_data['position_size_pct']:.0f}% of standard size")
            print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
            print(f"     Expected gain: {strategy_data['expected_gain_pct']:+.2f}%")
            print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
            if strategy_data['risk_reward_ratio']:
                print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")
        else:  # BEARISH
            print(f"\n  1. SELL/SHORT at ${strategy_data['entry_price']:,.2f}")
            print(f"     Position: {strategy_data['position_size_pct']:.0f}% of standard size")
            print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
            print(f"     Expected gain from short: {strategy_data['expected_gain_pct']:+.2f}%")
            print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")

        # Confidence interpretation
        print(f"\n  💡 Confidence Interpretation:")
        if strategy_data['confidence_level'] == "HIGH":
            print(f"     Strong consensus among models - high conviction trade")
            print(f"     {conf['agreement']:.0f}% of models agree on direction")
        elif strategy_data['confidence_level'] == "MEDIUM":
            print(f"     Moderate consensus - reduced position to manage risk")
            print(f"     {conf['agreement']:.0f}% of models agree on direction")
        else:  # LOW
            print(f"     Weak consensus - small position or wait for better setup")
            print(f"     {conf['agreement']:.0f}% of models agree on direction")

    else:  # NO_TRADE
        print(f"\n⚠️  No clear consensus among ensemble models")
        print(f"  Agreement: {conf['agreement']:.1f}% ({strategy_data['confidence_level']} confidence)")
        print(f"  Recommendation: Wait for higher confidence setup")
        print(f"\n  Models are nearly split:")
        print(f"    - {conf['pct_above']:.1f}% predict ABOVE current price")
        print(f"    - {conf['pct_below']:.1f}% predict BELOW current price")

    # Risk Analysis
    metrics = strategy_data['metrics']
    print(f"\n{'='*70}")
    print("RISK ANALYSIS")
    print(f"{'='*70}")
    print(f"\nForecast Range (Q25-Q75): ${metrics['q25']:,.2f} to ${metrics['q75']:,.2f}")
    print(f"Range as % of median: {metrics['forecast_range_pct']:.1f}%")
    print(f"\nWorst Case: ${metrics['min']:,.2f} ({metrics['worst_case_loss']:+.2f}%)")
    print(f"Best Case: ${metrics['max']:,.2f} ({metrics['best_case_gain']:+.2f}%)")

    print(f"\n{'='*70}")


def visualize_confidence_strategy(strategy_data: Dict, stats: Dict,
                                  save_path: str = 'confidence_strategy.png'):
    """
    Create visualization showing the confidence-weighted strategy.

    Args:
        strategy_data: Strategy analysis data
        stats: Forecast statistics
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    conf = strategy_data['confidence_metrics']
    metrics = strategy_data['metrics']

    # Plot 1: Forecast with confidence bands
    ax1 = axes[0, 0]
    days = np.arange(0, 14)

    ax1.axhline(current_price, color='green', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.7)

    ax1.plot(days, stats['median'], 'b-', linewidth=3, marker='o',
             markersize=6, label='Median Forecast', alpha=0.8)
    ax1.fill_between(days, stats['q25'], stats['q75'],
                     alpha=0.3, color='blue', label='Q25-Q75 Range')
    ax1.fill_between(days, stats['min'], stats['max'],
                     alpha=0.1, color='gray', label='Min-Max Range')

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{strategy_data["confidence_level"]} Confidence Forecast',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Model agreement distribution
    ax2 = axes[0, 1]

    final_predictions = stats['all_predictions'][:, -1]
    ax2.hist(final_predictions, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(current_price, color='red', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}')
    ax2.axvline(metrics['median'], color='green', linestyle='-', linewidth=2,
                label=f'Median: ${metrics["median"]:,.0f}')

    ax2.set_xlabel('Predicted Price ($)', fontsize=12)
    ax2.set_ylabel('Number of Models', fontsize=12)
    ax2.set_title('Ensemble Prediction Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text with confidence metrics
    ax2.text(0.05, 0.95,
             f'Above current: {conf["pct_above"]:.1f}%\nBelow current: {conf["pct_below"]:.1f}%\nStd Dev: ${conf["prediction_std"]:,.0f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    # Plot 3: Signal and position sizing
    ax3 = axes[1, 0]
    ax3.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Direction: {strategy_data['direction']}\n"
    strategy_text += f"Confidence: {strategy_data['confidence_level']}\n\n"
    strategy_text += f"Position Size: {strategy_data['position_size_pct']:.0f}%\n\n"

    if strategy_data['signal'] != "NO_TRADE":
        strategy_text += f"Entry: ${strategy_data['entry_price']:,.2f}\n"
        strategy_text += f"Target: ${strategy_data['target_price']:,.2f}\n"
        strategy_text += f"Stop Loss: ${strategy_data['stop_loss']:,.2f}\n"
        strategy_text += f"Expected: {strategy_data['expected_gain_pct']:+.2f}%\n"
    else:
        strategy_text += "No trade - insufficient confidence\n"
        strategy_text += f"Agreement: {conf['agreement']:.1f}%\n"

    ax3.text(0.1, 0.9, strategy_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Confidence levels comparison
    ax4 = axes[1, 1]

    confidence_levels = ['VERY_LOW\n(<55%)', 'LOW\n(55-65%)', 'MEDIUM\n(65-80%)', 'HIGH\n(80%+)']
    position_sizes = [0, 25, 50, 100]
    colors_conf = ['red', 'orange', 'yellow', 'green']

    bars = ax4.bar(confidence_levels, position_sizes, color=colors_conf, alpha=0.7,
                   edgecolor='black', linewidth=2)

    # Highlight current level
    current_level_idx = {'VERY_LOW': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    current_idx = current_level_idx.get(strategy_data['confidence_level'], 0)
    bars[current_idx].set_edgecolor('blue')
    bars[current_idx].set_linewidth(4)

    ax4.set_ylabel('Position Size (%)', fontsize=12)
    ax4.set_xlabel('Confidence Level', fontsize=12)
    ax4.set_title(f'Position Sizing by Confidence\nCurrent: {strategy_data["confidence_level"]} ({conf["agreement"]:.1f}%)',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, size in zip(bars, position_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{size}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Confidence-Weighted Ensemble Strategy analysis."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"CONFIDENCE-WEIGHTED ENSEMBLE STRATEGY - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    current_price = df_14day['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_confidence_weighted_strategy(stats_14day, current_price)

    # Print results
    print_confidence_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_confidence_strategy(strategy_data, stats_14day)

    print(f"\n{'='*70}")
    print("CONFIDENCE-WEIGHTED STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
