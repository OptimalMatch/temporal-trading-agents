"""
Multi-Timeframe Trend Alignment Strategy

Trains ensembles at multiple forecast horizons (3, 7, 14, 21 days) and looks for
trend alignment across all timeframes. Strong signals occur when all timeframes agree.

Strategy Logic:
- ALL BULLISH: All timeframes predict gains - strong buy signal
- ALL BEARISH: All timeframes predict losses - strong sell/avoid signal
- SHORT-TERM BULLISH: Only near-term positive - quick trade
- LONG-TERM BULLISH: Only long-term positive - wait for dip
- MIXED: Conflicting signals - no clear trade

Position sizing increases with number of aligned timeframes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
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
    format_strategy_output,
    get_default_ensemble_configs,
)
from strategies.strategy_cache import cached_strategy


def train_multiple_timeframes(symbol: str, ensemble_module,
                              horizons: List[int] = [3, 7, 14, 21], interval: str = '1d') -> Dict:
    """
    Train ensembles for multiple forecast horizons with error handling.

    Args:
        symbol: Trading symbol
        ensemble_module: Loaded ensemble module
        horizons: List of forecast horizons in periods (days or hours depending on interval)
        interval: Data interval ('1d' for daily, '1h' for hourly)

    Returns:
        Dictionary mapping horizon to (stats, df) tuple
    """
    results = {}

    for horizon in horizons:
        max_retries = 2
        retry_count = 0

        while retry_count < max_retries:
            try:
                configs = get_default_ensemble_configs(horizon)
                interval_label = "HOUR" if interval == '1h' else "DAY"
                stats, df = train_ensemble(symbol, horizon, configs, f"{horizon}-{interval_label}", ensemble_module, interval=interval)
                results[horizon] = (stats, df)
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\n⚠️  Training failed for {horizon}-day horizon (attempt {retry_count}/{max_retries})")
                    print(f"   Error: {str(e)}")
                    print(f"   Retrying...")
                else:
                    print(f"\n✗ Failed to train {horizon}-day horizon after {max_retries} attempts")
                    print(f"  Error: {str(e)}")
                    print(f"  Skipping this horizon and continuing...")
                    # Continue with other horizons instead of failing completely

    if not results:
        raise RuntimeError("Failed to train any timeframe models")

    return results


@cached_strategy
def analyze_multi_timeframe_strategy(timeframe_data: Dict, current_price: float) -> Dict:
    """
    Analyze trading opportunity based on multi-timeframe alignment.

    Args:
        timeframe_data: Dictionary mapping horizon to (stats, df) tuple
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Validate we have at least 2 timeframes to analyze
    if len(timeframe_data) < 2:
        print(f"\n⚠️  Warning: Only {len(timeframe_data)} timeframe(s) available")
        print(f"   Multi-timeframe analysis requires at least 2 horizons")
        if len(timeframe_data) == 0:
            raise ValueError("No timeframe data available for analysis")

    # Calculate metrics for each timeframe
    # Convert horizon keys to strings for MongoDB compatibility
    timeframe_metrics = {}
    for horizon, (stats, df) in timeframe_data.items():
        metrics = calculate_forecast_metrics(stats, current_price, horizon)
        timeframe_metrics[str(horizon)] = metrics  # Use string key for MongoDB

    # Determine bullish/bearish for each timeframe
    directions = {}
    for horizon_str, metrics in timeframe_metrics.items():
        directions[horizon_str] = "BULLISH" if metrics['median_change_pct'] > 0 else "BEARISH"

    # Count alignment
    bullish_count = sum(1 for d in directions.values() if d == "BULLISH")
    bearish_count = sum(1 for d in directions.values() if d == "BEARISH")
    total_count = len(directions)

    # Determine overall alignment
    if bullish_count == total_count:
        alignment = "ALL_BULLISH"
        signal = "STRONG_BUY"
        position_size = 1.0
    elif bearish_count == total_count:
        alignment = "ALL_BEARISH"
        signal = "STRONG_SELL"
        position_size = 0.0  # Stay out
    elif bullish_count >= total_count * 0.75:
        alignment = "MOSTLY_BULLISH"
        signal = "BUY"
        position_size = 0.75
    elif bearish_count >= total_count * 0.75:
        alignment = "MOSTLY_BEARISH"
        signal = "SELL"
        position_size = 0.0
    else:
        # Check for short-term vs long-term divergence
        short_term_bullish = all(directions[h] == "BULLISH" for h in [3, 7] if h in directions)
        long_term_bearish = all(directions[h] == "BEARISH" for h in [14, 21] if h in directions)

        if short_term_bullish and long_term_bearish:
            alignment = "SHORT_TERM_OPPORTUNITY"
            signal = "QUICK_TRADE"
            position_size = 0.5
        else:
            alignment = "MIXED"
            signal = "NO_CLEAR_SIGNAL"
            position_size = 0.25

    # Calculate trade parameters using 14-day forecast as primary
    primary_horizon = 14 if 14 in timeframe_metrics else max(timeframe_metrics.keys())
    primary_metrics = timeframe_metrics[primary_horizon]

    entry_price = current_price
    target_price = primary_metrics['median']
    expected_gain_pct = primary_metrics['median_change_pct']

    stop_loss = calculate_stop_loss(entry_price, primary_metrics['min'], cushion_pct=2.0)
    risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)

    strategy_data = {
        'signal': signal,
        'alignment': alignment,
        'directions': directions,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'total_count': total_count,
        'position_size': position_size,
        'position_size_pct': position_size * 100,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'expected_gain_pct': expected_gain_pct,
        'risk_reward_ratio': risk_reward,
        'current_price': current_price,
        'timeframe_metrics': timeframe_metrics,
        'primary_horizon': primary_horizon,
    }

    return strategy_data


def print_multi_timeframe_strategy(strategy_data: Dict):
    """Print formatted output for multi-timeframe strategy."""
    print(f"\n{'='*70}")
    print("MULTI-TIMEFRAME TREND ALIGNMENT STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Timeframe analysis
    print(f"\n{'='*70}")
    print("TIMEFRAME ALIGNMENT ANALYSIS")
    print(f"{'='*70}")

    directions = strategy_data['directions']
    metrics = strategy_data['timeframe_metrics']

    for horizon in sorted(directions.keys()):
        direction = directions[horizon]
        change = metrics[horizon]['median_change_pct']
        target = metrics[horizon]['median']

        symbol = "↑" if direction == "BULLISH" else "↓"
        color_code = "+" if direction == "BULLISH" else ""

        print(f"\n{horizon:2d}-Day Forecast: {symbol} {direction:8s}")
        print(f"  Target: ${target:,.2f} ({color_code}{change:+.2f}%)")
        print(f"  Range:  ${metrics[horizon]['q25']:,.2f} to ${metrics[horizon]['q75']:,.2f}")

    print(f"\n{'='*70}")
    print(f"Alignment: {strategy_data['alignment']}")
    print(f"  Bullish: {strategy_data['bullish_count']}/{strategy_data['total_count']} timeframes")
    print(f"  Bearish: {strategy_data['bearish_count']}/{strategy_data['total_count']} timeframes")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"{'='*70}")

    signal = strategy_data['signal']

    if signal in ["STRONG_BUY", "BUY"]:
        print(f"\n📊 Trading Plan:")
        print(f"\n  Timeframe Alignment: {strategy_data['alignment']}")
        print(f"  Position Size: {strategy_data['position_size_pct']:.0f}% of standard")

        print(f"\n  1. BUY NOW at ${strategy_data['entry_price']:,.2f}")

        print(f"\n  2. PRIMARY TARGET (Day {strategy_data['primary_horizon']})")
        print(f"     Price: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected gain: {strategy_data['expected_gain_pct']:+.2f}%")

        # Show intermediate targets
        print(f"\n  3. INTERMEDIATE TARGETS:")
        for horizon in sorted(directions.keys()):
            if directions[horizon] == "BULLISH":
                target = metrics[horizon]['median']
                change = metrics[horizon]['median_change_pct']
                print(f"     Day {horizon:2d}: ${target:,.2f} ({change:+.2f}%)")

        print(f"\n  4. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        # Strategy interpretation
        print(f"\n  💡 Strategy Interpretation:")
        if signal == "STRONG_BUY":
            print(f"     ALL timeframes aligned - highest conviction trade")
            print(f"     Strong trend across short, medium, and long-term")
        else:
            print(f"     Most timeframes aligned - good confidence")
            print(f"     {strategy_data['bullish_count']}/{strategy_data['total_count']} timeframes bullish")

    elif signal == "QUICK_TRADE":
        print(f"\n📊 Trading Plan:")
        print(f"\n  Timeframe Divergence: {strategy_data['alignment']}")
        print(f"  Short-term opportunity with long-term headwinds")
        print(f"  Position Size: {strategy_data['position_size_pct']:.0f}% (reduced for quick trade)")

        print(f"\n  1. BUY NOW at ${strategy_data['entry_price']:,.2f}")

        # Show only short-term targets
        print(f"\n  2. SHORT-TERM TARGETS (exit early!):")
        for horizon in [3, 7]:
            if horizon in directions and directions[horizon] == "BULLISH":
                target = metrics[horizon]['median']
                change = metrics[horizon]['median_change_pct']
                print(f"     Day {horizon}: ${target:,.2f} ({change:+.2f}%) ← Take profit here")

        print(f"\n  3. EXIT BEFORE: Day {strategy_data['primary_horizon']}")
        print(f"     Long-term forecast: {directions.get(14, 'N/A')}")

        print(f"\n  4. TIGHT STOP LOSS: ${strategy_data['stop_loss']:,.2f}")

        print(f"\n  ⚠️  WARNING: Short-term trade only!")
        print(f"     Exit at short-term targets - don't hold for Day 14/21")

    elif signal in ["STRONG_SELL", "SELL"]:
        print(f"\n⚠️  {strategy_data['alignment']}")
        print(f"  {strategy_data['bearish_count']}/{strategy_data['total_count']} timeframes predict decline")
        print(f"  Expected decline: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"\n  Recommendation: Stay out or consider shorting (advanced)")

        print(f"\n  Forecast Targets:")
        for horizon in sorted(directions.keys()):
            target = metrics[horizon]['median']
            change = metrics[horizon]['median_change_pct']
            print(f"    Day {horizon:2d}: ${target:,.2f} ({change:+.2f}%)")

    else:  # NO_CLEAR_SIGNAL
        print(f"\n⚠️  Mixed signals across timeframes")
        print(f"  Bullish: {strategy_data['bullish_count']}/{strategy_data['total_count']} timeframes")
        print(f"  Bearish: {strategy_data['bearish_count']}/{strategy_data['total_count']} timeframes")
        print(f"\n  Recommendation: Wait for better alignment")

        print(f"\n  Timeframe Breakdown:")
        for horizon in sorted(directions.keys()):
            direction = directions[horizon]
            change = metrics[horizon]['median_change_pct']
            print(f"    {horizon:2d}-day: {direction:8s} ({change:+.2f}%)")

    # Risk Analysis
    primary_metrics = strategy_data['timeframe_metrics'][strategy_data['primary_horizon']]
    print(f"\n{'='*70}")
    print("RISK ANALYSIS (Primary Timeframe)")
    print(f"{'='*70}")
    print(f"\nWorst Case: ${primary_metrics['min']:,.2f} ({primary_metrics['worst_case_loss']:+.2f}%)")
    print(f"Best Case: ${primary_metrics['max']:,.2f} ({primary_metrics['best_case_gain']:+.2f}%)")
    print(f"Forecast Range: {primary_metrics['forecast_range_pct']:.1f}%")

    print(f"\n{'='*70}")


def visualize_multi_timeframe_strategy(strategy_data: Dict, timeframe_data: Dict,
                                       save_path: str = 'multi_timeframe_strategy.png'):
    """
    Create visualization showing the multi-timeframe strategy.

    Args:
        strategy_data: Strategy analysis data
        timeframe_data: Dictionary mapping horizon to (stats, df) tuple
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    directions = strategy_data['directions']
    metrics = strategy_data['timeframe_metrics']

    # Plot 1: All timeframes together
    ax1 = axes[0, 0]

    ax1.axhline(current_price, color='black', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.7)

    colors = ['blue', 'green', 'red', 'purple']
    for i, horizon in enumerate(sorted(timeframe_data.keys())):
        stats, _ = timeframe_data[horizon]
        days = np.arange(0, horizon)
        color = colors[i % len(colors)]

        ax1.plot(days, stats['median'], color=color, linewidth=2,
                marker='o', markersize=4, label=f'{horizon}-Day', alpha=0.8)

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Multi-Timeframe Alignment: {strategy_data["alignment"]}',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Expected returns by timeframe
    ax2 = axes[0, 1]

    horizons = sorted(directions.keys())
    expected_returns = [metrics[h]['median_change_pct'] for h in horizons]
    colors_bars = ['green' if r > 0 else 'red' for r in expected_returns]

    bars = ax2.bar([f'{h}d' for h in horizons], expected_returns,
                   color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)

    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Timeframe', fontsize=12)
    ax2.set_ylabel('Expected Return (%)', fontsize=12)
    ax2.set_title('Expected Returns by Timeframe', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, ret in zip(bars, expected_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2,
                height + (0.5 if height > 0 else -0.5),
                f'{ret:+.1f}%', ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Plot 3: Alignment summary
    ax3 = axes[1, 0]
    ax3.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Alignment: {strategy_data['alignment']}\n\n"
    strategy_text += f"Timeframes:\n"
    for horizon in sorted(horizons):
        direction = directions[horizon]
        change = metrics[horizon]['median_change_pct']
        symbol = "↑" if direction == "BULLISH" else "↓"
        strategy_text += f"  {horizon:2d}-day: {symbol} {direction:8s} ({change:+.2f}%)\n"

    strategy_text += f"\nBullish: {strategy_data['bullish_count']}/{strategy_data['total_count']}\n"
    strategy_text += f"Bearish: {strategy_data['bearish_count']}/{strategy_data['total_count']}\n\n"

    if strategy_data['signal'] not in ["STRONG_SELL", "SELL", "NO_CLEAR_SIGNAL"]:
        strategy_text += f"Position Size: {strategy_data['position_size_pct']:.0f}%\n"
        strategy_text += f"Target: ${strategy_data['target_price']:,.2f}\n"
        strategy_text += f"Expected: {strategy_data['expected_gain_pct']:+.2f}%\n"

    ax3.text(0.1, 0.9, strategy_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Position sizing based on alignment
    ax4 = axes[1, 1]

    alignment_types = ['ALL\nBULLISH', 'MOSTLY\nBULLISH', 'MIXED', 'MOSTLY\nBEARISH', 'ALL\nBEARISH']
    position_sizes = [100, 75, 25, 0, 0]
    colors_pos = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']

    bars = ax4.bar(alignment_types, position_sizes, color=colors_pos, alpha=0.7,
                   edgecolor='black', linewidth=2)

    # Highlight current alignment
    alignment_map = {
        'ALL_BULLISH': 0,
        'MOSTLY_BULLISH': 1,
        'SHORT_TERM_OPPORTUNITY': 2,
        'MIXED': 2,
        'MOSTLY_BEARISH': 3,
        'ALL_BEARISH': 4
    }
    current_idx = alignment_map.get(strategy_data['alignment'], 2)
    bars[current_idx].set_edgecolor('blue')
    bars[current_idx].set_linewidth(4)

    ax4.set_ylabel('Position Size (%)', fontsize=12)
    ax4.set_xlabel('Alignment Type', fontsize=12)
    ax4.set_title(f'Position Sizing by Alignment\nCurrent: {strategy_data["alignment"]}',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, size in zip(bars, position_sizes):
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, height + 2,
                    f'{size}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Multi-Timeframe Trend Alignment Strategy analysis."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Train multiple timeframes
    print(f"\n{'='*70}")
    print(f"MULTI-TIMEFRAME STRATEGY ANALYSIS - {symbol}")
    print(f"{'='*70}")

    horizons = [3, 7, 14, 21]
    timeframe_data = train_multiple_timeframes(symbol, ensemble, horizons)

    # Get current price from any dataframe
    current_price = timeframe_data[14][1]['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_multi_timeframe_strategy(timeframe_data, current_price)

    # Print results
    print_multi_timeframe_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_multi_timeframe_strategy(strategy_data, timeframe_data)

    print(f"\n{'='*70}")
    print("MULTI-TIMEFRAME STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
