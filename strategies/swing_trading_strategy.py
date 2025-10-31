"""
Swing Trading with Intra-Forecast Opportunities Strategy

Uses daily predictions within the forecast window to identify multiple swing
trading opportunities. Instead of single entry/exit, captures multiple swings
within the 14-day forecast period.

Strategy Logic:
- Analyze daily forecast to find local peaks and troughs
- Identify profitable swing opportunities (peak - trough > threshold)
- Create multi-swing trading plan with partial position sizing
- Take profits at each predicted peak
- Re-enter at each predicted trough
- Risk management for each swing leg

Signals:
- MULTI_SWING: Multiple profitable swings identified
- SINGLE_SWING: One clear swing opportunity
- TREND_NO_SWINGS: Trend but no clear swings (hold through)
- NO_SWINGS: Flat or unclear, no swing opportunities
"""

import numpy as np
from scipy.signal import argrelextrema
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
    get_default_ensemble_configs,
)


def identify_swings(median_forecast: np.ndarray, min_swing_pct: float = 2.0) -> Dict:
    """
    Identify local peaks and troughs in the forecast for swing trading.

    Args:
        median_forecast: Array of median forecast values
        min_swing_pct: Minimum percentage change to qualify as a swing

    Returns:
        Dictionary with swing analysis
    """
    # Find local maxima (peaks) and minima (troughs)
    # Using order=2 means comparing with 2 neighbors on each side
    peaks = argrelextrema(median_forecast, np.greater, order=2)[0]
    troughs = argrelextrema(median_forecast, np.less, order=2)[0]

    # Add start and end points if they make sense
    if len(peaks) == 0 or (len(peaks) > 0 and median_forecast[0] > median_forecast[peaks[0]]):
        peaks = np.insert(peaks, 0, 0)
    if len(troughs) == 0 or (len(troughs) > 0 and median_forecast[0] < median_forecast[troughs[0]]):
        troughs = np.insert(troughs, 0, 0)

    # Add endpoint
    if len(peaks) == 0 or (len(peaks) > 0 and median_forecast[-1] > median_forecast[peaks[-1]]):
        peaks = np.append(peaks, len(median_forecast) - 1)
    if len(troughs) == 0 or (len(troughs) > 0 and median_forecast[-1] < median_forecast[troughs[-1]]):
        troughs = np.append(troughs, len(median_forecast) - 1)

    # Identify swing opportunities
    swings = []

    # Look for trough -> peak swings (buy at trough, sell at peak)
    for trough_idx in troughs:
        trough_price = median_forecast[trough_idx]

        # Find next peak after this trough
        future_peaks = peaks[peaks > trough_idx]
        if len(future_peaks) > 0:
            peak_idx = future_peaks[0]
            peak_price = median_forecast[peak_idx]

            # Calculate swing magnitude
            swing_pct = ((peak_price - trough_price) / trough_price) * 100

            if swing_pct >= min_swing_pct:
                swings.append({
                    'type': 'LONG',
                    'entry_day': int(trough_idx),
                    'entry_price': trough_price,
                    'exit_day': int(peak_idx),
                    'exit_price': peak_price,
                    'gain_pct': swing_pct,
                    'duration': int(peak_idx - trough_idx),
                })

    # Look for peak -> trough swings (short opportunities)
    for peak_idx in peaks:
        peak_price = median_forecast[peak_idx]

        # Find next trough after this peak
        future_troughs = troughs[troughs > peak_idx]
        if len(future_troughs) > 0:
            trough_idx = future_troughs[0]
            trough_price = median_forecast[trough_idx]

            # Calculate swing magnitude
            swing_pct = ((peak_price - trough_price) / peak_price) * 100

            if swing_pct >= min_swing_pct:
                swings.append({
                    'type': 'SHORT',
                    'entry_day': int(peak_idx),
                    'entry_price': peak_price,
                    'exit_day': int(trough_idx),
                    'exit_price': trough_price,
                    'gain_pct': swing_pct,
                    'duration': int(trough_idx - peak_idx),
                })

    # Sort swings by entry day
    swings.sort(key=lambda x: x['entry_day'])

    return {
        'swings': swings,
        'num_swings': len(swings),
        'peaks': peaks,
        'troughs': troughs,
        'total_swing_gain': sum(s['gain_pct'] for s in swings),
    }


def analyze_swing_trading_strategy(stats_14day: Dict, current_price: float) -> Dict:
    """
    Analyze swing trading opportunities within the forecast.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    median_forecast = stats_14day['median']

    # Identify swings
    swing_analysis = identify_swings(median_forecast, min_swing_pct=2.0)

    # Get forecast metrics
    forecast_metrics = calculate_forecast_metrics(stats_14day, current_price, 14)

    num_swings = swing_analysis['num_swings']
    swings = swing_analysis['swings']

    # Filter to only long swings that start from day 0 or near current price
    executable_swings = [s for s in swings if s['type'] == 'LONG' and s['entry_day'] <= 2]

    # Determine signal
    if len(executable_swings) >= 2:
        signal = "MULTI_SWING"
        strategy_type = "Multiple swing opportunities"
    elif len(executable_swings) == 1:
        signal = "SINGLE_SWING"
        strategy_type = "Single swing trade"
    elif forecast_metrics['median_change_pct'] > 5:
        signal = "TREND_NO_SWINGS"
        strategy_type = "Trend trade without swings"
    else:
        signal = "NO_SWINGS"
        strategy_type = "No clear swing opportunities"

    # Calculate position sizing for swings
    # Divide capital across multiple swings
    if len(executable_swings) > 0:
        position_per_swing = 1.0 / len(executable_swings)  # Split equally
    else:
        position_per_swing = 0.0

    # Calculate total expected return from all swings
    total_expected_return = sum(s['gain_pct'] * position_per_swing for s in executable_swings)

    strategy_data = {
        'signal': signal,
        'strategy_type': strategy_type,
        'swing_analysis': swing_analysis,
        'executable_swings': executable_swings,
        'num_executable_swings': len(executable_swings),
        'position_per_swing': position_per_swing,
        'position_per_swing_pct': position_per_swing * 100,
        'total_expected_return': total_expected_return,
        'current_price': current_price,
        'forecast_metrics': forecast_metrics,
    }

    return strategy_data


def print_swing_trading_strategy(strategy_data: Dict):
    """Print formatted output for swing trading strategy."""
    print(f"\n{'='*70}")
    print("SWING TRADING WITH INTRA-FORECAST OPPORTUNITIES")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Swing analysis summary
    sa = strategy_data['swing_analysis']
    print(f"\n{'='*70}")
    print("SWING ANALYSIS")
    print(f"{'='*70}")
    print(f"\nTotal Swings Identified: {sa['num_swings']}")
    print(f"Executable Swings (from current price): {strategy_data['num_executable_swings']}")
    print(f"Total Potential Gain: {sa['total_swing_gain']:.2f}%")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"STRATEGY: {strategy_data['strategy_type']}")
    print(f"{'='*70}")

    signal = strategy_data['signal']
    swings = strategy_data['executable_swings']

    if signal in ["MULTI_SWING", "SINGLE_SWING"]:
        print(f"\n📊 SWING TRADING PLAN:")
        print(f"\n  Total Capital Allocation: 100%")
        print(f"  Position per Swing: {strategy_data['position_per_swing_pct']:.1f}%")
        print(f"  Number of Swings: {len(swings)}")
        print(f"  Total Expected Return: {strategy_data['total_expected_return']:.2f}%")

        print(f"\n{'='*70}")
        print("SWING-BY-SWING BREAKDOWN")
        print(f"{'='*70}")

        for i, swing in enumerate(swings, 1):
            print(f"\n  SWING #{i}:")
            print(f"  {'='*60}")
            print(f"    Type: {swing['type']}")
            print(f"    Duration: {swing['duration']} days")

            print(f"\n    ENTRY (Day {swing['entry_day']}):")
            print(f"      Price: ${swing['entry_price']:,.2f}")
            print(f"      Action: {'BUY' if swing['type'] == 'LONG' else 'SHORT'} {strategy_data['position_per_swing_pct']:.1f}% of capital")

            print(f"\n    EXIT (Day {swing['exit_day']}):")
            print(f"      Price: ${swing['exit_price']:,.2f}")
            print(f"      Action: {'SELL' if swing['type'] == 'LONG' else 'COVER'}")
            print(f"      Gain: {swing['gain_pct']:+.2f}%")

            # Calculate stop loss for this swing
            if swing['type'] == 'LONG':
                stop_loss = swing['entry_price'] * 0.97
                print(f"\n    STOP LOSS: ${stop_loss:,.2f} (-3%)")
            else:
                stop_loss = swing['entry_price'] * 1.03
                print(f"\n    STOP LOSS: ${stop_loss:,.2f} (+3%)")

            # Position-weighted gain
            weighted_gain = swing['gain_pct'] * strategy_data['position_per_swing']
            print(f"\n    Position-Weighted Gain: {weighted_gain:+.2f}%")

        print(f"\n{'='*70}")
        print("SWING EXECUTION STRATEGY")
        print(f"{'='*70}")

        if signal == "MULTI_SWING":
            print(f"\n  💡 Multi-Swing Approach:")
            print(f"    1. Divide capital into {len(swings)} equal parts")
            print(f"    2. Enter each swing at the predicted trough")
            print(f"    3. Take profit at each predicted peak")
            print(f"    4. Immediately re-enter next swing if available")
            print(f"    5. Use tight stops on each individual swing")
            print(f"\n  Benefits:")
            print(f"    - Capture multiple opportunities within forecast")
            print(f"    - Reduce risk through position timing")
            print(f"    - Take profits multiple times")
            print(f"    - Compound gains across swings")

        else:  # SINGLE_SWING
            print(f"\n  💡 Single-Swing Approach:")
            print(f"    1. Enter full position at predicted trough")
            print(f"    2. Hold through to predicted peak")
            print(f"    3. Take profit at peak")
            print(f"    4. Simple buy-hold-sell strategy")

    elif signal == "TREND_NO_SWINGS":
        fm = strategy_data['forecast_metrics']
        print(f"\n  Strong trend detected but no clear swings")
        print(f"  Forecast: {fm['median_change_pct']:+.2f}%")
        print(f"\n  💡 Recommendation:")
        print(f"    - Use standard BUY AND HOLD strategy")
        print(f"    - No swing trading opportunities")
        print(f"    - Strong directional trend suggests holding full period")
        print(f"    - Entry: ${strategy_data['current_price']:,.2f}")
        print(f"    - Target: ${fm['median']:,.2f} (Day 14)")
        print(f"    - Expected: {fm['median_change_pct']:+.2f}%")

    else:  # NO_SWINGS
        print(f"\n  No profitable swing opportunities identified")
        print(f"  Forecast shows flat or uncertain movement")
        print(f"\n  💡 Recommendation:")
        print(f"    - Wait for clearer swing patterns")
        print(f"    - Minimum swing threshold: 2%")
        print(f"    - Current forecast lacks profitable swings")

    # Risk analysis
    print(f"\n{'='*70}")
    print("RISK ANALYSIS")
    print(f"{'='*70}")

    if len(swings) > 0:
        avg_swing_gain = np.mean([s['gain_pct'] for s in swings])
        avg_duration = np.mean([s['duration'] for s in swings])
        print(f"\nAverage Swing Gain: {avg_swing_gain:.2f}%")
        print(f"Average Swing Duration: {avg_duration:.1f} days")
        print(f"Total Executable Swings: {len(swings)}")
        print(f"Risk per Swing: 3% stop loss")
        print(f"Total Capital at Risk: {len(swings) * strategy_data['position_per_swing'] * 3:.1f}%")

    fm = strategy_data['forecast_metrics']
    print(f"\nForecast Uncertainty: {fm['forecast_range_pct']:.1f}%")
    print(f"Worst Case: ${fm['min']:,.2f} ({fm['worst_case_loss']:+.2f}%)")
    print(f"Best Case: ${fm['max']:,.2f} ({fm['best_case_gain']:+.2f}%)")

    print(f"\n{'='*70}")


def visualize_swing_trading_strategy(strategy_data: Dict, stats: Dict,
                                     save_path: str = 'swing_trading_strategy.png'):
    """Create visualization for swing trading strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    median = stats['median']
    sa = strategy_data['swing_analysis']
    swings = strategy_data['executable_swings']

    # Plot 1: Forecast with swings marked
    ax1 = axes[0, 0]
    days = np.arange(0, 14)

    ax1.plot(days, median, 'b-', linewidth=3, marker='o', markersize=6,
            label='Median Forecast', alpha=0.8)

    ax1.axhline(current_price, color='black', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.5)

    # Mark all peaks and troughs
    for peak in sa['peaks']:
        ax1.scatter([peak], [median[peak]], s=200, color='red', marker='^',
                   zorder=5, alpha=0.6, edgecolors='black', linewidths=2)

    for trough in sa['troughs']:
        ax1.scatter([trough], [median[trough]], s=200, color='green', marker='v',
                   zorder=5, alpha=0.6, edgecolors='black', linewidths=2)

    # Draw swing arrows
    for swing in swings:
        ax1.annotate('',
                    xy=(swing['exit_day'], swing['exit_price']),
                    xytext=(swing['entry_day'], swing['entry_price']),
                    arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen', alpha=0.7))

        # Add gain label
        mid_day = (swing['entry_day'] + swing['exit_day']) / 2
        mid_price = (swing['entry_price'] + swing['exit_price']) / 2
        ax1.text(mid_day, mid_price, f"+{swing['gain_pct']:.1f}%",
                fontsize=10, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Swing Trading Opportunities - {strategy_data["signal"]}',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Swing gains
    ax2 = axes[0, 1]

    if len(swings) > 0:
        swing_labels = [f"Swing {i+1}\n(D{s['entry_day']}-D{s['exit_day']})"
                       for i, s in enumerate(swings)]
        swing_gains = [s['gain_pct'] for s in swings]

        bars = ax2.bar(swing_labels, swing_gains, color='green', alpha=0.7,
                      edgecolor='black', linewidth=2)

        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Gain per Swing (%)', fontsize=12)
        ax2.set_title('Individual Swing Gains', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, gain in zip(bars, swing_gains):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                    f'{gain:+.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        # Add average line
        avg_gain = np.mean(swing_gains)
        ax2.axhline(avg_gain, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Avg: {avg_gain:.1f}%')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No Executable Swings', transform=ax2.transAxes,
                ha='center', va='center', fontsize=14)

    # Plot 3: Cumulative returns comparison
    ax3 = axes[1, 0]

    # Buy-and-hold return
    bah_returns = ((median - median[0]) / median[0]) * 100

    ax3.plot(days, bah_returns, 'b-', linewidth=2, marker='o', markersize=4,
            label='Buy & Hold', alpha=0.7)

    # Swing trading returns (cumulative)
    swing_returns = np.zeros(14)
    if len(swings) > 0:
        for swing in swings:
            # Add swing gain to days after the swing completes
            for day in range(swing['exit_day'], 14):
                swing_returns[day] += swing['gain_pct'] * strategy_data['position_per_swing']

    ax3.plot(days, swing_returns, 'g-', linewidth=2, marker='s', markersize=4,
            label='Swing Trading', alpha=0.7)

    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Days', fontsize=12)
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax3.set_title('Swing Trading vs Buy & Hold', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add final comparison
    final_bah = bah_returns[-1]
    final_swing = swing_returns[-1]
    ax3.text(0.05, 0.95,
            f'Final Returns:\nB&H: {final_bah:+.1f}%\nSwing: {final_swing:+.1f}%',
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=10)

    # Plot 4: Strategy summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Swing Summary:\n"
    strategy_text += f"  Total Swings: {sa['num_swings']}\n"
    strategy_text += f"  Executable: {len(swings)}\n"
    strategy_text += f"  Position/Swing: {strategy_data['position_per_swing_pct']:.1f}%\n\n"

    if len(swings) > 0:
        strategy_text += f"Expected Returns:\n"
        strategy_text += f"  Total: {strategy_data['total_expected_return']:.2f}%\n"
        strategy_text += f"  Avg/Swing: {np.mean([s['gain_pct'] for s in swings]):.2f}%\n"
        strategy_text += f"  Max Swing: {max([s['gain_pct'] for s in swings]):.2f}%\n"

    ax4.text(0.1, 0.9, strategy_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Swing Trading with Intra-Forecast Strategy."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"SWING TRADING STRATEGY - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    current_price = df_14day['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_swing_trading_strategy(stats_14day, current_price)

    # Print results
    print_swing_trading_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_swing_trading_strategy(strategy_data, stats_14day)

    print(f"\n{'='*70}")
    print("SWING TRADING STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
