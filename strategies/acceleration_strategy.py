"""
Acceleration/Deceleration Momentum Strategy

Analyzes changes in forecast momentum to identify accelerating or decelerating trends.
Compares rate of change between forecast periods to determine if gains are speeding up
or slowing down.

Strategy Logic:
- Calculate momentum for different periods (Day 0-7, Day 7-14)
- Compare momentum between periods to detect acceleration/deceleration
- Accelerating gains = stay in position, target further
- Decelerating gains = take profit early before reversal
- Acceleration from negative = strong reversal signal
- Deceleration to negative = exit immediately

Signals:
- ACCELERATING_GAINS: Momentum increasing, ride the trend
- DECELERATING_GAINS: Momentum slowing, take early profit
- ACCELERATION_REVERSAL: Strong reversal from decline to growth
- DECELERATION_REVERSAL: Trend reversing, exit positions
- STEADY_MOMENTUM: Constant momentum, hold as planned
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
    calculate_stop_loss,
    calculate_risk_reward_ratio,
    get_default_ensemble_configs,
)
from strategies.strategy_cache import cached_strategy


def calculate_momentum_metrics(stats_14day: Dict, current_price: float) -> Dict:
    """
    Calculate momentum and acceleration metrics.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with momentum metrics
    """
    median = stats_14day['median']

    # Calculate returns for each period
    period1_start = median[0]
    period1_end = median[6]  # Day 7
    period2_start = median[7]  # Day 7
    period2_end = median[13]  # Day 14

    # Period returns
    period1_return = ((period1_end - period1_start) / period1_start) * 100
    period2_return = ((period2_end - period2_start) / period2_start) * 100

    # Average daily returns
    period1_daily = period1_return / 7
    period2_daily = period2_return / 7

    # Acceleration (change in momentum)
    acceleration = period2_daily - period1_daily

    # Determine momentum state
    if period1_daily > 0 and period2_daily > period1_daily * 1.2:
        momentum_state = "ACCELERATING_GAINS"
    elif period1_daily > 0 and period2_daily > 0 and period2_daily < period1_daily * 0.8:
        momentum_state = "DECELERATING_GAINS"
    elif period1_daily < 0 and period2_daily > 0:
        momentum_state = "ACCELERATION_REVERSAL"
    elif period1_daily > 0 and period2_daily < 0:
        momentum_state = "DECELERATION_REVERSAL"
    elif abs(period2_daily - period1_daily) < period1_daily * 0.2:
        momentum_state = "STEADY_MOMENTUM"
    else:
        momentum_state = "MIXED"

    # Calculate overall trend strength
    total_return = ((median[-1] - median[0]) / median[0]) * 100
    avg_daily_return = total_return / 14

    return {
        'period1_return': period1_return,
        'period2_return': period2_return,
        'period1_daily': period1_daily,
        'period2_daily': period2_daily,
        'acceleration': acceleration,
        'momentum_state': momentum_state,
        'total_return': total_return,
        'avg_daily_return': avg_daily_return,
        'period1_end_price': period1_end,
        'period2_end_price': period2_end,
    }


@cached_strategy
def analyze_acceleration_strategy(stats_14day: Dict, current_price: float) -> Dict:
    """
    Analyze trading opportunity based on momentum acceleration/deceleration.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Calculate momentum metrics
    momentum = calculate_momentum_metrics(stats_14day, current_price)

    # Get forecast metrics
    forecast_metrics = calculate_forecast_metrics(stats_14day, current_price, 14)

    state = momentum['momentum_state']
    entry_price = current_price

    # Determine strategy based on momentum state
    if state == "ACCELERATING_GAINS":
        # Momentum increasing - stay in and target full period
        signal = "BUY_AND_HOLD_FULL"
        target_price = forecast_metrics['q75']  # Target optimistic scenario
        target_day = 14
        position_size = 1.25  # 125% - high conviction
        stop_loss = calculate_stop_loss(entry_price, forecast_metrics['q25'], cushion_pct=2.0)

        strategy_rationale = "Momentum accelerating - gains speeding up"
        action = "Buy now and hold full 14 days, momentum suggests continued gains"

    elif state == "DECELERATING_GAINS":
        # Momentum slowing - take profit early
        signal = "BUY_EARLY_EXIT"
        target_price = momentum['period1_end_price']  # Exit at Day 7
        target_day = 7
        position_size = 0.75  # 75% - reduced conviction
        stop_loss = entry_price * 0.97  # Tighter stop

        strategy_rationale = "Momentum decelerating - gains slowing down"
        action = "Buy now but exit early at Day 7 before momentum fades"

    elif state == "ACCELERATION_REVERSAL":
        # Strong reversal - aggressive buy
        signal = "STRONG_BUY_REVERSAL"
        target_price = forecast_metrics['q75']
        target_day = 14
        position_size = 1.5  # 150% - very high conviction
        stop_loss = entry_price * 0.93  # Wider stop for reversal

        strategy_rationale = "Strong reversal - accelerating from decline"
        action = "Strong buy signal - momentum shifting from negative to positive"

    elif state == "DECELERATION_REVERSAL":
        # Trend reversing - exit or short
        signal = "EXIT_REVERSAL"
        target_price = current_price * 0.95  # Expecting decline
        target_day = 7
        position_size = 0.0  # Don't enter, or exit existing
        stop_loss = entry_price * 1.05

        strategy_rationale = "Negative reversal - momentum turning negative"
        action = "Exit existing positions, avoid new longs, consider shorts"

    elif state == "STEADY_MOMENTUM":
        # Consistent momentum - standard trade
        signal = "BUY_STANDARD"
        target_price = forecast_metrics['median']
        target_day = 14
        position_size = 1.0  # 100% - standard
        stop_loss = calculate_stop_loss(entry_price, forecast_metrics['q25'], cushion_pct=3.0)

        strategy_rationale = "Steady consistent momentum"
        action = "Standard buy with normal position size"

    else:  # MIXED
        signal = "NO_CLEAR_SIGNAL"
        target_price = forecast_metrics['median']
        target_day = 14
        position_size = 0.5  # 50% - low conviction
        stop_loss = entry_price * 0.95

        strategy_rationale = "Mixed momentum signals"
        action = "Reduced position or wait for clearer momentum"

    # Calculate expected gain
    expected_gain_pct = ((target_price - entry_price) / entry_price) * 100

    # Risk/reward
    risk_reward = calculate_risk_reward_ratio(entry_price, target_price, stop_loss) if position_size > 0 else None

    strategy_data = {
        'signal': signal,
        'momentum_state': state,
        'entry_price': entry_price,
        'target_price': target_price,
        'target_day': target_day,
        'stop_loss': stop_loss,
        'expected_gain_pct': expected_gain_pct,
        'risk_reward_ratio': risk_reward,
        'position_size': position_size,
        'position_size_pct': position_size * 100,
        'current_price': current_price,
        'momentum': momentum,
        'forecast_metrics': forecast_metrics,
        'strategy_rationale': strategy_rationale,
        'action': action,
    }

    return strategy_data


def print_acceleration_strategy(strategy_data: Dict):
    """Print formatted output for acceleration/deceleration strategy."""
    print(f"\n{'='*70}")
    print("ACCELERATION/DECELERATION MOMENTUM STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Momentum analysis
    mom = strategy_data['momentum']
    print(f"\n{'='*70}")
    print("MOMENTUM ANALYSIS")
    print(f"{'='*70}")

    print(f"\nPeriod 1 (Days 0-7):")
    print(f"  Total Return: {mom['period1_return']:+.2f}%")
    print(f"  Avg Daily: {mom['period1_daily']:+.3f}%")
    print(f"  End Price: ${mom['period1_end_price']:,.2f}")

    print(f"\nPeriod 2 (Days 7-14):")
    print(f"  Total Return: {mom['period2_return']:+.2f}%")
    print(f"  Avg Daily: {mom['period2_daily']:+.3f}%")
    print(f"  End Price: ${mom['period2_end_price']:,.2f}")

    print(f"\nAcceleration:")
    print(f"  Change in Daily Momentum: {mom['acceleration']:+.3f}%")
    print(f"  Momentum State: {mom['momentum_state']}")

    if mom['acceleration'] > 0.05:
        print(f"  → Momentum is ACCELERATING (speeding up)")
    elif mom['acceleration'] < -0.05:
        print(f"  → Momentum is DECELERATING (slowing down)")
    else:
        print(f"  → Momentum is STEADY")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"MOMENTUM STATE: {strategy_data['momentum_state']}")
    print(f"{'='*70}")

    print(f"\n💡 {strategy_data['strategy_rationale']}")
    print(f"📊 {strategy_data['action']}")

    signal = strategy_data['signal']

    if signal in ["BUY_AND_HOLD_FULL", "STRONG_BUY_REVERSAL", "BUY_STANDARD"]:
        print(f"\n{'='*70}")
        print("TRADING PLAN")
        print(f"{'='*70}")

        print(f"\n  Position Size: {strategy_data['position_size_pct']:.0f}%")
        if strategy_data['position_size'] > 1:
            print(f"  (Oversized due to high conviction)")
        elif strategy_data['position_size'] < 1:
            print(f"  (Reduced due to uncertainty)")

        print(f"\n  1. ENTRY: ${strategy_data['entry_price']:,.2f}")
        print(f"     Action: BUY NOW")

        print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f} (Day {strategy_data['target_day']})")
        print(f"     Expected Gain: {strategy_data['expected_gain_pct']:+.2f}%")

        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        if strategy_data['risk_reward_ratio']:
            print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        # Specific guidance
        if signal == "BUY_AND_HOLD_FULL":
            print(f"\n  Strategy Notes:")
            print(f"    - Momentum accelerating in Period 2")
            print(f"    - Period 2 daily return: {mom['period2_daily']:+.2f}% > Period 1: {mom['period1_daily']:+.2f}%")
            print(f"    - Hold full 14 days to capture accelerating gains")
            print(f"    - Don't exit early - momentum suggests continued upside")

        elif signal == "STRONG_BUY_REVERSAL":
            print(f"\n  Strategy Notes:")
            print(f"    - Strong reversal signal detected")
            print(f"    - Period 1 was negative ({mom['period1_daily']:+.2f}%)")
            print(f"    - Period 2 turning positive ({mom['period2_daily']:+.2f}%)")
            print(f"    - Acceleration: {mom['acceleration']:+.2f}% daily")
            print(f"    - High conviction oversized position")

    elif signal == "BUY_EARLY_EXIT":
        print(f"\n{'='*70}")
        print("TRADING PLAN - EARLY EXIT STRATEGY")
        print(f"{'='*70}")

        print(f"\n  Position Size: {strategy_data['position_size_pct']:.0f}%")
        print(f"  (Reduced - taking profit before momentum fades)")

        print(f"\n  1. ENTRY: ${strategy_data['entry_price']:,.2f}")
        print(f"     Action: BUY NOW")

        print(f"\n  2. EARLY EXIT at Day {strategy_data['target_day']}: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected Gain: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"     ⚠️  EXIT BEFORE Day 14 - momentum decelerating")

        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f}")
        if strategy_data['risk_reward_ratio']:
            print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        print(f"\n  Strategy Notes:")
        print(f"    - Momentum decelerating in Period 2")
        print(f"    - Period 1 daily: {mom['period1_daily']:+.2f}% → Period 2 daily: {mom['period2_daily']:+.2f}%")
        print(f"    - Deceleration: {mom['acceleration']:+.2f}% daily")
        print(f"    - Exit at Day 7 before gains fade")
        print(f"    - Possible top forming around Day 7-10")

    elif signal == "EXIT_REVERSAL":
        print(f"\n{'='*70}")
        print("WARNING - NEGATIVE REVERSAL")
        print(f"{'='*70}")

        print(f"\n  ⚠️  MOMENTUM TURNING NEGATIVE")
        print(f"    Period 1: {mom['period1_daily']:+.2f}% daily")
        print(f"    Period 2: {mom['period2_daily']:+.2f}% daily")
        print(f"    Deceleration: {mom['acceleration']:+.2f}% daily")

        print(f"\n  Recommendation:")
        print(f"    - EXIT any existing long positions")
        print(f"    - DO NOT enter new longs")
        print(f"    - Consider SHORT positions (advanced)")
        print(f"    - Wait for momentum to stabilize")

    else:  # NO_CLEAR_SIGNAL
        print(f"\n  Mixed momentum signals")
        print(f"  Period 1: {mom['period1_daily']:+.2f}% daily")
        print(f"  Period 2: {mom['period2_daily']:+.2f}% daily")
        print(f"  Acceleration: {mom['acceleration']:+.2f}%")
        print(f"\n  Recommendation: Wait for clearer momentum pattern")

    # Risk analysis
    fm = strategy_data['forecast_metrics']
    print(f"\n{'='*70}")
    print("RISK ANALYSIS")
    print(f"{'='*70}")
    print(f"\nForecast Range (Q25-Q75): {fm['forecast_range_pct']:.1f}%")
    print(f"Worst Case: ${fm['min']:,.2f} ({fm['worst_case_loss']:+.2f}%)")
    print(f"Best Case: ${fm['max']:,.2f} ({fm['best_case_gain']:+.2f}%)")

    print(f"\n{'='*70}")


def visualize_acceleration_strategy(strategy_data: Dict, stats: Dict,
                                    save_path: str = 'acceleration_strategy.png'):
    """Create visualization for acceleration/deceleration strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    mom = strategy_data['momentum']
    median = stats['median']

    # Plot 1: Forecast with momentum periods
    ax1 = axes[0, 0]
    days = np.arange(0, 14)

    ax1.axhline(current_price, color='black', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.5)

    # Full forecast
    ax1.plot(days, median, 'b-', linewidth=3, marker='o', markersize=6,
            label='14-Day Forecast', alpha=0.8)

    # Highlight periods
    period1_days = np.arange(0, 7)
    period2_days = np.arange(7, 14)

    ax1.plot(period1_days, median[:7], 'g-', linewidth=5, alpha=0.4, label='Period 1 (Days 0-7)')
    ax1.plot(period2_days, median[7:], 'r-', linewidth=5, alpha=0.4, label='Period 2 (Days 7-14)')

    # Mark key points
    ax1.scatter([6], [mom['period1_end_price']], s=200, color='green',
               zorder=5, marker='o', edgecolors='black', linewidths=2)
    ax1.scatter([13], [mom['period2_end_price']], s=200, color='red',
               zorder=5, marker='s', edgecolors='black', linewidths=2)

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Momentum: {mom["momentum_state"]}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Daily returns comparison
    ax2 = axes[0, 1]

    periods = ['Period 1\n(Days 0-7)', 'Period 2\n(Days 7-14)']
    daily_returns = [mom['period1_daily'], mom['period2_daily']]
    colors = ['green' if r > 0 else 'red' for r in daily_returns]

    bars = ax2.bar(periods, daily_returns, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)

    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Avg Daily Return (%)', fontsize=12)
    ax2.set_title('Momentum Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels and acceleration arrow
    for bar, ret in zip(bars, daily_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2,
                height + (0.05 if height > 0 else -0.05),
                f'{ret:+.3f}%', ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

    # Acceleration arrow
    if mom['acceleration'] > 0.05:
        ax2.annotate('', xy=(1, mom['period2_daily']), xytext=(0, mom['period1_daily']),
                    arrowprops=dict(arrowstyle='->', lw=3, color='darkgreen'))
        ax2.text(0.5, max(daily_returns) * 1.2, f'Acceleration: {mom["acceleration"]:+.3f}%',
                ha='center', fontsize=11, color='darkgreen', fontweight='bold')
    elif mom['acceleration'] < -0.05:
        ax2.annotate('', xy=(1, mom['period2_daily']), xytext=(0, mom['period1_daily']),
                    arrowprops=dict(arrowstyle='->', lw=3, color='darkred'))
        ax2.text(0.5, min(daily_returns) * 1.2, f'Deceleration: {mom["acceleration"]:+.3f}%',
                ha='center', fontsize=11, color='darkred', fontweight='bold')

    # Plot 3: Cumulative returns
    ax3 = axes[1, 0]

    # Calculate cumulative returns for both periods
    cumulative = np.zeros(14)
    for i in range(14):
        cumulative[i] = ((median[i] - median[0]) / median[0]) * 100

    ax3.plot(days, cumulative, 'b-', linewidth=3, marker='o', markersize=6)
    ax3.axvline(6.5, color='gray', linestyle='--', linewidth=2, alpha=0.5,
               label='Period Boundary')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Shade periods
    ax3.axvspan(0, 7, alpha=0.1, color='green', label='Period 1')
    ax3.axvspan(7, 14, alpha=0.1, color='red', label='Period 2')

    ax3.set_xlabel('Days', fontsize=12)
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax3.set_title('Cumulative Return Trajectory', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Strategy summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Momentum State:\n  {mom['momentum_state']}\n\n"
    strategy_text += f"Period Analysis:\n"
    strategy_text += f"  Period 1: {mom['period1_return']:+.2f}% ({mom['period1_daily']:+.3f}%/day)\n"
    strategy_text += f"  Period 2: {mom['period2_return']:+.2f}% ({mom['period2_daily']:+.3f}%/day)\n"
    strategy_text += f"  Acceleration: {mom['acceleration']:+.3f}%/day\n\n"

    if strategy_data['position_size'] > 0:
        strategy_text += f"Trade Setup:\n"
        strategy_text += f"  Entry: ${strategy_data['entry_price']:,.2f}\n"
        strategy_text += f"  Target: ${strategy_data['target_price']:,.2f} (Day {strategy_data['target_day']})\n"
        strategy_text += f"  Expected: {strategy_data['expected_gain_pct']:+.2f}%\n"
        strategy_text += f"  Position: {strategy_data['position_size_pct']:.0f}%\n"

    ax4.text(0.1, 0.9, strategy_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Acceleration/Deceleration Momentum Strategy."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"ACCELERATION/DECELERATION STRATEGY - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    current_price = df_14day['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_acceleration_strategy(stats_14day, current_price)

    # Print results
    print_acceleration_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_acceleration_strategy(strategy_data, stats_14day)

    print(f"\n{'='*70}")
    print("ACCELERATION/DECELERATION STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
