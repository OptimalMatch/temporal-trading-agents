"""
Cryptocurrency Trading Strategy Analysis

Compares 7-day and 14-day forecasts to identify profitable trading opportunities:
- Buy the dip: Short-term decline followed by medium-term recovery
- Momentum play: Sustained upward trend
- Risk assessment: Analyze worst-case scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib.util

# Load the ensemble module
spec = importlib.util.spec_from_file_location("ensemble", "crypto_ensemble_forecast.py")
ensemble = importlib.util.module_from_spec(spec)
sys.modules["ensemble"] = ensemble
spec.loader.exec_module(ensemble)

import torch


def train_ensemble(symbol, forecast_horizon, configs, name):
    """Train ensemble for a specific forecast horizon."""
    print(f"\n{'='*70}")
    print(f"TRAINING {name} ENSEMBLE ({forecast_horizon}-day forecast)")
    print(f"{'='*70}")

    ensemble_models = []
    for config in configs:
        model_info = ensemble.train_ensemble_model(
            symbol=symbol,
            period='2y',
            lookback=config['lookback'],
            forecast_horizon=forecast_horizon,
            epochs=config['epochs'],
            focus=config['focus'],
            model_name=config['name']
        )
        ensemble_models.append(model_info)

    # Make predictions
    ensemble_stats, df_latest = ensemble.make_ensemble_predictions(
        ensemble_models, symbol, forecast_horizon
    )

    return ensemble_stats, df_latest


def analyze_trading_strategy(stats_7day, stats_14day, current_price):
    """
    Analyze trading opportunities by comparing 7-day and 14-day forecasts.
    """
    print(f"\n{'='*70}")
    print("TRADING STRATEGY ANALYSIS")
    print(f"{'='*70}")

    # Extract key data points
    day7_median = stats_7day['median'][-1]
    day7_q25 = stats_7day['q25'][-1]
    day7_q75 = stats_7day['q75'][-1]

    day14_median = stats_14day['median'][-1]
    day14_q25 = stats_14day['q25'][-1]
    day14_q75 = stats_14day['q75'][-1]

    # Calculate changes
    day7_change_pct = ((day7_median - current_price) / current_price) * 100
    day14_change_pct = ((day14_median - current_price) / current_price) * 100
    recovery_pct = ((day14_median - day7_median) / day7_median) * 100

    print(f"\nCurrent Price: ${current_price:,.2f}")
    print(f"\nDay 7 Forecast:")
    print(f"  Median: ${day7_median:,.2f} ({day7_change_pct:+.2f}%)")
    print(f"  Range: ${day7_q25:,.2f} to ${day7_q75:,.2f}")

    print(f"\nDay 14 Forecast:")
    print(f"  Median: ${day14_median:,.2f} ({day14_change_pct:+.2f}%)")
    print(f"  Range: ${day14_q25:,.2f} to ${day14_q75:,.2f}")

    print(f"\n{'='*70}")

    # Strategy identification
    strategy = None

    if day7_change_pct < 0 and day14_change_pct > 0:
        strategy = "BUY_THE_DIP"
        print("STRATEGY: BUY THE DIP")
        print(f"{'='*70}")
        print(f"\n✓ Short-term dip detected: {day7_change_pct:+.2f}%")
        print(f"✓ Medium-term recovery: {day14_change_pct:+.2f}%")
        print(f"✓ Expected gain from dip to recovery: {recovery_pct:+.2f}%")

        print(f"\n📊 Trading Plan:")
        print(f"  1. WAIT for dip around Day 7")
        print(f"     Target entry: ${day7_median:,.2f}")
        print(f"     Conservative entry (wait for): ${day7_q25:,.2f}")
        print(f"\n  2. BUY when price reaches Day 7 target")
        print(f"\n  3. SELL around Day 14")
        print(f"     Target exit: ${day14_median:,.2f}")
        print(f"     Optimistic exit: ${day14_q75:,.2f}")

        # Calculate profit scenarios
        conservative_profit = ((day14_median - day7_q25) / day7_q25) * 100
        median_profit = recovery_pct
        optimistic_profit = ((day14_q75 - day7_median) / day7_median) * 100

        print(f"\n💰 Profit Scenarios:")
        print(f"  Conservative: Buy at ${day7_q25:,.2f}, sell at ${day14_median:,.2f} = {conservative_profit:+.2f}%")
        print(f"  Median:       Buy at ${day7_median:,.2f}, sell at ${day14_median:,.2f} = {median_profit:+.2f}%")
        print(f"  Optimistic:   Buy at ${day7_median:,.2f}, sell at ${day14_q75:,.2f} = {optimistic_profit:+.2f}%")

    elif day7_change_pct > 0 and day14_change_pct > day7_change_pct:
        strategy = "MOMENTUM_PLAY"
        print("STRATEGY: MOMENTUM PLAY")
        print(f"{'='*70}")
        print(f"\n✓ Short-term gain: {day7_change_pct:+.2f}%")
        print(f"✓ Accelerating to Day 14: {day14_change_pct:+.2f}%")
        print(f"✓ Additional gain Day 7→14: {recovery_pct:+.2f}%")

        print(f"\n📊 Trading Plan:")
        print(f"  1. BUY NOW at ${current_price:,.2f}")
        print(f"\n  2. Consider taking partial profits at Day 7: ${day7_median:,.2f} ({day7_change_pct:+.2f}%)")
        print(f"\n  3. SELL remaining position around Day 14: ${day14_median:,.2f} ({day14_change_pct:+.2f}%)")

        print(f"\n💰 Profit Scenarios:")
        print(f"  Conservative: ${current_price:,.2f} → ${day14_q25:,.2f} = {((day14_q25-current_price)/current_price*100):+.2f}%")
        print(f"  Median:       ${current_price:,.2f} → ${day14_median:,.2f} = {day14_change_pct:+.2f}%")
        print(f"  Optimistic:   ${current_price:,.2f} → ${day14_q75:,.2f} = {((day14_q75-current_price)/current_price*100):+.2f}%")

    elif day7_change_pct < 0 and day14_change_pct < 0:
        strategy = "STAY_OUT"
        print("STRATEGY: STAY OUT / SHORT (Advanced)")
        print(f"{'='*70}")
        print(f"\n⚠ Both forecasts show decline")
        print(f"  Day 7:  {day7_change_pct:+.2f}%")
        print(f"  Day 14: {day14_change_pct:+.2f}%")
        print(f"\n💡 Recommendation: Wait for better entry point or consider shorting")

    else:
        strategy = "HOLD"
        print("STRATEGY: HOLD")
        print(f"{'='*70}")
        print(f"\n→ Moderate gains expected but no clear dip opportunity")
        print(f"  Day 7:  {day7_change_pct:+.2f}%")
        print(f"  Day 14: {day14_change_pct:+.2f}%")

    # Risk analysis
    print(f"\n{'='*70}")
    print("RISK ANALYSIS")
    print(f"{'='*70}")

    worst_case_7 = stats_7day['min'][-1]
    worst_case_14 = stats_14day['min'][-1]

    print(f"\nWorst-Case Scenarios:")
    print(f"  Day 7:  ${worst_case_7:,.2f} ({((worst_case_7-current_price)/current_price*100):+.2f}%)")
    print(f"  Day 14: ${worst_case_14:,.2f} ({((worst_case_14-current_price)/current_price*100):+.2f}%)")

    if strategy == "BUY_THE_DIP":
        max_drawdown = ((day7_q25 - current_price) / current_price) * 100
        stop_loss = day7_q25 * 0.97  # 3% below expected dip

        print(f"\nRisk Management:")
        print(f"  Maximum expected drawdown: {max_drawdown:.2f}%")
        print(f"  Suggested stop-loss: ${stop_loss:,.2f} (-{abs(((stop_loss-current_price)/current_price*100)):.2f}%)")
        print(f"  Risk/Reward ratio: 1:{abs(recovery_pct/max_drawdown):.2f}")

    elif strategy == "MOMENTUM_PLAY":
        stop_loss = current_price * 0.95  # 5% stop loss
        potential_gain = day14_change_pct
        risk = 5.0

        print(f"\nRisk Management:")
        print(f"  Suggested stop-loss: ${stop_loss:,.2f} (-5.0%)")
        print(f"  Potential gain: {potential_gain:+.2f}%")
        print(f"  Risk/Reward ratio: 1:{abs(potential_gain/risk):.2f}")

    # Model confidence
    print(f"\nModel Confidence:")
    prob_above_7 = np.mean(stats_7day['all_predictions'][:, -1] > current_price) * 100
    prob_above_14 = np.mean(stats_14day['all_predictions'][:, -1] > current_price) * 100

    print(f"  Day 7:  {int(prob_above_7)}% of models predict price ABOVE current")
    print(f"  Day 14: {int(prob_above_14)}% of models predict price ABOVE current")

    return {
        'strategy': strategy,
        'day7_median': day7_median,
        'day14_median': day14_median,
        'current_price': current_price,
        'stats_7day': stats_7day,
        'stats_14day': stats_14day
    }


def visualize_trading_strategy(strategy_data):
    """Create visualization showing the trading strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    current_price = strategy_data['current_price']
    stats_7 = strategy_data['stats_7day']
    stats_14 = strategy_data['stats_14day']

    # Plot 1: Combined 7-day and 14-day forecast
    ax1 = axes[0, 0]
    days_7 = np.arange(0, 7)
    days_14 = np.arange(0, 14)

    ax1.axhline(current_price, color='green', linestyle='--', linewidth=2,
                label=f'Current: ${current_price:,.0f}', alpha=0.7)

    # 7-day forecast
    ax1.plot(days_7, stats_7['median'], 'b-', linewidth=3, marker='o',
             markersize=6, label='7-Day Median', alpha=0.8)
    ax1.fill_between(days_7, stats_7['q25'], stats_7['q75'],
                     alpha=0.3, color='blue', label='7-Day 25th-75th %ile')

    # 14-day forecast
    ax1.plot(days_14, stats_14['median'], 'r-', linewidth=3, marker='s',
             markersize=6, label='14-Day Median', alpha=0.8)
    ax1.fill_between(days_14, stats_14['q25'], stats_14['q75'],
                     alpha=0.2, color='red', label='14-Day 25th-75th %ile')

    # Mark key points
    if strategy_data['strategy'] == "BUY_THE_DIP":
        ax1.scatter([6], [stats_7['median'][-1]], s=300, color='lime',
                   marker='v', zorder=5, label='BUY HERE', edgecolors='black', linewidths=2)
        ax1.scatter([13], [stats_14['median'][-1]], s=300, color='gold',
                   marker='^', zorder=5, label='SELL HERE', edgecolors='black', linewidths=2)
    elif strategy_data['strategy'] == "MOMENTUM_PLAY":
        ax1.scatter([0], [current_price], s=300, color='lime',
                   marker='v', zorder=5, label='BUY NOW', edgecolors='black', linewidths=2)
        ax1.scatter([13], [stats_14['median'][-1]], s=300, color='gold',
                   marker='^', zorder=5, label='SELL AT DAY 14', edgecolors='black', linewidths=2)

    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Trading Strategy: {strategy_data["strategy"].replace("_", " ")}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Daily returns comparison
    ax2 = axes[0, 1]
    returns_7 = np.diff(stats_7['median']) / stats_7['median'][:-1] * 100
    returns_14 = np.diff(stats_14['median']) / stats_14['median'][:-1] * 100

    ax2.bar(days_7[1:], returns_7, alpha=0.6, color='blue', label='7-Day Forecast', width=0.4)
    ax2.bar(days_14[1:] + 0.4, returns_14, alpha=0.6, color='red', label='14-Day Forecast', width=0.4)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.set_title('Expected Daily Returns', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Profit potential over time
    ax3 = axes[1, 0]

    if strategy_data['strategy'] == "BUY_THE_DIP":
        # Show profit from buying at day 7
        entry_price = stats_7['median'][-1]
        profit_curve = ((stats_14['median'][7:] - entry_price) / entry_price) * 100
        days_profit = np.arange(7, 14)

        ax3.plot(days_profit, profit_curve, 'g-', linewidth=3, marker='o', markersize=8)
        ax3.fill_between(days_profit, 0, profit_curve, alpha=0.3, color='green')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Day', fontsize=12)
        ax3.set_ylabel('Profit from Day 7 Entry (%)', fontsize=12)
        ax3.set_title(f'Profit Potential: Buy at Day 7 (${entry_price:,.0f})',
                     fontsize=14, fontweight='bold')

    elif strategy_data['strategy'] == "MOMENTUM_PLAY":
        profit_curve = ((stats_14['median'] - current_price) / current_price) * 100
        ax3.plot(days_14, profit_curve, 'g-', linewidth=3, marker='o', markersize=8)
        ax3.fill_between(days_14, 0, profit_curve, alpha=0.3, color='green')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Day', fontsize=12)
        ax3.set_ylabel('Profit from Current Price (%)', fontsize=12)
        ax3.set_title(f'Profit Potential: Buy Now (${current_price:,.0f})',
                     fontsize=14, fontweight='bold')

    ax3.grid(True, alpha=0.3)

    # Plot 4: Risk vs Reward
    ax4 = axes[1, 1]

    scenarios = ['Worst\nCase', 'Pessimistic\n(25th)', 'Median', 'Optimistic\n(75th)', 'Best\nCase']

    if strategy_data['strategy'] == "BUY_THE_DIP":
        entry = stats_7['median'][-1]
        profits = [
            ((stats_14['min'][-1] - entry) / entry) * 100,
            ((stats_14['q25'][-1] - entry) / entry) * 100,
            ((stats_14['median'][-1] - entry) / entry) * 100,
            ((stats_14['q75'][-1] - entry) / entry) * 100,
            ((stats_14['max'][-1] - entry) / entry) * 100,
        ]
    else:
        profits = [
            ((stats_14['min'][-1] - current_price) / current_price) * 100,
            ((stats_14['q25'][-1] - current_price) / current_price) * 100,
            ((stats_14['median'][-1] - current_price) / current_price) * 100,
            ((stats_14['q75'][-1] - current_price) / current_price) * 100,
            ((stats_14['max'][-1] - current_price) / current_price) * 100,
        ]

    colors = ['red' if p < 0 else 'green' for p in profits]
    bars = ax4.barh(scenarios, profits, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.axvline(0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Profit/Loss (%)', fontsize=12)
    ax4.set_title('Risk vs Reward Analysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for bar, profit in zip(bars, profits):
        width = bar.get_width()
        ax4.text(width + (0.2 if width > 0 else -0.2), bar.get_y() + bar.get_height()/2,
                f'{profit:+.1f}%', ha='left' if width > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('bitcoin_trading_strategy.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to 'bitcoin_trading_strategy.png'")


def main():
    symbol = 'BTC-USD'

    # 7-day ensemble config (optimized epochs for faster training)
    configs_7day = [
        {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Short-term Momentum'},
        {'lookback': 60, 'focus': 'balanced', 'epochs': 15, 'name': 'Medium-term Balanced'},
        {'lookback': 90, 'focus': 'balanced', 'epochs': 15, 'name': 'Long-term Trend'},
        {'lookback': 60, 'focus': 'mean_reversion', 'epochs': 15, 'name': 'Mean Reversion'},
        {'lookback': 45, 'focus': 'momentum', 'epochs': 15, 'name': 'Mid-term Momentum'},
    ]

    # 14-day ensemble config (reduced lookbacks, optimized epochs)
    configs_14day = [
        {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Short-term Momentum'},
        {'lookback': 45, 'focus': 'balanced', 'epochs': 15, 'name': 'Medium-term Balanced'},
        {'lookback': 60, 'focus': 'balanced', 'epochs': 15, 'name': 'Long-term Trend'},
        {'lookback': 45, 'focus': 'mean_reversion', 'epochs': 15, 'name': 'Mean Reversion'},
        {'lookback': 30, 'focus': 'momentum', 'epochs': 15, 'name': 'Mid-term Momentum'},
    ]

    # Train both ensembles
    stats_7day, df_7day = train_ensemble(symbol, 7, configs_7day, "7-DAY")
    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY")

    current_price = df_14day['Close'].iloc[-1]

    # Analyze trading strategy
    strategy_data = analyze_trading_strategy(stats_7day, stats_14day, current_price)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_trading_strategy(strategy_data)

    print(f"\n{'='*70}")
    print("TRADING STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\n⚠️  DISCLAIMER:")
    print("This analysis is for educational purposes only.")
    print("Cryptocurrency trading carries substantial risk.")
    print("Always do your own research and never invest more than you can afford to lose.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
