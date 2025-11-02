"""
Advanced Risk-Adjusted Strategy

Combines multiple risk metrics to provide comprehensive risk-adjusted trading signals.
Uses Sharpe ratio, Sortino ratio, Maximum Drawdown, VaR, and CVaR to determine
optimal position sizing and entry/exit points.

Strategy Logic:
- Calculate multiple risk metrics from ensemble predictions
- Sharpe Ratio: Return per unit of total volatility
- Sortino Ratio: Return per unit of downside volatility
- Maximum Drawdown: Worst peak-to-trough decline
- VaR (95%): Maximum expected loss at 95% confidence
- CVaR: Average loss beyond VaR threshold
- Position sizing based on risk-adjusted score

Signals:
- EXCELLENT_RISK_REWARD: All metrics favorable, large position
- GOOD_RISK_REWARD: Most metrics favorable, standard position
- MODERATE_RISK_REWARD: Mixed metrics, reduced position
- POOR_RISK_REWARD: Unfavorable metrics, small/no position
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
    get_default_ensemble_configs,
)
from strategies.strategy_cache import cached_strategy


def calculate_risk_metrics(stats: Dict, current_price: float, risk_free_rate: float = 0.05) -> Dict:
    """
    Calculate comprehensive risk metrics from forecast.

    Args:
        stats: Forecast statistics dictionary
        current_price: Current asset price
        risk_free_rate: Annual risk-free rate (default 5%)

    Returns:
        Dictionary with risk metrics
    """
    # Get all predictions and calculate returns
    all_predictions = stats['all_predictions']  # Shape: (n_models, n_days)
    final_predictions = all_predictions[:, -1]

    # Calculate returns for each model
    returns = ((final_predictions - current_price) / current_price) * 100

    # Expected return
    expected_return = np.mean(returns)

    # Total volatility (standard deviation)
    volatility = np.std(returns)

    # Downside volatility (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_volatility = np.std(downside_returns)
    else:
        downside_volatility = 0.01  # Small number to avoid division by zero

    # Sharpe Ratio (return per unit of total risk)
    # Annualized risk-free rate adjusted for 14-day period
    rf_14day = (risk_free_rate / 365) * 14
    sharpe_ratio = (expected_return - rf_14day) / volatility if volatility > 0 else 0

    # Sortino Ratio (return per unit of downside risk)
    sortino_ratio = (expected_return - rf_14day) / downside_volatility if downside_volatility > 0 else 0

    # Maximum Drawdown (worst peak-to-trough)
    median_path = stats['median']
    running_max = np.maximum.accumulate(median_path)
    drawdown = ((median_path - running_max) / running_max) * 100
    max_drawdown = np.min(drawdown)

    # Value at Risk (VaR) at 95% confidence
    var_95 = np.percentile(returns, 5)  # 5th percentile = 95% VaR

    # Conditional Value at Risk (CVaR) - average loss beyond VaR
    cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95

    # Win rate
    win_rate = (np.sum(returns > 0) / len(returns)) * 100

    # Risk-Adjusted Score (composite metric)
    # Normalize and combine metrics
    sharpe_score = min(max(sharpe_ratio, -2), 3) / 3  # Normalize to ~0-1
    sortino_score = min(max(sortino_ratio, -2), 3) / 3
    drawdown_score = min(max(max_drawdown / -20, 0), 1)  # Better if smaller
    var_score = min(max(var_95 / -20, 0), 1)  # Better if smaller (less negative)
    win_rate_score = win_rate / 100

    # Weighted average (can adjust weights)
    risk_adjusted_score = (
        0.25 * sharpe_score +
        0.25 * sortino_score +
        0.20 * drawdown_score +
        0.15 * var_score +
        0.15 * win_rate_score
    )

    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'downside_volatility': downside_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'risk_adjusted_score': risk_adjusted_score,
    }


@cached_strategy
def analyze_risk_adjusted_strategy(stats_14day: Dict, current_price: float) -> Dict:
    """
    Analyze trading opportunity based on comprehensive risk metrics.

    Args:
        stats_14day: 14-day forecast statistics
        current_price: Current asset price

    Returns:
        Dictionary with strategy recommendation
    """
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(stats_14day, current_price)

    # Get forecast and confidence metrics
    forecast_metrics = calculate_forecast_metrics(stats_14day, current_price, 14)
    confidence_metrics = calculate_ensemble_confidence(stats_14day, current_price)

    # Determine signal based on risk-adjusted score
    score = risk_metrics['risk_adjusted_score']

    if score >= 0.7:
        signal = "EXCELLENT_RISK_REWARD"
        position_size = 1.5  # 150%
        rationale = "Exceptional risk/reward profile"
    elif score >= 0.5:
        signal = "GOOD_RISK_REWARD"
        position_size = 1.0  # 100%
        rationale = "Favorable risk/reward profile"
    elif score >= 0.3:
        signal = "MODERATE_RISK_REWARD"
        position_size = 0.5  # 50%
        rationale = "Acceptable but moderate risk/reward"
    else:
        signal = "POOR_RISK_REWARD"
        position_size = 0.1  # 10% or avoid
        rationale = "Unfavorable risk/reward profile"

    # Trading parameters
    entry_price = current_price
    target_price = forecast_metrics['median']
    expected_gain_pct = forecast_metrics['median_change_pct']

    # Dynamic stop loss based on risk metrics
    # Use CVaR as basis for stop loss
    stop_loss_pct = min(max(risk_metrics['cvar_95'] * 0.8, -10), -3)  # Between -3% and -10%
    stop_loss = entry_price * (1 + stop_loss_pct / 100)

    # Risk/reward ratio
    potential_gain = target_price - entry_price
    potential_loss = entry_price - stop_loss
    risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0

    strategy_data = {
        'signal': signal,
        'rationale': rationale,
        'position_size': position_size,
        'position_size_pct': position_size * 100,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'stop_loss_pct': stop_loss_pct,
        'expected_gain_pct': expected_gain_pct,
        'risk_reward_ratio': risk_reward_ratio,
        'current_price': current_price,
        'risk_metrics': risk_metrics,
        'forecast_metrics': forecast_metrics,
        'confidence_metrics': confidence_metrics,
    }

    return strategy_data


def print_risk_adjusted_strategy(strategy_data: Dict):
    """Print formatted output for risk-adjusted strategy."""
    print(f"\n{'='*70}")
    print("ADVANCED RISK-ADJUSTED STRATEGY")
    print(f"{'='*70}")

    print(f"\nCurrent Price: ${strategy_data['current_price']:,.2f}")

    # Risk metrics
    rm = strategy_data['risk_metrics']
    print(f"\n{'='*70}")
    print("COMPREHENSIVE RISK ANALYSIS")
    print(f"{'='*70}")

    print(f"\nReturn Metrics:")
    print(f"  Expected Return: {rm['expected_return']:+.2f}%")
    print(f"  Win Rate: {rm['win_rate']:.1f}%")

    print(f"\nVolatility Metrics:")
    print(f"  Total Volatility: {rm['volatility']:.2f}%")
    print(f"  Downside Volatility: {rm['downside_volatility']:.2f}%")

    print(f"\nRisk-Adjusted Ratios:")
    print(f"  Sharpe Ratio: {rm['sharpe_ratio']:.3f}")
    if rm['sharpe_ratio'] > 1:
        print(f"    → Excellent (>1.0)")
    elif rm['sharpe_ratio'] > 0.5:
        print(f"    → Good (>0.5)")
    elif rm['sharpe_ratio'] > 0:
        print(f"    → Acceptable (>0)")
    else:
        print(f"    → Poor (<0)")

    print(f"\n  Sortino Ratio: {rm['sortino_ratio']:.3f}")
    if rm['sortino_ratio'] > 1.5:
        print(f"    → Excellent (>1.5)")
    elif rm['sortino_ratio'] > 0.7:
        print(f"    → Good (>0.7)")
    elif rm['sortino_ratio'] > 0:
        print(f"    → Acceptable (>0)")
    else:
        print(f"    → Poor (<0)")

    print(f"\nDownside Risk Metrics:")
    print(f"  Maximum Drawdown: {rm['max_drawdown']:+.2f}%")
    if rm['max_drawdown'] > -5:
        print(f"    → Low Risk (<-5%)")
    elif rm['max_drawdown'] > -10:
        print(f"    → Moderate Risk (-5% to -10%)")
    else:
        print(f"    → High Risk (>-10%)")

    print(f"\n  VaR (95%): {rm['var_95']:+.2f}%")
    print(f"    (Maximum expected loss at 95% confidence)")

    print(f"\n  CVaR (95%): {rm['cvar_95']:+.2f}%")
    print(f"    (Average loss beyond VaR threshold)")

    print(f"\nComposite Score:")
    print(f"  Risk-Adjusted Score: {rm['risk_adjusted_score']:.3f}")
    if rm['risk_adjusted_score'] >= 0.7:
        print(f"    → EXCELLENT (≥0.7)")
    elif rm['risk_adjusted_score'] >= 0.5:
        print(f"    → GOOD (≥0.5)")
    elif rm['risk_adjusted_score'] >= 0.3:
        print(f"    → MODERATE (≥0.3)")
    else:
        print(f"    → POOR (<0.3)")

    print(f"\n{'='*70}")
    print(f"SIGNAL: {strategy_data['signal']}")
    print(f"{'='*70}")
    print(f"\n💡 {strategy_data['rationale']}")

    signal = strategy_data['signal']

    if signal in ["EXCELLENT_RISK_REWARD", "GOOD_RISK_REWARD"]:
        print(f"\n{'='*70}")
        print("TRADING PLAN")
        print(f"{'='*70}")

        print(f"\n  Risk-Adjusted Position Size: {strategy_data['position_size_pct']:.0f}%")
        if strategy_data['position_size'] > 1:
            print(f"  (Oversized due to excellent risk metrics)")

        print(f"\n  1. ENTRY: ${strategy_data['entry_price']:,.2f}")
        print(f"     Action: BUY NOW")

        print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected Gain: {strategy_data['expected_gain_pct']:+.2f}%")
        print(f"     Win Probability: {rm['win_rate']:.1f}%")

        print(f"\n  3. DYNAMIC STOP LOSS: ${strategy_data['stop_loss']:,.2f} ({strategy_data['stop_loss_pct']:+.2f}%)")
        print(f"     Based on CVaR analysis")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        print(f"\n  Strategy Justification:")
        print(f"    ✓ Sharpe Ratio: {rm['sharpe_ratio']:.2f} (risk-adjusted return)")
        print(f"    ✓ Sortino Ratio: {rm['sortino_ratio']:.2f} (downside risk-adjusted)")
        print(f"    ✓ Max Drawdown: {rm['max_drawdown']:.2f}% (manageable)")
        print(f"    ✓ Win Rate: {rm['win_rate']:.1f}% (high probability)")
        print(f"    ✓ VaR: {rm['var_95']:.2f}% (controlled downside)")

    elif signal == "MODERATE_RISK_REWARD":
        print(f"\n{'='*70}")
        print("REDUCED POSITION TRADING PLAN")
        print(f"{'='*70}")

        print(f"\n  Risk-Adjusted Position Size: {strategy_data['position_size_pct']:.0f}%")
        print(f"  (Reduced due to moderate risk metrics)")

        print(f"\n  1. ENTRY: ${strategy_data['entry_price']:,.2f}")
        print(f"     Action: BUY with caution")

        print(f"\n  2. TARGET: ${strategy_data['target_price']:,.2f}")
        print(f"     Expected Gain: {strategy_data['expected_gain_pct']:+.2f}%")

        print(f"\n  3. STOP LOSS: ${strategy_data['stop_loss']:,.2f} ({strategy_data['stop_loss_pct']:+.2f}%)")
        print(f"     Risk/Reward: 1:{strategy_data['risk_reward_ratio']:.2f}")

        print(f"\n  Risk Concerns:")
        if rm['sharpe_ratio'] < 0.5:
            print(f"    ⚠ Low Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
        if rm['max_drawdown'] < -10:
            print(f"    ⚠ High Drawdown Risk: {rm['max_drawdown']:.2f}%")
        if rm['win_rate'] < 60:
            print(f"    ⚠ Moderate Win Rate: {rm['win_rate']:.1f}%")

    else:  # POOR_RISK_REWARD
        print(f"\n{'='*70}")
        print("WARNING - POOR RISK/REWARD")
        print(f"{'='*70}")

        print(f"\n  ⚠️  Risk metrics indicate unfavorable setup")
        print(f"  Recommended Position: {strategy_data['position_size_pct']:.0f}% (minimal or avoid)")

        print(f"\n  Risk Flags:")
        if rm['sharpe_ratio'] < 0:
            print(f"    ✗ Negative Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
        if rm['sortino_ratio'] < 0:
            print(f"    ✗ Negative Sortino Ratio: {rm['sortino_ratio']:.2f}")
        if rm['max_drawdown'] < -15:
            print(f"    ✗ Severe Drawdown Risk: {rm['max_drawdown']:.2f}%")
        if rm['var_95'] < -10:
            print(f"    ✗ High VaR: {rm['var_95']:.2f}%")
        if rm['win_rate'] < 50:
            print(f"    ✗ Low Win Rate: {rm['win_rate']:.1f}%")

        print(f"\n  💡 Recommendation:")
        print(f"     - Avoid entering this trade")
        print(f"     - Wait for better risk/reward setup")
        print(f"     - Risk-adjusted score: {rm['risk_adjusted_score']:.3f} (too low)")

    # Additional context
    print(f"\n{'='*70}")
    print("FORECAST CONTEXT")
    print(f"{'='*70}")
    fm = strategy_data['forecast_metrics']
    cm = strategy_data['confidence_metrics']
    print(f"\nForecast Range: {fm['forecast_range_pct']:.1f}%")
    print(f"Model Confidence: {cm['confidence_level']} ({cm['agreement']:.1f}% agreement)")
    print(f"Probability Above: {fm['prob_above']:.1f}%")

    print(f"\n{'='*70}")


def visualize_risk_adjusted_strategy(strategy_data: Dict, stats: Dict,
                                     save_path: str = 'risk_adjusted_strategy.png'):
    """Create visualization for risk-adjusted strategy."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    current_price = strategy_data['current_price']
    rm = strategy_data['risk_metrics']

    # Plot 1: Return distribution
    ax1 = axes[0, 0]

    all_predictions = stats['all_predictions'][:, -1]
    returns = ((all_predictions - current_price) / current_price) * 100

    ax1.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(rm['expected_return'], color='green', linestyle='-', linewidth=2,
                label=f'Expected: {rm["expected_return"]:+.1f}%')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(rm['var_95'], color='red', linestyle='--', linewidth=2,
                label=f'VaR(95%): {rm["var_95"]:+.1f}%')

    ax1.set_xlabel('Return (%)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Return Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Risk metrics radar
    ax2 = axes[0, 1]

    # Normalize metrics for radar chart
    metrics_norm = {
        'Sharpe': min(max(rm['sharpe_ratio'] / 2, 0), 1),
        'Sortino': min(max(rm['sortino_ratio'] / 2, 0), 1),
        'Win Rate': rm['win_rate'] / 100,
        'Drawdown': min(max((rm['max_drawdown'] + 20) / 20, 0), 1),
        'VaR': min(max((rm['var_95'] + 20) / 20, 0), 1),
    }

    categories = list(metrics_norm.keys())
    values = list(metrics_norm.values())
    values += values[:1]  # Complete the circle

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax2 = plt.subplot(2, 3, 2, projection='polar')
    ax2.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax2.fill(angles, values, alpha=0.25, color='blue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('Risk Metrics Profile', fontsize=13, fontweight='bold', pad=20)
    ax2.grid(True)

    # Plot 3: Drawdown analysis
    ax3 = axes[0, 2]

    median_path = stats['median']
    running_max = np.maximum.accumulate(median_path)
    drawdown = ((median_path - running_max) / running_max) * 100
    days = np.arange(14)

    ax3.fill_between(days, drawdown, 0, where=(drawdown < 0),
                     color='red', alpha=0.5, label='Drawdown')
    ax3.plot(days, drawdown, 'r-', linewidth=2)
    ax3.axhline(rm['max_drawdown'], color='darkred', linestyle='--', linewidth=2,
                label=f'Max DD: {rm["max_drawdown"]:.1f}%')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax3.set_xlabel('Days', fontsize=11)
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_title('Maximum Drawdown Analysis', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Risk/Reward comparison
    ax4 = axes[1, 0]

    metrics = ['Sharpe', 'Sortino', 'Win Rate', 'Score']
    values = [
        rm['sharpe_ratio'],
        rm['sortino_ratio'],
        rm['win_rate'] / 100,  # Normalize
        rm['risk_adjusted_score']
    ]
    colors = ['green' if v > 0.5 else 'orange' if v > 0 else 'red' for v in values]

    bars = ax4.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Good Threshold')

    ax4.set_xlabel('Value', fontsize=11)
    ax4.set_title('Risk-Adjusted Metrics', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center',
                fontsize=10, fontweight='bold')

    # Plot 5: Position sizing justification
    ax5 = axes[1, 1]

    score_thresholds = ['Poor\n(<0.3)', 'Moderate\n(0.3-0.5)', 'Good\n(0.5-0.7)', 'Excellent\n(≥0.7)']
    position_sizes = [10, 50, 100, 150]
    colors_pos = ['red', 'orange', 'lightgreen', 'darkgreen']

    bars = ax5.bar(score_thresholds, position_sizes, color=colors_pos, alpha=0.7,
                   edgecolor='black', linewidth=2)

    # Highlight current
    current_score = rm['risk_adjusted_score']
    if current_score >= 0.7:
        bars[3].set_edgecolor('blue')
        bars[3].set_linewidth(4)
    elif current_score >= 0.5:
        bars[2].set_edgecolor('blue')
        bars[2].set_linewidth(4)
    elif current_score >= 0.3:
        bars[1].set_edgecolor('blue')
        bars[1].set_linewidth(4)
    else:
        bars[0].set_edgecolor('blue')
        bars[0].set_linewidth(4)

    ax5.set_ylabel('Position Size (%)', fontsize=11)
    ax5.set_xlabel('Risk-Adjusted Score', fontsize=11)
    ax5.set_title(f'Position Sizing - Score: {current_score:.2f}', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, size in zip(bars, position_sizes):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 3,
                f'{size}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 6: Strategy summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    strategy_text = f"SIGNAL: {strategy_data['signal']}\n\n"
    strategy_text += f"Risk-Adjusted Score: {rm['risk_adjusted_score']:.3f}\n\n"
    strategy_text += f"Key Metrics:\n"
    strategy_text += f"  Sharpe: {rm['sharpe_ratio']:.2f}\n"
    strategy_text += f"  Sortino: {rm['sortino_ratio']:.2f}\n"
    strategy_text += f"  Max DD: {rm['max_drawdown']:.1f}%\n"
    strategy_text += f"  VaR(95%): {rm['var_95']:.1f}%\n"
    strategy_text += f"  Win Rate: {rm['win_rate']:.0f}%\n\n"

    if strategy_data['position_size'] > 0:
        strategy_text += f"Trade Setup:\n"
        strategy_text += f"  Position: {strategy_data['position_size_pct']:.0f}%\n"
        strategy_text += f"  Expected: {strategy_data['expected_gain_pct']:+.1f}%\n"
        strategy_text += f"  R/R: 1:{strategy_data['risk_reward_ratio']:.1f}\n"

    ax6.text(0.1, 0.9, strategy_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Strategy visualization saved to '{save_path}'")


def main():
    """Run the Advanced Risk-Adjusted Strategy."""
    symbol = 'BTC-USD'

    # Load ensemble module
    ensemble = load_ensemble_module("../examples/crypto_ensemble_forecast.py")

    # Get default configs for 14-day forecast
    configs_14day = get_default_ensemble_configs(14)

    # Train 14-day ensemble
    print(f"\n{'='*70}")
    print(f"ADVANCED RISK-ADJUSTED STRATEGY - {symbol}")
    print(f"{'='*70}")

    stats_14day, df_14day = train_ensemble(symbol, 14, configs_14day, "14-DAY", ensemble)

    current_price = df_14day['Close'].iloc[-1]

    # Analyze strategy
    strategy_data = analyze_risk_adjusted_strategy(stats_14day, current_price)

    # Print results
    print_risk_adjusted_strategy(strategy_data)

    # Visualize
    print(f"\n{'='*70}")
    print("CREATING STRATEGY VISUALIZATION")
    print(f"{'='*70}")
    visualize_risk_adjusted_strategy(strategy_data, stats_14day)

    print(f"\n{'='*70}")
    print("RISK-ADJUSTED STRATEGY ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
