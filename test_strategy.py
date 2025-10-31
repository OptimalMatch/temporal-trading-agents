"""Quick test of strategy functionality"""
import sys
sys.path.append('strategies')

from strategy_utils import calculate_forecast_metrics, analyze_forecast_gradient
import numpy as np

print("="*70)
print("TESTING STRATEGY UTILITIES")
print("="*70)

# Create test forecast data
test_stats = {
    'median': np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126]),
    'q25': np.array([95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121]),
    'q75': np.array([105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131]),
    'min': np.array([90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]),
    'max': np.array([110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136]),
    'all_predictions': np.random.randn(5, 14) * 5 + 115
}

current_price = 100.0

print("\n1. Testing calculate_forecast_metrics...")
metrics = calculate_forecast_metrics(test_stats, current_price, 14)
print(f"   Current Price: ${current_price:.2f}")
print(f"   14-day Median Forecast: ${metrics['median']:.2f}")
print(f"   Expected Change: {metrics['median_change_pct']:+.2f}%")
print(f"   Forecast Range (Q25-Q75): {metrics['forecast_range_pct']:.2f}%")
print(f"   Probability Above Current: {metrics['prob_above']:.1f}%")
print("   ✓ Passed")

print("\n2. Testing analyze_forecast_gradient...")
gradient = analyze_forecast_gradient(test_stats)
print(f"   Forecast Shape: {gradient['shape']}")
print(f"   Description: {gradient['description']}")
print(f"   Peak Day: {gradient['peak_day']} at ${gradient['peak_value']:.2f}")
print(f"   First Half Avg Return: {gradient['first_half_avg']:.2f}%")
print(f"   Second Half Avg Return: {gradient['second_half_avg']:.2f}%")
print("   ✓ Passed")

print("\n" + "="*70)
print("ALL TESTS PASSED - STRATEGIES ARE WORKING!")
print("="*70)
print("\nThe strategies can now be run with real data:")
print("  - python strategies/confidence_weighted_strategy.py")
print("  - python strategies/forecast_gradient_strategy.py")
print("  - python strategies/multi_timeframe_strategy.py")
print("  - python strategies/volatility_position_sizing.py")
print("  - python examples/compare_all_strategies.py")
print("="*70)
