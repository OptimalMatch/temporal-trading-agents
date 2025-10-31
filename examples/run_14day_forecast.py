"""
Run 14-day Bitcoin forecast using ensemble approach

Simply runs the ensemble system with forecast_horizon=14
"""

import sys
import importlib.util

# Load the ensemble module
spec = importlib.util.spec_from_file_location("ensemble", "crypto_ensemble_forecast.py")
ensemble = importlib.util.module_from_spec(spec)
sys.modules["ensemble"] = ensemble
spec.loader.exec_module(ensemble)

# Import what we need
import torch
import numpy as np

print("="*70)
print("14-DAY BITCOIN FORECAST - ENSEMBLE APPROACH")
print("="*70)
print("\nNote: Longer forecasts have higher uncertainty")
print("Use Day 1-7 predictions with higher confidence")
print("="*70)

symbol = 'BTC-USD'
forecast_horizon = 14  # 14 days instead of 7

# Define ensemble models (same as before but with 14-day horizon)
ensemble_configs = [
    {'lookback': 30, 'focus': 'momentum', 'epochs': 20, 'name': 'Short-term Momentum'},
    {'lookback': 60, 'focus': 'balanced', 'epochs': 25, 'name': 'Medium-term Balanced'},
    {'lookback': 90, 'focus': 'balanced', 'epochs': 25, 'name': 'Long-term Trend'},
    {'lookback': 60, 'focus': 'mean_reversion', 'epochs': 20, 'name': 'Mean Reversion'},
    {'lookback': 45, 'focus': 'momentum', 'epochs': 20, 'name': 'Mid-term Momentum'},
]

# Train ensemble
print("\n" + "="*70)
print("TRAINING ENSEMBLE MODELS (14-day forecast)")
print("="*70)

ensemble_models = []
for config in ensemble_configs:
    model_info = ensemble.train_ensemble_model(
        symbol=symbol,
        period='2y',
        lookback=config['lookback'],
        forecast_horizon=forecast_horizon,  # 14 days
        epochs=config['epochs'],
        focus=config['focus'],
        model_name=config['name']
    )
    ensemble_models.append(model_info)

# Make predictions
ensemble_stats, df_latest = ensemble.make_ensemble_predictions(ensemble_models, symbol, forecast_horizon)

# Print summary
current_price = df_latest['Close'].iloc[-1]
print(f"\n{'='*70}")
print("14-DAY ENSEMBLE FORECAST SUMMARY")
print(f"{'='*70}")
print(f"\nCurrent {symbol} Price: ${current_price:,.2f}")

print(f"\n14-Day Forecast Scenarios:")
print(f"  Best Case (Max):         ${ensemble_stats['max'][-1]:,.2f} ({((ensemble_stats['max'][-1]-current_price)/current_price*100):+.1f}%)")
print(f"  Optimistic (75th %ile):  ${ensemble_stats['q75'][-1]:,.2f} ({((ensemble_stats['q75'][-1]-current_price)/current_price*100):+.1f}%)")
print(f"  Most Likely (Median):    ${ensemble_stats['median'][-1]:,.2f} ({((ensemble_stats['median'][-1]-current_price)/current_price*100):+.1f}%)")
print(f"  Pessimistic (25th %ile): ${ensemble_stats['q25'][-1]:,.2f} ({((ensemble_stats['q25'][-1]-current_price)/current_price*100):+.1f}%)")
print(f"  Worst Case (Min):        ${ensemble_stats['min'][-1]:,.2f} ({((ensemble_stats['min'][-1]-current_price)/current_price*100):+.1f}%)")

# Show week-by-week breakdown
print(f"\n{'='*70}")
print("WEEK-BY-WEEK BREAKDOWN")
print(f"{'='*70}")

print("\nWeek 1 (Day 7):")
print(f"  Median: ${ensemble_stats['median'][6]:,.2f} ({((ensemble_stats['median'][6]-current_price)/current_price*100):+.1f}%)")
print(f"  Range: ${ensemble_stats['q25'][6]:,.2f} to ${ensemble_stats['q75'][6]:,.2f}")

print("\nWeek 2 (Day 14):")
print(f"  Median: ${ensemble_stats['median'][-1]:,.2f} ({((ensemble_stats['median'][-1]-current_price)/current_price*100):+.1f}%)")
print(f"  Range: ${ensemble_stats['q25'][-1]:,.2f} to ${ensemble_stats['q75'][-1]:,.2f}")

uncertainty_week1 = (ensemble_stats['std'][6] / ensemble_stats['median'][6]) * 100
uncertainty_week2 = (ensemble_stats['std'][-1] / ensemble_stats['median'][-1]) * 100

print(f"\nUncertainty Analysis:")
print(f"  Week 1 uncertainty: ±{uncertainty_week1:.1f}%")
print(f"  Week 2 uncertainty: ±{uncertainty_week2:.1f}%")
print(f"  Uncertainty increase: {uncertainty_week2 - uncertainty_week1:.1f}pp")

prob_above = np.mean(ensemble_stats['all_predictions'][:, -1] > current_price) * 100
prob_below = np.mean(ensemble_stats['all_predictions'][:, -1] < current_price) * 100

print(f"\nDay 14 Model Consensus:")
print(f"  {int(prob_above)}% of models predict price will be ABOVE ${current_price:,.0f}")
print(f"  {int(prob_below)}% of models predict price will be BELOW ${current_price:,.0f}")

# Visualize
print(f"\n{'='*70}")
print("CREATING VISUALIZATION")
print(f"{'='*70}")
ensemble.visualize_ensemble(ensemble_stats, df_latest, symbol)

# Rename output file
import os
os.rename('bitcoin_ensemble_forecast.png', 'bitcoin_14day_forecast.png')
print(f"✓ Visualization saved to 'bitcoin_14day_forecast.png'")

print(f"\n{'='*70}")
print("14-DAY FORECAST COMPLETE!")
print(f"{'='*70}")
print("\nKey Insights:")
print("✓ Uncertainty increases from Week 1 to Week 2")
print("✓ Use Week 1 forecasts with higher confidence")
print("✓ Week 2 forecasts provide directional guidance")
print("✓ Consider multiple scenarios when making decisions")
