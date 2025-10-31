# Temporal Trading Agents

A trading bot framework built on top of [temporal-forecasting](https://github.com/OptimalMatch/temporal) that uses advanced time series forecasting to generate trading signals and strategies.

## Overview

This repository contains trading agents and strategies that leverage the temporal-forecasting library for making trading decisions in cryptocurrency and stock markets. The agents use ensemble forecasting methods and multi-horizon predictions to identify trading opportunities.

## Project Structure

```
temporal-trading-agents/
├── agents/           # Trading agent implementations
├── strategies/       # Trading strategies using forecasts
├── examples/         # Example scripts and demonstrations
├── data/            # Data storage and cache
└── requirements.txt # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/temporal-trading-agents.git
cd temporal-trading-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install from the temporal-forecasting repository directly:
```bash
pip install git+https://github.com/OptimalMatch/temporal.git
```

## Quick Start

### Example: Crypto Trading Strategy

```python
from strategies.crypto_trading_strategy import run_trading_strategy

# Run a 14-day forecast-based trading strategy
run_trading_strategy(
    ticker="BTC-USD",
    forecast_days=14,
    lookback_days=90
)
```

### Example: Stock Forecasting

```python
# See examples/stock_forecasting.py
python examples/stock_forecasting.py
```

## Available Examples

### Forecasting Examples
- `examples/crypto_14day_forecast.py` - 14-day cryptocurrency forecasting
- `examples/crypto_ensemble_forecast.py` - Ensemble forecasting for crypto
- `examples/crypto_forecasting_improved.py` - Enhanced crypto forecasting
- `examples/stock_forecasting.py` - Stock market forecasting
- `examples/run_14day_forecast.py` - Utility script for running forecasts

### Strategy Examples
- `examples/compare_all_strategies.py` - Compare all 4 advanced strategies on same asset
- `strategies/crypto_trading_strategy.py` - Buy-the-dip strategy (7-day vs 14-day comparison)
- `strategies/forecast_gradient_strategy.py` - Curve shape analysis strategy
- `strategies/confidence_weighted_strategy.py` - Model agreement-based strategy
- `strategies/multi_timeframe_strategy.py` - Multiple horizon alignment strategy
- `strategies/volatility_position_sizing.py` - Uncertainty-based position sizing

## Features

- **Multi-horizon Forecasting**: Generate predictions for 3, 7, 14, and 21-day timeframes
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Advanced Trading Strategies**: 5 distinct strategies leveraging forecast capabilities
- **Dynamic Position Sizing**: Adjust position sizes based on confidence and volatility
- **Risk Management**: Built-in stop-loss and risk/reward calculations
- **Comprehensive Visualization**: Rich charts for forecasts and trading decisions
- **Strategy Comparison**: Compare multiple strategies side-by-side

## Trading Strategies

### 1. Buy-the-Dip Strategy (`crypto_trading_strategy.py`)
**Original strategy comparing 7-day and 14-day forecasts**

Identifies four trading scenarios:
- **Buy the Dip**: 7-day decline followed by 14-day recovery
- **Momentum Play**: Sustained upward trend across both horizons
- **Stay Out**: Both forecasts show decline
- **Hold**: Moderate gains without clear pattern

**Usage:**
```bash
cd strategies
python crypto_trading_strategy.py
```

### 2. Forecast Gradient Strategy (`forecast_gradient_strategy.py`)
**Analyzes the SHAPE of the forecast curve**

Identifies and trades based on forecast patterns:
- **U-Shaped**: Buy at predicted trough, sell at recovery
- **Inverted-U**: Buy now, exit at peak before decline
- **Steep Rise**: Strong momentum play
- **Gradual Rise**: Buy and hold for full period
- **Decline**: Stay out or short

**Key Features:**
- Analyzes daily returns and curve gradients
- Identifies optimal entry/exit timing based on curve shape
- Detects peaks and troughs within forecast horizon

**Usage:**
```bash
cd strategies
python forecast_gradient_strategy.py
```

### 3. Confidence-Weighted Ensemble Strategy (`confidence_weighted_strategy.py`)
**Uses model agreement to determine position sizing**

Position sizing based on ensemble consensus:
- **HIGH confidence (80%+ agreement)**: 100% position
- **MEDIUM confidence (65-80%)**: 50% position
- **LOW confidence (55-65%)**: 25% position
- **VERY LOW (<55%)**: No trade

**Key Features:**
- Measures percentage of models predicting above/below current price
- Calculates prediction standard deviation and coefficient of variation
- Adjusts position size dynamically based on model agreement

**Usage:**
```bash
cd strategies
python confidence_weighted_strategy.py
```

### 4. Multi-Timeframe Trend Alignment (`multi_timeframe_strategy.py`)
**Compares 3, 7, 14, and 21-day forecasts for alignment**

Trading signals based on timeframe consensus:
- **All Bullish**: Strong buy (100% position)
- **Mostly Bullish**: Buy (75% position)
- **Short-term Opportunity**: Quick trade when short-term bullish but long-term bearish (50%)
- **Mixed**: No clear signal (25% or wait)
- **All Bearish**: Stay out

**Key Features:**
- Trains separate ensembles for 4 different horizons
- Identifies trend alignment across timeframes
- Higher conviction when all timeframes agree
- Detects divergences between short and long-term forecasts

**Usage:**
```bash
cd strategies
python multi_timeframe_strategy.py
```

### 5. Volatility-Based Position Sizing (`volatility_position_sizing.py`)
**Adjusts position size based on forecast uncertainty**

Dynamic position sizing using forecast volatility:
- **VERY LOW volatility (<5% range)**: 150-200% position
- **LOW volatility (5-10%)**: 100-150% position
- **MEDIUM volatility (10-15%)**: 50-100% position
- **HIGH volatility (15-20%)**: 25-50% position
- **VERY HIGH volatility (>20%)**: 0-25% position

**Key Features:**
- Uses Q25-Q75 forecast range as volatility measure
- Implements Kelly Criterion for optimal sizing
- Combines volatility-based and Kelly sizing
- Calculates position-adjusted risk/reward

**Usage:**
```bash
cd strategies
python volatility_position_sizing.py
```

### Compare All Strategies
Run all strategies and see their recommendations side-by-side:

```bash
cd examples
python compare_all_strategies.py
```

This will:
- Train models for all required horizons
- Run all 4 advanced strategies
- Display strategy consensus and recommended action
- Create comprehensive comparison visualization

## Strategy Utilities

The `strategies/strategy_utils.py` module provides shared functionality:
- Ensemble training and prediction
- Forecast metrics calculation (returns, volatility, confidence)
- Risk management (stop-loss, risk/reward ratios)
- Position sizing algorithms
- Gradient and curve shape analysis
- Visualization helpers

## Development Roadmap

- [ ] Live trading integration (Alpaca, Binance, etc.)
- [ ] Advanced risk management
- [ ] Portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Real-time data feeds
- [ ] Paper trading mode
- [ ] Performance analytics dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies and stocks carries risk. Always do your own research and never invest more than you can afford to lose. Past performance does not guarantee future results.
