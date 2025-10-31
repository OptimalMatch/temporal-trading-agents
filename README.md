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

- `examples/crypto_14day_forecast.py` - 14-day cryptocurrency forecasting
- `examples/crypto_ensemble_forecast.py` - Ensemble forecasting for crypto
- `examples/crypto_forecasting_improved.py` - Enhanced crypto forecasting
- `examples/stock_forecasting.py` - Stock market forecasting
- `examples/run_14day_forecast.py` - Utility script for running forecasts
- `strategies/crypto_trading_strategy.py` - Trading strategy implementation

## Features

- **Multi-horizon Forecasting**: Generate predictions for 7-day and 14-day timeframes
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Trading Signals**: Automated buy/hold/sell signal generation
- **Backtesting**: Evaluate strategy performance on historical data
- **Visualization**: Rich charts for forecasts and trading decisions

## Trading Strategies

The current implementation includes:

1. **Forecast-based Strategy**: Uses price predictions to generate signals
2. **Ensemble Strategy**: Combines multiple forecast horizons
3. **Comparison Strategy**: Analyzes differences between short and long-term forecasts

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
