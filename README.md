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

## Docker Deployment

The project includes a complete Docker setup with REST API backend, MCP server for AI agent access, and MongoDB database.

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Server    │────▶│  Backend API     │────▶│   MongoDB       │
│  (Port: stdio)  │     │  (Port: 8000)    │     │ (Port: 27017)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
      AI Agents              REST API              Database Storage
```

**Services:**
- **Backend API**: FastAPI REST endpoints for all 8 trading strategies
- **MCP Server**: Model Context Protocol server for AI agent integration
- **MongoDB**: Database for storing analyses, consensus results, and forecasts

### Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/yourusername/temporal-trading-agents.git
cd temporal-trading-agents
```

2. Build and start all services:
```bash
docker-compose up --build
```

3. Access the API:
```bash
# Health check
curl http://localhost:8000/health

# Run gradient strategy analysis
curl -X POST "http://localhost:8000/api/v1/analyze/gradient?symbol=BTC-USD"

# Run comprehensive 8-strategy consensus analysis
curl -X POST http://localhost:8000/api/v1/analyze/consensus \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC-USD", "horizons": [3, 7, 14, 21]}'
```

### Docker Services

**Backend API** (Port 8000):
- Health check: `GET /health`
- Gradient strategy: `POST /api/v1/analyze/gradient?symbol=SYMBOL`
- Confidence strategy: `POST /api/v1/analyze/confidence?symbol=SYMBOL`
- All 8 strategies consensus: `POST /api/v1/analyze/consensus`
- Analysis history: `GET /api/v1/history/analyses/{symbol}`
- Consensus history: `GET /api/v1/history/consensus/{symbol}`
- Symbol analytics: `GET /api/v1/analytics/{symbol}`

**MCP Server** (AI Agent Access):
Available tools:
- `analyze_gradient_strategy` - Run forecast gradient analysis
- `analyze_confidence_strategy` - Run confidence-weighted analysis
- `analyze_all_strategies` - Run all 8 strategies with consensus
- `get_analysis_history` - Get historical strategy analyses
- `get_consensus_history` - Get historical consensus results
- `get_symbol_analytics` - Get comprehensive symbol analytics

**MongoDB** (Port 27017):
Collections:
- `strategy_analyses` - Individual strategy analysis results
- `consensus_results` - Multi-strategy consensus results
- `model_trainings` - Model training records
- `price_forecasts` - Forecast data
- `users` - User accounts (future)
- `api_keys` - API keys (future)

### Environment Configuration

Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
```

Key environment variables:
- `MONGODB_URL` - MongoDB connection string (default: `mongodb://mongodb:27017`)
- `BACKEND_URL` - Backend API URL for MCP server (default: `http://backend:8000`)
- `DEFAULT_FORECAST_HORIZONS` - Default horizons for analysis (default: `3,7,14,21`)
- `CORS_ORIGINS` - Allowed CORS origins (comma-separated)

### Docker Commands

```bash
# Start services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f mcp-server
docker-compose logs -f mongodb

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes database data)
docker-compose down -v

# Rebuild services
docker-compose up --build

# Check service health
docker-compose ps
```

### API Examples

**Run Gradient Strategy:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/gradient?symbol=ETH-USD"
```

**Run All 8 Strategies with Consensus:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze/consensus \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "horizons": [3, 7, 14, 21]
  }'
```

**Get Analysis History:**
```bash
curl "http://localhost:8000/api/v1/history/analyses/BTC-USD?limit=10"
```

**Get Symbol Analytics:**
```bash
curl "http://localhost:8000/api/v1/analytics/BTC-USD"
```

### MCP Server Usage

The MCP server allows AI agents (like Claude) to access trading strategies through a standardized protocol.

**Claude Desktop Configuration:**

Add to your Claude Desktop `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "temporal-trading": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "temporal-trading-mcp",
        "python",
        "mcp-server/server.py"
      ]
    }
  }
}
```

**Example Agent Usage:**

With the MCP server configured, AI agents can use commands like:
```
Analyze BTC-USD using all 8 trading strategies
Get consensus history for ETH-USD
What are the analytics for BTC-USD?
```

### Database Management

**Access MongoDB shell:**
```bash
docker exec -it temporal-trading-mongodb mongosh
```

**MongoDB commands:**
```javascript
// Switch to database
use temporal_trading

// View collections
show collections

// Query strategy analyses
db.strategy_analyses.find({symbol: "BTC-USD"}).sort({created_at: -1}).limit(5)

// Query consensus results
db.consensus_results.find({symbol: "BTC-USD"}).sort({created_at: -1}).limit(5)

// Get analytics
db.strategy_analyses.aggregate([
  {$match: {symbol: "BTC-USD"}},
  {$group: {_id: "$strategy_type", count: {$sum: 1}}}
])
```

### Troubleshooting

**Backend fails to start:**
- Check MongoDB is healthy: `docker-compose ps`
- View backend logs: `docker-compose logs backend`
- Ensure port 8000 is not in use: `lsof -i :8000`

**MongoDB connection issues:**
- Verify MongoDB is running: `docker-compose ps mongodb`
- Check MongoDB logs: `docker-compose logs mongodb`
- Wait for MongoDB health check to pass (may take 10-20 seconds on first start)

**MCP server issues:**
- Ensure backend is healthy: `curl http://localhost:8000/health`
- View MCP logs: `docker-compose logs mcp-server`
- Check BACKEND_URL environment variable

**Out of memory errors:**
- PyTorch models require significant memory
- Increase Docker memory limit (Docker Desktop: Settings → Resources → Memory)
- Recommended: 8GB+ RAM allocation

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
- `examples/compare_all_strategies.py` - Compare first 4 advanced strategies on same asset
- `examples/compare_all_8_strategies.py` - Compare ALL 8 advanced strategies with consensus analysis
- `strategies/crypto_trading_strategy.py` - Buy-the-dip strategy (7-day vs 14-day comparison)
- `strategies/forecast_gradient_strategy.py` - Curve shape analysis strategy
- `strategies/confidence_weighted_strategy.py` - Model agreement-based strategy
- `strategies/multi_timeframe_strategy.py` - Multiple horizon alignment strategy
- `strategies/volatility_position_sizing.py` - Uncertainty-based position sizing
- `strategies/mean_reversion_strategy.py` - Mean reversion with forecast confirmation
- `strategies/acceleration_strategy.py` - Acceleration/deceleration momentum strategy
- `strategies/swing_trading_strategy.py` - Intra-forecast swing opportunities
- `strategies/risk_adjusted_strategy.py` - Advanced risk-adjusted strategy (Sharpe, Sortino, VaR, CVaR)

## Features

- **Multi-horizon Forecasting**: Generate predictions for 3, 7, 14, and 21-day timeframes
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Advanced Trading Strategies**: 9 distinct strategies leveraging forecast capabilities
- **Dynamic Position Sizing**: Adjust position sizes based on confidence, volatility, and risk metrics
- **Risk Management**: Built-in stop-loss, risk/reward calculations, VaR, CVaR, Sharpe/Sortino ratios
- **Comprehensive Visualization**: Rich charts for forecasts and trading decisions
- **Strategy Comparison**: Compare all 8 advanced strategies side-by-side with consensus analysis

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

### 6. Mean Reversion with Forecast Confirmation (`mean_reversion_strategy.py`)
**Trades mean reversion only when the forecast confirms the reversion will occur**

Trading signals based on mean reversion + forecast alignment:
- **OVERSOLD_REVERT**: Price below SMA, Z-score indicates oversold, forecast confirms upward reversion
- **OVERBOUGHT_REVERT**: Price above SMA, Z-score indicates overbought, forecast confirms downward reversion
- **FALSE_SIGNAL**: Mean reversion setup but forecast contradicts the expected move
- **NO_SIGNAL**: No mean reversion opportunity

**Key Features:**
- Calculates 20-day and 50-day Simple Moving Averages
- Uses Z-score analysis to identify overbought/oversold conditions
- Validates mean reversion signals against forecast direction
- Filters out false signals where price appears mean-reverting but forecast disagrees
- Measures deviation magnitude for position sizing

**Mean Reversion Metrics:**
- Z-score < -1.5 (oversold) or > 1.5 (overbought)
- SMA crossover confirmation
- Forecast must predict movement toward the mean

**Usage:**
```bash
cd strategies
python mean_reversion_strategy.py
```

### 7. Acceleration/Deceleration Momentum Strategy (`acceleration_strategy.py`)
**Analyzes changes in momentum to identify accelerating or decelerating trends**

Trading signals based on momentum acceleration:
- **ACCELERATING_GAINS**: Momentum is increasing, suggesting strengthening trend
- **DECELERATING_GAINS**: Momentum is slowing, potential trend exhaustion
- **ACCELERATION_REVERSAL**: Negative momentum becoming more negative (bearish acceleration)
- **DECELERATION_REVERSAL**: Negative momentum slowing (potential bottoming)

**Key Features:**
- Compares first 7 days vs. second 7 days of 14-day forecast
- Calculates daily returns for each period
- Measures acceleration (change in momentum between periods)
- Identifies momentum shifts that precede trend changes
- Higher position sizes for stronger acceleration

**Momentum Metrics:**
- Period 1 (Days 0-7) daily return rate
- Period 2 (Days 7-14) daily return rate
- Acceleration = Period 2 rate - Period 1 rate
- Positive acceleration (>0.2% daily) = strong signal
- Negative acceleration (<-0.2% daily) = trend exhaustion

**Usage:**
```bash
cd strategies
python acceleration_strategy.py
```

### 8. Swing Trading with Intra-Forecast Strategy (`swing_trading_strategy.py`)
**Identifies multiple swing opportunities within the forecast window**

Trading signals based on forecast peaks and troughs:
- **EXCELLENT_SWING_OPP**: Multiple profitable swings detected within forecast period
- **GOOD_SWING_OPP**: 1-2 profitable swings available
- **POOR_SWING_OPP**: Swings detected but profit potential too small
- **NO_SWING**: No clear swing pattern in forecast

**Key Features:**
- Uses scipy.signal.argrelextrema to detect local peaks and troughs
- Identifies multiple entry/exit opportunities within forecast window
- Calculates profit potential for each swing
- Requires minimum 2% price movement for valid swing
- Provides specific entry/exit prices and expected returns

**Swing Detection:**
- Local maxima (peaks) and minima (troughs) in forecast
- Minimum swing amplitude: 2% price change
- Expected return calculation for each swing opportunity
- Filters noise using relative extrema detection (order=2)

**Usage:**
```bash
cd strategies
python swing_trading_strategy.py
```

### 9. Advanced Risk-Adjusted Strategy (`risk_adjusted_strategy.py`)
**Comprehensive risk analysis using multiple risk metrics and modern portfolio theory**

Trading signals based on risk-adjusted returns:
- **EXCELLENT_RISK_ADJUSTED**: High Sharpe (>1.5), low drawdown, favorable VaR/CVaR
- **GOOD_RISK_ADJUSTED**: Positive Sharpe (>0.8), acceptable risk metrics
- **MODERATE_RISK**: Positive expected return but elevated risk
- **POOR_RISK_ADJUSTED**: Negative Sharpe or unacceptable risk profile

**Key Features:**
- **Sharpe Ratio**: Risk-adjusted return using total volatility
- **Sortino Ratio**: Risk-adjusted return using downside volatility only
- **Maximum Drawdown**: Worst peak-to-trough decline in forecast
- **Value at Risk (VaR 95%)**: Expected loss at 95% confidence level
- **Conditional VaR (CVaR)**: Expected loss in worst 5% of scenarios
- **Risk-Adjusted Score**: Weighted composite of all metrics

**Risk Metrics:**
- Sharpe Ratio > 1.5 = Excellent
- Sortino Ratio > 1.0 = Good downside protection
- Max Drawdown < -5% = Elevated risk
- VaR and CVaR for tail risk assessment
- Position sizing based on composite risk score

**Usage:**
```bash
cd strategies
python risk_adjusted_strategy.py
```

### Compare All Strategies

**Compare First 4 Strategies:**
```bash
cd examples
python compare_all_strategies.py
```

**Compare ALL 8 Advanced Strategies (Recommended):**
```bash
cd examples
python compare_all_8_strategies.py
```

The comprehensive 8-strategy comparison will:
- Train ensemble models for all required horizons (3, 7, 14, 21 days)
- Run all 8 advanced strategies on the same asset
- Analyze consensus across strategies (bullish/bearish/neutral counts)
- Calculate average position sizing recommendations
- Provide clear recommended action based on strategy agreement
- Create comprehensive visualization with:
  - 14-day forecast plot
  - Strategy consensus pie chart
  - Position sizing comparison
  - Signal summary table
  - Key insights from each strategy

**Consensus Signals:**
- **STRONG BUY**: 6+ strategies bullish
- **BUY**: 5+ strategies bullish
- **MODERATE BUY**: 4+ strategies bullish
- **SELL/AVOID**: 4+ strategies bearish
- **MIXED SIGNALS**: No clear consensus

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
