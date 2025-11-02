# Parameter Optimization and Regime Analysis

## Overview
This document outlines the design and implementation plan for two key features:
1. **Parameter Optimization** - Grid search and optimization for backtest parameters
2. **Regime Analysis** - Market regime detection and regime-specific performance metrics

---

## 1. Parameter Optimization

### Current Configurable Parameters

From `backend/models.py` - `BacktestConfig`:
```python
position_size_pct: float = 10.0       # % of portfolio per position
min_edge_bps: float = 55.0            # Minimum edge to trade
initial_capital: float = 100000.0     # Starting capital
```

From `backend/backtesting_engine.py` - Consensus voting thresholds (hardcoded):
```python
if bullish_pct >= 0.80:    # Strong buy threshold
if bullish_pct >= 0.60:    # Buy threshold
if bullish_pct >= 0.50:    # Moderate buy threshold
if bearish_pct >= 0.60:    # Sell threshold
if bearish_pct >= 0.50:    # Moderate sell threshold
```

### Proposed Optimizable Parameters

#### High Impact Parameters
1. **Position Sizing**
   - Current: Fixed 10% per trade
   - Optimize: 5%, 10%, 15%, 20%, 25%
   - Impact: Risk/reward tradeoff, drawdown control

2. **Consensus Voting Thresholds**
   - Strong buy threshold: 70%, 75%, 80%, 85%
   - Buy threshold: 55%, 60%, 65%, 70%
   - Moderate buy threshold: 45%, 50%, 55%
   - Impact: Trade frequency, signal quality

3. **Minimum Edge Requirement**
   - Current: 55 bps (3x transaction costs)
   - Optimize: 30, 40, 50, 60, 70, 80 bps
   - Impact: Trade selectivity, cost coverage

#### Medium Impact Parameters
4. **Stop Loss Percentage**
   - Range: 2%, 3%, 5%, 7%, 10%
   - Impact: Risk management, win rate

5. **Take Profit Percentage**
   - Range: 5%, 7%, 10%, 15%, 20%
   - Impact: Profit capture, trade duration

6. **Maximum Concurrent Positions**
   - Range: 1, 2, 3, unlimited
   - Impact: Diversification, capital utilization

### Implementation Design

#### 1. Enhanced Config Model
```python
class OptimizableParams(BaseModel):
    """Parameters that can be optimized"""
    position_size_pct: float = 10.0
    min_edge_bps: float = 55.0
    strong_buy_threshold: float = 0.80
    buy_threshold: float = 0.60
    moderate_buy_threshold: float = 0.50
    sell_threshold: float = 0.60
    moderate_sell_threshold: float = 0.50
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_concurrent_positions: Optional[int] = None

class ParameterGrid(BaseModel):
    """Grid of parameters to test"""
    position_size_pct: List[float] = [5, 10, 15, 20]
    min_edge_bps: List[float] = [30, 50, 70]
    strong_buy_threshold: List[float] = [0.75, 0.80, 0.85]
    # ... other parameters

class OptimizationRequest(BaseModel):
    """Request to run parameter optimization"""
    backtest_config: BacktestConfig
    parameter_grid: ParameterGrid
    optimization_metric: str = "sharpe_ratio"  # or total_return, profit_factor, etc.
    top_n_results: int = 10

class OptimizationResult(BaseModel):
    """Single optimization run result"""
    run_id: str
    parameters: OptimizableParams
    metrics: BacktestMetrics
    rank: int  # Ranking by optimization metric
```

#### 2. Optimization Engine
```python
class ParameterOptimizer:
    """Grid search parameter optimizer"""

    def optimize(
        self,
        base_config: BacktestConfig,
        parameter_grid: ParameterGrid,
        price_data: pd.DataFrame
    ) -> List[OptimizationResult]:
        """
        Run grid search over parameter combinations.

        Uses:
        - Parallel execution via ThreadPoolExecutor
        - Strategy result caching for speed
        - Progress tracking and early stopping
        """

        # Generate all parameter combinations
        combinations = self._generate_combinations(parameter_grid)

        # Run backtests in parallel
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for params in combinations:
                config = self._apply_params(base_config, params)
                future = executor.submit(self._run_single_backtest, config, price_data)
                futures.append((future, params))

            for future, params in futures:
                try:
                    backtest_result = future.result()
                    results.append(OptimizationResult(
                        run_id=backtest_result.run_id,
                        parameters=params,
                        metrics=backtest_result.metrics
                    ))
                except Exception as e:
                    logger.error(f"Backtest failed for {params}: {e}")

        # Rank by optimization metric
        results = self._rank_results(results, optimization_metric)

        return results[:top_n_results]
```

#### 3. API Endpoints
```python
@app.post("/api/v1/optimize/parameters")
async def optimize_parameters(request: OptimizationRequest):
    """Run parameter optimization"""

@app.get("/api/v1/optimize/{optimization_id}")
async def get_optimization_results(optimization_id: str):
    """Get optimization results"""
```

### Performance Considerations

With caching (PR #27):
- Each parameter combination reuses cached strategy results where `(stats, price)` matches
- Expected speedup: **5-10x faster** than without caching
- Example: 100 parameter combinations × 250 bars = 25,000 signal computations
  - Without caching: ~25 seconds
  - With caching (~70% hit rate): ~7 seconds

Grid search complexity:
- 4 position sizes × 3 edge thresholds × 3 voting thresholds = **36 combinations**
- With 4 parallel workers: ~2 minutes total (vs 8 minutes sequential)

---

## 2. Regime Analysis

### Market Regime Types

#### Volatility Regimes
1. **Low Volatility** (VIX < 15 or realized vol < 15%)
   - Characteristics: Steady trends, mean-reversion works
   - Best strategies: Mean reversion, swing trading

2. **Medium Volatility** (VIX 15-25 or realized vol 15-30%)
   - Characteristics: Normal markets
   - Best strategies: All strategies balanced

3. **High Volatility** (VIX > 25 or realized vol > 30%)
   - Characteristics: Large swings, trend-following works
   - Best strategies: Momentum, volatility sizing

#### Trend Regimes
1. **Strong Uptrend** (Price > 50 SMA, 50 SMA > 200 SMA, slope > threshold)
   - Best: Momentum, gradient strategies

2. **Weak Uptrend** (Price > 50 SMA, flat slope)
   - Best: Swing trading, confidence-weighted

3. **Sideways/Range** (Price oscillating around moving averages)
   - Best: Mean reversion, swing trading

4. **Downtrend** (Price < 50 SMA, 50 SMA < 200 SMA)
   - Best: Avoid longs, risk-adjusted short signals

#### Volume Regimes
1. **High Volume** (Volume > 1.5x average)
   - Stronger signals, better liquidity

2. **Low Volume** (Volume < 0.5x average)
   - Weaker signals, higher slippage

### Implementation Design

#### 1. Regime Detection Module
```python
class MarketRegime(str, Enum):
    """Market regime types"""
    LOW_VOL_UPTREND = "low_vol_uptrend"
    LOW_VOL_DOWNTREND = "low_vol_downtrend"
    HIGH_VOL_UPTREND = "high_vol_uptrend"
    HIGH_VOL_DOWNTREND = "high_vol_downtrend"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"

class RegimeMetrics(BaseModel):
    """Metrics for a specific regime"""
    regime: MarketRegime
    bars_in_regime: int
    regime_pct: float  # % of backtest in this regime
    total_return: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float

class RegimeAnalysis(BaseModel):
    """Complete regime breakdown"""
    regime_metrics: List[RegimeMetrics]
    regime_transitions: int  # How often regime changed
    dominant_regime: MarketRegime  # Most common regime
    best_performing_regime: MarketRegime
    worst_performing_regime: MarketRegime

def detect_regime(
    price_data: pd.DataFrame,
    lookback_window: int = 50
) -> MarketRegime:
    """
    Detect current market regime based on:
    - Realized volatility (std of returns)
    - Trend direction (SMA crossovers, slope)
    - Volume patterns
    """

    # Calculate indicators
    returns = price_data['close'].pct_change()
    volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized

    sma_50 = price_data['close'].rolling(50).mean().iloc[-1]
    sma_200 = price_data['close'].rolling(200).mean().iloc[-1]
    current_price = price_data['close'].iloc[-1]

    # Trend detection
    is_uptrend = (current_price > sma_50) and (sma_50 > sma_200)
    is_downtrend = (current_price < sma_50) and (sma_50 < sma_200)
    is_sideways = not is_uptrend and not is_downtrend

    # Volatility classification
    is_high_vol = volatility > 0.30  # 30% annualized
    is_low_vol = volatility < 0.15   # 15% annualized

    # Determine regime
    if is_uptrend and is_low_vol:
        return MarketRegime.LOW_VOL_UPTREND
    elif is_uptrend and is_high_vol:
        return MarketRegime.HIGH_VOL_UPTREND
    elif is_downtrend and is_low_vol:
        return MarketRegime.LOW_VOL_DOWNTREND
    elif is_downtrend and is_high_vol:
        return MarketRegime.HIGH_VOL_DOWNTREND
    elif is_sideways and is_low_vol:
        return MarketRegime.SIDEWAYS_LOW_VOL
    else:  # sideways high vol
        return MarketRegime.SIDEWAYS_HIGH_VOL
```

#### 2. Regime Tracking in Backtests
```python
class BacktestEngine:
    def run_backtest_with_regime_analysis(self, price_data: pd.DataFrame):
        """Enhanced backtest with regime tracking"""

        regime_trades = defaultdict(list)  # Track trades by regime
        regime_bars = defaultdict(int)     # Count bars in each regime
        current_regime = None
        regime_changes = 0

        for idx, row in price_data.iterrows():
            # Detect regime at current bar
            historical_df = price_data.loc[:idx]
            if len(historical_df) >= 200:  # Need enough data for indicators
                new_regime = detect_regime(historical_df)

                if new_regime != current_regime:
                    regime_changes += 1
                    current_regime = new_regime

                regime_bars[current_regime] += 1

            # Execute trading logic (existing code)
            # ...

            # Track trades by regime
            if trade_executed:
                regime_trades[current_regime].append(trade)

        # Calculate regime-specific metrics
        regime_metrics = []
        total_bars = len(price_data)

        for regime, trades in regime_trades.items():
            metrics = self._calculate_regime_metrics(
                regime, trades, regime_bars[regime], total_bars
            )
            regime_metrics.append(metrics)

        # Add to backtest results
        backtest_result.regime_analysis = RegimeAnalysis(
            regime_metrics=regime_metrics,
            regime_transitions=regime_changes,
            dominant_regime=max(regime_bars, key=regime_bars.get),
            best_performing_regime=max(regime_metrics, key=lambda m: m.sharpe_ratio).regime,
            worst_performing_regime=min(regime_metrics, key=lambda m: m.sharpe_ratio).regime
        )
```

#### 3. Enhanced Backtest Results Model
```python
class BacktestRun(BaseModel):
    """Complete backtest run record"""
    run_id: str
    name: str
    config: BacktestConfig
    status: BacktestStatus
    metrics: Optional[BacktestMetrics] = None
    period_metrics: List[BacktestPeriodMetrics] = []

    # NEW: Regime analysis
    regime_analysis: Optional[RegimeAnalysis] = None

    trades: List[BacktestTrade] = []
    equity_curve: List[EquityPoint] = []
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
```

### Regime-Aware Strategy Selection

Future enhancement: Adjust strategy weights based on detected regime
```python
def get_regime_weighted_consensus(
    strategies: Dict[str, Dict],
    current_regime: MarketRegime
) -> str:
    """
    Weight strategy votes based on regime-specific performance.

    Example: In HIGH_VOL_UPTREND, give 2x weight to:
    - Volatility sizing
    - Acceleration/momentum

    Reduce weight for:
    - Mean reversion (less effective in strong trends)
    """

    regime_weights = {
        MarketRegime.HIGH_VOL_UPTREND: {
            'volatility': 2.0,
            'acceleration': 2.0,
            'gradient': 1.5,
            'mean_reversion': 0.5,
        },
        MarketRegime.SIDEWAYS_LOW_VOL: {
            'mean_reversion': 2.0,
            'swing': 2.0,
            'volatility': 0.5,
            'acceleration': 0.5,
        },
        # ... other regimes
    }

    weights = regime_weights.get(current_regime, {})

    # Apply weights to votes
    weighted_bullish = sum(
        weights.get(name, 1.0)
        for name in bullish_strategies
    )

    # Use weighted votes for consensus
    # ...
```

---

## Implementation Priority

### Phase 1: Basic Parameter Optimization
- [ ] Add `OptimizableParams` to `BacktestConfig`
- [ ] Update `_get_consensus_signal` to use configurable thresholds
- [ ] Implement grid search with 3-4 key parameters
- [ ] Add `/api/v1/optimize/parameters` endpoint
- [ ] Build simple UI for optimization results

### Phase 2: Regime Analysis
- [ ] Implement `detect_regime()` function
- [ ] Add regime tracking to backtesting loop
- [ ] Calculate regime-specific metrics
- [ ] Add `RegimeAnalysis` to `BacktestRun` model
- [ ] Display regime breakdown in UI

### Phase 3: Advanced Features
- [ ] Regime-weighted consensus voting
- [ ] Adaptive parameters based on regime
- [ ] Regime transition analysis
- [ ] Regime-specific optimization

---

## Performance & Scalability

### Parameter Optimization
- **With caching (70% hit rate):** 36 combinations in ~2 minutes
- **Without caching:** 36 combinations in ~8 minutes
- **Parallelization:** 4 workers (limited by ThreadPoolExecutor)

### Regime Analysis
- **Computational overhead:** ~5-10% slowdown (indicator calculations)
- **Memory overhead:** Negligible (just tracking regime state)
- **Value:** High - reveals when/why strategy works

---

## Expected Benefits

### Parameter Optimization
1. **Higher Sharpe ratios** (20-40% improvement from optimized parameters)
2. **Better risk management** (optimized stop losses and position sizes)
3. **Reduced drawdowns** (optimized thresholds reduce bad trades)
4. **Data-driven decisions** (objective optimization vs guessing)

### Regime Analysis
1. **Transparency** - Understand when strategy works/fails
2. **Risk awareness** - Avoid trading in unfavorable regimes
3. **Strategy selection** - Enable/disable strategies based on regime
4. **Performance attribution** - Separate alpha from regime effects

---

## Next Steps

1. **Decide priority:** Parameter optimization vs Regime analysis first?
2. **UI considerations:** How should optimization results be displayed?
3. **Storage:** Should we save all optimization runs or just top N?
4. **Real-time:** Should regime detection work in paper trading?
