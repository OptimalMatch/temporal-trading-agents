"""
Backtesting Engine for Consensus Trading Strategies

Implements realistic backtesting with:
- Transaction cost modeling
- Walk-forward validation
- Position tracking with P&L
- Performance metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# Add parent directory to path for strategy imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.models import (
    BacktestConfig, BacktestRun, BacktestTrade, BacktestMetrics,
    BacktestPeriodMetrics, BacktestStatus, TransactionCostConfig,
    WalkForwardConfig
)

# Import strategies
from strategies.forecast_gradient_strategy import analyze_gradient_strategy
from strategies.confidence_weighted_strategy import analyze_confidence_weighted_strategy
from strategies.multi_timeframe_strategy import analyze_multi_timeframe_strategy
from strategies.volatility_position_sizing import analyze_volatility_position_sizing
from strategies.mean_reversion_strategy import analyze_mean_reversion_strategy
from strategies.acceleration_strategy import analyze_acceleration_strategy
from strategies.swing_trading_strategy import analyze_swing_trading_strategy
from strategies.risk_adjusted_strategy import analyze_risk_adjusted_strategy
from strategies.strategy_utils import load_ensemble_module, train_ensemble, get_default_ensemble_configs
from strategies.strategy_cache import get_strategy_cache

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Models realistic transaction costs for trading.
    Based on the framework in docs/backtesting/
    """

    def __init__(self, config: TransactionCostConfig):
        self.config = config

    def calculate_cost(
        self,
        price: float,
        shares: int,
        order_type: str = "market",
        adv: float = 1_000_000
    ) -> float:
        """
        Calculate total transaction cost in dollars.

        Args:
            price: Execution price
            shares: Number of shares
            order_type: 'market' or 'limit'
            adv: Average daily volume (for market impact)

        Returns:
            Total cost in dollars (always positive)
        """
        notional = price * shares

        # Exchange fees
        exchange_cost = notional * (self.config.taker_fee_bps / 10000)

        # Spread cost (market orders cross the spread)
        spread_cost = notional * (self.config.half_spread_bps / 10000) if order_type == "market" else 0

        # Slippage: increases with order size relative to daily volume
        pct_of_adv = (shares / adv) * 100 if adv > 0 else 0
        slippage_bps = self.config.slippage_coefficient * pct_of_adv
        slippage_cost = notional * (slippage_bps / 10000)

        # Adverse selection
        adverse_cost = notional * (self.config.adverse_selection_bps / 10000)

        # Regulatory fees
        sec_cost = notional * (self.config.sec_fee_bps / 10000)

        total_cost = exchange_cost + spread_cost + slippage_cost + adverse_cost + sec_cost
        return total_cost

    def round_trip_cost_bps(self, shares: int, price: float, adv: float) -> float:
        """Calculate round-trip cost in basis points (buy + sell)."""
        notional = price * shares
        buy_cost = self.calculate_cost(price, shares, "market", adv)
        sell_cost = self.calculate_cost(price, shares, "market", adv)
        total_cost_bps = ((buy_cost + sell_cost) / notional) * 10000
        return total_cost_bps


class Position:
    """Track position state and P&L"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.shares = 0.0  # Support fractional shares for crypto
        self.avg_cost_basis = 0.0
        self.realized_pnl = 0.0
        self.total_transaction_costs = 0.0
        self.trades: List[Dict[str, Any]] = []

    def update(self, trade_dict: Dict[str, Any]):
        """Update position with new trade"""
        self.trades.append(trade_dict)
        self.total_transaction_costs += trade_dict['transaction_cost']

        if trade_dict['side'] == 'buy':
            # Update cost basis
            total_cost = self.shares * self.avg_cost_basis + trade_dict['notional']
            self.shares += trade_dict['shares']
            self.avg_cost_basis = total_cost / self.shares if self.shares > 0 else 0

        elif trade_dict['side'] == 'sell':
            # Allow small rounding errors in fractional shares
            if self.shares >= trade_dict['shares'] - 0.00001:
                # Realize P&L
                pnl = (trade_dict['price'] - self.avg_cost_basis) * trade_dict['shares']
                self.realized_pnl += pnl
                self.shares -= trade_dict['shares']
                # Prevent negative shares from rounding errors
                if abs(self.shares) < 0.00001:
                    self.shares = 0.0
            else:
                raise ValueError(f"Cannot sell {trade_dict['shares']} shares, only have {self.shares}")

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price"""
        if self.shares == 0:
            return 0.0
        return (current_price - self.avg_cost_basis) * self.shares

    def total_pnl(self, current_price: float) -> float:
        """Total P&L including costs"""
        return self.realized_pnl + self.unrealized_pnl(current_price) - self.total_transaction_costs


class WalkForwardValidator:
    """Implements walk-forward validation to detect overfitting"""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits for walk-forward validation.

        Returns list of (train_df, test_df) tuples.
        """
        splits = []
        total_days = len(data)

        current_day = self.config.train_window_days

        while current_day + self.config.test_window_days <= total_days:
            train_start = current_day - self.config.train_window_days
            train_end = current_day
            test_end = current_day + self.config.test_window_days

            train_df = data.iloc[train_start:train_end].copy()
            test_df = data.iloc[train_end:test_end].copy()

            splits.append((train_df, test_df))

            current_day += self.config.retrain_frequency_days

        return splits


class BacktestEngine:
    """
    Main backtesting engine that runs consensus strategies on historical data
    """

    def __init__(
        self,
        config: BacktestConfig,
        consensus_analyzer=None,  # Function to get consensus signal
        enable_regime_analysis: bool = True
    ):
        self.config = config
        self.consensus_analyzer = consensus_analyzer
        self.cost_model = TransactionCostModel(config.transaction_costs)
        self.enable_regime_analysis = enable_regime_analysis

        # State
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.consensus_stats: Optional[Dict] = None  # Trained model stats for consensus (14-day)
        self.multi_horizon_stats: Optional[Dict] = None  # Multi-horizon stats for multi-timeframe strategy

        # Regime tracking
        self.regime_tracker: Optional['RegimeTracker'] = None
        if self.enable_regime_analysis:
            from backend.regime_analysis import RegimeDetector, RegimeTracker
            self.regime_tracker = RegimeTracker(RegimeDetector())

    def reset(self):
        """Reset backtest state (but preserve trained model stats)"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        # Note: consensus_stats is NOT reset - it should persist across periods

        # Reset regime tracker
        if self.enable_regime_analysis:
            from backend.regime_analysis import RegimeDetector, RegimeTracker
            self.regime_tracker = RegimeTracker(RegimeDetector())

    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        target_notional: float,
        price: float,
        adv: float,
        strategy_signal: Optional[str] = None
    ) -> Optional[BacktestTrade]:
        """
        Execute a trade with realistic costs.

        Returns BacktestTrade if successful, None if trade couldn't be executed
        """
        # Detect if this is crypto (supports fractional shares) or stock (whole shares only)
        is_crypto = '-' in symbol  # Crypto symbols are like BTC-USD, ETH-USD

        # Calculate shares from target notional
        if is_crypto:
            # Crypto supports fractional shares
            shares = target_notional / price
            if shares < 0.00001:  # Minimum fractional amount
                print(f"    ❌ Cannot execute {side}: amount too small ({shares:.8f})")
                return None
        else:
            # Stocks require whole shares
            shares = int(target_notional / price)
            if shares == 0:
                print(f"    ❌ Cannot execute {side}: target_notional=${target_notional:.2f}, price=${price:.2f}, would buy {target_notional/price:.4f} shares (need at least 1)")
                return None

        # Calculate transaction cost
        cost = self.cost_model.calculate_cost(
            price=price,
            shares=shares,
            order_type="market",
            adv=adv
        )

        # Check if we have enough cash (for buys)
        if side == 'buy':
            total_cost = price * shares + cost
            if total_cost > self.cash:
                # Reduce position size to fit cash
                if is_crypto:
                    shares = (self.cash - cost) / price
                    if shares < 0.00001:
                        return None  # Can't afford minimum amount
                else:
                    shares = int((self.cash - cost) / price)
                    if shares <= 0:
                        return None  # Can't afford even 1 share

        # Calculate actual notional (in case shares were adjusted)
        actual_notional = price * shares

        # Create trade record
        trade = BacktestTrade(
            backtest_run_id="",  # Will be set later
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            shares=shares,
            price=price,
            notional=actual_notional,
            transaction_cost=cost,
            strategy_signal=strategy_signal
        )

        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        trade_dict = trade.dict()
        self.positions[symbol].update(trade_dict)

        # Update cash
        if side == 'buy':
            self.cash -= (trade.notional + cost)
        else:
            self.cash += (trade.notional - cost)

        self.trades.append(trade)
        return trade

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos.shares * current_prices.get(symbol, 0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value

    def record_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Record equity curve point"""
        # Calculate position value (capital deployed)
        position_value = sum(
            pos.shares * current_prices.get(symbol, 0)
            for symbol, pos in self.positions.items()
        )

        equity = self.cash + position_value

        # Calculate drawdown
        if len(self.equity_curve) > 0:
            max_equity = max(point['equity'] for point in self.equity_curve)
            max_equity = max(max_equity, equity)
        else:
            max_equity = equity

        drawdown = (equity - max_equity) / max_equity if max_equity > 0 else 0

        self.equity_curve.append({
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'drawdown': drawdown,
            'position_value': position_value,
            'cash': self.cash
        })

    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics"""
        if len(self.equity_curve) < 2:
            return BacktestMetrics(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                max_drawdown=0, avg_drawdown=0, win_rate=0, profit_factor=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                avg_win=0, avg_loss=0, total_costs=0, costs_pct_of_capital=0
            )

        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['equity'].pct_change()

        # Total and annualized return
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1

        days = (pd.to_datetime(df['timestamp'].iloc[-1]) -
                pd.to_datetime(df['timestamp'].iloc[0])).days
        years = days / 252 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility and Sharpe
        daily_vol = df['returns'].std()
        annualized_vol = daily_vol * np.sqrt(252)
        risk_free_rate = 0.04
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

        # Drawdown metrics
        max_drawdown = df['drawdown'].min()
        avg_drawdown = df['drawdown'].mean()

        # Trade analysis
        winning_trades = []
        losing_trades = []

        # Group trades by round-trip
        for position in self.positions.values():
            if position.realized_pnl > 0:
                winning_trades.append(position.realized_pnl)
            elif position.realized_pnl < 0:
                losing_trades.append(abs(position.realized_pnl))

        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        gross_profit = sum(winning_trades)
        gross_loss = sum(losing_trades)
        # Cap profit factor at 999 to avoid JSON serialization issues with infinity
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0  # Cap instead of infinity
        else:
            profit_factor = 0

        # Transaction costs
        total_costs = sum(t.transaction_cost for t in self.trades)
        costs_pct = total_costs / self.initial_capital

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_costs=total_costs,
            costs_pct_of_capital=costs_pct
        )

    def run_simple_backtest(
        self,
        price_data: pd.DataFrame,
        run_id: str,
        full_price_history: pd.DataFrame = None
    ) -> BacktestRun:
        """
        Run a simple backtest without walk-forward validation.

        Args:
            price_data: DataFrame with columns [date, open, high, low, close, volume]
            run_id: Unique ID for this backtest run
            full_price_history: Optional full historical data for regime detection (includes data before price_data)

        Returns:
            BacktestRun with results
        """
        print(f"\n{'='*60}")
        print(f"STARTING SIMPLE BACKTEST FOR {self.config.symbol}")
        print(f"Date range: {price_data['date'].iloc[0]} to {price_data['date'].iloc[-1]}")
        print(f"Total bars: {len(price_data)}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")
        logger.info(f"Starting simple backtest for {self.config.symbol}")
        logger.info(f"Date range: {price_data['date'].iloc[0]} to {price_data['date'].iloc[-1]}")
        logger.info(f"Total bars: {len(price_data)}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Train consensus model if not already trained
        if self.consensus_stats is None:
            try:
                print(f"\n{'='*60}")
                print(f"TRAINING CONSENSUS MODEL FOR BACKTEST")
                print(f"{'='*60}\n")
                logger.info("Training consensus model for backtest")

                # Load ensemble module and train model
                ensemble = load_ensemble_module("examples/crypto_ensemble_forecast.py")
                configs = get_default_ensemble_configs(horizon=14)

                # Train ensemble model
                stats, _ = train_ensemble(
                    symbol=self.config.symbol,
                    forecast_horizon=14,
                    configs=configs,
                    name="Backtest-Consensus",
                    ensemble_module=ensemble
                )

                # Store trained model stats for use during testing
                self.consensus_stats = stats
                logger.info("Model training completed")
                print(f"✅ Model training completed\n")

                # Train multi-horizon models for multi-timeframe strategy
                print(f"{'='*60}")
                print(f"TRAINING MULTI-HORIZON MODELS FOR MULTI-TIMEFRAME STRATEGY")
                print(f"{'='*60}\n")

                multi_horizon_stats = {}
                horizons = [7, 14, 30]

                for horizon in horizons:
                    try:
                        print(f"Training {horizon}-day horizon model...")
                        configs_h = get_default_ensemble_configs(horizon=horizon)
                        stats_h, _ = train_ensemble(
                            symbol=self.config.symbol,
                            forecast_horizon=horizon,
                            configs=configs_h,
                            name=f"Backtest-Horizon-{horizon}d",
                            ensemble_module=ensemble
                        )
                        multi_horizon_stats[horizon] = (stats_h, None)  # (stats, df) - df not needed
                        print(f"✅ {horizon}-day model training completed\n")
                    except Exception as e:
                        logger.warning(f"Training {horizon}-day horizon failed: {e}")
                        print(f"⚠️  {horizon}-day model training failed: {e}\n")

                self.multi_horizon_stats = multi_horizon_stats if len(multi_horizon_stats) >= 2 else None
                if self.multi_horizon_stats:
                    print(f"✅ Multi-horizon training completed ({len(multi_horizon_stats)} horizons)\n")
                else:
                    print(f"⚠️  Multi-horizon training incomplete, skipping multi-timeframe strategy\n")

            except Exception as e:
                logger.error(f"Model training failed: {e}")
                logger.warning("Falling back to dummy signals")
                print(f"⚠️  Model training failed: {e}")
                print(f"⚠️  Falling back to dummy signals\n")
                self.consensus_stats = None
                self.multi_horizon_stats = None

        self.reset()
        current_position = None
        bar_counter = 0  # Use sequential counter for signal generation
        buy_signals = 0
        sell_signals = 0

        for idx, row in price_data.iterrows():
            current_price = row['close']
            current_date = pd.to_datetime(row['date'])
            adv = row.get('volume', 1_000_000)

            # Get historical data up to current bar for mean reversion strategy
            historical_df = price_data.loc[:idx].copy() if len(price_data.loc[:idx]) >= 20 else None

            # Get trading signal - use consensus if model stats available, otherwise dummy
            if self.consensus_stats is not None:
                signal = self._get_consensus_signal(
                    self.consensus_stats,
                    current_price,
                    historical_df=historical_df,
                    multi_horizon_stats=self.multi_horizon_stats,
                    params=self.config.optimizable
                )
            else:
                signal = self._get_dummy_signal(bar_counter, current_price)

            if signal['action'] == 'buy':
                buy_signals += 1
            elif signal['action'] == 'sell':
                sell_signals += 1

            bar_counter += 1

            # Trading logic based on consensus signal
            portfolio_value = self.get_portfolio_value({self.config.symbol: current_price})

            if signal['action'] == 'buy' and current_position != 'long':
                # Close short if exists, go long
                if current_position == 'short':
                    position = self.positions.get(self.config.symbol)
                    if position and position.shares > 0:
                        self.execute_trade(
                            timestamp=current_date,
                            symbol=self.config.symbol,
                            side='buy',
                            target_notional=position.shares * current_price,
                            price=current_price,
                            adv=adv,
                            strategy_signal=signal['strategy']
                        )

                # Open long position
                position_size = portfolio_value * (self.config.optimizable.position_size_pct / 100)
                trade = self.execute_trade(
                    timestamp=current_date,
                    symbol=self.config.symbol,
                    side='buy',
                    target_notional=position_size,
                    price=current_price,
                    adv=adv,
                    strategy_signal=signal['strategy']
                )
                if trade:
                    print(f"  ✅ BUY: {trade.shares} shares @ ${current_price:.2f} = ${trade.notional:.2f}")
                    logger.info(f"  BUY: {trade.shares} shares @ ${current_price:.2f} (${trade.notional:.2f})")
                    current_position = 'long'
                else:
                    print(f"  ⚠️ BUY signal but no trade executed (insufficient cash or other issue)")

            elif signal['action'] == 'sell' and current_position == 'long':
                # Close long position
                position = self.positions.get(self.config.symbol)
                if position and position.shares > 0:
                    trade = self.execute_trade(
                        timestamp=current_date,
                        symbol=self.config.symbol,
                        side='sell',
                        target_notional=position.shares * current_price,
                        price=current_price,
                        adv=adv,
                        strategy_signal=signal['strategy']
                    )
                    if trade:
                        print(f"  ✅ SELL: {trade.shares} shares @ ${current_price:.2f} = ${trade.notional:.2f}")
                        logger.info(f"  SELL: {trade.shares} shares @ ${current_price:.2f} (${trade.notional:.2f})")
                    current_position = None

            # Record equity
            self.record_equity(current_date, {self.config.symbol: current_price})

            # Update regime tracking
            if self.regime_tracker:
                # Calculate bar return if we have equity curve
                bar_return = None
                if len(self.equity_curve) >= 2:
                    prev_equity = self.equity_curve[-2]['equity']
                    curr_equity = self.equity_curve[-1]['equity']
                    bar_return = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0

                # Use full price history if provided (for walk-forward), otherwise use current historical data
                regime_price_data = None
                if full_price_history is not None:
                    # Include full history up to current bar for walk-forward periods
                    regime_price_data = full_price_history[full_price_history['date'] <= row['date']].copy()
                elif historical_df is not None:
                    # Use standard historical data for simple backtests
                    regime_price_data = historical_df

                # Track regime for this bar
                if regime_price_data is not None and len(regime_price_data) >= 20:
                    last_trade = self.trades[-1].dict() if self.trades else None
                    self.regime_tracker.update(
                        timestamp=current_date,
                        price_data=regime_price_data,
                        trade=last_trade,
                        bar_return=bar_return
                    )

        # Force-liquidate any open positions at the end of the period
        # This ensures each period ends flat (100% cash, no positions)
        final_price = price_data.iloc[-1]['close']
        final_date = pd.to_datetime(price_data.iloc[-1]['date'])
        final_adv = price_data.iloc[-1].get('volume', 1_000_000)

        open_positions = [(symbol, pos) for symbol, pos in self.positions.items() if pos.shares > 0]
        if open_positions:
            print(f"\n⚠️  Force-liquidating {len(open_positions)} open position(s) at period end")
            logger.info(f"Force-liquidating {len(open_positions)} open position(s) at period end")

            for symbol, position in open_positions:
                trade = self.execute_trade(
                    timestamp=final_date,
                    symbol=symbol,
                    side='sell',
                    target_notional=position.shares * final_price,
                    price=final_price,
                    adv=final_adv,
                    strategy_signal='force_liquidation'
                )
                if trade:
                    print(f"  ✅ FORCE SELL: {trade.shares} shares @ ${final_price:.2f} = ${trade.notional:.2f}")
                    logger.info(f"  FORCE SELL: {trade.shares} shares @ ${final_price:.2f} (${trade.notional:.2f})")

        # Calculate final metrics
        metrics = self.calculate_metrics()

        # Update trade run IDs
        for trade in self.trades:
            trade.backtest_run_id = run_id

        print(f"\n{'='*60}")
        print(f"BACKTEST COMPLETED FOR {self.config.symbol}")
        print(f"Signal stats: {buy_signals} buy signals, {sell_signals} sell signals")
        print(f"Trades executed: {len(self.trades)}")
        print(f"Total return: {metrics.total_return*100:.2f}%")
        print(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Max drawdown: {metrics.max_drawdown*100:.2f}%")
        print(f"{'='*60}\n")
        logger.info(f"Signal stats: {buy_signals} buy signals, {sell_signals} sell signals")
        logger.info(f"Backtest completed: {len(self.trades)} trades executed")
        logger.info(f"Total return: {metrics.total_return*100:.2f}%")
        logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {metrics.max_drawdown*100:.2f}%")

        # Log cache statistics
        cache_stats = get_strategy_cache().stats()
        logger.info(f"Strategy cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate_pct']}% hit rate)")

        # Get regime analysis if enabled
        regime_analysis = None
        if self.regime_tracker:
            regime_analysis = self.regime_tracker.get_regime_statistics()
            logger.info(f"Regime analysis: {len(regime_analysis.get('regime_statistics', {}))} regimes detected")

        return BacktestRun(
            run_id=run_id,
            name=f"Backtest {self.config.symbol}",
            config=self.config,
            status=BacktestStatus.COMPLETED,
            metrics=metrics,
            regime_analysis=regime_analysis,
            trades=self.trades,
            equity_curve=self.equity_curve,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

    def run_walkforward_backtest(
        self,
        price_data: pd.DataFrame,
        run_id: str
    ) -> BacktestRun:
        """
        Run walk-forward validation backtest.

        This is the proper way to test for overfitting.
        """
        logger.info(f"Starting walk-forward backtest for {self.config.symbol}")

        validator = WalkForwardValidator(self.config.walk_forward)
        splits = validator.generate_splits(price_data)

        logger.info(f"Generated {len(splits)} walk-forward periods")

        # If no splits generated, fall back to simple backtest
        if len(splits) == 0:
            logger.warning(f"Date range too short for walk-forward validation. Minimum required: {self.config.walk_forward.train_window_days + self.config.walk_forward.test_window_days} days")
            logger.warning("Falling back to simple backtest without walk-forward validation")
            return self.run_simple_backtest(price_data, run_id)

        all_period_metrics = []
        all_trades = []
        all_equity_points = []

        # Train model once at the beginning
        # Note: For MVP, we train once and reuse across all periods
        # TODO: In future, implement per-period training with data-aware training function
        try:
            print(f"\n{'='*60}")
            print(f"TRAINING CONSENSUS MODEL FOR BACKTEST")
            print(f"Training on full dataset to get model for consensus signals")
            print(f"{'='*60}\n")

            logger.info("Training consensus model for backtest")

            # Load ensemble module and train model
            ensemble = load_ensemble_module("examples/crypto_ensemble_forecast.py")
            configs = get_default_ensemble_configs(horizon=14)

            # Train ensemble model
            # Note: This will fetch recent data for training
            # In future versions, we should train per-period using only data available at that time
            stats, _ = train_ensemble(
                symbol=self.config.symbol,
                forecast_horizon=14,
                configs=configs,
                name="Backtest-Consensus",
                ensemble_module=ensemble
            )

            # Store trained model stats for use during testing
            self.consensus_stats = stats
            logger.info("Model training completed")

            # Train multi-horizon models for multi-timeframe strategy
            print(f"{'='*60}")
            print(f"TRAINING MULTI-HORIZON MODELS FOR MULTI-TIMEFRAME STRATEGY")
            print(f"{'='*60}\n")

            multi_horizon_stats = {}
            horizons = [7, 14, 30]

            for horizon in horizons:
                try:
                    print(f"Training {horizon}-day horizon model...")
                    configs_h = get_default_ensemble_configs(horizon=horizon)
                    stats_h, _ = train_ensemble(
                        symbol=self.config.symbol,
                        forecast_horizon=horizon,
                        configs=configs_h,
                        name=f"Backtest-Horizon-{horizon}d",
                        ensemble_module=ensemble
                    )
                    multi_horizon_stats[horizon] = (stats_h, None)  # (stats, df) - df not needed
                    print(f"✅ {horizon}-day model training completed\n")
                except Exception as e:
                    logger.warning(f"Training {horizon}-day horizon failed: {e}")
                    print(f"⚠️  {horizon}-day model training failed: {e}\n")

            self.multi_horizon_stats = multi_horizon_stats if len(multi_horizon_stats) >= 2 else None
            if self.multi_horizon_stats:
                print(f"✅ Multi-horizon training completed ({len(multi_horizon_stats)} horizons)\n")
            else:
                print(f"⚠️  Multi-horizon training incomplete, skipping multi-timeframe strategy\n")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.warning("Falling back to dummy signals for all periods")
            self.consensus_stats = None
            self.multi_horizon_stats = None

        for period_num, (train_df, test_df) in enumerate(splits, 1):
            logger.info(f"Period {period_num}/{len(splits)}")
            logger.info(f"  Train: {train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]}")
            logger.info(f"  Test: {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}")

            # Run backtest on test period using trained model
            # Save regime_tracker before reset to preserve accumulated data across periods
            saved_regime_tracker = self.regime_tracker
            self.reset()
            self.regime_tracker = saved_regime_tracker  # Restore to continue accumulating

            # Create full historical data for regime detection (from start of all data to end of test period)
            test_end_date = test_df['date'].iloc[-1]
            full_history_for_regime = price_data[price_data['date'] <= test_end_date].copy()

            period_result = self.run_simple_backtest(test_df, run_id, full_price_history=full_history_for_regime)

            # Calculate capital deployment metrics from equity curve
            position_values = [point.get('position_value', 0) for point in period_result.equity_curve]
            equity_values = [point['equity'] for point in period_result.equity_curve]

            avg_capital_deployed = float(np.mean(position_values)) if position_values else 0
            peak_capital_deployed = float(np.max(position_values)) if position_values else 0
            avg_equity = float(np.mean(equity_values)) if equity_values else self.initial_capital
            capital_utilization = (avg_capital_deployed / avg_equity) if avg_equity > 0 else 0

            # Record period metrics
            period_metrics = BacktestPeriodMetrics(
                backtest_run_id=run_id,
                period_number=period_num,
                train_start=pd.to_datetime(train_df['date'].iloc[0]),
                train_end=pd.to_datetime(train_df['date'].iloc[-1]),
                test_start=pd.to_datetime(test_df['date'].iloc[0]),
                test_end=pd.to_datetime(test_df['date'].iloc[-1]),
                total_return=period_result.metrics.total_return,
                annualized_return=period_result.metrics.annualized_return,
                sharpe_ratio=period_result.metrics.sharpe_ratio,
                max_drawdown=period_result.metrics.max_drawdown,
                win_rate=period_result.metrics.win_rate,
                total_trades=period_result.metrics.total_trades,
                total_costs=period_result.metrics.total_costs,
                final_equity=period_result.equity_curve[-1]['equity'] if period_result.equity_curve else self.initial_capital,
                avg_capital_deployed=avg_capital_deployed,
                peak_capital_deployed=peak_capital_deployed,
                capital_utilization=capital_utilization
            )

            all_period_metrics.append(period_metrics)
            all_trades.extend(period_result.trades)
            all_equity_points.extend(period_result.equity_curve)

        # Calculate aggregate metrics across all periods
        period_returns = [p.total_return for p in all_period_metrics]
        period_sharpes = [p.sharpe_ratio for p in all_period_metrics]

        # Calculate aggregate metrics manually from period results
        # (can't use self.calculate_metrics() because positions were reset)

        # Compound returns across periods
        cumulative_return = 1.0
        for p in all_period_metrics:
            cumulative_return *= (1 + p.total_return)
        total_return = cumulative_return - 1

        # Calculate time-weighted metrics
        total_days = (all_period_metrics[-1].test_end - all_period_metrics[0].test_start).days
        years = total_days / 252 if total_days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Aggregate Sharpe using period sharpes
        median_sharpe = float(np.median(period_sharpes))

        # Aggregate drawdown
        worst_drawdown = min(p.max_drawdown for p in all_period_metrics)
        avg_drawdown = np.mean([p.max_drawdown for p in all_period_metrics])

        # Trade statistics from all trades
        total_trades = len(all_trades)

        # Analyze P&L from trades by matching buy/sell pairs
        buy_trades = [t for t in all_trades if t.side == 'buy']
        sell_trades = [t for t in all_trades if t.side == 'sell']

        wins = []
        losses = []

        # Match each sell to its corresponding buy (FIFO within each period)
        for sell in sell_trades:
            # Find the most recent buy before this sell with matching shares
            matching_buy = None
            for buy in reversed(buy_trades):
                if abs(buy.shares - sell.shares) < 0.00001:  # Floating point tolerance
                    # Check if this buy hasn't been matched yet and is before the sell
                    if buy.timestamp < sell.timestamp:
                        matching_buy = buy
                        break

            if matching_buy:
                # Calculate P&L for this round trip
                entry_cost = matching_buy.notional + matching_buy.transaction_cost
                exit_proceeds = sell.notional - sell.transaction_cost
                pnl = exit_proceeds - entry_cost

                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(abs(pnl))

        # Calculate win/loss statistics
        winning_trades_count = len(wins)
        losing_trades_count = len(losses)
        total_round_trips = winning_trades_count + losing_trades_count

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0
        # Cap profit factor at 999 to avoid JSON serialization issues with infinity
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0  # Cap instead of infinity
        else:
            profit_factor = 0

        win_rate = winning_trades_count / total_round_trips if total_round_trips > 0 else 0

        # Use period-level statistics as backup
        total_period_trades = sum(p.total_trades for p in all_period_metrics)
        winning_periods = sum(1 for p in all_period_metrics if p.total_return > 0)
        period_win_rate = winning_periods / len(all_period_metrics) if all_period_metrics else 0

        # Transaction costs
        total_costs = sum(p.total_costs for p in all_period_metrics)
        costs_pct = total_costs / self.initial_capital

        # Calculate volatility from period returns
        period_returns_array = np.array(period_returns)
        period_volatility = np.std(period_returns_array) if len(period_returns_array) > 1 else 0
        # Annualize volatility (assuming ~6 periods per year given 63-day test windows)
        annualized_vol = period_volatility * np.sqrt(6) if period_volatility > 0 else 0

        aggregate_metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=median_sharpe,
            max_drawdown=worst_drawdown,
            avg_drawdown=avg_drawdown,
            win_rate=win_rate if total_round_trips > 0 else period_win_rate,  # Use actual trade win rate
            profit_factor=profit_factor,
            total_trades=total_period_trades,
            winning_trades=winning_trades_count if total_round_trips > 0 else winning_periods,
            losing_trades=losing_trades_count if total_round_trips > 0 else (len(all_period_metrics) - winning_periods),
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_costs=total_costs,
            costs_pct_of_capital=costs_pct,
            median_period_sharpe=median_sharpe,
            period_win_rate=period_win_rate,
            worst_period_drawdown=worst_drawdown
        )

        # Store trades and equity for reference
        self.trades = all_trades
        self.equity_curve = all_equity_points

        logger.info("Walk-forward backtest completed")
        logger.info(f"  Total return: {aggregate_metrics.total_return*100:.2f}%")
        logger.info(f"  Annualized return: {aggregate_metrics.annualized_return*100:.2f}%")
        logger.info(f"  Median period Sharpe: {aggregate_metrics.median_period_sharpe:.2f}")
        logger.info(f"  Win rate: {aggregate_metrics.win_rate*100:.1f}%")
        logger.info(f"  Profit factor: {aggregate_metrics.profit_factor:.2f}")
        logger.info(f"  Total trades: {aggregate_metrics.total_trades} ({winning_trades_count}W / {losing_trades_count}L)")
        logger.info(f"  Avg win: ${avg_win:.2f}, Avg loss: ${avg_loss:.2f}")
        logger.info(f"  Period win rate: {aggregate_metrics.period_win_rate*100:.0f}% ({winning_periods}/{len(all_period_metrics)} periods)")

        # Log cache statistics
        cache_stats = get_strategy_cache().stats()
        logger.info(f"  Strategy cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate_pct']}% hit rate)")

        # Get aggregated regime analysis across all walk-forward periods
        regime_analysis = None
        if self.regime_tracker:
            regime_analysis = self.regime_tracker.get_regime_statistics()
            logger.info(f"  Regime changes: {regime_analysis.get('total_regime_changes', 0)}")
            regime_stats = regime_analysis.get('regime_statistics', {})
            if regime_stats:
                logger.info(f"  Detected {len(regime_stats)} different market regimes")

        return BacktestRun(
            run_id=run_id,
            name=f"Walk-Forward Backtest {self.config.symbol}",
            config=self.config,
            status=BacktestStatus.COMPLETED,
            metrics=aggregate_metrics,
            period_metrics=all_period_metrics,
            regime_analysis=regime_analysis,
            trades=all_trades,
            equity_curve=all_equity_points,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

    def _get_consensus_signal(self, stats: Dict, current_price: float, historical_df: pd.DataFrame = None,
                            multi_horizon_stats: Dict = None, params: 'OptimizableParams' = None) -> Dict[str, Any]:
        """
        Generate consensus trading signal by running all strategies and voting.

        Args:
            stats: Trained model forecast statistics (14-day horizon)
            current_price: Current asset price
            historical_df: Historical price DataFrame for mean reversion (optional)
            multi_horizon_stats: Dictionary of stats for multiple horizons (optional)
            params: Optimizable parameters for configurable thresholds (optional)

        Returns:
            Dictionary with action ('buy', 'sell', 'hold'), strategy name, and consensus info

        Note: Using 6-8 strategies for backtest consensus depending on available data:
        - Core 6: Forecast Gradient, Confidence-Weighted, Volatility Sizing,
                 Acceleration, Swing Trading, Risk-Adjusted
        - Optional: Mean Reversion (if historical_df provided)
        - Optional: Multi-Timeframe (if multi_horizon_stats provided)
        """
        # Use provided params or default values
        if params is None:
            from backend.models import OptimizableParams
            params = OptimizableParams()
        try:
            # Run core 6 strategies that work with single-horizon stats
            results = {}

            try:
                results['gradient'] = analyze_gradient_strategy(stats, current_price)
            except Exception as e:
                logger.warning(f"Gradient strategy failed: {e}")
                results['gradient'] = {'signal': 'ERROR', 'position_size_pct': 0}

            try:
                results['confidence'] = analyze_confidence_weighted_strategy(stats, current_price)
            except Exception as e:
                logger.warning(f"Confidence strategy failed: {e}")
                results['confidence'] = {'signal': 'ERROR', 'position_size_pct': 0}

            try:
                results['volatility'] = analyze_volatility_position_sizing(stats, current_price)
            except Exception as e:
                logger.warning(f"Volatility strategy failed: {e}")
                results['volatility'] = {'signal': 'ERROR', 'position_size_pct': 0}

            try:
                results['acceleration'] = analyze_acceleration_strategy(stats, current_price)
            except Exception as e:
                logger.warning(f"Acceleration strategy failed: {e}")
                results['acceleration'] = {'signal': 'ERROR', 'position_size_pct': 0}

            try:
                results['swing'] = analyze_swing_trading_strategy(stats, current_price)
            except Exception as e:
                logger.warning(f"Swing strategy failed: {e}")
                results['swing'] = {'signal': 'ERROR', 'position_size_pct': 0}

            try:
                results['risk_adjusted'] = analyze_risk_adjusted_strategy(stats, current_price)
            except Exception as e:
                logger.warning(f"Risk-adjusted strategy failed: {e}")
                results['risk_adjusted'] = {'signal': 'ERROR', 'position_size_pct': 0}

            # Add Mean Reversion if historical data provided
            if historical_df is not None and len(historical_df) > 0:
                try:
                    results['mean_reversion'] = analyze_mean_reversion_strategy(stats, historical_df, current_price)
                except Exception as e:
                    logger.warning(f"Mean Reversion strategy failed: {e}")
                    results['mean_reversion'] = {'signal': 'ERROR', 'position_size_pct': 0}

            # Add Multi-Timeframe if multi-horizon stats provided
            if multi_horizon_stats is not None:
                try:
                    results['multi_timeframe'] = analyze_multi_timeframe_strategy(multi_horizon_stats, current_price)
                except Exception as e:
                    logger.warning(f"Multi-Timeframe strategy failed: {e}")
                    results['multi_timeframe'] = {'signal': 'ERROR', 'position_size_pct': 0}

            # Apply consensus voting logic (adapted from compare_all_8_strategies.py)
            # Using 6-8 strategies depending on available data
            strategies = {
                'Forecast Gradient': results['gradient'],
                'Confidence-Weighted': results['confidence'],
                'Volatility Sizing': results['volatility'],
                'Acceleration': results['acceleration'],
                'Swing Trading': results['swing'],
                'Risk-Adjusted': results['risk_adjusted'],
            }

            # Add optional strategies if they ran
            if 'mean_reversion' in results:
                strategies['Mean Reversion'] = results['mean_reversion']
            if 'multi_timeframe' in results:
                strategies['Multi-Timeframe'] = results['multi_timeframe']

            # Categorize signals
            bullish_keywords = ['BUY', 'BULLISH', 'MOMENTUM', 'REVERT', 'REVERSAL', 'EXCELLENT', 'GOOD']
            bearish_keywords = ['SELL', 'BEARISH', 'OUT', 'STAY', 'EXIT', 'POOR', 'FALSE']

            bullish_strategies = []
            bearish_strategies = []
            neutral_strategies = []
            error_strategies = []

            for name, data in strategies.items():
                signal = data.get('signal', 'ERROR')

                # Skip ERROR signals
                if signal == 'ERROR':
                    error_strategies.append(name)
                    continue

                if any(keyword in signal for keyword in bullish_keywords) and 'POOR' not in signal and 'FALSE' not in signal:
                    bullish_strategies.append(name)
                elif any(keyword in signal for keyword in bearish_keywords) or 'NO' in signal:
                    bearish_strategies.append(name)
                else:
                    neutral_strategies.append(name)

            # Count votes (excluding ERROR strategies)
            total = len(strategies) - len(error_strategies)
            bullish_count = len(bullish_strategies)
            bearish_count = len(bearish_strategies)

            # Determine consensus action using configurable thresholds
            # Thresholds are percentage-based and work for any number of strategies (6-8)
            bullish_pct = bullish_count / total if total > 0 else 0
            bearish_pct = bearish_count / total if total > 0 else 0

            # Use configurable thresholds from params
            if bullish_pct >= params.strong_buy_threshold:
                action = 'buy'
                consensus = 'STRONG_BUY'
            elif bullish_pct >= params.buy_threshold:
                action = 'buy'
                consensus = 'BUY'
            elif bullish_pct >= params.moderate_buy_threshold:
                action = 'buy'
                consensus = 'MODERATE_BUY'
            elif bearish_pct >= params.sell_threshold:
                action = 'sell'
                consensus = 'SELL_AVOID'
            elif bearish_pct >= params.moderate_sell_threshold:
                action = 'sell'
                consensus = 'MODERATE_SELL'
            else:
                action = 'hold'
                consensus = 'MIXED'

            return {
                'action': action,
                'strategy': f'consensus_{consensus}',
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': len(neutral_strategies),
                'error_count': len(error_strategies),
                'total_count': total,
                'consensus': consensus
            }

        except Exception as e:
            logger.error(f"Consensus signal generation failed: {e}")
            # Fallback to hold on error
            return {
                'action': 'hold',
                'strategy': 'consensus_error',
                'error': str(e)
            }

    def _get_dummy_signal(self, idx: int, price: float) -> Dict[str, str]:
        """
        Dummy signal generator for testing.
        In production, this will call the actual consensus analyzer.

        TODO: Replace with actual consensus analysis integration
        """
        # More active trading for testing - buy every 5 bars, sell after 3 bars
        cycle_position = idx % 10

        if cycle_position == 0:
            return {'action': 'buy', 'strategy': 'dummy_momentum'}
        elif cycle_position == 4:
            return {'action': 'sell', 'strategy': 'dummy_momentum'}
        elif cycle_position == 8:
            return {'action': 'buy', 'strategy': 'dummy_momentum'}
        else:
            return {'action': 'hold', 'strategy': 'dummy_momentum'}
