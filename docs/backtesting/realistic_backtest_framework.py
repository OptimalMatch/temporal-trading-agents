"""
Realistic Backtesting Framework for Transformer Trading Models

This framework explicitly models all the ways retail/small funds get destroyed:
- Transaction costs (fees + spread + slippage)
- Adverse selection (you're slow, market makers pick you off)
- Regime changes (your model trained on 2020 fails in 2022)
- Overfitting detection (walk-forward validation)
- Capacity constraints (your edge disappears as you scale)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class TransactionCostModel:
    """
    Models the TRUE costs of trading, not the fantasy version.
    """
    # Exchange fees (per share or % of notional)
    taker_fee_bps: float = 5.0  # 5 bps for taking liquidity
    maker_rebate_bps: float = 0.0  # You don't get rebates (not a market maker)
    
    # Bid-ask spread cost
    half_spread_bps: float = 2.0  # Cross half-spread on market orders
    
    # Slippage model: worse as order size increases
    slippage_coefficient: float = 0.1  # bps per $100k notional
    
    # Adverse selection: you're slow, so filled orders are biased against you
    adverse_selection_bps: float = 2.0
    
    # SEC fees, other regulatory
    sec_fee_bps: float = 0.23  # SEC Section 31 fee
    
    def calculate_cost(self, 
                       price: float, 
                       shares: int, 
                       order_type: OrderType,
                       adv: float) -> float:
        """
        Calculate total transaction cost in dollars.
        
        Args:
            price: Execution price
            shares: Number of shares
            order_type: Market or limit order
            adv: Average daily volume (to estimate market impact)
        
        Returns:
            Total cost in dollars (always positive)
        """
        notional = price * shares
        
        # Base exchange fees
        if order_type == OrderType.MARKET:
            exchange_cost = notional * (self.taker_fee_bps / 10000)
        else:
            # Limit orders: might get rebate if you're a market maker (you're not)
            exchange_cost = notional * (self.taker_fee_bps / 10000)
        
        # Spread cost (market orders cross the spread)
        spread_cost = notional * (self.half_spread_bps / 10000) if order_type == OrderType.MARKET else 0
        
        # Slippage: increases with order size relative to daily volume
        pct_of_adv = (shares / adv) * 100  # % of daily volume
        slippage_bps = self.slippage_coefficient * pct_of_adv
        slippage_cost = notional * (slippage_bps / 10000)
        
        # Adverse selection: you get picked off by faster traders
        adverse_cost = notional * (self.adverse_selection_bps / 10000)
        
        # Regulatory fees
        sec_cost = notional * (self.sec_fee_bps / 10000)
        
        total_cost = exchange_cost + spread_cost + slippage_cost + adverse_cost + sec_cost
        
        return total_cost
    
    def round_trip_cost_bps(self, shares: int, price: float, adv: float) -> float:
        """
        Calculate round-trip cost in basis points (buy + sell).
        """
        notional = price * shares
        buy_cost = self.calculate_cost(price, shares, OrderType.MARKET, adv)
        sell_cost = self.calculate_cost(price, shares, OrderType.MARKET, adv)
        
        total_cost_bps = ((buy_cost + sell_cost) / notional) * 10000
        return total_cost_bps


@dataclass
class Trade:
    """Record of a single trade"""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    shares: int
    price: float
    transaction_cost: float
    
    @property
    def notional(self) -> float:
        return self.price * self.shares


class Position:
    """Track position state and P&L"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.shares = 0
        self.avg_cost_basis = 0.0
        self.realized_pnl = 0.0
        self.total_transaction_costs = 0.0
        self.trades: List[Trade] = []
    
    def update(self, trade: Trade):
        """Update position with new trade"""
        self.trades.append(trade)
        self.total_transaction_costs += trade.transaction_cost
        
        if trade.side == 'buy':
            # Update cost basis
            total_cost = self.shares * self.avg_cost_basis + trade.notional
            self.shares += trade.shares
            self.avg_cost_basis = total_cost / self.shares if self.shares > 0 else 0
        
        elif trade.side == 'sell':
            if self.shares >= trade.shares:
                # Realize P&L
                pnl = (trade.price - self.avg_cost_basis) * trade.shares
                self.realized_pnl += pnl
                self.shares -= trade.shares
            else:
                raise ValueError(f"Cannot sell {trade.shares} shares, only have {self.shares}")
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price"""
        if self.shares == 0:
            return 0.0
        return (current_price - self.avg_cost_basis) * self.shares
    
    def total_pnl(self, current_price: float) -> float:
        """Total P&L including costs"""
        return self.realized_pnl + self.unrealized_pnl(current_price) - self.total_transaction_costs


class WalkForwardValidator:
    """
    Implements walk-forward validation to detect overfitting.
    
    The ONLY way to know if your model has real edge vs curve-fitting.
    """
    def __init__(self,
                 train_window_days: int = 252,  # 1 year training
                 test_window_days: int = 63,     # 1 quarter testing
                 retrain_frequency_days: int = 21):  # Monthly retrain
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.retrain_frequency = retrain_frequency_days
    
    def generate_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits for walk-forward validation.
        
        Returns list of (train_df, test_df) tuples.
        """
        splits = []
        total_days = len(data)
        
        current_day = self.train_window
        
        while current_day + self.test_window <= total_days:
            train_start = current_day - self.train_window
            train_end = current_day
            test_end = current_day + self.test_window
            
            train_df = data.iloc[train_start:train_end]
            test_df = data.iloc[train_end:test_end]
            
            splits.append((train_df, test_df))
            
            current_day += self.retrain_frequency
        
        return splits


class RealisticBacktest:
    """
    Backtesting engine that doesn't lie to you.
    """
    def __init__(self,
                 initial_capital: float = 100000,
                 cost_model: Optional[TransactionCostModel] = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.positions: dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        
    def execute_trade(self,
                     timestamp: pd.Timestamp,
                     symbol: str,
                     side: str,
                     target_notional: float,
                     price: float,
                     adv: float):
        """
        Execute a trade with realistic costs.
        
        Args:
            timestamp: When the trade occurs
            symbol: Ticker symbol
            side: 'buy' or 'sell'
            target_notional: Dollar amount to trade
            price: Execution price
            adv: Average daily volume
        """
        # Calculate shares from target notional
        shares = int(target_notional / price)
        if shares == 0:
            return
        
        # Calculate transaction cost
        cost = self.cost_model.calculate_cost(
            price=price,
            shares=shares,
            order_type=OrderType.MARKET,
            adv=adv
        )
        
        # Check if we have enough cash (for buys)
        if side == 'buy':
            total_cost = price * shares + cost
            if total_cost > self.cash:
                # Reduce position size to fit cash
                shares = int((self.cash - cost) / price)
                if shares <= 0:
                    return  # Can't afford even 1 share
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            shares=shares,
            price=price,
            transaction_cost=cost
        )
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        self.positions[symbol].update(trade)
        
        # Update cash
        if side == 'buy':
            self.cash -= (trade.notional + cost)
        else:
            self.cash += (trade.notional - cost)
        
        self.trades.append(trade)
    
    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos.shares * current_prices.get(symbol, 0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def record_equity(self, timestamp: pd.Timestamp, current_prices: dict[str, float]):
        """Record equity curve point"""
        equity = self.get_portfolio_value(current_prices)
        self.equity_curve.append((timestamp, equity))
    
    def get_metrics(self) -> dict:
        """
        Calculate performance metrics that matter.
        """
        if len(self.equity_curve) < 2:
            return {}
        
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df['returns'] = df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        years = days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        daily_vol = df['returns'].std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assume 4% risk-free rate)
        risk_free_rate = 0.04
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for t in self.trades if self._is_winning_trade(t))
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Total transaction costs
        total_costs = sum(t.transaction_cost for t in self.trades)
        costs_pct_of_capital = total_costs / self.initial_capital
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_transaction_costs': total_costs,
            'costs_pct_of_capital': costs_pct_of_capital,
            'final_equity': df['equity'].iloc[-1]
        }
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was profitable (simplified)"""
        # This is a simplified check - in reality you'd track full round-trips
        position = self.positions.get(trade.symbol)
        if position:
            return position.total_pnl(trade.price) > 0
        return False


# Example usage and testing framework
if __name__ == "__main__":
    print("=" * 80)
    print("REALISTIC BACKTESTING FRAMEWORK")
    print("=" * 80)
    
    # Example: Calculate transaction costs
    cost_model = TransactionCostModel()
    
    print("\n1. TRANSACTION COST EXAMPLES")
    print("-" * 80)
    
    # Small trade
    small_trade_cost = cost_model.round_trip_cost_bps(
        shares=100,
        price=100,
        adv=1_000_000  # 1M shares average daily volume
    )
    print(f"Small trade (100 shares @ $100, 0.01% of ADV):")
    print(f"  Round-trip cost: {small_trade_cost:.2f} bps")
    print(f"  On $10k notional: ${(small_trade_cost/10000)*10000:.2f}")
    
    # Large trade (market impact matters)
    large_trade_cost = cost_model.round_trip_cost_bps(
        shares=10_000,
        price=100,
        adv=1_000_000  # 1% of daily volume - significant impact
    )
    print(f"\nLarge trade (10,000 shares @ $100, 1% of ADV):")
    print(f"  Round-trip cost: {large_trade_cost:.2f} bps")
    print(f"  On $1M notional: ${(large_trade_cost/10000)*1_000_000:.2f}")
    
    print("\n2. MINIMUM EDGE REQUIREMENTS")
    print("-" * 80)
    print(f"To be profitable, your model needs to predict moves > {small_trade_cost:.1f} bps")
    print(f"For 2:1 edge-to-cost ratio: Need >{small_trade_cost*2:.1f} bps predicted edge")
    print(f"For 3:1 ratio (recommended): Need >{small_trade_cost*3:.1f} bps predicted edge")
    
    print("\n3. WALK-FORWARD VALIDATION SETUP")
    print("-" * 80)
    validator = WalkForwardValidator(
        train_window_days=252,
        test_window_days=63,
        retrain_frequency_days=21
    )
    print(f"Training window: {validator.train_window} days (1 year)")
    print(f"Testing window: {validator.test_window} days (1 quarter)")
    print(f"Retrain every: {validator.retrain_frequency} days (monthly)")
    print("\nThis ensures your model:")
    print("  - Never sees future data")
    print("  - Tests on multiple market regimes")
    print("  - Retrains regularly (markets evolve)")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Integrate this with your Temporal model")
    print("2. Run walk-forward validation on historical data")
    print("3. Calculate metrics with REALISTIC transaction costs")
    print("4. Only deploy if Sharpe > 1.0 after costs in walk-forward test")
    print("=" * 80)

