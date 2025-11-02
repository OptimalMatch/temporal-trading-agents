"""
Integration between Temporal transformer model and realistic backtesting.

This shows how to properly test if your model has real edge.
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from realistic_backtest_framework import (
    RealisticBacktest, 
    TransactionCostModel,
    WalkForwardValidator
)

class TemporalTradingStrategy:
    """
    Wraps your Temporal model with realistic trading logic.
    """
    def __init__(self,
                 model,  # Your trained Temporal model
                 lookback_periods: int = 96,
                 forecast_horizon: int = 24,
                 min_edge_bps: float = 55,  # 3x transaction costs
                 position_size_pct: float = 0.1):  # 10% of portfolio per position
        
        self.model = model
        self.lookback = lookback_periods
        self.horizon = forecast_horizon
        self.min_edge_bps = min_edge_bps
        self.position_size_pct = position_size_pct
        
        self.model.eval()
    
    def generate_signal(self, 
                        historical_prices: np.ndarray,
                        current_price: float) -> Tuple[str, float]:
        """
        Generate trading signal from model prediction.
        
        Args:
            historical_prices: Array of shape (lookback, features)
            current_price: Current price
        
        Returns:
            (signal, confidence) where signal in ['buy', 'sell', 'hold']
        """
        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(historical_prices).unsqueeze(0)  # Add batch dim
            
            # Generate forecast
            forecast = self.model.forecast(x)  # (1, horizon, features)
            
            # Extract predicted prices
            predicted_prices = forecast[0, :, 0].numpy()  # Assuming first feature is price
            
            # Calculate expected return over forecast horizon
            predicted_final_price = predicted_prices[-1]
            expected_return_bps = ((predicted_final_price / current_price) - 1) * 10000
            
            # Decision logic: only trade if edge > minimum threshold
            if expected_return_bps > self.min_edge_bps:
                return 'buy', abs(expected_return_bps)
            elif expected_return_bps < -self.min_edge_bps:
                return 'sell', abs(expected_return_bps)
            else:
                return 'hold', 0.0
    
    def calculate_position_size(self, 
                               portfolio_value: float,
                               confidence: float) -> float:
        """
        Calculate position size based on confidence and portfolio value.
        
        Uses a simple % of portfolio approach. In production, you'd want
        Kelly criterion or more sophisticated risk management.
        """
        base_size = portfolio_value * self.position_size_pct
        
        # Scale by confidence (normalized to 0-1 range)
        # Higher confidence = larger position (up to 2x base)
        confidence_multiplier = min(1 + (confidence / 100), 2.0)
        
        return base_size * confidence_multiplier


def run_walkforward_backtest(
    data: pd.DataFrame,
    model_factory,  # Function that creates and trains a model
    lookback: int = 96,
    horizon: int = 24,
    initial_capital: float = 100000
) -> dict:
    """
    Run walk-forward validation with your Temporal model.
    
    Args:
        data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        model_factory: Function(train_data) -> trained_model
        lookback: Historical window for model
        horizon: Forecast horizon
        initial_capital: Starting capital
    
    Returns:
        Dictionary with performance metrics across all test periods
    """
    
    validator = WalkForwardValidator(
        train_window_days=252,
        test_window_days=63,
        retrain_frequency_days=21
    )
    
    cost_model = TransactionCostModel()
    
    # Store results from each walk-forward period
    all_period_results = []
    
    splits = validator.generate_splits(data)
    
    print(f"Running {len(splits)} walk-forward periods...")
    print("=" * 80)
    
    for i, (train_df, test_df) in enumerate(splits):
        print(f"\nPeriod {i+1}/{len(splits)}")
        print(f"Train: {train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]}")
        print(f"Test:  {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")
        
        # Train model on training data
        print("  Training model...")
        model = model_factory(train_df)
        
        # Create strategy
        strategy = TemporalTradingStrategy(
            model=model,
            lookback_periods=lookback,
            forecast_horizon=horizon
        )
        
        # Initialize backtest for this period
        backtest = RealisticBacktest(
            initial_capital=initial_capital,
            cost_model=cost_model
        )
        
        # Run backtest on test period
        print("  Running backtest...")
        current_position = None
        
        for idx in range(lookback, len(test_df)):
            # Get historical data
            hist_data = test_df.iloc[idx-lookback:idx]['close'].values.reshape(-1, 1)
            current_price = test_df.iloc[idx]['close']
            current_time = test_df.iloc[idx]['timestamp']
            adv = test_df.iloc[idx].get('volume', 1_000_000)  # Average daily volume
            
            # Generate signal
            signal, confidence = strategy.generate_signal(hist_data, current_price)
            
            # Trading logic
            portfolio_value = backtest.get_portfolio_value({'STOCK': current_price})
            
            if signal == 'buy' and current_position != 'long':
                # Close short if exists, go long
                if current_position == 'short':
                    position_size = strategy.calculate_position_size(portfolio_value, confidence)
                    backtest.execute_trade(
                        timestamp=current_time,
                        symbol='STOCK',
                        side='buy',
                        target_notional=position_size,
                        price=current_price,
                        adv=adv
                    )
                
                # Open long
                position_size = strategy.calculate_position_size(portfolio_value, confidence)
                backtest.execute_trade(
                    timestamp=current_time,
                    symbol='STOCK',
                    side='buy',
                    target_notional=position_size,
                    price=current_price,
                    adv=adv
                )
                current_position = 'long'
            
            elif signal == 'sell' and current_position != 'short':
                # Close long if exists
                if current_position == 'long':
                    position = backtest.positions.get('STOCK')
                    if position and position.shares > 0:
                        backtest.execute_trade(
                            timestamp=current_time,
                            symbol='STOCK',
                            side='sell',
                            target_notional=position.shares * current_price,
                            price=current_price,
                            adv=adv
                        )
                current_position = None
            
            # Record equity
            backtest.record_equity(current_time, {'STOCK': current_price})
        
        # Get metrics for this period
        metrics = backtest.get_metrics()
        metrics['period'] = i + 1
        metrics['train_start'] = train_df['timestamp'].iloc[0]
        metrics['test_start'] = test_df['timestamp'].iloc[0]
        metrics['test_end'] = test_df['timestamp'].iloc[-1]
        
        all_period_results.append(metrics)
        
        print(f"  Results:")
        print(f"    Return: {metrics['total_return']*100:.2f}%")
        print(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"    Max DD: {metrics['max_drawdown']*100:.2f}%")
        print(f"    Trades: {metrics['total_trades']}")
        print(f"    Costs:  ${metrics['total_transaction_costs']:.2f}")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS ALL PERIODS")
    print("=" * 80)
    
    df_results = pd.DataFrame(all_period_results)
    
    aggregate_stats = {
        'num_periods': len(all_period_results),
        'avg_return': df_results['total_return'].mean(),
        'median_return': df_results['total_return'].median(),
        'avg_sharpe': df_results['sharpe_ratio'].mean(),
        'median_sharpe': df_results['sharpe_ratio'].median(),
        'avg_max_drawdown': df_results['max_drawdown'].mean(),
        'worst_drawdown': df_results['max_drawdown'].min(),
        'win_rate_periods': (df_results['total_return'] > 0).sum() / len(df_results),
        'total_trades': df_results['total_trades'].sum(),
        'total_costs': df_results['total_transaction_costs'].sum(),
        'periods': all_period_results
    }
    
    print(f"\nAverage Return: {aggregate_stats['avg_return']*100:.2f}%")
    print(f"Median Return:  {aggregate_stats['median_return']*100:.2f}%")
    print(f"Average Sharpe: {aggregate_stats['avg_sharpe']:.2f}")
    print(f"Median Sharpe:  {aggregate_stats['median_sharpe']:.2f}")
    print(f"Worst Drawdown: {aggregate_stats['worst_drawdown']*100:.2f}%")
    print(f"Period Win Rate: {aggregate_stats['win_rate_periods']*100:.0f}%")
    print(f"Total Trades: {aggregate_stats['total_trades']}")
    print(f"Total Costs:  ${aggregate_stats['total_costs']:.2f}")
    
    # The verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if aggregate_stats['median_sharpe'] > 1.0:
        print("✓ PASS: Model shows consistent edge after transaction costs")
        print("  → Consider paper trading for 6 months before live deployment")
    elif aggregate_stats['median_sharpe'] > 0.5:
        print("⚠ MARGINAL: Model has weak edge, risky for live trading")
        print("  → Need more work on feature engineering or regime detection")
    else:
        print("✗ FAIL: Model has no edge after transaction costs")
        print("  → This is curve-fitting. Do not trade with real money.")
    
    if aggregate_stats['win_rate_periods'] < 0.5:
        print("\n⚠ WARNING: Model loses money in >50% of test periods")
        print("  → Likely overfitting to training data or specific regime")
    
    return aggregate_stats


# Example: How to use this with your Temporal model
def example_model_factory(train_data: pd.DataFrame):
    """
    Example factory function that creates and trains your Temporal model.
    Replace this with your actual training code.
    """
    # Import your Temporal model
    # from temporal import Temporal
    # from temporal.trainer import TemporalTrainer, TimeSeriesDataset
    
    # This is pseudocode - adapt to your actual implementation
    """
    model = Temporal(
        input_dim=1,
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        d_ff=1024,
        forecast_horizon=24
    )
    
    # Create dataset from train_data
    dataset = TimeSeriesDataset(
        train_data['close'].values,
        lookback=96,
        forecast_horizon=24
    )
    
    # Train model
    trainer = TemporalTrainer(model, ...)
    trainer.fit(dataset, num_epochs=50)
    
    return model
    """
    
    # For now, return a dummy model that just predicts current price
    class DummyModel:
        def eval(self): pass
        def forecast(self, x):
            # Just return input as forecast (no edge)
            return x[:, -24:, :]  # Last 24 periods
    
    return DummyModel()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TEMPORAL MODEL INTEGRATION EXAMPLE")
    print("=" * 80)
    
    print("\nTo run full walk-forward backtest:")
    print("-" * 80)
    print("""
# Load your historical data
data = pd.read_csv('historical_prices.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Define your model factory
def train_temporal_model(train_df):
    # Your training code here
    model = Temporal(...)
    # ... train model on train_df ...
    return model

# Run walk-forward backtest
results = run_walkforward_backtest(
    data=data,
    model_factory=train_temporal_model,
    lookback=96,
    horizon=24,
    initial_capital=100000
)

# Analyze results
if results['median_sharpe'] > 1.0:
    print("Model has edge - proceed to paper trading")
else:
    print("Back to the drawing board")
    """)
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
1. TRANSACTION COSTS MATTER
   - Your model must predict moves >55 bps to be profitable
   - Small edges (10-20 bps) get destroyed by fees
   
2. WALK-FORWARD VALIDATION IS MANDATORY
   - In-sample metrics (training MSE) are meaningless
   - Only out-of-sample performance matters
   - Model must work across multiple regimes
   
3. REGIME CHANGES WILL KILL YOU
   - A model trained in 2019 fails in 2020 (COVID)
   - A model trained in 2020 fails in 2022 (rate hikes)
   - Must retrain frequently and detect regime shifts
   
4. REALISTIC POSITION SIZING
   - Can't use 100% of capital (need cash buffer)
   - Large positions have worse slippage
   - Must account for inability to exit instantly
   
5. THE BAR FOR "GOOD ENOUGH"
   - Sharpe > 1.0 after costs = might be real
   - Sharpe < 0.5 after costs = curve-fitting
   - Win rate on periods > 60% = consistency
   - Max drawdown < 20% = survivable
    """)

