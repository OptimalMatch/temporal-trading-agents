"""
Market Regime Analysis Module

Detects market regimes based on volatility and trend characteristics,
and tracks strategy performance across different market conditions.
"""
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime types based on volatility and trend"""
    LOW_VOL_UPTREND = "low_vol_uptrend"
    LOW_VOL_DOWNTREND = "low_vol_downtrend"
    LOW_VOL_SIDEWAYS = "low_vol_sideways"
    HIGH_VOL_UPTREND = "high_vol_uptrend"
    HIGH_VOL_DOWNTREND = "high_vol_downtrend"
    HIGH_VOL_SIDEWAYS = "high_vol_sideways"
    MED_VOL_UPTREND = "med_vol_uptrend"
    MED_VOL_DOWNTREND = "med_vol_downtrend"
    MED_VOL_SIDEWAYS = "med_vol_sideways"
    UNKNOWN = "unknown"


class RegimeDetector:
    """Detects market regimes based on technical indicators"""

    def __init__(
        self,
        low_vol_threshold: float = 0.15,
        high_vol_threshold: float = 0.30,
        trend_sma_short: int = 50,
        trend_sma_long: int = 200,
        trend_threshold: float = 0.02
    ):
        """
        Initialize regime detector.

        Args:
            low_vol_threshold: Annualized volatility threshold for low vol (default: 15%)
            high_vol_threshold: Annualized volatility threshold for high vol (default: 30%)
            trend_sma_short: Short-term SMA period for trend detection (default: 50)
            trend_sma_long: Long-term SMA period for trend detection (default: 200)
            trend_threshold: Threshold for sideways market detection (default: 2%)
        """
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.trend_sma_short = trend_sma_short
        self.trend_sma_long = trend_sma_long
        self.trend_threshold = trend_threshold

    def detect_regime(
        self,
        price_data: pd.DataFrame,
        lookback_window: int = 20
    ) -> MarketRegime:
        """
        Detect current market regime based on recent price action.

        Args:
            price_data: DataFrame with price data (must have 'close' column and datetime index)
            lookback_window: Number of periods for volatility calculation (default: 20)

        Returns:
            Current market regime
        """
        if len(price_data) < max(self.trend_sma_long, lookback_window):
            return MarketRegime.UNKNOWN

        try:
            # Calculate volatility (annualized)
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(lookback_window).std().iloc[-1] * np.sqrt(252)

            # Classify volatility
            if volatility < self.low_vol_threshold:
                vol_regime = "low"
            elif volatility > self.high_vol_threshold:
                vol_regime = "high"
            else:
                vol_regime = "med"

            # Calculate trend indicators
            sma_short = price_data['close'].rolling(self.trend_sma_short).mean().iloc[-1]
            sma_long = price_data['close'].rolling(self.trend_sma_long).mean().iloc[-1]
            current_price = price_data['close'].iloc[-1]

            # Determine trend direction
            if current_price > sma_short and sma_short > sma_long:
                # Strong uptrend
                trend_regime = "uptrend"
            elif current_price < sma_short and sma_short < sma_long:
                # Strong downtrend
                trend_regime = "downtrend"
            else:
                # Check if sideways (price oscillating around SMAs)
                price_vs_short = abs(current_price - sma_short) / sma_short
                if price_vs_short < self.trend_threshold:
                    trend_regime = "sideways"
                else:
                    # Weak trend or transitioning
                    if current_price > sma_short:
                        trend_regime = "uptrend"
                    else:
                        trend_regime = "downtrend"

            # Combine volatility and trend to determine regime
            regime_map = {
                ("low", "uptrend"): MarketRegime.LOW_VOL_UPTREND,
                ("low", "downtrend"): MarketRegime.LOW_VOL_DOWNTREND,
                ("low", "sideways"): MarketRegime.LOW_VOL_SIDEWAYS,
                ("med", "uptrend"): MarketRegime.MED_VOL_UPTREND,
                ("med", "downtrend"): MarketRegime.MED_VOL_DOWNTREND,
                ("med", "sideways"): MarketRegime.MED_VOL_SIDEWAYS,
                ("high", "uptrend"): MarketRegime.HIGH_VOL_UPTREND,
                ("high", "downtrend"): MarketRegime.HIGH_VOL_DOWNTREND,
                ("high", "sideways"): MarketRegime.HIGH_VOL_SIDEWAYS,
            }

            return regime_map.get((vol_regime, trend_regime), MarketRegime.UNKNOWN)

        except Exception as e:
            logger.error(f"Failed to detect regime: {e}")
            return MarketRegime.UNKNOWN

    def get_regime_description(self, regime: MarketRegime) -> str:
        """Get human-readable description of a regime"""
        descriptions = {
            MarketRegime.LOW_VOL_UPTREND: "Low Volatility Uptrend - Steady gains with low risk",
            MarketRegime.LOW_VOL_DOWNTREND: "Low Volatility Downtrend - Gradual decline with low vol",
            MarketRegime.LOW_VOL_SIDEWAYS: "Low Volatility Range - Calm, range-bound market",
            MarketRegime.MED_VOL_UPTREND: "Medium Volatility Uptrend - Normal bull market",
            MarketRegime.MED_VOL_DOWNTREND: "Medium Volatility Downtrend - Normal bear market",
            MarketRegime.MED_VOL_SIDEWAYS: "Medium Volatility Range - Choppy consolidation",
            MarketRegime.HIGH_VOL_UPTREND: "High Volatility Uptrend - Volatile rally",
            MarketRegime.HIGH_VOL_DOWNTREND: "High Volatility Downtrend - Volatile sell-off",
            MarketRegime.HIGH_VOL_SIDEWAYS: "High Volatility Range - Chaotic, whipsaw market",
            MarketRegime.UNKNOWN: "Unknown - Insufficient data",
        }
        return descriptions.get(regime, "Unknown regime")


class RegimeTracker:
    """Tracks regime changes and statistics during a backtest"""

    def __init__(self, detector: RegimeDetector):
        self.detector = detector
        self.regime_bars = {}  # Count of bars in each regime
        self.regime_trades = {}  # Trades executed in each regime
        self.regime_returns = {}  # Returns by regime
        self.regime_changes = []  # List of (timestamp, old_regime, new_regime)
        self.current_regime = MarketRegime.UNKNOWN

    def update(
        self,
        timestamp: datetime,
        price_data: pd.DataFrame,
        trade: Optional[Dict] = None,
        bar_return: Optional[float] = None
    ) -> MarketRegime:
        """
        Update regime tracking with current market data.

        Args:
            timestamp: Current timestamp
            price_data: Historical price data up to current point
            trade: Optional trade that was executed at this timestamp
            bar_return: Optional return for this bar

        Returns:
            Current market regime
        """
        # Detect current regime
        new_regime = self.detector.detect_regime(price_data)

        # Track regime change
        if new_regime != self.current_regime:
            if self.current_regime != MarketRegime.UNKNOWN:
                self.regime_changes.append((timestamp, self.current_regime, new_regime))
            self.current_regime = new_regime

        # Count bars in this regime
        if new_regime not in self.regime_bars:
            self.regime_bars[new_regime] = 0
            self.regime_trades[new_regime] = []
            self.regime_returns[new_regime] = []

        self.regime_bars[new_regime] += 1

        # Track trade if provided
        if trade is not None:
            self.regime_trades[new_regime].append(trade)

        # Track return if provided
        if bar_return is not None:
            self.regime_returns[new_regime].append(bar_return)

        return new_regime

    def get_regime_statistics(self) -> Dict:
        """
        Calculate statistics for each regime.

        Returns:
            Dictionary with regime statistics
        """
        total_bars = sum(self.regime_bars.values())

        regime_stats = {}
        for regime in self.regime_bars.keys():
            bars = self.regime_bars[regime]
            trades = self.regime_trades.get(regime, [])
            returns = self.regime_returns.get(regime, [])

            # Calculate metrics
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

            total_pnl = sum(t.get('pnl', 0) for t in trades)
            avg_return = np.mean(returns) if returns else 0

            regime_stats[regime.value] = {
                'regime': regime.value,
                'description': RegimeDetector().get_regime_description(regime),
                'bars_in_regime': bars,
                'regime_pct': (bars / total_bars * 100) if total_bars > 0 else 0,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades) * 100) if trades else 0,
                'total_pnl': total_pnl,
                'avg_return': avg_return,
                'avg_trade_pnl': (total_pnl / len(trades)) if trades else 0,
            }

        return {
            'regime_statistics': regime_stats,
            'total_regime_changes': len(self.regime_changes),
            'regime_changes': [
                {
                    'timestamp': str(ts),
                    'from_regime': old.value,
                    'to_regime': new.value
                }
                for ts, old, new in self.regime_changes[:50]  # Limit to first 50 changes
            ]
        }
