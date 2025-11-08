"""
Pydantic models for temporal trading agents backend.
"""
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import uuid


class StrategyType(str, Enum):
    """Available strategy types"""
    GRADIENT = "gradient"
    CONFIDENCE = "confidence"
    TIMEFRAME = "timeframe"
    VOLATILITY = "volatility"
    MEAN_REVERSION = "mean_reversion"
    ACCELERATION = "acceleration"
    SWING = "swing"
    RISK_ADJUSTED = "risk_adjusted"
    ALL = "all"  # Run all strategies


class AnalysisStatus(str, Enum):
    """Status of strategy analysis"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== Request Models ====================

class StrategyAnalysisRequest(BaseModel):
    """Request to analyze a trading strategy"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC-USD, AAPL)")
    strategies: List[StrategyType] = Field(..., description="List of strategies to run")
    horizons: Optional[List[int]] = Field(default=[3, 7, 14, 21], description="Forecast horizons in days")

    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC-USD",
                "strategies": ["gradient", "confidence", "timeframe"],
                "horizons": [3, 7, 14, 21]
            }
        }


class ConsensusRequest(BaseModel):
    """Request for consensus analysis across strategies"""
    symbol: str = Field(..., description="Trading symbol")
    horizons: Optional[List[int]] = Field(default=[3, 7, 14, 21])


# ==================== Response Models ====================

class ForecastStats(BaseModel):
    """Forecast statistics"""
    median: List[float]
    q25: List[float]
    q75: List[float]
    min: List[float]
    max: List[float]


class PricePoint(BaseModel):
    """Single price data point"""
    date: str  # ISO format date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class HistoricalPriceData(BaseModel):
    """Historical price data stored separately for efficiency"""
    symbol: str
    source: str = "polygon"  # Data source (polygon, yfinance, etc.)
    prices: List[PricePoint]  # Daily price points
    first_date: str  # First date in dataset
    last_date: str  # Last date in dataset (most recent)
    total_days: int  # Total number of data points
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = {}


class ModelPrediction(BaseModel):
    """Individual model forecast"""
    name: str
    prices: List[float]
    final_change_pct: float


class ForecastData(BaseModel):
    """Comprehensive forecast data for visualization"""
    historical_prices: List[float]  # Last 60 days of historical data
    historical_days: int  # Number of historical days included
    forecast_horizon: int  # Number of forecast days
    current_price: float
    ensemble_median: List[float]
    ensemble_q25: List[float]
    ensemble_q75: List[float]
    ensemble_min: List[float]
    ensemble_max: List[float]
    individual_models: List[ModelPrediction]  # Each model's forecast
    forecast_days: List[int]  # [1, 2, 3, ..., N] for x-axis


class StrategySignal(BaseModel):
    """Individual strategy signal"""
    signal: str
    position_size_pct: float
    confidence: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    rationale: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class AnalysisStarted(BaseModel):
    """Response when analysis is started"""
    analysis_id: str
    symbol: str
    strategy_type: StrategyType
    status: AnalysisStatus
    message: str = "Analysis started in background"


class StrategyResult(BaseModel):
    """Result from a single strategy analysis"""
    strategy_type: StrategyType
    symbol: str
    current_price: float
    signal: StrategySignal
    forecast_stats: Optional[ForecastStats] = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: Optional[int] = None


class ConsensusAnalysis(BaseModel):
    """Consensus across multiple strategies"""
    symbol: str
    current_price: float
    consensus: str
    strength: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_count: int
    bullish_strategies: List[str]
    bearish_strategies: List[str]
    neutral_strategies: List[str]
    avg_position: float
    strategies: Dict[str, StrategySignal]
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==================== Database Models ====================

class StrategyAnalysis(BaseModel):
    """Stored strategy analysis in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    strategy_type: StrategyType
    current_price: float
    signal: StrategySignal
    forecast_stats: Optional[ForecastStats] = None
    forecast_data: Optional[ForecastData] = None  # Visualization data
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: Optional[int] = None
    logs: Optional[List[str]] = []  # Captured console logs during execution
    metadata: Dict[str, Any] = {}


class ConsensusResult(BaseModel):
    """Stored consensus result in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    current_price: float
    consensus: str
    strength: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_count: int
    bullish_strategies: List[str]
    bearish_strategies: List[str]
    neutral_strategies: List[str]
    avg_position: float
    strategy_results: List[str]  # References to StrategyAnalysis IDs
    forecast_data: Optional[ForecastData] = None  # Visualization data
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: Optional[int] = None
    logs: Optional[List[str]] = []  # Captured console logs during execution
    metadata: Dict[str, Any] = {}


class ModelTraining(BaseModel):
    """Model training record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    horizon_days: int
    model_config: Dict[str, Any]
    validation_loss: float
    training_loss: float
    epochs: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    training_time_ms: int
    metadata: Dict[str, Any] = {}


class PriceForecast(BaseModel):
    """Price forecast from ensemble"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    current_price: float
    horizon_days: int
    forecast_stats: ForecastStats
    model_training_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = {}


# ==================== API Health ====================

class HealthCheck(BaseModel):
    """API health check response"""
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    database_connected: bool
    strategies_available: List[str]


# ==================== User Models (for future authentication) ====================

class User(BaseModel):
    """User account"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None


class ApiKey(BaseModel):
    """API key for external access"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    key: str
    name: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    usage_count: int = 0


# ==================== Scheduled Tasks ====================

class ScheduleFrequency(str, Enum):
    """Scheduling frequency options"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"  # Cron expression


class ScheduledTask(BaseModel):
    """Scheduled analysis task"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    symbol: str
    strategy_type: StrategyType
    frequency: ScheduleFrequency
    cron_expression: Optional[str] = None  # For CUSTOM frequency
    horizons: List[int] = [3, 7, 14, 21]
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    metadata: Dict[str, Any] = {}


class ScheduledTaskCreate(BaseModel):
    """Request to create a scheduled task"""
    name: str
    symbol: str
    strategy_type: StrategyType
    frequency: ScheduleFrequency
    cron_expression: Optional[str] = None
    horizons: Optional[List[int]] = [3, 7, 14, 21]


class ScheduledTaskUpdate(BaseModel):
    """Request to update a scheduled task"""
    name: Optional[str] = None
    is_active: Optional[bool] = None
    frequency: Optional[ScheduleFrequency] = None
    cron_expression: Optional[str] = None
    horizons: Optional[List[int]] = None


# ==================== WebSocket Models ====================

class ProgressUpdate(BaseModel):
    """Progress update for WebSocket streaming"""
    task_id: str
    symbol: str
    strategy_type: Optional[str] = None
    status: str  # training, analyzing, completed, error
    progress: float  # 0-100
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==================== Data Synchronization Models ====================

class SyncJobStatus(str, Enum):
    """Status of data sync job"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSyncJob(BaseModel):
    """Data synchronization job for downloading market data"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    period: str  # e.g., '2y', '5y'
    interval: str = '1d'
    status: SyncJobStatus = SyncJobStatus.PENDING
    progress_percent: float = 0.0
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Control flags
    pause_requested: bool = False
    cancel_requested: bool = False


class TickerWatchlist(BaseModel):
    """Ticker in the watchlist for automatic synchronization"""
    symbol: str
    period: str  # e.g., '2y' for crypto, '5y' for stocks
    interval: str = '1d'
    enabled: bool = True
    auto_sync: bool = True  # Automatically sync daily deltas
    priority: int = 0  # Higher priority tickers sync first
    tags: List[str] = []  # e.g., ['crypto', 'high-volume', 'watchlist']
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_synced_at: Optional[datetime] = None
    next_sync_at: Optional[datetime] = None


class DataInventory(BaseModel):
    """Inventory of what market data we have cached"""
    symbol: str
    period: str
    interval: str
    total_days: int = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    file_size_bytes: int = 0
    file_count: int = 0  # Number of S3 files processed
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_complete: bool = False  # Whether we have all available data
    missing_dates: List[str] = []  # List of missing date ranges

    # Auto-scheduling fields
    auto_schedule_enabled: bool = False  # Whether auto delta-sync + analysis is enabled
    schedule_frequency: str = "daily"  # Frequency: "daily", "12h", "6h", or cron expression
    last_auto_sync_at: Optional[datetime] = None  # Last time auto-sync ran
    last_auto_analysis_at: Optional[datetime] = None  # Last time auto-analysis ran
    next_scheduled_sync: Optional[datetime] = None  # Next scheduled sync time
    scheduler_job_id: Optional[str] = None  # APScheduler job ID for tracking


class WatchlistAddRequest(BaseModel):
    """Request to add ticker to watchlist"""
    symbol: str
    period: Optional[str] = None  # Auto-detect if not provided
    interval: str = '1d'
    auto_sync: bool = True
    priority: int = 0
    tags: List[str] = []


# ==================== Backtesting Models ====================

class BacktestStatus(str, Enum):
    """Status of backtest run"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransactionCostConfig(BaseModel):
    """Transaction cost model configuration"""
    taker_fee_bps: float = 5.0  # Exchange taker fees
    maker_rebate_bps: float = 0.0  # Market maker rebates (usually 0 for retail)
    half_spread_bps: float = 2.0  # Half of bid-ask spread
    slippage_coefficient: float = 0.1  # Slippage per $100k notional
    adverse_selection_bps: float = 2.0  # Adverse selection cost
    sec_fee_bps: float = 0.23  # SEC regulatory fees


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration"""
    enabled: bool = True
    train_window_days: int = 252  # 1 year training
    test_window_days: int = 63  # 1 quarter testing
    retrain_frequency_days: int = 21  # Monthly retraining


class OptimizableParams(BaseModel):
    """Parameters that can be optimized"""
    position_size_pct: float = 10.0  # % of portfolio per position
    min_edge_bps: float = 55.0  # Minimum edge to trade
    strong_buy_threshold: float = 0.80  # >= 80% bullish → strong buy
    buy_threshold: float = 0.60  # >= 60% bullish → buy
    moderate_buy_threshold: float = 0.50  # >= 50% bullish → moderate buy
    sell_threshold: float = 0.60  # >= 60% bearish → sell
    moderate_sell_threshold: float = 0.50  # >= 50% bearish → moderate sell


class BacktestConfig(BaseModel):
    """Complete backtest configuration"""
    symbol: str
    start_date: str  # ISO format YYYY-MM-DD
    end_date: str  # ISO format YYYY-MM-DD
    initial_capital: float = 100000.0

    # Optimizable parameters
    optimizable: OptimizableParams = Field(default_factory=OptimizableParams)

    # Legacy fields (kept for backwards compatibility)
    position_size_pct: float = 10.0  # Deprecated - use optimizable.position_size_pct
    min_edge_bps: float = 55.0  # Deprecated - use optimizable.min_edge_bps

    transaction_costs: TransactionCostConfig = Field(default_factory=TransactionCostConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    use_consensus: bool = True  # Use consensus strategies
    individual_strategies: List[StrategyType] = []  # Or test individual strategies

    # Strategy selection for consensus (all enabled by default)
    enabled_strategies: List[str] = [
        'gradient', 'confidence', 'volatility', 'acceleration',
        'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'
    ]


class BacktestTrade(BaseModel):
    """Individual trade in backtest"""
    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    backtest_run_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    shares: float  # Support fractional shares for crypto
    price: float
    notional: float
    transaction_cost: float
    strategy_signal: Optional[str] = None  # Which strategy triggered this
    metadata: Dict[str, Any] = {}


class BacktestPeriodMetrics(BaseModel):
    """Metrics for a single walk-forward period"""
    period_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    backtest_run_id: str
    period_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_costs: float
    final_equity: float
    avg_capital_deployed: Optional[float] = None  # Average capital in positions
    peak_capital_deployed: Optional[float] = None  # Maximum capital in positions
    capital_utilization: Optional[float] = None  # Avg deployed / avg portfolio value


class BacktestMetrics(BaseModel):
    """Aggregate backtest metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    avg_drawdown: float
    win_rate: float  # % of winning trades
    profit_factor: float  # Gross profit / gross loss
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    total_costs: float
    costs_pct_of_capital: float
    # Walk-forward specific metrics
    median_period_sharpe: Optional[float] = None
    period_win_rate: Optional[float] = None  # % of periods with positive return
    worst_period_drawdown: Optional[float] = None


class BacktestRun(BaseModel):
    """Complete backtest run record"""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # User-friendly name
    config: BacktestConfig
    status: BacktestStatus = BacktestStatus.PENDING
    metrics: Optional[BacktestMetrics] = None
    period_metrics: List[BacktestPeriodMetrics] = []  # Walk-forward periods
    regime_analysis: Optional[Dict[str, Any]] = None  # Market regime breakdown
    trades: List[BacktestTrade] = []
    equity_curve: List[Dict[str, Any]] = []  # [{timestamp, equity, drawdown}, ...]
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestCreateRequest(BaseModel):
    """Request to create and run a backtest"""
    name: str
    config: BacktestConfig


class BacktestSummary(BaseModel):
    """Summary of backtest for list view"""
    run_id: str
    name: str
    symbol: str
    start_date: str
    end_date: str
    status: BacktestStatus
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ==================== Parameter Optimization Models ====================

class ParameterGrid(BaseModel):
    """Grid of parameters to test during optimization"""
    position_size_pct: List[float] = [5.0, 10.0, 15.0, 20.0]
    min_edge_bps: List[float] = [30.0, 50.0, 70.0]
    strong_buy_threshold: List[float] = [0.75, 0.80, 0.85]
    buy_threshold: List[float] = [0.55, 0.60, 0.65]
    moderate_buy_threshold: List[float] = [0.45, 0.50, 0.55]
    sell_threshold: List[float] = [0.55, 0.60, 0.65]
    moderate_sell_threshold: List[float] = [0.45, 0.50, 0.55]

    # Strategy combinations to test (optional, defaults to all strategies enabled)
    # Each element is a list of strategy names to enable for that test
    # Example: [['gradient', 'confidence'], ['all'], ['gradient', 'volatility', 'swing']]
    enabled_strategies: Optional[List[List[str]]] = None


class OptimizationMetric(str, Enum):
    """Metric to optimize"""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize


class OptimizationRequest(BaseModel):
    """Request to run parameter optimization"""
    name: str  # User-friendly name for this optimization run
    base_config: BacktestConfig  # Base configuration (symbol, dates, etc.)
    parameter_grid: ParameterGrid  # Parameters to optimize
    optimization_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO
    top_n_results: int = 10  # Number of top results to return


class OptimizationResult(BaseModel):
    """Result for a single parameter combination"""
    parameters: OptimizableParams
    metrics: BacktestMetrics
    backtest_run_id: str
    rank: Optional[int] = None  # Ranking by optimization metric
    metric_value: float  # Value of the optimization metric


class OptimizationStatus(str, Enum):
    """Status of optimization run"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationRun(BaseModel):
    """Complete optimization run record"""
    optimization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    base_config: BacktestConfig
    parameter_grid: ParameterGrid
    optimization_metric: OptimizationMetric
    status: OptimizationStatus = OptimizationStatus.PENDING

    # Results
    total_combinations: int = 0  # Total parameter combinations to test
    completed_combinations: int = 0  # Completed so far
    results: List[OptimizationResult] = []  # All results
    top_results: List[OptimizationResult] = []  # Top N results
    best_parameters: Optional[OptimizableParams] = None  # Best found parameters

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None


# ==================== Paper Trading Models ====================

class PaperTradingStatus(str, Enum):
    """Status of paper trading session"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class PaperTradingConfig(BaseModel):
    """Paper trading configuration"""
    symbol: str
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0
    min_edge_bps: float = 55.0
    transaction_costs: TransactionCostConfig = Field(default_factory=TransactionCostConfig)
    use_consensus: bool = True
    check_interval_minutes: int = 60  # How often to check for signals
    auto_execute: bool = True  # Automatically execute signals


class PaperTrade(BaseModel):
    """Individual paper trade"""
    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    shares: float  # Support fractional shares for crypto
    price: float
    notional: float
    transaction_cost: float
    strategy_signal: Optional[str] = None
    was_executed: bool  # True if simulated, False if rejected
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}

    @field_serializer('timestamp')
    def serialize_datetime(self, dt: datetime, _info):
        """Serialize datetime with timezone"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')


class PaperPosition(BaseModel):
    """Current paper trading position"""
    symbol: str
    shares: float  # Support fractional shares for crypto
    entry_price: float
    entry_timestamp: datetime
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class SignalLog(BaseModel):
    """Log entry for signal checks"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_price: Optional[float] = None
    signal: Optional[str] = None  # 'BUY', 'SELL', 'HOLD', or None
    consensus_score: Optional[float] = None
    expected_return_bps: Optional[float] = None
    action_taken: str  # 'executed', 'rejected', 'no_signal', 'error'
    reason: str
    details: Dict[str, Any] = {}

    @field_serializer('timestamp')
    def serialize_datetime(self, dt: datetime, _info):
        """Serialize datetime with timezone"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')


class PaperTradingSession(BaseModel):
    """Paper trading session"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    config: PaperTradingConfig
    status: PaperTradingStatus = PaperTradingStatus.ACTIVE
    cash: float  # Current cash balance
    starting_capital: float
    current_equity: float
    total_pnl: float
    total_pnl_pct: float
    positions: List[PaperPosition] = []
    trades: List[PaperTrade] = []
    signal_logs: List[SignalLog] = []
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    last_signal_check: Optional[datetime] = None
    next_signal_check: Optional[datetime] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stopped_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @field_serializer('last_signal_check', 'next_signal_check', 'started_at', 'stopped_at')
    def serialize_datetime(self, dt: Optional[datetime], _info):
        """Serialize datetime with timezone"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')


class PaperTradingCreateRequest(BaseModel):
    """Request to create paper trading session"""
    name: str
    config: PaperTradingConfig


class PaperTradingSummary(BaseModel):
    """Summary for paper trading list"""
    session_id: str
    name: str
    symbol: str
    status: PaperTradingStatus
    current_equity: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    started_at: datetime

    @field_serializer('started_at')
    def serialize_datetime(self, dt: datetime, _info):
        """Serialize datetime with timezone"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')
