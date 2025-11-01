"""
Pydantic models for temporal trading agents backend.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
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
    executed_at: datetime = Field(default_factory=datetime.utcnow)
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
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


# ==================== API Health ====================

class HealthCheck(BaseModel):
    """API health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class ApiKey(BaseModel):
    """API key for external access"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    key: str
    name: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)


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
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    added_at: datetime = Field(default_factory=datetime.utcnow)
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
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_complete: bool = False  # Whether we have all available data
    missing_dates: List[str] = []  # List of missing date ranges


class WatchlistAddRequest(BaseModel):
    """Request to add ticker to watchlist"""
    symbol: str
    period: Optional[str] = None  # Auto-detect if not provided
    interval: str = '1d'
    auto_sync: bool = True
    priority: int = 0
    tags: List[str] = []
