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


class StrategySignal(BaseModel):
    """Individual strategy signal"""
    signal: str
    position_size_pct: float
    confidence: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    rationale: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


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
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[int] = None
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
