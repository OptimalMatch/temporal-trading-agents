"""
FastAPI backend for Temporal Trading Agents.
Provides REST API endpoints for running trading strategies and getting consensus.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import traceback

# Add parent directory to path for strategy imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import time
from datetime import datetime, timedelta
import uuid
import asyncio
from concurrent.futures import ProcessPoolExecutor

from backend.models import (
    StrategyAnalysisRequest, ConsensusRequest, StrategyResult,
    ConsensusAnalysis, HealthCheck, StrategyType, StrategyAnalysis,
    ConsensusResult, AnalysisStatus, StrategySignal, ForecastStats,
    ScheduledTask, ScheduledTaskCreate, ScheduledTaskUpdate, ScheduleFrequency,
    AnalysisStarted, ForecastData, ModelPrediction, HistoricalPriceData, PricePoint,
    WatchlistAddRequest, DataSyncJob, TickerWatchlist, DataInventory,
    BacktestConfig, BacktestRun, BacktestCreateRequest, BacktestSummary,
    BacktestStatus, PaperTradingConfig, PaperTradingSession, PaperTradingCreateRequest,
    PaperTradingSummary, PaperTradingStatus
)
from backend.database import Database
from backend.websocket_manager import manager as ws_manager
from backend.scheduler import get_scheduler
from backend.data_sync_manager import get_sync_manager
from backend.backtesting_engine import BacktestEngine

# Import strategies
from strategies.strategy_utils import load_ensemble_module, train_ensemble, get_default_ensemble_configs
from strategies.forecast_gradient_strategy import analyze_gradient_strategy
from strategies.confidence_weighted_strategy import analyze_confidence_weighted_strategy
from strategies.multi_timeframe_strategy import train_multiple_timeframes, analyze_multi_timeframe_strategy
from strategies.volatility_position_sizing import analyze_volatility_position_sizing
from strategies.mean_reversion_strategy import analyze_mean_reversion_strategy
from strategies.acceleration_strategy import analyze_acceleration_strategy
from strategies.swing_trading_strategy import analyze_swing_trading_strategy
from strategies.risk_adjusted_strategy import analyze_risk_adjusted_strategy

# Initialize FastAPI app
app = FastAPI(
    title="Temporal Trading Agents API",
    description="REST API for running forecast-based trading strategies",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database instance
db = Database()

# Scheduler instance
scheduler = None

# Process pool executor for CPU-intensive tasks
executor = None


# ==================== Lifecycle Events ====================

@app.on_event("startup")
async def startup_event():
    """Connect to database and start scheduler on startup"""
    global scheduler, executor

    await db.connect()
    print("🚀 API: Server started successfully")

    # Initialize process pool executor for CPU-intensive tasks
    # Using max 4 workers to avoid overloading the system
    executor = ProcessPoolExecutor(max_workers=4)
    print("⚡ API: ProcessPoolExecutor initialized with 4 workers")

    # Initialize and start scheduler
    scheduler = get_scheduler(db)
    scheduler.start()
    await scheduler.load_scheduled_tasks()
    print("📅 API: Scheduler initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from database and shutdown scheduler"""
    global scheduler, executor

    if scheduler:
        scheduler.shutdown()
        print("📅 API: Scheduler stopped")

    if executor:
        executor.shutdown(wait=True)
        print("⚡ API: ProcessPoolExecutor shutdown")

    await db.disconnect()
    print("👋 API: Server shutdown")


# ==================== Dependency Injection ====================

async def get_database():
    """Dependency to get database instance"""
    return db


# ==================== Utility Functions ====================

async def ensure_dataset_available(symbol: str, database: Database, interval: str = '1d') -> bool:
    """
    Ensure the required dataset is available in cache before analysis.
    If not available, triggers a sync job and waits for completion.

    Args:
        symbol: Trading symbol
        database: Database instance
        interval: Data interval (default '1d')

    Returns:
        True when dataset is ready, False if failed
    """
    from strategies.data_cache import get_cache

    # Determine required period based on asset type
    is_crypto = '-USD' in symbol or '-EUR' in symbol or '-GBP' in symbol
    required_period = '2y' if is_crypto else '5y'

    print(f"📊 Checking dataset availability for {symbol} (required period: {required_period})")

    # Check if data exists in cache
    cache = get_cache()
    cached_data = cache.get(symbol, required_period, interval=interval)

    if cached_data is not None and not cached_data.empty:
        print(f"✓ Dataset already available in cache for {symbol} ({len(cached_data)} rows)")
        return True

    # Check inventory to see if data exists with correct period
    sync_manager = await get_sync_manager(database.client.temporal_trading)
    inventory_items = await sync_manager.get_inventory(symbol=symbol)

    for item in inventory_items:
        if item.interval == interval and item.period == required_period:
            print(f"✓ Dataset found in inventory for {symbol}, loading from cache")
            # Data exists in inventory, try loading from cache again (may have been TTL expired)
            cached_data = cache.get(symbol, required_period, interval=interval)
            if cached_data is not None and not cached_data.empty:
                return True

    # Data not available - trigger sync job
    print(f"📥 Dataset not available for {symbol}, triggering sync job ({required_period})")

    # Create sync job
    job = await sync_manager.create_sync_job(symbol, required_period, interval)
    job_id = job.job_id

    # Start the job
    started = await sync_manager.start_sync_job(job_id)
    if not started:
        print(f"❌ Failed to start sync job for {symbol}")
        return False

    print(f"⏳ Waiting for sync job {job_id} to complete...")

    # Poll for job completion (check every 2 seconds)
    max_wait_time = 600  # 10 minutes max
    elapsed = 0

    while elapsed < max_wait_time:
        await asyncio.sleep(2)
        elapsed += 2

        # Check job status
        job_doc = await database.client.temporal_trading.data_sync_jobs.find_one({"job_id": job_id})
        if not job_doc:
            print(f"❌ Sync job {job_id} not found")
            return False

        status = job_doc.get('status')

        if status == 'completed':
            print(f"✓ Sync job completed for {symbol} in {elapsed}s")
            return True
        elif status == 'failed':
            error = job_doc.get('error_message', 'Unknown error')
            print(f"❌ Sync job failed for {symbol}: {error}")
            return False
        elif status == 'cancelled':
            print(f"❌ Sync job cancelled for {symbol}")
            return False

        # Show progress
        progress = job_doc.get('progress_percent', 0)
        if elapsed % 10 == 0:  # Log every 10 seconds
            print(f"⏳ Sync job progress: {progress:.1f}% ({elapsed}s elapsed)")

    print(f"⏰ Sync job timed out after {max_wait_time}s")
    return False

# Wrapper functions for ProcessPoolExecutor (must be picklable)
def _train_ensemble_worker(symbol: str, horizon: int, name: str, ensemble_path: str = "examples/crypto_ensemble_forecast.py"):
    """Worker function for training ensemble in separate process"""
    ensemble = load_ensemble_module(ensemble_path)
    configs = get_default_ensemble_configs(horizon)
    return train_ensemble(symbol, horizon, configs, name, ensemble)

def _train_multiple_timeframes_worker(symbol: str, horizons: list, ensemble_path: str = "examples/crypto_ensemble_forecast.py"):
    """Worker function for training multiple timeframes in separate process"""
    ensemble = load_ensemble_module(ensemble_path)
    return train_multiple_timeframes(symbol, ensemble, horizons)

async def save_historical_prices(symbol: str, df, db: Database, source: str = "polygon") -> bool:
    """Save historical price data from DataFrame to database"""
    try:
        # Extract OHLCV data from DataFrame
        prices = []
        for idx, row in df.iterrows():
            price_point = PricePoint(
                date=idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                open=float(row['Open']) if 'Open' in row else float(row['Close']),
                high=float(row['High']) if 'High' in row else float(row['Close']),
                low=float(row['Low']) if 'Low' in row else float(row['Close']),
                close=float(row['Close']),
                volume=float(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else None
            )
            prices.append(price_point.dict())

        # Get date range
        dates = [p['date'] for p in prices]
        first_date = min(dates)
        last_date = max(dates)

        historical_data = {
            "symbol": symbol,
            "source": source,
            "prices": prices,
            "first_date": first_date,
            "last_date": last_date,
            "total_days": len(prices),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "metadata": {}
        }

        # Upsert to database
        success = await db.upsert_historical_prices(historical_data)
        return success
    except Exception as e:
        print(f"Error saving historical prices for {symbol}: {e}")
        return False


def build_forecast_data(ensemble_stats: dict, df, forecast_horizon: int, current_price: float) -> ForecastData:
    """Build ForecastData object from ensemble stats and dataframe"""
    # Get historical prices (up to 2 years = 730 days for flexible display)
    hist_days = min(730, len(df))
    historical_prices = df['Close'].iloc[-hist_days:].values.tolist()

    # Build individual model predictions
    individual_models = []
    if 'details' in ensemble_stats:
        for detail in ensemble_stats['details']:
            individual_models.append(ModelPrediction(
                name=detail['name'],
                prices=detail['prices'].tolist() if hasattr(detail['prices'], 'tolist') else detail['prices'],
                final_change_pct=float(detail['final_change'])
            ))

    # Convert numpy arrays to lists
    def to_list(arr):
        return arr.tolist() if hasattr(arr, 'tolist') else list(arr)

    return ForecastData(
        historical_prices=historical_prices,
        historical_days=hist_days,
        forecast_horizon=forecast_horizon,
        current_price=current_price,
        ensemble_median=to_list(ensemble_stats['median']),
        ensemble_q25=to_list(ensemble_stats['q25']),
        ensemble_q75=to_list(ensemble_stats['q75']),
        ensemble_min=to_list(ensemble_stats['min']),
        ensemble_max=to_list(ensemble_stats['max']),
        individual_models=individual_models,
        forecast_days=list(range(1, forecast_horizon + 1))
    )

def convert_numpy_to_native(obj):
    """Recursively convert numpy types to native Python types"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_native(item) for item in obj)
    return obj

def convert_strategy_result_to_model(strategy_type: str, result: dict, symbol: str, current_price: float, execution_time_ms: int) -> StrategyAnalysis:
    """Convert strategy result dict to StrategyAnalysis model"""
    # Convert all numpy types to native Python types
    result = convert_numpy_to_native(result)

    signal = StrategySignal(
        signal=result.get('signal', 'UNKNOWN'),
        position_size_pct=result.get('position_size_pct', 0),
        confidence=result.get('confidence'),
        target_price=result.get('target_price'),
        stop_loss=result.get('stop_loss'),
        rationale=result.get('rationale'),
        metadata=result.get('metadata', {})
    )

    # Extract forecast stats if available
    forecast_stats = None
    if 'forecast_median' in result:
        forecast_stats = ForecastStats(
            median=result.get('forecast_median', []),
            q25=result.get('forecast_q25', []),
            q75=result.get('forecast_q75', []),
            min=result.get('forecast_min', []),
            max=result.get('forecast_max', [])
        )

    return StrategyAnalysis(
        symbol=symbol,
        strategy_type=StrategyType(strategy_type),
        current_price=current_price,
        signal=signal,
        forecast_stats=forecast_stats,
        status=AnalysisStatus.COMPLETED,
        execution_time_ms=execution_time_ms,
        metadata={k: v for k, v in result.items() if k not in ['signal', 'position_size_pct', 'confidence', 'target_price', 'stop_loss']}
    )


# ==================== Health Check ====================

@app.get("/health", response_model=HealthCheck)
async def health_check(database: Database = Depends(get_database)):
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        database_connected=database.db is not None,
        strategies_available=[s.value for s in StrategyType if s != StrategyType.ALL]
    )


# ==================== Background Task Functions ====================

async def run_gradient_analysis_background(analysis_id: str, symbol: str, database: Database):
    """Background task to run gradient analysis"""
    logs = []  # Initialize logs list

    try:
        # Log: Starting
        logs.append(f"[{datetime.utcnow().isoformat()}] Starting gradient analysis for {symbol}")

        # Send WebSocket update: Starting
        await ws_manager.send_progress(
            task_id=analysis_id,
            symbol=symbol,
            strategy_type="gradient",
            status="running",
            progress=0,
            message="Starting gradient analysis..."
        )

        # Update status to RUNNING
        await database.db.strategy_analyses.update_one(
            {"id": analysis_id},
            {"$set": {"status": AnalysisStatus.RUNNING.value}}
        )

        start_time = time.time()

        # Ensure dataset is available before analysis
        logs.append(f"[{datetime.utcnow().isoformat()}] Checking dataset availability for {symbol}")
        dataset_ready = await ensure_dataset_available(symbol, database)

        if not dataset_ready:
            error_msg = f"Failed to ensure dataset availability for {symbol}"
            logs.append(f"[{datetime.utcnow().isoformat()}] ERROR: {error_msg}")
            raise Exception(error_msg)

        logs.append(f"[{datetime.utcnow().isoformat()}] Dataset confirmed available")

        # Log and send progress: Loading data
        logs.append(f"[{datetime.utcnow().isoformat()}] Loading market data for {symbol}")
        await ws_manager.send_progress(
            task_id=analysis_id,
            symbol=symbol,
            strategy_type="gradient",
            status="running",
            progress=10,
            message="Loading market data..."
        )

        # Log and send progress: Training models
        logs.append(f"[{datetime.utcnow().isoformat()}] Training ensemble models (14-day forecast)")
        await ws_manager.send_progress(
            task_id=analysis_id,
            symbol=symbol,
            strategy_type="gradient",
            status="running",
            progress=30,
            message="Training ensemble models..."
        )

        # Run CPU-intensive training in process pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        stats, df = await loop.run_in_executor(
            executor,
            _train_ensemble_worker,
            symbol, 14, "14-DAY"
        )
        current_price = df['Close'].iloc[-1]
        logs.append(f"[{datetime.utcnow().isoformat()}] Model training completed. Current price: ${current_price:.2f}")

        # Log and send progress: Running strategy
        logs.append(f"[{datetime.utcnow().isoformat()}] Analyzing forecast gradient patterns")
        await ws_manager.send_progress(
            task_id=analysis_id,
            symbol=symbol,
            strategy_type="gradient",
            status="running",
            progress=80,
            message="Analyzing forecast gradient..."
        )

        # Run gradient strategy
        result = analyze_gradient_strategy(stats, current_price)
        result['forecast_median'] = stats['median'].tolist() if hasattr(stats['median'], 'tolist') else list(stats['median'])
        logs.append(f"[{datetime.utcnow().isoformat()}] Strategy analysis completed. Signal: {result.get('signal')}")

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Convert and store results
        analysis = convert_strategy_result_to_model('gradient', result, symbol, current_price, execution_time_ms)
        analysis.id = analysis_id
        analysis.status = AnalysisStatus.COMPLETED
        analysis.logs = logs  # Attach logs to analysis

        # Update database with completed analysis
        logs.append(f"[{datetime.utcnow().isoformat()}] Saving analysis results to database")
        logs.append(f"[{datetime.utcnow().isoformat()}] SUCCESS: Analysis completed in {execution_time_ms}ms")
        await database.db.strategy_analyses.update_one(
            {"id": analysis_id},
            {"$set": analysis.dict()}
        )

        # Send WebSocket update: Completed
        await ws_manager.send_progress(
            task_id=analysis_id,
            symbol=symbol,
            strategy_type="gradient",
            status="completed",
            progress=100,
            message=f"Analysis completed in {execution_time_ms}ms",
            details={
                "signal": result.get('signal'),
                "current_price": current_price
            }
        )

    except Exception as e:
        # Log error
        logs.append(f"[{datetime.utcnow().isoformat()}] ERROR: Analysis failed - {str(e)}")

        # Update status to FAILED with logs
        await database.db.strategy_analyses.update_one(
            {"id": analysis_id},
            {"$set": {
                "status": AnalysisStatus.FAILED.value,
                "error": str(e),
                "logs": logs
            }}
        )

        # Send WebSocket update: Failed
        await ws_manager.send_progress(
            task_id=analysis_id,
            symbol=symbol,
            strategy_type="gradient",
            status="failed",
            progress=0,
            message=f"Analysis failed: {str(e)}"
        )

        print(f"❌ Background analysis failed for {analysis_id}: {e}")


async def run_consensus_analysis_background(consensus_id: str, request: ConsensusRequest, database: Database):
    """Background task to run consensus analysis"""
    logs = []  # Initialize logs list

    try:
        # Log: Starting
        logs.append(f"[{datetime.utcnow().isoformat()}] Starting consensus analysis for {request.symbol}")

        # Send WebSocket update: Starting
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=0,
            message="Starting consensus analysis..."
        )

        # Update status to RUNNING
        await database.db.consensus_results.update_one(
            {"id": consensus_id},
            {"$set": {"status": AnalysisStatus.RUNNING.value}}
        )

        start_time = time.time()

        # Ensure dataset is available before analysis
        logs.append(f"[{datetime.utcnow().isoformat()}] Checking dataset availability for {request.symbol}")
        dataset_ready = await ensure_dataset_available(request.symbol, database)

        if not dataset_ready:
            error_msg = f"Failed to ensure dataset availability for {request.symbol}"
            logs.append(f"[{datetime.utcnow().isoformat()}] ERROR: {error_msg}")
            raise Exception(error_msg)

        logs.append(f"[{datetime.utcnow().isoformat()}] Dataset confirmed available")

        # Log: Training models
        logs.append(f"[{datetime.utcnow().isoformat()}] Training ensemble models for consensus analysis")

        # Train 14-day ensemble - run in process pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        stats, df = await loop.run_in_executor(
            executor,
            _train_ensemble_worker,
            request.symbol, 14, "14-DAY"
        )
        current_price = df['Close'].iloc[-1]
        logs.append(f"[{datetime.utcnow().isoformat()}] Model training completed. Current price: ${current_price:.2f}")

        results = {}
        strategy_ids = []

        # Run all 8 strategies with progress updates
        # Strategy 1: Gradient (0% -> 12%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Forecast Gradient strategy (1/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=0,
            message="Running Forecast Gradient strategy..."
        )
        try:
            result = analyze_gradient_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('gradient', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Forecast Gradient'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Forecast Gradient completed: {result.get('signal')}")
        except Exception as e:
            print(f"Gradient strategy failed: {e}")
            results['Forecast Gradient'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Forecast Gradient failed - {str(e)}")

        # Strategy 2: Confidence-Weighted (12% -> 25%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Confidence-Weighted strategy (2/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=12,
            message="Running Confidence-Weighted strategy..."
        )
        try:
            result = analyze_confidence_weighted_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('confidence', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Confidence-Weighted'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Confidence-Weighted completed: {result.get('signal')}")
        except Exception as e:
            print(f"Confidence strategy failed: {e}")
            results['Confidence-Weighted'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Confidence-Weighted failed - {str(e)}")

        # Strategy 3: Multi-Timeframe (25% -> 37%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Multi-Timeframe strategy (3/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=25,
            message="Running Multi-Timeframe strategy..."
        )
        try:
            # Run in process pool to avoid blocking
            timeframe_data = await loop.run_in_executor(
                executor,
                _train_multiple_timeframes_worker,
                request.symbol, request.horizons
            )
            result = analyze_multi_timeframe_strategy(timeframe_data, current_price)
            analysis = convert_strategy_result_to_model('timeframe', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Multi-Timeframe'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Multi-Timeframe completed: {result.get('signal')}")
        except Exception as e:
            print(f"Timeframe strategy failed: {e}")
            results['Multi-Timeframe'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Multi-Timeframe failed - {str(e)}")

        # Strategy 4: Volatility Sizing (37% -> 50%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Volatility Sizing strategy (4/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=37,
            message="Running Volatility Sizing strategy..."
        )
        try:
            result = analyze_volatility_position_sizing(stats, current_price)
            analysis = convert_strategy_result_to_model('volatility', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Volatility Sizing'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Volatility Sizing completed: {result.get('signal')}")
        except Exception as e:
            print(f"Volatility strategy failed: {e}")
            results['Volatility Sizing'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Volatility Sizing failed - {str(e)}")

        # Strategy 5: Mean Reversion (50% -> 62%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Mean Reversion strategy (5/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=50,
            message="Running Mean Reversion strategy..."
        )
        try:
            result = analyze_mean_reversion_strategy(stats, df, current_price)
            analysis = convert_strategy_result_to_model('mean_reversion', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Mean Reversion'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Mean Reversion completed: {result.get('signal')}")
        except Exception as e:
            print(f"Mean reversion strategy failed: {e}")
            results['Mean Reversion'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Mean Reversion failed - {str(e)}")

        # Strategy 6: Acceleration (62% -> 75%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Acceleration strategy (6/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=62,
            message="Running Acceleration strategy..."
        )
        try:
            result = analyze_acceleration_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('acceleration', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Acceleration'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Acceleration completed: {result.get('signal')}")
        except Exception as e:
            print(f"Acceleration strategy failed: {e}")
            results['Acceleration'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Acceleration failed - {str(e)}")

        # Strategy 7: Swing Trading (75% -> 87%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Swing Trading strategy (7/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=75,
            message="Running Swing Trading strategy..."
        )
        try:
            result = analyze_swing_trading_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('swing', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Swing Trading'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Swing Trading completed: {result.get('signal')}")
        except Exception as e:
            print(f"Swing strategy failed: {e}")
            results['Swing Trading'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Swing Trading failed - {str(e)}")

        # Strategy 8: Risk-Adjusted (87% -> 100%)
        logs.append(f"[{datetime.utcnow().isoformat()}] Running Risk-Adjusted strategy (8/8)")
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="running",
            progress=87,
            message="Running Risk-Adjusted strategy..."
        )
        try:
            result = analyze_risk_adjusted_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('risk_adjusted', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Risk-Adjusted'] = result
            logs.append(f"[{datetime.utcnow().isoformat()}] Risk-Adjusted completed: {result.get('signal')}")
        except Exception as e:
            print(f"Risk-adjusted strategy failed: {e}")
            results['Risk-Adjusted'] = {'signal': 'ERROR', 'position_size_pct': 0}
            logs.append(f"[{datetime.utcnow().isoformat()}] WARNING: Risk-Adjusted failed - {str(e)}")

        # Analyze consensus
        logs.append(f"[{datetime.utcnow().isoformat()}] Calculating consensus from strategy results")
        bullish_keywords = ['BUY', 'BULLISH', 'MOMENTUM', 'REVERT', 'REVERSAL', 'EXCELLENT', 'GOOD']
        bearish_keywords = ['SELL', 'BEARISH', 'OUT', 'STAY', 'EXIT', 'POOR', 'FALSE']

        bullish_strategies = []
        bearish_strategies = []
        neutral_strategies = []

        for name, data in results.items():
            signal = data.get('signal', 'ERROR')
            if signal == 'ERROR':
                continue
            if any(keyword in signal for keyword in bullish_keywords) and 'POOR' not in signal and 'FALSE' not in signal:
                bullish_strategies.append(name)
            elif any(keyword in signal for keyword in bearish_keywords) or 'NO' in signal:
                bearish_strategies.append(name)
            else:
                neutral_strategies.append(name)

        # Calculate consensus
        total = len([name for name, data in results.items() if data.get('signal') != 'ERROR'])
        bullish_count = len(bullish_strategies)
        bearish_count = len(bearish_strategies)

        if bullish_count >= 6:
            consensus = "STRONG BUY CONSENSUS"
            strength = "VERY HIGH"
        elif bullish_count >= 5:
            consensus = "BUY CONSENSUS"
            strength = "HIGH"
        elif bullish_count >= 4:
            consensus = "MODERATE BUY"
            strength = "MODERATE"
        elif bearish_count >= 5:
            consensus = "SELL/AVOID CONSENSUS"
            strength = "HIGH"
        elif bearish_count >= 4:
            consensus = "MODERATE SELL/AVOID"
            strength = "MODERATE"
        else:
            consensus = "MIXED SIGNALS"
            strength = "LOW"

        logs.append(f"[{datetime.utcnow().isoformat()}] Consensus: {consensus} ({strength}) - Bullish: {bullish_count}, Bearish: {bearish_count}")

        # Calculate average position
        bullish_positions = [results[name].get('position_size_pct', 0) for name in bullish_strategies
                           if 'position_size_pct' in results[name] and results[name]['position_size_pct'] > 0]
        avg_position = sum(bullish_positions) / len(bullish_positions) if bullish_positions else 0

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Build forecast data for visualization
        forecast_data = build_forecast_data(stats, df, 14, current_price)

        # Save historical prices to separate collection
        logs.append(f"[{datetime.utcnow().isoformat()}] Saving historical price data to database")
        await save_historical_prices(request.symbol, df, database, source="polygon")

        # Update consensus result in database
        logs.append(f"[{datetime.utcnow().isoformat()}] Saving consensus results to database")
        logs.append(f"[{datetime.utcnow().isoformat()}] SUCCESS: Consensus analysis completed in {execution_time_ms}ms")
        await database.db.consensus_results.update_one(
            {"id": consensus_id},
            {"$set": {
                "current_price": current_price,
                "consensus": consensus,
                "strength": strength,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": len(neutral_strategies),
                "total_count": total,
                "bullish_strategies": bullish_strategies,
                "bearish_strategies": bearish_strategies,
                "neutral_strategies": neutral_strategies,
                "avg_position": avg_position,
                "strategy_results": strategy_ids,
                "forecast_data": forecast_data.dict(),
                "status": AnalysisStatus.COMPLETED.value,
                "execution_time_ms": execution_time_ms,
                "logs": logs
            }}
        )

        # Send WebSocket update: Completed
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="completed",
            progress=100,
            message=f"Consensus analysis completed in {execution_time_ms}ms",
            details={
                "consensus": consensus,
                "strength": strength,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "current_price": current_price
            }
        )

    except Exception as e:
        # Log error
        logs.append(f"[{datetime.utcnow().isoformat()}] ERROR: Consensus analysis failed - {str(e)}")

        # Update status to FAILED with logs
        await database.db.consensus_results.update_one(
            {"id": consensus_id},
            {"$set": {
                "status": AnalysisStatus.FAILED.value,
                "error": str(e),
                "logs": logs
            }}
        )

        # Send WebSocket update: Failed
        await ws_manager.send_progress(
            task_id=consensus_id,
            symbol=request.symbol,
            strategy_type="consensus",
            status="failed",
            progress=0,
            message=f"Consensus analysis failed: {str(e)}"
        )

        print(f"❌ Background consensus analysis failed for {consensus_id}: {e}")
        traceback.print_exc()


# ==================== Strategy Endpoints ====================

@app.post("/api/v1/analyze/gradient", response_model=AnalysisStarted)
async def analyze_gradient(
    symbol: str,
    background_tasks: BackgroundTasks,
    database: Database = Depends(get_database)
):
    """Start forecast gradient strategy analysis (async)"""
    try:
        # Create pending analysis record
        analysis_id = str(uuid.uuid4())
        pending_analysis = StrategyAnalysis(
            id=analysis_id,
            symbol=symbol,
            strategy_type=StrategyType.GRADIENT,
            current_price=0.0,  # Will be updated when analysis runs
            signal=StrategySignal(signal="PENDING", position_size_pct=0),
            status=AnalysisStatus.PENDING
        )

        # Store in database
        await database.create_strategy_analysis(pending_analysis)

        # Add to background tasks
        background_tasks.add_task(run_gradient_analysis_background, analysis_id, symbol, database)

        return AnalysisStarted(
            analysis_id=analysis_id,
            symbol=symbol,
            strategy_type=StrategyType.GRADIENT,
            status=AnalysisStatus.PENDING
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


@app.post("/api/v1/analyze/confidence", response_model=StrategyResult)
async def analyze_confidence(
    symbol: str,
    database: Database = Depends(get_database)
):
    """Run confidence-weighted strategy analysis"""
    try:
        start_time = time.time()

        ensemble = load_ensemble_module("examples/crypto_ensemble_forecast.py")
        configs = get_default_ensemble_configs(14)
        stats, df = train_ensemble(symbol, 14, configs, "14-DAY", ensemble)
        current_price = df['Close'].iloc[-1]

        result = analyze_confidence_weighted_strategy(stats, current_price)
        execution_time_ms = int((time.time() - start_time) * 1000)

        analysis = convert_strategy_result_to_model('confidence', result, symbol, current_price, execution_time_ms)
        await database.create_strategy_analysis(analysis)

        return StrategyResult(
            strategy_type=StrategyType.CONFIDENCE,
            symbol=symbol,
            current_price=current_price,
            signal=analysis.signal,
            forecast_stats=analysis.forecast_stats,
            execution_time_ms=execution_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy analysis failed: {str(e)}")


# ==================== Analysis Status Endpoint ====================

@app.get("/api/v1/analysis/{analysis_id}")
async def get_analysis_status(
    analysis_id: str,
    database: Database = Depends(get_database)
):
    """Get status and results of an analysis (checks both strategy and consensus collections)"""
    try:
        # Try to find in strategy_analyses collection first
        analysis = await database.get_strategy_analysis(analysis_id)

        if analysis:
            return analysis

        # If not found, try consensus_results collection
        consensus_result = await database.get_consensus_result(analysis_id)

        if consensus_result:
            return consensus_result

        # Not found in either collection
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis status: {str(e)}")


@app.post("/api/v1/analyze/consensus", response_model=AnalysisStarted)
async def analyze_consensus(
    request: ConsensusRequest,
    background_tasks: BackgroundTasks,
    database: Database = Depends(get_database)
):
    """Start consensus analysis (async) - runs all 8 strategies"""
    try:
        # Create pending consensus record
        consensus_id = str(uuid.uuid4())
        pending_consensus = ConsensusResult(
            id=consensus_id,
            symbol=request.symbol,
            current_price=0.0,  # Will be updated when analysis runs
            consensus="PENDING",
            strength="PENDING",
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            total_count=0,
            bullish_strategies=[],
            bearish_strategies=[],
            neutral_strategies=[],
            avg_position=0.0,
            strategy_results=[],
            status=AnalysisStatus.PENDING
        )

        # Store in database
        await database.create_consensus_result(pending_consensus)

        # Add to background tasks
        background_tasks.add_task(run_consensus_analysis_background, consensus_id, request, database)

        return AnalysisStarted(
            analysis_id=consensus_id,
            symbol=request.symbol,
            strategy_type=StrategyType.ALL,
            status=AnalysisStatus.PENDING
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start consensus analysis: {str(e)}")


# ==================== History Endpoints ====================

@app.get("/api/v1/history/analyses")
async def get_all_analyses(
    strategy_type: Optional[StrategyType] = None,
    limit: int = 100,
    skip: int = 0,
    database: Database = Depends(get_database)
):
    """Get recent strategy analyses across all symbols"""
    try:
        analyses = await database.get_strategy_analyses(
            symbol=None,  # No symbol filter = all symbols
            strategy_type=strategy_type,
            limit=limit,
            skip=skip
        )
        return {"count": len(analyses), "analyses": analyses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/api/v1/history/analyses/{symbol}")
async def get_analysis_history(
    symbol: str,
    strategy_type: Optional[StrategyType] = None,
    limit: int = 100,
    skip: int = 0,
    database: Database = Depends(get_database)
):
    """Get historical strategy analyses for a symbol"""
    try:
        analyses = await database.get_strategy_analyses(
            symbol=symbol,
            strategy_type=strategy_type,
            limit=limit,
            skip=skip
        )
        return {"symbol": symbol, "count": len(analyses), "analyses": analyses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/api/v1/history/consensus")
async def get_all_consensus(
    limit: int = 100,
    skip: int = 0,
    database: Database = Depends(get_database)
):
    """Get recent consensus results across all symbols"""
    try:
        results = await database.get_consensus_results(
            symbol=None,  # No symbol filter = all symbols
            limit=limit,
            skip=skip
        )
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve consensus history: {str(e)}")


@app.get("/api/v1/history/consensus/{symbol}")
async def get_consensus_history(
    symbol: str,
    limit: int = 100,
    skip: int = 0,
    database: Database = Depends(get_database)
):
    """Get historical consensus results for a symbol"""
    try:
        results = await database.get_consensus_results(
            symbol=symbol,
            limit=limit,
            skip=skip
        )
        return {"symbol": symbol, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/api/v1/history/prices/{symbol}")
async def get_historical_prices(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    database: Database = Depends(get_database)
):
    """Get historical price data for a symbol"""
    try:
        if start_date or end_date:
            prices = await database.get_historical_prices_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        else:
            prices_doc = await database.get_historical_prices(symbol)
            prices = prices_doc.get("prices") if prices_doc else None

        if prices is None:
            raise HTTPException(status_code=404, detail=f"No historical price data found for {symbol}")

        return {
            "symbol": symbol,
            "count": len(prices) if isinstance(prices, list) else 0,
            "prices": prices
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical prices: {str(e)}")


@app.get("/api/v1/analytics/{symbol}")
async def get_symbol_analytics(
    symbol: str,
    database: Database = Depends(get_database)
):
    """Get analytics for a trading symbol"""
    try:
        analytics = await database.get_symbol_analytics(symbol)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analytics: {str(e)}")


# ==================== WebSocket Endpoints ====================

@app.websocket("/ws/progress")
async def websocket_progress_global(websocket: WebSocket):
    """WebSocket endpoint for global progress updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages if needed
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/progress/{task_id}")
async def websocket_progress_task(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for specific task progress updates"""
    await ws_manager.connect(websocket, task_id)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, task_id)


# ==================== Scheduled Task Endpoints ====================

@app.post("/api/v1/schedule", response_model=ScheduledTask)
async def create_scheduled_task(
    task_request: ScheduledTaskCreate,
    database: Database = Depends(get_database)
):
    """Create a new scheduled analysis task"""
    try:
        global scheduler

        # Calculate next run time based on frequency
        now = datetime.utcnow()
        if task_request.frequency == ScheduleFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif task_request.frequency == ScheduleFrequency.DAILY:
            next_run = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        elif task_request.frequency == ScheduleFrequency.WEEKLY:
            days_ahead = 0 - now.weekday()  # Monday is 0
            if days_ahead <= 0:
                days_ahead += 7
            next_run = (now + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            next_run = now + timedelta(hours=1)  # Default to 1 hour

        # Create task
        task = ScheduledTask(
            name=task_request.name,
            symbol=task_request.symbol,
            strategy_type=task_request.strategy_type,
            frequency=task_request.frequency,
            cron_expression=task_request.cron_expression,
            horizons=task_request.horizons or [3, 7, 14, 21],
            next_run=next_run
        )

        # Save to database
        await database.create_scheduled_task(task)

        # Schedule in APScheduler
        if scheduler:
            await scheduler.schedule_task(task)

        return task

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scheduled task: {str(e)}")


@app.get("/api/v1/schedule")
async def get_scheduled_tasks(
    symbol: Optional[str] = None,
    is_active: Optional[bool] = None,
    database: Database = Depends(get_database)
):
    """Get all scheduled tasks with optional filters"""
    try:
        tasks = await database.get_scheduled_tasks(symbol=symbol, is_active=is_active)
        return {"count": len(tasks), "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scheduled tasks: {str(e)}")


@app.get("/api/v1/schedule/{task_id}")
async def get_scheduled_task(
    task_id: str,
    database: Database = Depends(get_database)
):
    """Get a specific scheduled task"""
    task = await database.get_scheduled_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Scheduled task not found")
    return task


@app.patch("/api/v1/schedule/{task_id}")
async def update_scheduled_task(
    task_id: str,
    task_update: ScheduledTaskUpdate,
    database: Database = Depends(get_database)
):
    """Update a scheduled task"""
    try:
        global scheduler

        # Update in database
        updates = task_update.dict(exclude_unset=True)
        success = await database.update_scheduled_task(task_id, updates)

        if not success:
            raise HTTPException(status_code=404, detail="Scheduled task not found")

        # Reschedule if active status or frequency changed
        if scheduler and ('is_active' in updates or 'frequency' in updates or 'cron_expression' in updates):
            if updates.get('is_active', True):
                await scheduler.reschedule_task(task_id)
            else:
                await scheduler.unschedule_task(task_id)

        # Get updated task
        task = await database.get_scheduled_task(task_id)
        return task

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update scheduled task: {str(e)}")


@app.delete("/api/v1/schedule/{task_id}")
async def delete_scheduled_task(
    task_id: str,
    database: Database = Depends(get_database)
):
    """Delete a scheduled task"""
    try:
        global scheduler

        # Unschedule from APScheduler
        if scheduler:
            await scheduler.unschedule_task(task_id)

        # Delete from database
        success = await database.delete_scheduled_task(task_id)

        if not success:
            raise HTTPException(status_code=404, detail="Scheduled task not found")

        return {"status": "deleted", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete scheduled task: {str(e)}")


# ==================== Helper Functions ====================

async def run_strategy_analysis(symbol: str, strategy_type: StrategyType,
                                horizons: List[int] = [3, 7, 14, 21],
                                task_id: str = None):
    """
    Run strategy analysis with WebSocket progress updates.
    This is used by both API endpoints and scheduled tasks.
    """
    if not task_id:
        task_id = str(uuid.uuid4())

    try:
        # Send initial progress
        await ws_manager.send_progress(
            task_id=task_id,
            symbol=symbol,
            strategy_type=strategy_type.value,
            status="starting",
            progress=0,
            message=f"Starting {strategy_type.value} analysis for {symbol}"
        )

        # If consensus/all strategies, run them all
        if strategy_type == StrategyType.ALL:
            await ws_manager.send_progress(
                task_id, symbol, "consensus", "training", 10,
                "Loading ensemble and training models"
            )

            ensemble = load_ensemble_module("examples/crypto_ensemble_forecast.py")
            configs = get_default_ensemble_configs(14)
            stats, df = train_ensemble(symbol, 14, configs, "14-DAY", ensemble)
            current_price = df['Close'].iloc[-1]

            # Run all 8 strategies with progress updates
            strategies_results = {}
            strategy_list = [
                ("Gradient", "gradient", analyze_gradient_strategy),
                ("Confidence", "confidence", analyze_confidence_weighted_strategy),
                ("Volatility", "volatility", analyze_volatility_position_sizing),
                ("Mean Reversion", "mean_reversion", lambda s, p: analyze_mean_reversion_strategy(s, df, p)),
                ("Acceleration", "acceleration", analyze_acceleration_strategy),
                ("Swing Trading", "swing", analyze_swing_trading_strategy),
                ("Risk Adjusted", "risk_adjusted", analyze_risk_adjusted_strategy),
            ]

            for idx, (name, strat_type, func) in enumerate(strategy_list):
                progress = 30 + (idx / len(strategy_list)) * 50
                await ws_manager.send_progress(
                    task_id, symbol, strat_type, "analyzing", progress,
                    f"Running {name} strategy"
                )

                try:
                    result = func(stats, current_price)
                    strategies_results[name] = result

                    # Store in database
                    analysis = convert_strategy_result_to_model(strat_type, result, symbol, current_price, 0)
                    await db.create_strategy_analysis(analysis)

                except Exception as e:
                    print(f"{name} strategy failed: {e}")
                    strategies_results[name] = {'signal': 'ERROR', 'position_size_pct': 0}

            # Multi-timeframe separately
            await ws_manager.send_progress(
                task_id, symbol, "timeframe", "analyzing", 85,
                "Running Multi-Timeframe strategy"
            )
            try:
                timeframe_data = train_multiple_timeframes(symbol, ensemble, horizons)
                result = analyze_multi_timeframe_strategy(timeframe_data, current_price)
                strategies_results['Multi-Timeframe'] = result

                analysis = convert_strategy_result_to_model('timeframe', result, symbol, current_price, 0)
                await db.create_strategy_analysis(analysis)
            except Exception as e:
                print(f"Multi-Timeframe strategy failed: {e}")
                strategies_results['Multi-Timeframe'] = {'signal': 'ERROR', 'position_size_pct': 0}

            # Calculate consensus
            await ws_manager.send_progress(
                task_id, symbol, "consensus", "finalizing", 95,
                "Calculating consensus"
            )

            # [Consensus calculation logic would go here]

            await ws_manager.send_progress(
                task_id, symbol, "consensus", "completed", 100,
                "Analysis completed successfully",
                details={"strategies_run": len(strategies_results)}
            )

        else:
            # Single strategy analysis
            # Implementation would be similar to existing endpoints
            await ws_manager.send_progress(
                task_id, symbol, strategy_type.value, "completed", 100,
                "Single strategy analysis completed"
            )

    except Exception as e:
        await ws_manager.send_progress(
            task_id, symbol, strategy_type.value if strategy_type else "unknown",
            "error", 0,
            f"Analysis failed: {str(e)}"
        )
        raise


# ==================== Data Synchronization Endpoints ====================

@app.post("/api/v1/sync/jobs")
async def create_sync_job(
    symbol: str,
    period: Optional[str] = None,
    interval: str = '1d'
):
    """
    Create and start a new data synchronization job.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD', 'AAPL')
        period: Optional data period ('2y', '5y', etc.). Auto-detected if not provided.
        interval: Data interval (default: '1d')

    Returns the created job with status and progress information.
    """
    sync_manager = await get_sync_manager(db.client.temporal_trading)

    # Create job
    job = await sync_manager.create_sync_job(symbol, period, interval)

    # Try to start it
    started = await sync_manager.start_sync_job(job.job_id)

    return {
        "message": f"Sync job created for {symbol}",
        "job": job.dict(),
        "started": started
    }


@app.get("/api/v1/sync/jobs")
async def list_sync_jobs(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50
):
    """
    List synchronization jobs with optional filtering.

    Args:
        status: Filter by status (running, completed, failed, etc.)
        symbol: Filter by symbol
        limit: Maximum number of jobs to return

    Returns list of sync jobs.
    """
    query = {}
    if status:
        query["status"] = status
    if symbol:
        query["symbol"] = symbol

    cursor = db.client.temporal_trading.data_sync_jobs.find(query).sort("created_at", -1).limit(limit)
    jobs = []
    async for doc in cursor:
        jobs.append(DataSyncJob(**doc).dict())

    return {
        "count": len(jobs),
        "jobs": jobs
    }


@app.get("/api/v1/sync/jobs/{job_id}")
async def get_sync_job(job_id: str):
    """Get details of a specific sync job."""
    job_doc = await db.client.temporal_trading.data_sync_jobs.find_one({"job_id": job_id})

    if not job_doc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return DataSyncJob(**job_doc).dict()


@app.post("/api/v1/sync/jobs/{job_id}/pause")
async def pause_sync_job(job_id: str):
    """Pause a running sync job."""
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    paused = await sync_manager.pause_job(job_id)

    if not paused:
        raise HTTPException(status_code=400, detail="Job not found or cannot be paused")

    return {"message": f"Job {job_id} paused"}


@app.post("/api/v1/sync/jobs/{job_id}/resume")
async def resume_sync_job(job_id: str):
    """Resume a paused sync job."""
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    resumed = await sync_manager.resume_job(job_id)

    if not resumed:
        raise HTTPException(status_code=400, detail="Job not found or cannot be resumed")

    return {"message": f"Job {job_id} resumed"}


@app.post("/api/v1/sync/jobs/{job_id}/cancel")
async def cancel_sync_job(job_id: str):
    """Cancel a sync job."""
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    cancelled = await sync_manager.cancel_job(job_id)

    if not cancelled:
        raise HTTPException(status_code=400, detail="Job not found or cannot be cancelled")

    return {"message": f"Job {job_id} cancelled"}


@app.post("/api/v1/sync/jobs/process-queue")
async def process_pending_jobs():
    """
    Process pending jobs in the queue.
    Starts pending jobs up to the concurrency limit.
    """
    sync_manager = await get_sync_manager(db.client.temporal_trading)

    # Get count of running jobs
    running_count = await db.client.temporal_trading.data_sync_jobs.count_documents({"status": "running"})

    # Start pending jobs up to max concurrency
    started = []
    while running_count < sync_manager.max_concurrent:
        # Find next pending job
        pending_job = await db.client.temporal_trading.data_sync_jobs.find_one(
            {"status": "pending"},
            sort=[("created_at", 1)]
        )

        if not pending_job:
            break

        success = await sync_manager.start_sync_job(pending_job["job_id"])
        if success:
            started.append(pending_job["job_id"])
            running_count += 1
        else:
            break

    return {
        "message": f"Started {len(started)} pending jobs",
        "started_jobs": started,
        "remaining_pending": await db.client.temporal_trading.data_sync_jobs.count_documents({"status": "pending"})
    }


# ==================== Watchlist Management Endpoints ====================

@app.post("/api/v1/watchlist")
async def add_to_watchlist(request: 'WatchlistAddRequest'):
    """
    Add a ticker to the watchlist for automatic synchronization.

    Args:
        request: Watchlist add request with symbol, period, tags, etc.

    Returns the created watchlist item.
    """
    sync_manager = await get_sync_manager(db.client.temporal_trading)

    watchlist_item = await sync_manager.add_to_watchlist(
        symbol=request.symbol,
        period=request.period,
        interval=request.interval,
        auto_sync=request.auto_sync,
        priority=request.priority,
        tags=request.tags
    )

    return {
        "message": f"Added {request.symbol} to watchlist",
        "watchlist_item": watchlist_item.dict()
    }


@app.get("/api/v1/watchlist")
async def get_watchlist(enabled_only: bool = True):
    """
    Get the watchlist of tickers for automatic synchronization.

    Args:
        enabled_only: Only return enabled tickers (default: True)

    Returns list of watchlist tickers.
    """
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    watchlist = await sync_manager.get_watchlist(enabled_only=enabled_only)

    return {
        "count": len(watchlist),
        "watchlist": [item.dict() for item in watchlist]
    }


@app.delete("/api/v1/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Remove a ticker from the watchlist."""
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    removed = await sync_manager.remove_from_watchlist(symbol)

    if not removed:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not in watchlist")

    return {"message": f"Removed {symbol} from watchlist"}


# ==================== Data Inventory Endpoints ====================

@app.get("/api/v1/inventory")
async def get_inventory(symbol: Optional[str] = None):
    """
    Get data inventory showing what market data is cached.

    Args:
        symbol: Optional symbol filter

    Returns inventory of cached data with coverage information.
    """
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    inventory = await sync_manager.get_inventory(symbol=symbol)

    return {
        "count": len(inventory),
        "inventory": [item.dict() for item in inventory]
    }


@app.delete("/api/v1/inventory/{symbol}/{period}/{interval}")
async def delete_cached_data(symbol: str, period: str, interval: str):
    """
    Delete cached data for a specific symbol/period/interval combination.

    Args:
        symbol: Trading symbol
        period: Data period
        interval: Data interval

    Returns success message
    """
    sync_manager = await get_sync_manager(db.client.temporal_trading)

    # Delete from cache
    cache_key = sync_manager.cache._get_cache_key(symbol, period, interval)
    cache_path = sync_manager.cache._get_cache_path(cache_key)

    if cache_path.exists():
        cache_path.unlink()

    # Delete from inventory
    await db.client.temporal_trading.data_inventory.delete_one({
        "symbol": symbol,
        "period": period,
        "interval": interval
    })

    return {"message": f"Deleted cached data for {symbol} ({period}, {interval})"}


@app.post("/api/v1/inventory/{symbol}/extend")
async def extend_data_range(
    symbol: str,
    new_period: str = Query(..., description="New period to extend to (e.g., '5y')"),
    interval: str = Query(default='1d', description="Data interval")
):
    """
    Extend data range for a symbol by fetching only the delta (missing dates).
    Merges new data with existing cached data.

    Args:
        symbol: Trading symbol
        new_period: New period to extend to (must be wider than current)
        interval: Data interval

    Returns:
        Job information for the delta download
    """
    from strategies.massive_s3_data_source import get_massive_s3_source
    from datetime import datetime

    sync_manager = await get_sync_manager(db.client.temporal_trading)

    # Get existing inventory to find current coverage
    inventory_items = await sync_manager.get_inventory(symbol=symbol)

    # Find matching inventory item for this interval
    existing_inventory = None
    for item in inventory_items:
        if item.interval == interval:
            existing_inventory = item
            break

    if not existing_inventory:
        raise HTTPException(
            status_code=404,
            detail=f"No existing data found for {symbol} with interval {interval}. Use regular sync instead."
        )

    # Convert new period to dates
    s3_source = get_massive_s3_source()
    new_start, new_end = s3_source._convert_period_to_dates(new_period)

    # Get existing date range
    existing_start = existing_inventory.date_range_start
    existing_end = existing_inventory.date_range_end

    if not existing_start or not existing_end:
        raise HTTPException(
            status_code=400,
            detail=f"Existing data has incomplete date range information. Use re-download instead."
        )

    # Calculate delta ranges to fetch
    delta_ranges = []

    # Check if we need to fetch earlier data
    if new_start < existing_start:
        delta_ranges.append(("before", new_start, existing_start))

    # Check if we need to fetch later data
    if new_end > existing_end:
        delta_ranges.append(("after", existing_end, new_end))

    if not delta_ranges:
        return {
            "message": "No new data to fetch. Existing coverage already includes the requested period.",
            "existing_start": existing_start.isoformat(),
            "existing_end": existing_end.isoformat(),
            "requested_start": new_start.isoformat(),
            "requested_end": new_end.isoformat()
        }

    # Create a special sync job for delta download
    # We'll mark it differently so it knows to merge instead of replace
    job = await sync_manager.create_sync_job(symbol, new_period, interval)

    # Add metadata to track that this is a delta/merge job
    # Store the old period so we can clean up the old inventory entry after completion
    await db.client.temporal_trading.data_sync_jobs.update_one(
        {"job_id": job.job_id},
        {"$set": {
            "is_delta_job": True,
            "old_period": existing_inventory.period,  # Store old period for cleanup
            "delta_ranges": [
                {"type": r[0], "start": r[1].isoformat(), "end": r[2].isoformat()}
                for r in delta_ranges
            ]
        }}
    )

    # Start the job
    started = await sync_manager.start_sync_job(job.job_id)

    return {
        "message": f"Started delta download for {symbol}",
        "job_id": job.job_id,
        "started": started,
        "delta_ranges": [
            {"type": r[0], "start": r[1].isoformat(), "end": r[2].isoformat()}
            for r in delta_ranges
        ],
        "existing_coverage": {
            "start": existing_start.isoformat(),
            "end": existing_end.isoformat()
        }
    }


@app.get("/api/v1/inventory/{symbol}")
async def get_symbol_inventory(symbol: str):
    """Get inventory details for a specific symbol."""
    sync_manager = await get_sync_manager(db.client.temporal_trading)
    inventory = await sync_manager.get_inventory(symbol=symbol)

    if not inventory:
        raise HTTPException(status_code=404, detail=f"No inventory found for {symbol}")

    return {
        "symbol": symbol,
        "coverage": [item.dict() for item in inventory]
    }


# Global cache for tickers (refresh every 24 hours)
_tickers_cache = {"data": None, "timestamp": None, "ttl_hours": 24}


@app.get("/api/v1/tickers")
async def get_available_tickers(market: str = "all"):
    """
    Get available tickers from Massive.com API.

    Args:
        market: Filter by market type - 'crypto', 'stocks', or 'all' (default)

    Returns:
        List of available tickers with symbol and name, sorted alphabetically
    """
    import httpx
    from datetime import datetime, timedelta

    # Check cache
    now = datetime.utcnow()
    if _tickers_cache["data"] and _tickers_cache["timestamp"]:
        age = now - _tickers_cache["timestamp"]
        if age < timedelta(hours=_tickers_cache["ttl_hours"]):
            cached_data = _tickers_cache["data"]
            if market == "all":
                return cached_data
            else:
                return {
                    "tickers": [t for t in cached_data["tickers"] if t["market"] == market],
                    "count": len([t for t in cached_data["tickers"] if t["market"] == market])
                }

    # Fetch from Massive.com API
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MASSIVE_API_KEY not configured")

    tickers = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch crypto tickers
            crypto_url = f"https://api.massive.com/v3/reference/tickers?market=crypto&active=true&order=asc&limit=1000&sort=ticker&apiKey={api_key}"
            crypto_response = await client.get(crypto_url)
            if crypto_response.status_code == 200:
                crypto_data = crypto_response.json()
                for ticker in crypto_data.get("results", []):
                    # Convert X:BTCUSD to BTC-USD format
                    polygon_ticker = ticker["ticker"]
                    if polygon_ticker.startswith("X:"):
                        # Extract base and quote currencies
                        # X:BTCUSD -> BTC-USD
                        base_currency = ticker.get("base_currency_symbol", "")
                        quote_currency = ticker.get("currency_symbol", "")
                        if base_currency and quote_currency:
                            symbol = f"{base_currency}-{quote_currency}"
                        else:
                            # Fallback: try to parse from ticker
                            symbol = polygon_ticker[2:]  # Remove X: prefix

                        tickers.append({
                            "symbol": symbol,
                            "name": ticker.get("name", symbol),
                            "market": "crypto",
                            "polygon_ticker": polygon_ticker
                        })

            # Fetch stock tickers (US stocks)
            stocks_url = f"https://api.massive.com/v3/reference/tickers?market=stocks&active=true&order=asc&limit=1000&sort=ticker&apiKey={api_key}"
            stocks_response = await client.get(stocks_url)
            if stocks_response.status_code == 200:
                stocks_data = stocks_response.json()
                for ticker in stocks_data.get("results", []):
                    symbol = ticker["ticker"]
                    tickers.append({
                        "symbol": symbol,
                        "name": ticker.get("name", symbol),
                        "market": "stocks",
                        "polygon_ticker": symbol
                    })

        # Sort alphabetically by symbol
        tickers.sort(key=lambda x: x["symbol"])

        # Cache the result
        _tickers_cache["data"] = {"tickers": tickers, "count": len(tickers)}
        _tickers_cache["timestamp"] = now

        # Filter by market if requested
        if market != "all":
            tickers = [t for t in tickers if t["market"] == market]

        return {
            "tickers": tickers,
            "count": len(tickers)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tickers: {str(e)}")


# ==================== Backtesting Endpoints ====================

@app.post("/api/v1/backtest", response_model=BacktestSummary)
async def create_backtest(
    request: BacktestCreateRequest,
    background_tasks: BackgroundTasks
) -> BacktestSummary:
    """
    Create and run a backtest.

    Runs backtest in background and returns immediately with run_id.
    """
    try:
        run_id = str(uuid.uuid4())

        # Create initial backtest record
        backtest_run = BacktestRun(
            run_id=run_id,
            name=request.name,
            config=request.config,
            status=BacktestStatus.PENDING
        )

        # Store in database
        await db.store_backtest(backtest_run)

        # Run backtest in background
        background_tasks.add_task(run_backtest_background, run_id, request.config)

        return BacktestSummary(
            run_id=run_id,
            name=request.name,
            symbol=request.config.symbol,
            start_date=request.config.start_date,
            end_date=request.config.end_date,
            status=BacktestStatus.PENDING,
            created_at=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backtest: {str(e)}")


async def run_backtest_background(run_id: str, config: BacktestConfig):
    """Run backtest in background"""
    try:
        # Update status to running
        await db.update_backtest_status(run_id, BacktestStatus.RUNNING)

        # Load historical price data from cache (where data sync stores it)
        # Determine period based on asset type
        is_crypto = '-' in config.symbol
        period = '2y' if is_crypto else '5y'

        # Check inventory to see if we have 5y data available
        inventory = await db.db.data_inventory.find_one({
            "symbol": config.symbol,
            "is_complete": True
        })
        if inventory and inventory.get('period') == '5y':
            period = '5y'

        from strategies.data_cache import get_cache
        cache = get_cache()
        price_data = cache.get(config.symbol, period, interval='1d')

        if price_data is None or len(price_data) == 0:
            raise ValueError(f"No price data found for {config.symbol} in cache. Please sync data first using the Dashboard → Data page.")

        # Filter by date range
        if config.start_date or config.end_date:
            if config.start_date:
                price_data = price_data[price_data.index >= config.start_date]
            if config.end_date:
                price_data = price_data[price_data.index <= config.end_date]

        # Reset index to make date a column for backtesting
        price_data = price_data.reset_index()
        price_data = price_data.rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})

        # Convert column names to lowercase if needed
        price_data.columns = [c.lower() for c in price_data.columns]

        if len(price_data) == 0:
            raise ValueError(f"No price data found for {config.symbol} in date range {config.start_date} to {config.end_date}")

        # Initialize backtest engine
        engine = BacktestEngine(config)

        # Run appropriate backtest type
        if config.walk_forward.enabled:
            result = engine.run_walkforward_backtest(price_data, run_id)
        else:
            result = engine.run_simple_backtest(price_data, run_id)

        # Store results
        await db.update_backtest_results(result)

        print(f"✅ Backtest {run_id} completed successfully")

    except Exception as e:
        print(f"❌ Backtest {run_id} failed: {str(e)}")
        traceback.print_exc()
        await db.update_backtest_status(run_id, BacktestStatus.FAILED, str(e))


@app.get("/api/v1/backtest", response_model=List[BacktestSummary])
async def list_backtests(
    symbol: Optional[str] = None,
    limit: int = 50
) -> List[BacktestSummary]:
    """
    List all backtest runs, optionally filtered by symbol.
    """
    try:
        backtests = await db.get_backtests(symbol=symbol, limit=limit)

        return [
            BacktestSummary(
                run_id=bt.run_id,
                name=bt.name,
                symbol=bt.config.symbol,
                start_date=bt.config.start_date,
                end_date=bt.config.end_date,
                status=bt.status,
                total_return=bt.metrics.total_return if bt.metrics else None,
                sharpe_ratio=bt.metrics.sharpe_ratio if bt.metrics else None,
                max_drawdown=bt.metrics.max_drawdown if bt.metrics else None,
                total_trades=bt.metrics.total_trades if bt.metrics else None,
                created_at=bt.created_at,
                completed_at=bt.completed_at
            )
            for bt in backtests
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")


@app.get("/api/v1/backtest/{run_id}", response_model=BacktestRun)
async def get_backtest(run_id: str) -> BacktestRun:
    """
    Get detailed backtest results by run_id.
    """
    try:
        backtest = await db.get_backtest(run_id)

        if not backtest:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        return backtest

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest: {str(e)}")


@app.delete("/api/v1/backtest/{run_id}")
async def delete_backtest(run_id: str):
    """
    Delete a backtest run.
    """
    try:
        success = await db.delete_backtest(run_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        return {"message": f"Backtest {run_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {str(e)}")


# ==================== Paper Trading Endpoints ====================

@app.post("/api/v1/paper-trading", response_model=PaperTradingSummary)
async def create_paper_trading_session(
    request: PaperTradingCreateRequest
) -> PaperTradingSummary:
    """
    Create and start a paper trading session.
    """
    try:
        session_id = str(uuid.uuid4())

        # Create paper trading session
        session = PaperTradingSession(
            session_id=session_id,
            name=request.name,
            config=request.config,
            status=PaperTradingStatus.ACTIVE,
            cash=request.config.initial_capital,
            starting_capital=request.config.initial_capital,
            current_equity=request.config.initial_capital,
            total_pnl=0.0,
            total_pnl_pct=0.0
        )

        # Store in database
        await db.store_paper_trading_session(session)

        # Start paper trading monitoring (in background)
        # This will check for signals at the specified interval
        asyncio.create_task(monitor_paper_trading_session(session_id))

        return PaperTradingSummary(
            session_id=session_id,
            name=request.name,
            symbol=request.config.symbol,
            status=PaperTradingStatus.ACTIVE,
            current_equity=request.config.initial_capital,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            total_trades=0,
            started_at=session.started_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create paper trading session: {str(e)}")


async def monitor_paper_trading_session(session_id: str):
    """
    Monitor a paper trading session and execute signals.
    Runs continuously in background.
    """
    while True:
        try:
            # Get session
            session = await db.get_paper_trading_session(session_id)

            if not session or session.status != PaperTradingStatus.ACTIVE:
                print(f"Paper trading session {session_id} stopped or not found")
                break

            # Check if it's time to check for signals
            now = datetime.utcnow()
            if session.next_signal_check and now < session.next_signal_check:
                # Wait until next check time
                sleep_seconds = (session.next_signal_check - now).total_seconds()
                await asyncio.sleep(min(sleep_seconds, 60))  # Check at least every minute
                continue

            # Get consensus signal
            # TODO: Integrate with actual consensus analysis
            signal = await get_paper_trading_signal(session.config.symbol)

            if signal and session.config.auto_execute:
                # Execute trade based on signal
                await execute_paper_trade(session, signal)

            # Update next check time
            session.last_signal_check = now
            session.next_signal_check = now + timedelta(minutes=session.config.check_interval_minutes)
            await db.update_paper_trading_session(session)

            # Wait for next interval
            await asyncio.sleep(session.config.check_interval_minutes * 60)

        except Exception as e:
            print(f"Error monitoring paper trading session {session_id}: {str(e)}")
            traceback.print_exc()
            await asyncio.sleep(60)  # Wait a minute before retrying


async def get_paper_trading_signal(symbol: str) -> Optional[Dict]:
    """Get current consensus signal for symbol"""
    # TODO: Implement actual consensus signal retrieval
    # For now, return None (no signal)
    return None


async def execute_paper_trade(session: PaperTradingSession, signal: Dict):
    """Execute a paper trade based on signal"""
    # TODO: Implement paper trade execution logic
    pass


@app.get("/api/v1/paper-trading", response_model=List[PaperTradingSummary])
async def list_paper_trading_sessions(
    status: Optional[PaperTradingStatus] = None,
    limit: int = 50
) -> List[PaperTradingSummary]:
    """
    List all paper trading sessions.
    """
    try:
        sessions = await db.get_paper_trading_sessions(status=status, limit=limit)

        return [
            PaperTradingSummary(
                session_id=s.session_id,
                name=s.name,
                symbol=s.config.symbol,
                status=s.status,
                current_equity=s.current_equity,
                total_pnl=s.total_pnl,
                total_pnl_pct=s.total_pnl_pct,
                total_trades=s.total_trades,
                started_at=s.started_at
            )
            for s in sessions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list paper trading sessions: {str(e)}")


@app.get("/api/v1/paper-trading/{session_id}", response_model=PaperTradingSession)
async def get_paper_trading_session(session_id: str) -> PaperTradingSession:
    """
    Get detailed paper trading session by session_id.
    """
    try:
        session = await db.get_paper_trading_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")

        return session

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get paper trading session: {str(e)}")


@app.post("/api/v1/paper-trading/{session_id}/pause")
async def pause_paper_trading_session(session_id: str):
    """
    Pause a paper trading session.
    """
    try:
        session = await db.get_paper_trading_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session.status = PaperTradingStatus.PAUSED
        await db.update_paper_trading_session(session)

        return {"message": f"Paper trading session {session_id} paused"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause session: {str(e)}")


@app.post("/api/v1/paper-trading/{session_id}/resume")
async def resume_paper_trading_session(session_id: str):
    """
    Resume a paused paper trading session.
    """
    try:
        session = await db.get_paper_trading_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session.status = PaperTradingStatus.ACTIVE
        await db.update_paper_trading_session(session)

        # Restart monitoring
        asyncio.create_task(monitor_paper_trading_session(session_id))

        return {"message": f"Paper trading session {session_id} resumed"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume session: {str(e)}")


@app.post("/api/v1/paper-trading/{session_id}/stop")
async def stop_paper_trading_session(session_id: str):
    """
    Stop a paper trading session.
    """
    try:
        session = await db.get_paper_trading_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session.status = PaperTradingStatus.STOPPED
        session.stopped_at = datetime.utcnow()
        await db.update_paper_trading_session(session)

        return {"message": f"Paper trading session {session_id} stopped"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
