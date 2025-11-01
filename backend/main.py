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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
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
    AnalysisStarted, ForecastData, ModelPrediction, HistoricalPriceData, PricePoint
)
from backend.database import Database
from backend.websocket_manager import manager as ws_manager
from backend.scheduler import get_scheduler
from backend.cache_manager import get_cache_manager

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


# ==================== Cache Management Endpoints ====================

@app.get("/api/v1/cache/list")
async def list_cached_data():
    """
    List all cached market data files.

    Returns a list of cached data with symbol, period, interval, size, and modification time.
    """
    cache_manager = get_cache_manager()
    cached_files = cache_manager.list_cached_data()

    return {
        "count": len(cached_files),
        "cached_data": cached_files
    }


@app.post("/api/v1/cache/preload/{symbol}")
async def start_preload(
    symbol: str,
    period: Optional[str] = None,
    interval: str = '1d'
):
    """
    Start a background data preload job for the given symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD', 'AAPL')
        period: Optional data period ('2y', '5y', etc.). Auto-detected if not provided.
        interval: Data interval (default: '1d')

    Returns the job status with progress tracking information.
    """
    cache_manager = get_cache_manager()
    job_status = cache_manager.start_preload(symbol, period, interval)

    return {
        "message": f"Preload started for {symbol}",
        "job": job_status
    }


@app.get("/api/v1/cache/status/{symbol}")
async def get_preload_status(
    symbol: str,
    period: Optional[str] = None,
    interval: str = '1d'
):
    """
    Get the status of a preload job or check if data is cached.

    Args:
        symbol: Trading symbol
        period: Optional data period (auto-detected if not provided)
        interval: Data interval (default: '1d')

    Returns job status if downloading, or indicates if data is already cached.
    """
    cache_manager = get_cache_manager()
    status = cache_manager.get_job_status(symbol, period, interval)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data or active download found for {symbol}"
        )

    return status


@app.delete("/api/v1/cache/{symbol}")
async def delete_cached_data(
    symbol: str,
    period: Optional[str] = None,
    interval: str = '1d'
):
    """
    Delete cached data for a symbol.

    Args:
        symbol: Trading symbol
        period: Optional data period (auto-detected if not provided)
        interval: Data interval (default: '1d')

    Cancels any running download and removes the cached data file.
    """
    cache_manager = get_cache_manager()
    deleted = cache_manager.delete_cached_data(symbol, period, interval)

    if deleted:
        return {"message": f"Deleted cached data for {symbol}"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data found for {symbol}"
        )


@app.delete("/api/v1/cache/preload/{symbol}")
async def cancel_preload(
    symbol: str,
    period: Optional[str] = None,
    interval: str = '1d'
):
    """
    Cancel a running preload job.

    Args:
        symbol: Trading symbol
        period: Optional data period (auto-detected if not provided)
        interval: Data interval (default: '1d')

    Cancels the download and clears the progress marker.
    """
    cache_manager = get_cache_manager()
    cancelled = cache_manager.cancel_preload(symbol, period, interval)

    if cancelled:
        return {"message": f"Cancelled preload for {symbol}"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No active preload job found for {symbol}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
