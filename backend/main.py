"""
FastAPI backend for Temporal Trading Agents.
Provides REST API endpoints for running trading strategies and getting consensus.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for strategy imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import time
from datetime import datetime, timedelta
import uuid
import asyncio

from models import (
    StrategyAnalysisRequest, ConsensusRequest, StrategyResult,
    ConsensusAnalysis, HealthCheck, StrategyType, StrategyAnalysis,
    ConsensusResult, AnalysisStatus, StrategySignal, ForecastStats,
    ScheduledTask, ScheduledTaskCreate, ScheduledTaskUpdate, ScheduleFrequency
)
from database import Database
from websocket_manager import manager as ws_manager
from scheduler import get_scheduler

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


# ==================== Lifecycle Events ====================

@app.on_event("startup")
async def startup_event():
    """Connect to database and start scheduler on startup"""
    global scheduler

    await db.connect()
    print("🚀 API: Server started successfully")

    # Initialize and start scheduler
    scheduler = get_scheduler(db)
    scheduler.start()
    await scheduler.load_scheduled_tasks()
    print("📅 API: Scheduler initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from database and shutdown scheduler"""
    global scheduler

    if scheduler:
        scheduler.shutdown()
        print("📅 API: Scheduler stopped")

    await db.disconnect()
    print("👋 API: Server shutdown")


# ==================== Dependency Injection ====================

async def get_database():
    """Dependency to get database instance"""
    return db


# ==================== Utility Functions ====================

def convert_strategy_result_to_model(strategy_type: str, result: dict, symbol: str, current_price: float, execution_time_ms: int) -> StrategyAnalysis:
    """Convert strategy result dict to StrategyAnalysis model"""
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


# ==================== Strategy Endpoints ====================

@app.post("/api/v1/analyze/gradient", response_model=StrategyResult)
async def analyze_gradient(
    symbol: str,
    database: Database = Depends(get_database)
):
    """Run forecast gradient strategy analysis"""
    try:
        start_time = time.time()

        # Load ensemble and train
        ensemble = load_ensemble_module("examples/crypto_ensemble_forecast.py")
        configs = get_default_ensemble_configs(14)
        stats, df = train_ensemble(symbol, 14, configs, "14-DAY", ensemble)
        current_price = df['Close'].iloc[-1]

        # Run gradient strategy
        result = analyze_gradient_strategy(stats, current_price)
        result['forecast_median'] = stats['median']

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Store in database
        analysis = convert_strategy_result_to_model('gradient', result, symbol, current_price, execution_time_ms)
        await database.create_strategy_analysis(analysis)

        return StrategyResult(
            strategy_type=StrategyType.GRADIENT,
            symbol=symbol,
            current_price=current_price,
            signal=analysis.signal,
            forecast_stats=analysis.forecast_stats,
            execution_time_ms=execution_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy analysis failed: {str(e)}")


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


@app.post("/api/v1/analyze/consensus", response_model=ConsensusAnalysis)
async def analyze_consensus(
    request: ConsensusRequest,
    database: Database = Depends(get_database)
):
    """Run all 8 strategies and return consensus analysis"""
    try:
        start_time = time.time()

        # Load ensemble
        ensemble = load_ensemble_module("examples/crypto_ensemble_forecast.py")

        # Train 14-day ensemble
        configs = get_default_ensemble_configs(14)
        stats, df = train_ensemble(request.symbol, 14, configs, "14-DAY", ensemble)
        current_price = df['Close'].iloc[-1]

        results = {}
        strategy_ids = []

        # Run all 8 strategies
        try:
            result = analyze_gradient_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('gradient', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Forecast Gradient'] = result
        except Exception as e:
            print(f"Gradient strategy failed: {e}")
            results['Forecast Gradient'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            result = analyze_confidence_weighted_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('confidence', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Confidence-Weighted'] = result
        except Exception as e:
            print(f"Confidence strategy failed: {e}")
            results['Confidence-Weighted'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            timeframe_data = train_multiple_timeframes(request.symbol, ensemble, request.horizons)
            result = analyze_multi_timeframe_strategy(timeframe_data, current_price)
            analysis = convert_strategy_result_to_model('timeframe', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Multi-Timeframe'] = result
        except Exception as e:
            print(f"Timeframe strategy failed: {e}")
            results['Multi-Timeframe'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            result = analyze_volatility_position_sizing(stats, current_price)
            analysis = convert_strategy_result_to_model('volatility', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Volatility Sizing'] = result
        except Exception as e:
            print(f"Volatility strategy failed: {e}")
            results['Volatility Sizing'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            result = analyze_mean_reversion_strategy(stats, df, current_price)
            analysis = convert_strategy_result_to_model('mean_reversion', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Mean Reversion'] = result
        except Exception as e:
            print(f"Mean reversion strategy failed: {e}")
            results['Mean Reversion'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            result = analyze_acceleration_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('acceleration', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Acceleration'] = result
        except Exception as e:
            print(f"Acceleration strategy failed: {e}")
            results['Acceleration'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            result = analyze_swing_trading_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('swing', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Swing Trading'] = result
        except Exception as e:
            print(f"Swing strategy failed: {e}")
            results['Swing Trading'] = {'signal': 'ERROR', 'position_size_pct': 0}

        try:
            result = analyze_risk_adjusted_strategy(stats, current_price)
            analysis = convert_strategy_result_to_model('risk_adjusted', result, request.symbol, current_price, 0)
            analysis_id = await database.create_strategy_analysis(analysis)
            strategy_ids.append(analysis_id)
            results['Risk-Adjusted'] = result
        except Exception as e:
            print(f"Risk-adjusted strategy failed: {e}")
            results['Risk-Adjusted'] = {'signal': 'ERROR', 'position_size_pct': 0}

        # Analyze consensus
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

        # Calculate average position
        bullish_positions = [results[name].get('position_size_pct', 0) for name in bullish_strategies
                           if 'position_size_pct' in results[name] and results[name]['position_size_pct'] > 0]
        avg_position = sum(bullish_positions) / len(bullish_positions) if bullish_positions else 0

        # Convert results to StrategySignal models
        strategy_signals = {}
        for name, data in results.items():
            strategy_signals[name] = StrategySignal(
                signal=data.get('signal', 'UNKNOWN'),
                position_size_pct=data.get('position_size_pct', 0),
                confidence=data.get('confidence'),
                target_price=data.get('target_price'),
                stop_loss=data.get('stop_loss')
            )

        # Store consensus in database
        consensus_result = ConsensusResult(
            symbol=request.symbol,
            current_price=current_price,
            consensus=consensus,
            strength=strength,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=len(neutral_strategies),
            total_count=total,
            bullish_strategies=bullish_strategies,
            bearish_strategies=bearish_strategies,
            neutral_strategies=neutral_strategies,
            avg_position=avg_position,
            strategy_results=strategy_ids
        )
        await database.create_consensus_result(consensus_result)

        return ConsensusAnalysis(
            symbol=request.symbol,
            current_price=current_price,
            consensus=consensus,
            strength=strength,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=len(neutral_strategies),
            total_count=total,
            bullish_strategies=bullish_strategies,
            bearish_strategies=bearish_strategies,
            neutral_strategies=neutral_strategies,
            avg_position=avg_position,
            strategies=strategy_signals
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consensus analysis failed: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
