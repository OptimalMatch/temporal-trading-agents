"""
Auto-Optimize Manager - Backend-orchestrated multi-stage parameter optimization

This module orchestrates the complete 4-stage auto-optimization workflow:
  Stage 1: Min Edge Discovery
  Stage 2: Position Sizing
  Stage 3: Strategy Selection
  Stage 4: Fine Tuning

The workflow runs entirely on the backend, survives browser disconnects,
and stores all state in MongoDB.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.models import (
    AutoOptimizeRun, AutoOptimizeConfig, AutoOptimizeStatus,
    AutoOptimizeStageResult, OptimizationRun, BacktestConfig,
    ParameterGrid, OptimizableParams, BacktestMetrics
)
from backend.parameter_optimizer import ParameterOptimizer

logger = logging.getLogger(__name__)


class AutoOptimizeManager:
    """Manages backend-orchestrated auto-optimization workflows"""

    def __init__(self, db: AsyncIOMotorDatabase, optimizer: ParameterOptimizer):
        self.db = db
        self.optimizer = optimizer
        self.collection = db.auto_optimize_runs

    async def create_auto_optimize(self, name: str, config: AutoOptimizeConfig) -> AutoOptimizeRun:
        """Create and start a new auto-optimize workflow"""
        run = AutoOptimizeRun(
            name=name,
            config=config,
            status=AutoOptimizeStatus.PENDING
        )

        # Store in MongoDB
        await self.collection.insert_one(run.model_dump())
        logger.info(f"Created auto-optimize run {run.auto_optimize_id}: {name}")

        # Start workflow in background
        asyncio.create_task(self._run_workflow(run.auto_optimize_id))

        return run

    async def get_auto_optimize(self, auto_optimize_id: str) -> Optional[AutoOptimizeRun]:
        """Get auto-optimize run by ID"""
        doc = await self.collection.find_one({"auto_optimize_id": auto_optimize_id})
        if doc:
            return AutoOptimizeRun(**doc)
        return None

    async def list_auto_optimizes(self, limit: int = 50) -> List[AutoOptimizeRun]:
        """List recent auto-optimize runs"""
        cursor = self.collection.find().sort("created_at", -1).limit(limit)
        runs = []
        async for doc in cursor:
            runs.append(AutoOptimizeRun(**doc))
        return runs

    async def cancel_auto_optimize(self, auto_optimize_id: str) -> bool:
        """Cancel a running auto-optimize workflow"""
        run = await self.get_auto_optimize(auto_optimize_id)
        if not run:
            return False

        if run.status != AutoOptimizeStatus.RUNNING:
            return False

        # Update status to cancelled
        run.status = AutoOptimizeStatus.CANCELLED
        run.completed_at = datetime.now(timezone.utc)
        run.error_message = "Cancelled by user"
        await self._update_run(run)

        logger.info(f"Auto-optimize run {auto_optimize_id} cancelled")
        return True

    async def _is_cancelled(self, auto_optimize_id: str) -> bool:
        """Check if workflow has been cancelled"""
        run = await self.get_auto_optimize(auto_optimize_id)
        return run.status == AutoOptimizeStatus.CANCELLED if run else False

    async def _run_workflow(self, auto_optimize_id: str):
        """Run the complete 4-stage workflow"""
        try:
            # Load run
            run = await self.get_auto_optimize(auto_optimize_id)
            if not run:
                logger.error(f"Auto-optimize run {auto_optimize_id} not found")
                return

            # Update status
            run.status = AutoOptimizeStatus.RUNNING
            run.started_at = datetime.now(timezone.utc)
            await self._update_run(run)

            logger.info(f"Starting auto-optimize workflow {auto_optimize_id}")

            # Stage 1: Min Edge Discovery
            stage1_result = await self._run_stage_1(run)
            run.stages.append(stage1_result)
            run.current_stage = 1
            await self._update_run(run)

            # Check for cancellation
            if await self._is_cancelled(auto_optimize_id):
                logger.info(f"Auto-optimize workflow {auto_optimize_id} cancelled after Stage 1")
                return

            # Stage 2: Position Sizing
            stage2_result = await self._run_stage_2(run, stage1_result)
            run.stages.append(stage2_result)
            run.current_stage = 2
            await self._update_run(run)

            # Check for cancellation
            if await self._is_cancelled(auto_optimize_id):
                logger.info(f"Auto-optimize workflow {auto_optimize_id} cancelled after Stage 2")
                return

            # Stage 3: Strategy Selection
            stage3_result = await self._run_stage_3(run, stage1_result, stage2_result)
            run.stages.append(stage3_result)
            run.current_stage = 3
            await self._update_run(run)

            # Check for cancellation
            if await self._is_cancelled(auto_optimize_id):
                logger.info(f"Auto-optimize workflow {auto_optimize_id} cancelled after Stage 3")
                return

            # Stage 4: Fine Tuning
            stage4_result = await self._run_stage_4(run, stage1_result, stage2_result, stage3_result)
            run.stages.append(stage4_result)
            run.current_stage = 4
            await self._update_run(run)

            # Finalize
            run.status = AutoOptimizeStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.execution_time_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)
            run.optimal_params = stage4_result.best_params
            run.optimal_metrics = stage4_result.best_metrics
            run.baseline_sharpe = run.stages[0].best_metrics.sharpe_ratio if run.stages else None
            if run.baseline_sharpe and run.baseline_sharpe > 0:
                run.improvement_pct = (stage4_result.best_metrics.sharpe_ratio / run.baseline_sharpe - 1) * 100

            await self._update_run(run)
            logger.info(f"Auto-optimize workflow {auto_optimize_id} completed successfully")

        except Exception as e:
            logger.error(f"Auto-optimize workflow {auto_optimize_id} failed: {e}", exc_info=True)
            run.status = AutoOptimizeStatus.FAILED
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            await self._update_run(run)

    async def _run_stage_1(self, run: AutoOptimizeRun) -> AutoOptimizeStageResult:
        """Stage 1: Min Edge Discovery"""
        logger.info(f"Running Stage 1: Min Edge Discovery for {run.auto_optimize_id}")

        stage_result = AutoOptimizeStageResult(
            stage_id=1,
            stage_name="Min Edge Discovery",
            optimization_id="",
            started_at=datetime.now(timezone.utc)
        )

        # Configure optimization
        opt_config = BacktestConfig(
            symbol=run.config.symbol,
            start_date=run.config.start_date,
            end_date=run.config.end_date,
            initial_capital=run.config.initial_capital,
            enabled_strategies=run.config.enabled_strategies,
            walk_forward_enabled=True,
            walk_forward_config={
                "train_window_days": 365,
                "test_window_days": 63,
                "retrain_frequency_days": 21
            }
        )

        param_grid = ParameterGrid(
            position_size_pct=[15.0],
            min_edge_bps=[25, 50, 75, 100, 125, 150, 175, 200],
            strong_buy_threshold=[0.75, 0.80],
            buy_threshold=[0.60, 0.65],
            moderate_buy_threshold=[0.50, 0.55],
            sell_threshold=[0.60],
            moderate_sell_threshold=[0.50]
        )

        # Run optimization
        opt_run = await self.optimizer.run_optimization(
            name=f"Auto-Opt Stage 1: {run.name}",
            base_config=opt_config,
            parameter_grid=param_grid,
            optimization_metric="sharpe_ratio",
            top_n_results=5
        )

        # Extract best result
        if opt_run.top_results:
            stage_result.optimization_id = opt_run.optimization_id
            stage_result.best_params = opt_run.top_results[0].parameters
            stage_result.best_metrics = opt_run.top_results[0].metrics
            stage_result.completed_at = datetime.now(timezone.utc)
        else:
            raise ValueError("Stage 1 optimization produced no results")

        logger.info(f"Stage 1 completed: best_min_edge={stage_result.best_params.min_edge_bps}")
        return stage_result

    async def _run_stage_2(self, run: AutoOptimizeRun, stage1: AutoOptimizeStageResult) -> AutoOptimizeStageResult:
        """Stage 2: Position Sizing (using Stage 1 winners)"""
        logger.info(f"Running Stage 2: Position Sizing for {run.auto_optimize_id}")

        stage_result = AutoOptimizeStageResult(
            stage_id=2,
            stage_name="Position Sizing",
            optimization_id="",
            started_at=datetime.now(timezone.utc)
        )

        opt_config = BacktestConfig(
            symbol=run.config.symbol,
            start_date=run.config.start_date,
            end_date=run.config.end_date,
            initial_capital=run.config.initial_capital,
            enabled_strategies=run.config.enabled_strategies,
            walk_forward_enabled=True,
            walk_forward_config={
                "train_window_days": 365,
                "test_window_days": 63,
                "retrain_frequency_days": 21
            }
        )

        # Use winners from Stage 1
        param_grid = ParameterGrid(
            position_size_pct=[5, 10, 15, 20, 25, 30],
            min_edge_bps=[stage1.best_params.min_edge_bps],
            strong_buy_threshold=[stage1.best_params.strong_buy_threshold],
            buy_threshold=[stage1.best_params.buy_threshold],
            moderate_buy_threshold=[stage1.best_params.moderate_buy_threshold],
            sell_threshold=[0.60],
            moderate_sell_threshold=[0.50]
        )

        opt_run = await self.optimizer.run_optimization(
            name=f"Auto-Opt Stage 2: {run.name}",
            base_config=opt_config,
            parameter_grid=param_grid,
            optimization_metric="sharpe_ratio",
            top_n_results=5
        )

        if opt_run.top_results:
            stage_result.optimization_id = opt_run.optimization_id
            stage_result.best_params = opt_run.top_results[0].parameters
            stage_result.best_metrics = opt_run.top_results[0].metrics
            stage_result.completed_at = datetime.now(timezone.utc)
        else:
            raise ValueError("Stage 2 optimization produced no results")

        logger.info(f"Stage 2 completed: best_position_size={stage_result.best_params.position_size_pct}")
        return stage_result

    async def _run_stage_3(
        self,
        run: AutoOptimizeRun,
        stage1: AutoOptimizeStageResult,
        stage2: AutoOptimizeStageResult
    ) -> AutoOptimizeStageResult:
        """Stage 3: Strategy Selection (testing each strategy individually)"""
        logger.info(f"Running Stage 3: Strategy Selection for {run.auto_optimize_id}")

        stage_result = AutoOptimizeStageResult(
            stage_id=3,
            stage_name="Strategy Selection",
            optimization_id="",
            started_at=datetime.now(timezone.utc)
        )

        # Test each strategy individually
        strategy_results = []
        for strategy in run.config.enabled_strategies:
            logger.info(f"  Testing strategy: {strategy}")

            opt_config = BacktestConfig(
                symbol=run.config.symbol,
                start_date=run.config.start_date,
                end_date=run.config.end_date,
                initial_capital=run.config.initial_capital,
                enabled_strategies=[strategy],  # Only this strategy
                walk_forward_enabled=True,
                walk_forward_config={
                    "train_window_days": 365,
                    "test_window_days": 63,
                    "retrain_frequency_days": 21
                }
            )

            param_grid = ParameterGrid(
                position_size_pct=[stage2.best_params.position_size_pct],
                min_edge_bps=[stage1.best_params.min_edge_bps],
                strong_buy_threshold=[0.75],
                buy_threshold=[0.60],
                moderate_buy_threshold=[0.50],
                sell_threshold=[0.60],
                moderate_sell_threshold=[0.50]
            )

            opt_run = await self.optimizer.run_optimization(
                name=f"Auto-Opt Stage 3: {strategy}",
                base_config=opt_config,
                parameter_grid=param_grid,
                optimization_metric="sharpe_ratio",
                top_n_results=1
            )

            if opt_run.top_results:
                strategy_results.append({
                    "strategy": strategy,
                    "sharpe": opt_run.top_results[0].metrics.sharpe_ratio,
                    "params": opt_run.top_results[0].parameters,
                    "metrics": opt_run.top_results[0].metrics
                })

        if not strategy_results:
            raise ValueError("Stage 3 produced no strategy results")

        # Sort by Sharpe ratio and take top 3
        strategy_results.sort(key=lambda x: x["sharpe"], reverse=True)
        best_strategy = strategy_results[0]

        stage_result.best_params = best_strategy["params"]
        stage_result.best_metrics = best_strategy["metrics"]
        stage_result.completed_at = datetime.now(timezone.utc)

        # Store top 3 strategies in optimization_id field (as JSON for now)
        top_3_strategies = [r["strategy"] for r in strategy_results[:3]]
        stage_result.optimization_id = ",".join(top_3_strategies)

        logger.info(f"Stage 3 completed: best_strategy={best_strategy['strategy']}, top_3={top_3_strategies}")
        return stage_result

    async def _run_stage_4(
        self,
        run: AutoOptimizeRun,
        stage1: AutoOptimizeStageResult,
        stage2: AutoOptimizeStageResult,
        stage3: AutoOptimizeStageResult
    ) -> AutoOptimizeStageResult:
        """Stage 4: Fine Tuning (narrow ranges around winners)"""
        logger.info(f"Running Stage 4: Fine Tuning for {run.auto_optimize_id}")

        stage_result = AutoOptimizeStageResult(
            stage_id=4,
            stage_name="Fine Tuning",
            optimization_id="",
            started_at=datetime.now(timezone.utc)
        )

        # Get top 3 strategies from Stage 3
        top_strategies = stage3.optimization_id.split(",")

        best_min_edge = stage1.best_params.min_edge_bps
        best_position_size = stage2.best_params.position_size_pct

        opt_config = BacktestConfig(
            symbol=run.config.symbol,
            start_date=run.config.start_date,
            end_date=run.config.end_date,
            initial_capital=run.config.initial_capital,
            enabled_strategies=top_strategies,
            walk_forward_enabled=True,
            walk_forward_config={
                "train_window_days": 365,
                "test_window_days": 63,
                "retrain_frequency_days": 21
            }
        )

        # Narrow ranges around winners
        param_grid = ParameterGrid(
            position_size_pct=[
                max(5, best_position_size - 5),
                max(5, best_position_size - 2),
                best_position_size,
                best_position_size + 2,
                min(30, best_position_size + 5)
            ],
            min_edge_bps=[
                max(10, best_min_edge - 20),
                max(10, best_min_edge - 10),
                best_min_edge,
                best_min_edge + 10,
                best_min_edge + 20
            ],
            strong_buy_threshold=[0.70, 0.75, 0.80],
            buy_threshold=[0.55, 0.60, 0.65],
            moderate_buy_threshold=[0.45, 0.50, 0.55],
            sell_threshold=[0.60],
            moderate_sell_threshold=[0.50]
        )

        opt_run = await self.optimizer.run_optimization(
            name=f"Auto-Opt Stage 4: {run.name}",
            base_config=opt_config,
            parameter_grid=param_grid,
            optimization_metric="sharpe_ratio",
            top_n_results=10
        )

        if opt_run.top_results:
            stage_result.optimization_id = opt_run.optimization_id
            stage_result.best_params = opt_run.top_results[0].parameters
            stage_result.best_metrics = opt_run.top_results[0].metrics
            stage_result.completed_at = datetime.now(timezone.utc)
        else:
            raise ValueError("Stage 4 optimization produced no results")

        logger.info(f"Stage 4 completed: Sharpe={stage_result.best_metrics.sharpe_ratio:.2f}")
        return stage_result

    async def _update_run(self, run: AutoOptimizeRun):
        """Update run in database"""
        await self.collection.update_one(
            {"auto_optimize_id": run.auto_optimize_id},
            {"$set": run.model_dump()}
        )
