"""
Parameter Optimization Engine

Implements grid search and other optimization methods for backtest parameters.
Uses parallel execution and strategy result caching for performance.
"""
import itertools
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.models import (
    BacktestConfig, OptimizableParams, ParameterGrid, OptimizationMetric,
    OptimizationResult, OptimizationRun, OptimizationStatus,
    BacktestMetrics
)
from backend.backtesting_engine import BacktestEngine

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Grid search parameter optimizer for backtest configurations"""

    def __init__(self, max_workers: int = 4):
        """
        Initialize parameter optimizer.

        Args:
            max_workers: Maximum number of parallel backtest workers
        """
        self.max_workers = max_workers

    def generate_parameter_combinations(self, grid: ParameterGrid) -> List[OptimizableParams]:
        """
        Generate all parameter combinations from the grid.

        Args:
            grid: Parameter grid defining search space

        Returns:
            List of all parameter combinations
        """
        # Extract all parameter values
        param_names = []
        param_values = []

        if grid.position_size_pct:
            param_names.append('position_size_pct')
            param_values.append(grid.position_size_pct)

        if grid.min_edge_bps:
            param_names.append('min_edge_bps')
            param_values.append(grid.min_edge_bps)

        if grid.strong_buy_threshold:
            param_names.append('strong_buy_threshold')
            param_values.append(grid.strong_buy_threshold)

        if grid.buy_threshold:
            param_names.append('buy_threshold')
            param_values.append(grid.buy_threshold)

        if grid.moderate_buy_threshold:
            param_names.append('moderate_buy_threshold')
            param_values.append(grid.moderate_buy_threshold)

        if grid.sell_threshold:
            param_names.append('sell_threshold')
            param_values.append(grid.sell_threshold)

        if grid.moderate_sell_threshold:
            param_names.append('moderate_sell_threshold')
            param_values.append(grid.moderate_sell_threshold)

        # Generate all combinations using Cartesian product
        combinations = []
        for values in itertools.product(*param_values):
            params_dict = dict(zip(param_names, values))
            combinations.append(OptimizableParams(**params_dict))

        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations

    def run_single_backtest(
        self,
        base_config: BacktestConfig,
        params: OptimizableParams,
        price_data: pd.DataFrame
    ) -> Tuple[OptimizableParams, BacktestMetrics, str]:
        """
        Run a single backtest with given parameters.

        Args:
            base_config: Base backtest configuration
            params: Parameters to test
            price_data: Price data for backtesting

        Returns:
            Tuple of (parameters, metrics, run_id)
        """
        # Create config with optimized parameters
        config = base_config.copy(deep=True)
        config.optimizable = params

        # Run backtest
        engine = BacktestEngine(config)

        # Generate a unique run ID for this optimization iteration
        import uuid
        run_id = str(uuid.uuid4())

        try:
            if config.walk_forward.enabled:
                result = engine.run_walkforward_backtest(price_data, run_id)
            else:
                result = engine.run_backtest(price_data, run_id)

            return (params, result.metrics, result.run_id)

        except Exception as e:
            logger.error(f"Backtest failed for params {params}: {e}")
            raise

    def extract_metric_value(
        self,
        metrics: BacktestMetrics,
        optimization_metric: OptimizationMetric
    ) -> float:
        """
        Extract the optimization metric value from backtest metrics.

        Args:
            metrics: Backtest metrics
            optimization_metric: Metric to optimize

        Returns:
            Metric value
        """
        if optimization_metric == OptimizationMetric.SHARPE_RATIO:
            return metrics.sharpe_ratio
        elif optimization_metric == OptimizationMetric.TOTAL_RETURN:
            return metrics.total_return
        elif optimization_metric == OptimizationMetric.PROFIT_FACTOR:
            return metrics.profit_factor
        elif optimization_metric == OptimizationMetric.WIN_RATE:
            return metrics.win_rate
        elif optimization_metric == OptimizationMetric.MAX_DRAWDOWN:
            # For max drawdown, we want to minimize (more negative is worse)
            # So negate it for ranking (less negative = better rank)
            return -metrics.max_drawdown
        else:
            raise ValueError(f"Unknown optimization metric: {optimization_metric}")

    def rank_results(
        self,
        results: List[OptimizationResult],
        optimization_metric: OptimizationMetric
    ) -> List[OptimizationResult]:
        """
        Rank results by optimization metric.

        Args:
            results: List of optimization results
            optimization_metric: Metric used for optimization

        Returns:
            Sorted and ranked results (best first)
        """
        # Sort by metric value (descending for most metrics, ascending for drawdown)
        sorted_results = sorted(
            results,
            key=lambda r: r.metric_value,
            reverse=True  # Higher is better (even for drawdown since we negated it)
        )

        # Assign ranks
        for i, result in enumerate(sorted_results, 1):
            result.rank = i

        return sorted_results

    def optimize(
        self,
        base_config: BacktestConfig,
        parameter_grid: ParameterGrid,
        price_data: pd.DataFrame,
        optimization_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO,
        top_n: int = 10,
        progress_callback: callable = None
    ) -> OptimizationRun:
        """
        Run parameter optimization using grid search.

        Args:
            base_config: Base backtest configuration
            parameter_grid: Grid of parameters to test
            price_data: Price data for backtesting
            optimization_metric: Metric to optimize
            top_n: Number of top results to return
            progress_callback: Optional callback function called with (completed, total) after each backtest

        Returns:
            Optimization run with results
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting parameter optimization with {self.max_workers} workers")

        # Generate all parameter combinations
        param_combinations = self.generate_parameter_combinations(parameter_grid)

        # Generate strategy combinations (default to base config if not specified)
        strategy_combinations = parameter_grid.enabled_strategies if parameter_grid.enabled_strategies else [base_config.enabled_strategies]

        # Create full combinations (params × strategies)
        all_combinations = []
        for params in param_combinations:
            for strategies in strategy_combinations:
                all_combinations.append((params, strategies))

        total_combinations = len(all_combinations)

        # Create optimization run record
        optimization_run = OptimizationRun(
            name=f"Optimization {base_config.symbol} {start_time.strftime('%Y-%m-%d %H:%M')}",
            base_config=base_config,
            parameter_grid=parameter_grid,
            optimization_metric=optimization_metric,
            status=OptimizationStatus.RUNNING,
            total_combinations=total_combinations,
            started_at=start_time
        )

        # Run backtests in parallel
        results = []
        completed = 0

        logger.info(f"Running {total_combinations} backtest combinations (params × strategies)...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {}
            for params, strategies in all_combinations:
                # Create a config with these specific strategies
                config = base_config.copy(deep=True)
                config.enabled_strategies = strategies
                future = executor.submit(self.run_single_backtest, config, params, price_data)
                future_to_config[future] = (params, strategies)

            # Process results as they complete
            for future in as_completed(future_to_config):
                try:
                    params, metrics, run_id = future.result()
                    metric_value = self.extract_metric_value(metrics, optimization_metric)

                    result = OptimizationResult(
                        parameters=params,
                        metrics=metrics,
                        backtest_run_id=run_id,
                        metric_value=metric_value
                    )
                    results.append(result)

                    completed += 1
                    optimization_run.completed_combinations = completed

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed, total_combinations)

                    if completed % 10 == 0 or completed == total_combinations:
                        logger.info(f"Progress: {completed}/{total_combinations} combinations completed")

                except Exception as e:
                    params, strategies = future_to_config[future]
                    logger.error(f"Backtest failed for {params}: {e}")
                    print(f"❌ Backtest failed for params: {params}")
                    print(f"   Error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with other combinations even if one fails
                    completed += 1  # Count failed attempts too
                    if progress_callback:
                        progress_callback(completed, total_combinations)

        # Rank results
        ranked_results = self.rank_results(results, optimization_metric)

        # Store all results and top N
        optimization_run.results = ranked_results
        optimization_run.top_results = ranked_results[:top_n]
        optimization_run.best_parameters = ranked_results[0].parameters if ranked_results else None

        # Update status
        optimization_run.status = OptimizationStatus.COMPLETED
        optimization_run.completed_at = datetime.utcnow()
        optimization_run.execution_time_ms = int(
            (optimization_run.completed_at - start_time).total_seconds() * 1000
        )

        logger.info(f"Optimization completed in {optimization_run.execution_time_ms}ms")
        logger.info(f"Best parameters: {optimization_run.best_parameters}")
        logger.info(f"Best {optimization_metric.value}: {ranked_results[0].metric_value:.4f}")

        return optimization_run

    async def run_optimization(
        self,
        name: str,
        base_config: BacktestConfig,
        parameter_grid: ParameterGrid,
        optimization_metric: str = "sharpe_ratio",
        top_n_results: int = 10
    ) -> OptimizationRun:
        """
        Async wrapper for optimize() method.
        Runs optimization in a thread pool executor to avoid blocking the event loop.

        Args:
            name: Name for this optimization run
            base_config: Base backtest configuration
            parameter_grid: Grid of parameters to test
            optimization_metric: Metric to optimize (string name)
            top_n_results: Number of top results to return

        Returns:
            Optimization run with results
        """
        import asyncio
        from strategies.data_cache import get_cache

        # Convert string metric to enum
        metric_enum = OptimizationMetric(optimization_metric)

        # Load price data from cache
        cache = get_cache()
        is_crypto = '-' in base_config.symbol
        period = '2y' if is_crypto else '5y'
        price_data = cache.get(base_config.symbol, period, interval='1d')

        if price_data is None or len(price_data) == 0:
            raise ValueError(f"No price data found for {base_config.symbol} in cache. Please sync data first.")

        # Filter by date range
        if base_config.start_date or base_config.end_date:
            if base_config.start_date:
                price_data = price_data[price_data.index >= base_config.start_date]
            if base_config.end_date:
                price_data = price_data[price_data.index <= base_config.end_date]

        # Reset index and normalize column names
        price_data = price_data.reset_index()
        price_data = price_data.rename(columns={
            'Date': 'date', 'Close': 'close', 'Open': 'open',
            'High': 'high', 'Low': 'low', 'Volume': 'volume'
        })
        price_data.columns = [c.lower() for c in price_data.columns]

        if len(price_data) == 0:
            raise ValueError(f"No price data in date range {base_config.start_date} to {base_config.end_date}")

        # Run the synchronous optimize method in a thread pool
        loop = asyncio.get_event_loop()
        optimization_run = await loop.run_in_executor(
            None,
            self.optimize,
            base_config,
            parameter_grid,
            price_data,
            metric_enum,
            top_n_results
        )

        # Update name
        optimization_run.name = name

        return optimization_run
