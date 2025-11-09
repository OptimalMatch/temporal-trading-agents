"""
MongoDB database interface for temporal trading agents.
Adapted from claude-workflow-manager/backend/database.py
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os
from bson import ObjectId

from backend.models import (
    StrategyAnalysis, ConsensusResult, ModelTraining,
    PriceForecast, User, ApiKey, StrategyType, ScheduledTask,
    BacktestRun, BacktestStatus, PaperTradingSession, PaperTradingStatus
)


class Database:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        """Connect to MongoDB database"""
        try:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://mongodb:27017")
            print(f"📊 DATABASE: Connecting to MongoDB at: {mongodb_url}")

            self.client = AsyncIOMotorClient(mongodb_url)
            self.db = self.client.temporal_trading

            # Test the connection
            await self.client.admin.command('ping')
            print("✅ DATABASE: MongoDB connection successful")

            # Create indexes
            await self._create_indexes()
            print("✅ DATABASE: Indexes created successfully")

        except Exception as e:
            print(f"❌ DATABASE: Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            raise

    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            print("📊 DATABASE: Disconnected from MongoDB")

    async def _create_indexes(self):
        """Create database indexes for performance"""
        # Strategy Analysis indexes
        await self.db.strategy_analyses.create_index("symbol")
        await self.db.strategy_analyses.create_index("strategy_type")
        await self.db.strategy_analyses.create_index("created_at")
        await self.db.strategy_analyses.create_index([("symbol", 1), ("created_at", -1)])

        # Consensus Results indexes
        await self.db.consensus_results.create_index("symbol")
        await self.db.consensus_results.create_index("created_at")
        await self.db.consensus_results.create_index([("symbol", 1), ("created_at", -1)])

        # Model Training indexes
        await self.db.model_trainings.create_index("symbol")
        await self.db.model_trainings.create_index("horizon_days")
        await self.db.model_trainings.create_index("created_at")

        # Price Forecasts indexes
        await self.db.price_forecasts.create_index("symbol")
        await self.db.price_forecasts.create_index("horizon_days")
        await self.db.price_forecasts.create_index("created_at")
        await self.db.price_forecasts.create_index([("symbol", 1), ("horizon_days", 1), ("created_at", -1)])

        # User indexes (for future auth)
        await self.db.users.create_index("username", unique=True)
        await self.db.users.create_index("email", unique=True)

        # API Key indexes (for future auth)
        await self.db.api_keys.create_index("key", unique=True)
        await self.db.api_keys.create_index("user_id")

        # Scheduled Tasks indexes
        await self.db.scheduled_tasks.create_index("symbol")
        await self.db.scheduled_tasks.create_index("is_active")
        await self.db.scheduled_tasks.create_index("next_run")
        await self.db.scheduled_tasks.create_index([("is_active", 1), ("next_run", 1)])

        # Historical Price Data indexes
        await self.db.historical_prices.create_index("symbol", unique=True)
        await self.db.historical_prices.create_index("last_date")
        await self.db.historical_prices.create_index("updated_at")

        # Backtest indexes
        await self.db.backtests.create_index("run_id", unique=True)
        await self.db.backtests.create_index("symbol")
        await self.db.backtests.create_index("status")
        await self.db.backtests.create_index("created_at")
        await self.db.backtests.create_index([("symbol", 1), ("created_at", -1)])

        # Paper Trading indexes
        await self.db.paper_trading_sessions.create_index("session_id", unique=True)
        await self.db.paper_trading_sessions.create_index("symbol")
        await self.db.paper_trading_sessions.create_index("status")
        await self.db.paper_trading_sessions.create_index("started_at")

    # ==================== Strategy Analysis Methods ====================

    async def create_strategy_analysis(self, analysis: StrategyAnalysis) -> str:
        """Create a new strategy analysis record"""
        analysis_dict = analysis.dict()
        result = await self.db.strategy_analyses.insert_one(analysis_dict)
        return str(result.inserted_id)

    async def get_strategy_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Get a strategy analysis by ID"""
        try:
            query = {"id": analysis_id}
            analysis = await self.db.strategy_analyses.find_one(query)
            if analysis:
                del analysis["_id"]
            return analysis
        except Exception as e:
            print(f"Error retrieving analysis {analysis_id}: {e}")
            return None

    async def get_strategy_analyses(
        self,
        symbol: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """Get strategy analyses with optional filters"""
        query = {}
        if symbol:
            query["symbol"] = symbol
        if strategy_type:
            query["strategy_type"] = strategy_type.value

        cursor = self.db.strategy_analyses.find(query).sort("created_at", -1).skip(skip).limit(limit)
        analyses = []
        async for analysis in cursor:
            del analysis["_id"]
            analyses.append(analysis)
        return analyses

    async def get_latest_strategy_analysis(
        self,
        symbol: str,
        strategy_type: StrategyType
    ) -> Optional[Dict]:
        """Get the most recent analysis for a symbol and strategy"""
        query = {
            "symbol": symbol,
            "strategy_type": strategy_type.value,
            "status": "completed"
        }
        analysis = await self.db.strategy_analyses.find_one(query, sort=[("created_at", -1)])
        if analysis:
            del analysis["_id"]
        return analysis

    # ==================== Consensus Results Methods ====================

    async def create_consensus_result(self, consensus: ConsensusResult) -> str:
        """Create a new consensus result"""
        consensus_dict = consensus.dict()
        result = await self.db.consensus_results.insert_one(consensus_dict)
        return str(result.inserted_id)

    async def get_consensus_result(self, consensus_id: str) -> Optional[Dict]:
        """Get a consensus result by ID"""
        try:
            query = {"id": consensus_id}
            consensus = await self.db.consensus_results.find_one(query)
            if consensus:
                del consensus["_id"]
            return consensus
        except Exception as e:
            print(f"Error retrieving consensus {consensus_id}: {e}")
            return None

    async def get_consensus_results(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """Get consensus results with optional filters"""
        query = {}
        if symbol:
            query["symbol"] = symbol

        cursor = self.db.consensus_results.find(query).sort("created_at", -1).skip(skip).limit(limit)
        results = []
        async for result in cursor:
            del result["_id"]
            results.append(result)
        return results

    async def get_latest_consensus(self, symbol: str, interval: str = None) -> Optional[Dict]:
        """Get the most recent consensus for a symbol, optionally filtered by interval"""
        query = {"symbol": symbol}
        if interval:
            query["interval"] = interval

        consensus = await self.db.consensus_results.find_one(
            query,
            sort=[("created_at", -1)]
        )
        if consensus:
            del consensus["_id"]
        return consensus

    # ==================== Model Training Methods ====================

    async def create_model_training(self, training: ModelTraining) -> str:
        """Record a model training session"""
        training_dict = training.dict()
        result = await self.db.model_trainings.insert_one(training_dict)
        return str(result.inserted_id)

    async def get_model_trainings(
        self,
        symbol: Optional[str] = None,
        horizon_days: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get model training records"""
        query = {}
        if symbol:
            query["symbol"] = symbol
        if horizon_days:
            query["horizon_days"] = horizon_days

        cursor = self.db.model_trainings.find(query).sort("created_at", -1).limit(limit)
        trainings = []
        async for training in cursor:
            del training["_id"]
            trainings.append(training)
        return trainings

    # ==================== Price Forecast Methods ====================

    async def create_price_forecast(self, forecast: PriceForecast) -> str:
        """Create a new price forecast"""
        forecast_dict = forecast.dict()
        result = await self.db.price_forecasts.insert_one(forecast_dict)
        return str(result.inserted_id)

    async def get_price_forecasts(
        self,
        symbol: Optional[str] = None,
        horizon_days: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get price forecasts"""
        query = {}
        if symbol:
            query["symbol"] = symbol
        if horizon_days:
            query["horizon_days"] = horizon_days

        cursor = self.db.price_forecasts.find(query).sort("created_at", -1).limit(limit)
        forecasts = []
        async for forecast in cursor:
            del forecast["_id"]
            forecasts.append(forecast)
        return forecasts

    async def get_latest_forecast(
        self,
        symbol: str,
        horizon_days: int
    ) -> Optional[Dict]:
        """Get the most recent forecast for a symbol and horizon"""
        forecast = await self.db.price_forecasts.find_one(
            {"symbol": symbol, "horizon_days": horizon_days},
            sort=[("created_at", -1)]
        )
        if forecast:
            del forecast["_id"]
        return forecast

    # ==================== User Methods (for future auth) ====================

    async def create_user(self, user: User) -> str:
        """Create a new user"""
        user_dict = user.dict()
        result = await self.db.users.insert_one(user_dict)
        return str(result.inserted_id)

    async def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        user = await self.db.users.find_one({"username": username, "is_active": True})
        if user:
            del user["_id"]
        return user

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        user = await self.db.users.find_one({"email": email, "is_active": True})
        if user:
            del user["_id"]
        return user

    # ==================== API Key Methods (for future auth) ====================

    async def create_api_key(self, api_key: ApiKey) -> str:
        """Create a new API key"""
        api_key_dict = api_key.dict()
        result = await self.db.api_keys.insert_one(api_key_dict)
        return str(result.inserted_id)

    async def get_api_key(self, key: str) -> Optional[Dict]:
        """Get API key by key value"""
        api_key = await self.db.api_keys.find_one({"key": key, "is_active": True})
        if api_key:
            del api_key["_id"]
        return api_key

    async def increment_api_key_usage(self, key: str) -> bool:
        """Increment API key usage count"""
        result = await self.db.api_keys.update_one(
            {"key": key},
            {
                "$inc": {"usage_count": 1},
                "$set": {"last_used": datetime.now(timezone.utc)}
            }
        )
        return result.modified_count > 0

    # ==================== Analytics Methods ====================

    async def get_symbol_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get analytics for a trading symbol"""
        try:
            # Count total analyses
            total_analyses = await self.db.strategy_analyses.count_documents({"symbol": symbol})

            # Get strategy breakdown
            pipeline = [
                {"$match": {"symbol": symbol}},
                {"$group": {
                    "_id": "$strategy_type",
                    "count": {"$sum": 1}
                }}
            ]
            strategy_counts = {}
            async for result in self.db.strategy_analyses.aggregate(pipeline):
                strategy_counts[result["_id"]] = result["count"]

            # Get consensus history
            consensus_history = await self.db.consensus_results.count_documents({"symbol": symbol})

            # Get latest consensus
            latest_consensus = await self.get_latest_consensus(symbol)

            return {
                "symbol": symbol,
                "total_analyses": total_analyses,
                "strategy_breakdown": strategy_counts,
                "consensus_count": consensus_history,
                "latest_consensus": latest_consensus
            }
        except Exception as e:
            print(f"Error getting analytics for {symbol}: {e}")
            return {
                "symbol": symbol,
                "total_analyses": 0,
                "strategy_breakdown": {},
                "consensus_count": 0,
                "latest_consensus": None
            }

    # ==================== Scheduled Tasks Methods ====================

    async def create_scheduled_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task"""
        task_dict = task.dict()
        result = await self.db.scheduled_tasks.insert_one(task_dict)
        return str(result.inserted_id)

    async def get_scheduled_task(self, task_id: str) -> Optional[Dict]:
        """Get a scheduled task by ID"""
        try:
            query = {"id": task_id}
            task = await self.db.scheduled_tasks.find_one(query)
            if task:
                del task["_id"]
            return task
        except Exception as e:
            print(f"Error retrieving scheduled task {task_id}: {e}")
            return None

    async def get_scheduled_tasks(
        self,
        symbol: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get scheduled tasks with optional filters"""
        query = {}
        if symbol:
            query["symbol"] = symbol
        if is_active is not None:
            query["is_active"] = is_active

        cursor = self.db.scheduled_tasks.find(query).sort("created_at", -1).limit(limit)
        tasks = []
        async for task in cursor:
            del task["_id"]
            tasks.append(task)
        return tasks

    async def update_scheduled_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scheduled task"""
        updates["updated_at"] = datetime.now(timezone.utc)
        result = await self.db.scheduled_tasks.update_one(
            {"id": task_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def delete_scheduled_task(self, task_id: str) -> bool:
        """Delete a scheduled task"""
        result = await self.db.scheduled_tasks.delete_one({"id": task_id})
        return result.deleted_count > 0

    async def update_task_run_time(self, task_id: str, next_run: datetime) -> bool:
        """Update task run time and increment run count"""
        result = await self.db.scheduled_tasks.update_one(
            {"id": task_id},
            {
                "$set": {
                    "last_run": datetime.now(timezone.utc),
                    "next_run": next_run,
                    "updated_at": datetime.now(timezone.utc)
                },
                "$inc": {"run_count": 1}
            }
        )
        return result.modified_count > 0

    async def get_due_tasks(self) -> List[Dict]:
        """Get tasks that are due to run"""
        now = datetime.now(timezone.utc)
        query = {
            "is_active": True,
            "next_run": {"$lte": now}
        }
        cursor = self.db.scheduled_tasks.find(query)
        tasks = []
        async for task in cursor:
            del task["_id"]
            tasks.append(task)
        return tasks

    # ==================== Historical Price Data Methods ====================

    async def upsert_historical_prices(self, historical_data: Dict[str, Any]) -> bool:
        """Insert or update historical price data for a symbol"""
        try:
            result = await self.db.historical_prices.update_one(
                {"symbol": historical_data["symbol"]},
                {"$set": historical_data},
                upsert=True
            )
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            print(f"Error upserting historical prices: {e}")
            return False

    async def get_historical_prices(self, symbol: str) -> Optional[Dict]:
        """Get historical price data for a symbol"""
        try:
            prices = await self.db.historical_prices.find_one({"symbol": symbol})
            if prices:
                del prices["_id"]
            return prices
        except Exception as e:
            print(f"Error retrieving historical prices for {symbol}: {e}")
            return None

    async def get_historical_prices_range(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """Get historical prices within a date range"""
        try:
            prices_doc = await self.get_historical_prices(symbol)
            if not prices_doc or "prices" not in prices_doc:
                return None

            prices = prices_doc["prices"]

            # Filter by date range if provided
            if start_date or end_date:
                filtered = []
                for price_point in prices:
                    price_date = price_point["date"]
                    if start_date and price_date < start_date:
                        continue
                    if end_date and price_date > end_date:
                        continue
                    filtered.append(price_point)
                return filtered

            return prices
        except Exception as e:
            print(f"Error retrieving historical prices range for {symbol}: {e}")
            return None

    async def get_historical_prices_dataframe(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """Get historical prices as pandas DataFrame for backtesting"""
        try:
            import pandas as pd

            prices = await self.get_historical_prices_range(symbol, start_date, end_date)
            if not prices:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(prices)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

            return df
        except Exception as e:
            print(f"Error retrieving historical prices dataframe for {symbol}: {e}")
            return None

    # ==================== Backtesting Methods ====================

    async def store_backtest(self, backtest: BacktestRun) -> bool:
        """Store a backtest run"""
        try:
            backtest_dict = backtest.dict()
            await self.db.backtests.insert_one(backtest_dict)
            return True
        except Exception as e:
            print(f"Error storing backtest: {e}")
            return False

    async def get_backtest(self, run_id: str) -> Optional[BacktestRun]:
        """Get backtest by run_id"""
        try:
            backtest = await self.db.backtests.find_one({"run_id": run_id})
            if backtest:
                del backtest["_id"]
                return BacktestRun(**backtest)
            return None
        except Exception as e:
            print(f"Error retrieving backtest {run_id}: {e}")
            return None

    async def get_backtests(
        self,
        symbol: Optional[str] = None,
        status: Optional[BacktestStatus] = None,
        limit: int = 50
    ) -> List[BacktestRun]:
        """Get list of backtests with optional filters (summary only, no trades/equity)"""
        try:
            query = {}
            if symbol:
                query["config.symbol"] = symbol
            if status:
                query["status"] = status

            # Use exclusion-only projection to avoid loading massive arrays
            # MongoDB doesn't allow mixing inclusion/exclusion (except _id)
            projection = {
                "_id": 0,
                "trades": 0,
                "equity_curve": 0
            }

            cursor = self.db.backtests.find(query, projection).sort("created_at", -1).limit(limit)
            backtests = []
            async for backtest in cursor:
                # Add empty arrays for fields that are excluded but required by model
                backtest["trades"] = []
                backtest["equity_curve"] = []
                backtest["period_metrics"] = []
                backtests.append(BacktestRun(**backtest))
            return backtests
        except Exception as e:
            print(f"Error retrieving backtests: {e}")
            return []

    async def update_backtest_status(
        self,
        run_id: str,
        status: BacktestStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update backtest status"""
        try:
            updates = {"status": status}
            if status == BacktestStatus.RUNNING:
                updates["started_at"] = datetime.now(timezone.utc)
            elif status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED]:
                updates["completed_at"] = datetime.now(timezone.utc)
            if error_message:
                updates["error_message"] = error_message

            result = await self.db.backtests.update_one(
                {"run_id": run_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating backtest status: {e}")
            return False

    async def update_backtest_results(self, backtest: BacktestRun) -> bool:
        """Update backtest with results"""
        try:
            backtest_dict = backtest.dict()
            result = await self.db.backtests.update_one(
                {"run_id": backtest.run_id},
                {"$set": backtest_dict}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating backtest results: {e}")
            return False

    async def delete_backtest(self, run_id: str) -> bool:
        """Delete a backtest"""
        try:
            result = await self.db.backtests.delete_one({"run_id": run_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting backtest: {e}")
            return False

    # ==================== Parameter Optimization Methods ====================

    async def store_optimization_run(self, optimization: 'OptimizationRun') -> bool:
        """Store an optimization run"""
        try:
            from backend.models import OptimizationRun
            opt_dict = optimization.dict()
            await self.db.optimizations.insert_one(opt_dict)
            return True
        except Exception as e:
            print(f"Error storing optimization run: {e}")
            return False

    async def get_optimization_run(self, optimization_id: str) -> Optional['OptimizationRun']:
        """Get optimization run by optimization_id"""
        try:
            from backend.models import OptimizationRun
            opt = await self.db.optimizations.find_one({"optimization_id": optimization_id}, {"_id": 0})
            if opt:
                return OptimizationRun(**opt)
            return None
        except Exception as e:
            print(f"Error getting optimization run: {e}")
            return None

    async def get_optimization_runs(self, limit: int = 50) -> List['OptimizationRun']:
        """Get recent optimization runs"""
        try:
            from backend.models import OptimizationRun
            cursor = self.db.optimizations.find(
                {},
                {"_id": 0}
            ).sort("created_at", -1).limit(limit)

            optimizations = []
            async for opt in cursor:
                optimizations.append(OptimizationRun(**opt))
            return optimizations
        except Exception as e:
            print(f"Error getting optimization runs: {e}")
            return []

    async def update_optimization_status(
        self,
        optimization_id: str,
        status: 'OptimizationStatus',
        error_message: Optional[str] = None
    ) -> bool:
        """Update optimization run status"""
        try:
            from backend.models import OptimizationStatus
            update_dict = {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc)
            }
            if error_message:
                update_dict["error_message"] = error_message
            if status == OptimizationStatus.COMPLETED or status == OptimizationStatus.FAILED:
                update_dict["completed_at"] = datetime.now(timezone.utc)

            await self.db.optimizations.update_one(
                {"optimization_id": optimization_id},
                {"$set": update_dict}
            )
            return True
        except Exception as e:
            print(f"Error updating optimization status: {e}")
            return False

    async def delete_optimization_run(self, optimization_id: str) -> bool:
        """Delete an optimization run"""
        try:
            result = await self.db.optimizations.delete_one(
                {"optimization_id": optimization_id}
            )
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting optimization run: {e}")
            return False

    # ==================== Paper Trading Methods ====================

    async def store_paper_trading_session(self, session: PaperTradingSession) -> bool:
        """Store a paper trading session"""
        try:
            session_dict = session.dict()
            await self.db.paper_trading_sessions.insert_one(session_dict)
            return True
        except Exception as e:
            print(f"Error storing paper trading session: {e}")
            return False

    async def get_paper_trading_session(self, session_id: str) -> Optional[PaperTradingSession]:
        """Get paper trading session by session_id"""
        try:
            session = await self.db.paper_trading_sessions.find_one({"session_id": session_id})
            if session:
                del session["_id"]
                return PaperTradingSession(**session)
            return None
        except Exception as e:
            print(f"Error retrieving paper trading session {session_id}: {e}")
            return None

    async def get_paper_trading_sessions(
        self,
        symbol: Optional[str] = None,
        status: Optional[PaperTradingStatus] = None,
        limit: int = 50
    ) -> List[PaperTradingSession]:
        """Get list of paper trading sessions with optional filters"""
        try:
            query = {}
            if symbol:
                query["config.symbol"] = symbol
            if status:
                query["status"] = status

            cursor = self.db.paper_trading_sessions.find(query).sort("started_at", -1).limit(limit)
            sessions = []
            async for session in cursor:
                del session["_id"]
                sessions.append(PaperTradingSession(**session))
            return sessions
        except Exception as e:
            print(f"Error retrieving paper trading sessions: {e}")
            return []

    async def update_paper_trading_session(self, session: PaperTradingSession) -> bool:
        """Update paper trading session"""
        try:
            session_dict = session.dict()
            result = await self.db.paper_trading_sessions.update_one(
                {"session_id": session.session_id},
                {"$set": session_dict}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating paper trading session: {e}")
            return False

    async def delete_paper_trading_session(self, session_id: str) -> bool:
        """Delete a paper trading session"""
        try:
            result = await self.db.paper_trading_sessions.delete_one({"session_id": session_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting paper trading session: {e}")
            return False

    # ========================================
    # Experiment Methods
    # ========================================

    async def store_experiment(self, experiment) -> bool:
        """Store an experiment"""
        try:
            experiment_dict = experiment.dict()
            await self.db.experiments.insert_one(experiment_dict)
            return True
        except Exception as e:
            print(f"Error storing experiment: {e}")
            return False

    async def get_experiment(self, experiment_id: str):
        """Get experiment by experiment_id"""
        try:
            from backend.models import Experiment
            experiment = await self.db.experiments.find_one({"experiment_id": experiment_id})
            if experiment:
                del experiment["_id"]
                return Experiment(**experiment)
            return None
        except Exception as e:
            print(f"Error retrieving experiment {experiment_id}: {e}")
            return None

    async def get_experiments(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ):
        """Get list of experiments with optional filters"""
        try:
            from backend.models import Experiment
            query = {}
            if symbol:
                query["symbol"] = symbol
            if status:
                query["status"] = status

            cursor = self.db.experiments.find(query).sort("created_at", -1).limit(limit)
            experiments = []
            async for experiment in cursor:
                del experiment["_id"]
                experiments.append(Experiment(**experiment))
            return experiments
        except Exception as e:
            print(f"Error retrieving experiments: {e}")
            return []

    async def update_experiment(self, experiment) -> bool:
        """Update experiment"""
        try:
            experiment_dict = experiment.dict()
            result = await self.db.experiments.update_one(
                {"experiment_id": experiment.experiment_id},
                {"$set": experiment_dict}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating experiment: {e}")
            return False

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        try:
            result = await self.db.experiments.delete_one({"experiment_id": experiment_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting experiment: {e}")
            return False

    # ==================== Remote Instance Management ====================

    async def create_remote_instance(self, instance) -> str:
        """Create a new remote instance configuration"""
        try:
            instance_dict = instance.dict()
            await self.db.remote_instances.insert_one(instance_dict)
            return instance.id
        except Exception as e:
            print(f"Error creating remote instance: {e}")
            raise

    async def get_remote_instance(self, instance_id: str) -> Optional[Dict]:
        """Get a remote instance by ID"""
        try:
            result = await self.db.remote_instances.find_one({"id": instance_id}, {"_id": 0})
            return result
        except Exception as e:
            print(f"Error getting remote instance: {e}")
            return None

    async def get_remote_instances(self, enabled_only: bool = False) -> List[Dict]:
        """Get all remote instances"""
        try:
            query = {"enabled": True} if enabled_only else {}
            cursor = self.db.remote_instances.find(query, {"_id": 0}).sort("created_at", -1)
            instances = await cursor.to_list(length=100)
            return instances
        except Exception as e:
            print(f"Error getting remote instances: {e}")
            return []

    async def update_remote_instance(self, instance_id: str, update_data: Dict) -> bool:
        """Update a remote instance"""
        try:
            # Remove None values
            update_data = {k: v for k, v in update_data.items() if v is not None}
            if not update_data:
                return False

            result = await self.db.remote_instances.update_one(
                {"id": instance_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating remote instance: {e}")
            return False

    async def delete_remote_instance(self, instance_id: str) -> bool:
        """Delete a remote instance"""
        try:
            # Also delete associated remote forecasts
            await self.db.remote_forecasts.delete_many({"remote_instance_id": instance_id})

            result = await self.db.remote_instances.delete_one({"id": instance_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting remote instance: {e}")
            return False

    async def create_remote_forecast(self, forecast) -> str:
        """Store an imported forecast from remote instance"""
        try:
            forecast_dict = forecast.dict()
            await self.db.remote_forecasts.insert_one(forecast_dict)
            return forecast.id
        except Exception as e:
            print(f"Error creating remote forecast: {e}")
            raise

    async def get_remote_forecasts(
        self,
        symbol: Optional[str] = None,
        remote_instance_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get imported forecasts from remote instances"""
        try:
            query = {}
            if symbol:
                query["symbol"] = symbol
            if remote_instance_id:
                query["remote_instance_id"] = remote_instance_id

            cursor = self.db.remote_forecasts.find(query, {"_id": 0}).sort("imported_at", -1).limit(limit)
            forecasts = await cursor.to_list(length=limit)
            return forecasts
        except Exception as e:
            print(f"Error getting remote forecasts: {e}")
            return []

    # ==================== HuggingFace Configuration Methods ====================

    async def create_hf_config(self, config: Dict) -> str:
        """Create a HuggingFace configuration"""
        try:
            # config is already a dict from model_dump()
            # Generate ID if not present
            if 'id' not in config:
                config['id'] = str(uuid.uuid4())

            # Add timestamps if not present
            if 'created_at' not in config:
                config['created_at'] = datetime.now(timezone.utc)
            if 'updated_at' not in config:
                config['updated_at'] = datetime.now(timezone.utc)

            # Create unique index on symbol+interval
            await self.db.hf_configs.create_index(
                [("symbol", 1), ("interval", 1)],
                unique=True
            )
            await self.db.hf_configs.insert_one(config)
            return config['id']
        except Exception as e:
            print(f"Error creating HF config: {e}")
            raise

    async def get_hf_config(self, config_id: str) -> Optional[Dict]:
        """Get a HuggingFace configuration by ID"""
        try:
            config = await self.db.hf_configs.find_one({"id": config_id}, {"_id": 0})
            return config
        except Exception as e:
            print(f"Error getting HF config: {e}")
            return None

    async def get_hf_config_by_symbol_interval(
        self,
        symbol: str,
        interval: str
    ) -> Optional[Dict]:
        """Get a HuggingFace configuration by symbol and interval"""
        try:
            config = await self.db.hf_configs.find_one(
                {"symbol": symbol, "interval": interval},
                {"_id": 0}
            )
            return config
        except Exception as e:
            print(f"Error getting HF config by symbol/interval: {e}")
            return None

    async def get_all_hf_configs(
        self,
        enabled_only: bool = False
    ) -> List[Dict]:
        """Get all HuggingFace configurations"""
        try:
            query = {}
            if enabled_only:
                query["enabled"] = True

            cursor = self.db.hf_configs.find(query, {"_id": 0}).sort("created_at", -1)
            configs = await cursor.to_list(length=None)
            return configs
        except Exception as e:
            print(f"Error getting HF configs: {e}")
            return []

    async def update_hf_config(
        self,
        config_id: str,
        update_data: Dict
    ) -> bool:
        """Update a HuggingFace configuration"""
        try:
            from datetime import datetime, timezone
            update_data["updated_at"] = datetime.now(timezone.utc)

            result = await self.db.hf_configs.update_one(
                {"id": config_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating HF config: {e}")
            return False

    async def delete_hf_config(self, config_id: str) -> bool:
        """Delete a HuggingFace configuration"""
        try:
            result = await self.db.hf_configs.delete_one({"id": config_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting HF config: {e}")
            return False

    async def update_hf_config_export_timestamp(
        self,
        config_id: str
    ) -> bool:
        """Update the last_export timestamp for a config"""
        try:
            from datetime import datetime, timezone
            result = await self.db.hf_configs.update_one(
                {"id": config_id},
                {
                    "$set": {
                        "last_export": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating HF config export timestamp: {e}")
            return False

    async def update_hf_config_import_timestamp(
        self,
        config_id: str
    ) -> bool:
        """Update the last_import timestamp for a config"""
        try:
            from datetime import datetime, timezone
            result = await self.db.hf_configs.update_one(
                {"id": config_id},
                {
                    "$set": {
                        "last_import": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating HF config import timestamp: {e}")
            return False
