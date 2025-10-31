"""
MongoDB database interface for temporal trading agents.
Adapted from claude-workflow-manager/backend/database.py
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from bson import ObjectId

from backend.models import (
    StrategyAnalysis, ConsensusResult, ModelTraining,
    PriceForecast, User, ApiKey, StrategyType, ScheduledTask
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

    async def get_latest_consensus(self, symbol: str) -> Optional[Dict]:
        """Get the most recent consensus for a symbol"""
        consensus = await self.db.consensus_results.find_one(
            {"symbol": symbol},
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
                "$set": {"last_used": datetime.utcnow()}
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
        updates["updated_at"] = datetime.utcnow()
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
                    "last_run": datetime.utcnow(),
                    "next_run": next_run,
                    "updated_at": datetime.utcnow()
                },
                "$inc": {"run_count": 1}
            }
        )
        return result.modified_count > 0

    async def get_due_tasks(self) -> List[Dict]:
        """Get tasks that are due to run"""
        now = datetime.utcnow()
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
