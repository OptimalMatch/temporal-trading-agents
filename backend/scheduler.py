"""
Background scheduler for running scheduled trading analysis tasks.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
from typing import Dict
import asyncio

from backend.database import Database
from backend.models import ScheduledTask, ScheduleFrequency, StrategyType


class TaskScheduler:
    """Manages scheduled trading analysis tasks"""

    def __init__(self, database: Database):
        self.db = database
        self.scheduler = AsyncIOScheduler()
        self.task_jobs: Dict[str, str] = {}  # task_id -> job_id mapping

    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        print("📅 SCHEDULER: Task scheduler started")

    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        print("📅 SCHEDULER: Task scheduler shutdown")

    async def load_scheduled_tasks(self):
        """Load all active scheduled tasks from database"""
        tasks = await self.db.get_scheduled_tasks(is_active=True)
        print(f"📅 SCHEDULER: Loading {len(tasks)} scheduled tasks")

        for task_dict in tasks:
            try:
                task = ScheduledTask(**task_dict)
                await self.schedule_task(task)
            except Exception as e:
                print(f"📅 SCHEDULER: Error loading task {task_dict.get('id')}: {e}")

    async def schedule_task(self, task: ScheduledTask):
        """Schedule a task based on its frequency"""
        # Remove existing job if any
        if task.id in self.task_jobs:
            self.scheduler.remove_job(self.task_jobs[task.id])

        # Determine trigger based on frequency
        trigger = self._get_trigger(task)

        if not trigger:
            print(f"📅 SCHEDULER: Invalid trigger for task {task.id}")
            return

        # Schedule the job
        job = self.scheduler.add_job(
            self._execute_task,
            trigger=trigger,
            args=[task.id],
            id=f"task_{task.id}",
            replace_existing=True
        )

        self.task_jobs[task.id] = job.id

        # Calculate and store next run time
        next_run = job.next_run_time
        if next_run:
            await self.db.update_scheduled_task(task.id, {"next_run": next_run})
            print(f"📅 SCHEDULER: Task '{task.name}' scheduled, next run: {next_run}")

    def _get_trigger(self, task: ScheduledTask):
        """Get APScheduler trigger for task frequency"""
        if task.frequency == ScheduleFrequency.HOURLY:
            return CronTrigger(minute=0)  # Every hour at minute 0

        elif task.frequency == ScheduleFrequency.DAILY:
            return CronTrigger(hour=9, minute=0)  # 9 AM daily

        elif task.frequency == ScheduleFrequency.WEEKLY:
            return CronTrigger(day_of_week='mon', hour=9, minute=0)  # Monday 9 AM

        elif task.frequency == ScheduleFrequency.CUSTOM and task.cron_expression:
            try:
                return CronTrigger.from_crontab(task.cron_expression)
            except Exception as e:
                print(f"📅 SCHEDULER: Invalid cron expression: {e}")
                return None

        return None

    async def _execute_task(self, task_id: str):
        """Execute a scheduled task"""
        try:
            print(f"📅 SCHEDULER: Executing task {task_id}")

            # Get task details
            task_dict = await self.db.get_scheduled_task(task_id)
            if not task_dict:
                print(f"📅 SCHEDULER: Task {task_id} not found")
                return

            task = ScheduledTask(**task_dict)

            if not task.is_active:
                print(f"📅 SCHEDULER: Task {task_id} is inactive, skipping")
                return

            # Import here to avoid circular dependency
            from main import run_strategy_analysis

            # Run the strategy analysis
            print(f"📅 SCHEDULER: Running {task.strategy_type.value} for {task.symbol}")
            await run_strategy_analysis(
                symbol=task.symbol,
                strategy_type=task.strategy_type,
                horizons=task.horizons,
                task_id=task_id
            )

            # Update run times
            job = self.scheduler.get_job(f"task_{task_id}")
            if job and job.next_run_time:
                await self.db.update_task_run_time(task_id, job.next_run_time)

            print(f"📅 SCHEDULER: Task {task_id} completed successfully")

        except Exception as e:
            print(f"📅 SCHEDULER: Error executing task {task_id}: {e}")

    async def unschedule_task(self, task_id: str):
        """Remove a task from the schedule"""
        if task_id in self.task_jobs:
            job_id = self.task_jobs[task_id]
            self.scheduler.remove_job(job_id)
            del self.task_jobs[task_id]
            print(f"📅 SCHEDULER: Task {task_id} unscheduled")

    async def reschedule_task(self, task_id: str):
        """Reschedule a task (useful after updates)"""
        task_dict = await self.db.get_scheduled_task(task_id)
        if task_dict:
            task = ScheduledTask(**task_dict)
            await self.schedule_task(task)


# Global scheduler instance
scheduler_instance = None


def get_scheduler(database: Database) -> TaskScheduler:
    """Get or create scheduler instance"""
    global scheduler_instance
    if scheduler_instance is None:
        scheduler_instance = TaskScheduler(database)
    return scheduler_instance
