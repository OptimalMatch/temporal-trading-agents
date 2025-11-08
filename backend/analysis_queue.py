"""
Analysis queue manager to prevent concurrent GPU-intensive consensus analysis.
Ensures only one analysis runs at a time to avoid GPU overload.
"""
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import uuid


class AnalysisQueue:
    """
    Manages a queue of consensus analysis jobs to prevent concurrent GPU usage.
    Only allows one analysis to run at a time.
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self._current_job: Optional[str] = None
        self._is_processing = False
        self._lock = asyncio.Lock()
        self._stats = {
            "total_queued": 0,
            "total_completed": 0,
            "total_failed": 0
        }

    async def start_processing(self):
        """Start the queue processor"""
        if self._is_processing:
            print("⚠️  ANALYSIS QUEUE: Already processing")
            return

        self._is_processing = True
        print("🎯 ANALYSIS QUEUE: Started processing")

        # Run the processor in the background
        asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Process queued analysis jobs one at a time"""
        while self._is_processing:
            try:
                # Wait for next job (with timeout to check if we should stop)
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                symbol = job["symbol"]
                job_id = job["job_id"]
                queued_at = job["queued_at"]
                callback = job["callback"]

                # Calculate wait time
                wait_time = (datetime.now(timezone.utc) - queued_at).total_seconds()

                print(f"🎯 ANALYSIS QUEUE: Processing {symbol} (job_id={job_id}, waited {wait_time:.1f}s)")
                self._current_job = job_id

                # Execute the analysis
                try:
                    await callback(symbol)
                    self._stats["total_completed"] += 1
                    print(f"✅ ANALYSIS QUEUE: Completed {symbol}")
                except Exception as e:
                    self._stats["total_failed"] += 1
                    print(f"❌ ANALYSIS QUEUE: Failed {symbol}: {e}")

                # Mark job as done
                self._queue.task_done()
                self._current_job = None

            except Exception as e:
                print(f"❌ ANALYSIS QUEUE: Error processing queue: {e}")
                await asyncio.sleep(1)

    async def enqueue_analysis(self, symbol: str, callback) -> str:
        """
        Enqueue a consensus analysis job.

        Args:
            symbol: Trading symbol
            callback: Async function to call with (symbol) when ready to run

        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "symbol": symbol,
            "queued_at": datetime.now(timezone.utc),
            "callback": callback
        }

        await self._queue.put(job)
        self._stats["total_queued"] += 1

        queue_size = self._queue.qsize()
        print(f"📋 ANALYSIS QUEUE: Enqueued {symbol} (job_id={job_id}, queue_size={queue_size})")

        return job_id

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_size": self._queue.qsize(),
            "current_job": self._current_job,
            "is_processing": self._is_processing,
            "total_queued": self._stats["total_queued"],
            "total_completed": self._stats["total_completed"],
            "total_failed": self._stats["total_failed"]
        }

    async def stop_processing(self):
        """Stop the queue processor"""
        self._is_processing = False
        print("🛑 ANALYSIS QUEUE: Stopped processing")


# Global queue instance
_analysis_queue: Optional[AnalysisQueue] = None


def get_analysis_queue() -> AnalysisQueue:
    """Get or create the global analysis queue instance"""
    global _analysis_queue
    if _analysis_queue is None:
        _analysis_queue = AnalysisQueue()
    return _analysis_queue
