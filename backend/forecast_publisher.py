"""
Forecast Publisher - Publishes forecasts to external systems via webhooks
"""

import asyncio
import httpx
import json
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl
import logging

logger = logging.getLogger(__name__)


class WebhookConfig(BaseModel):
    """Webhook configuration"""
    url: str
    enabled: bool = True
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30  # seconds
    retry_count: int = 3
    retry_delay: int = 5  # seconds


class ForecastPublisher:
    """Publishes forecasts to registered webhooks"""

    def __init__(self, webhooks: List[WebhookConfig] = None):
        self.webhooks = webhooks or []
        logger.info(f"ForecastPublisher initialized with {len(self.webhooks)} webhooks")

    def add_webhook(self, webhook: WebhookConfig):
        """Add a webhook to the publisher"""
        self.webhooks.append(webhook)
        logger.info(f"Added webhook: {webhook.url}")

    def remove_webhook(self, url: str):
        """Remove a webhook by URL"""
        self.webhooks = [w for w in self.webhooks if w.url != url]
        logger.info(f"Removed webhook: {url}")

    async def publish_forecast(
        self,
        forecast_data: dict,
        event_type: str = "forecast.created"
    ) -> Dict[str, bool]:
        """
        Publish forecast to all registered webhooks

        Args:
            forecast_data: The forecast data to publish
            event_type: Event type (e.g., "forecast.created", "forecast.updated")

        Returns:
            Dict mapping webhook URLs to success status
        """
        if not self.webhooks:
            logger.debug("No webhooks configured, skipping publish")
            return {}

        # Filter enabled webhooks
        enabled_webhooks = [w for w in self.webhooks if w.enabled]

        if not enabled_webhooks:
            logger.debug("No enabled webhooks, skipping publish")
            return {}

        logger.info(f"Publishing {event_type} to {len(enabled_webhooks)} webhooks")

        # Create payload
        payload = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": forecast_data,
            "version": "1.0"
        }

        # Publish to all webhooks concurrently
        tasks = [
            self._publish_to_webhook(webhook, payload)
            for webhook in enabled_webhooks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result map
        result_map = {}
        for webhook, result in zip(enabled_webhooks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to publish to {webhook.url}: {result}")
                result_map[webhook.url] = False
            else:
                result_map[webhook.url] = result

        success_count = sum(1 for v in result_map.values() if v)
        logger.info(f"Published to {success_count}/{len(enabled_webhooks)} webhooks successfully")

        return result_map

    async def _publish_to_webhook(self, webhook: WebhookConfig, payload: dict) -> bool:
        """
        Publish to a single webhook with retry logic

        Args:
            webhook: Webhook configuration
            payload: Data to send

        Returns:
            True if successful, False otherwise
        """
        headers = {
            "Content-Type": "application/json",
            "X-Forecast-Publisher": "temporal-trading-agents",
            "X-Event-Type": payload.get("event", "unknown")
        }

        # Add custom headers
        if webhook.headers:
            headers.update(webhook.headers)

        # Retry logic
        for attempt in range(webhook.retry_count):
            try:
                async with httpx.AsyncClient(timeout=webhook.timeout) as client:
                    response = await client.post(
                        webhook.url,
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()

                    logger.info(
                        f"Successfully published to {webhook.url} "
                        f"(status: {response.status_code}, attempt: {attempt + 1})"
                    )
                    return True

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"HTTP error publishing to {webhook.url}: "
                    f"{e.response.status_code} - {e.response.text} "
                    f"(attempt {attempt + 1}/{webhook.retry_count})"
                )

            except httpx.RequestError as e:
                logger.warning(
                    f"Request error publishing to {webhook.url}: {e} "
                    f"(attempt {attempt + 1}/{webhook.retry_count})"
                )

            except Exception as e:
                logger.error(
                    f"Unexpected error publishing to {webhook.url}: {e} "
                    f"(attempt {attempt + 1}/{webhook.retry_count})"
                )

            # Wait before retry (except on last attempt)
            if attempt < webhook.retry_count - 1:
                await asyncio.sleep(webhook.retry_delay)

        logger.error(f"Failed to publish to {webhook.url} after {webhook.retry_count} attempts")
        return False

    async def test_webhook(self, webhook_url: str) -> Dict:
        """
        Test a webhook with a sample payload

        Args:
            webhook_url: URL to test

        Returns:
            Dict with test results
        """
        test_payload = {
            "event": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "message": "Test webhook notification from temporal-trading-agents",
                "symbol": "BTC-USD",
                "test": True
            },
            "version": "1.0"
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    webhook_url,
                    json=test_payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Forecast-Publisher": "temporal-trading-agents",
                        "X-Event-Type": "test"
                    }
                )

                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "message": "Webhook test successful"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Webhook test failed"
            }


def _serialize_datetime(dt) -> Optional[str]:
    """Helper to serialize datetime objects to ISO format strings"""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    # If already a string, return as-is
    return str(dt)


def create_export_payload(consensus_result: dict) -> dict:
    """
    Create standardized export payload from consensus result

    Args:
        consensus_result: Raw consensus result from database

    Returns:
        Standardized export format with all datetimes serialized
    """
    forecast_data = consensus_result.get('forecast_data', {})
    forecast_stats = consensus_result.get('forecast_stats', {})

    export_data = {
        "id": consensus_result.get('id'),
        "version": "1.0",
        "timestamp": _serialize_datetime(consensus_result.get('created_at')),
        "symbol": consensus_result.get('symbol'),
        "interval": consensus_result.get('interval', '1d'),
        "forecast": {
            "horizon_days": forecast_data.get('forecast_horizon', 14),
            "consensus": consensus_result.get('consensus', 'UNKNOWN'),
            "confidence": consensus_result.get('confidence', 0),
            "median": forecast_data.get('ensemble_median', []),
            "q25": forecast_stats.get('q25', []),
            "q75": forecast_stats.get('q75', []),
            "min": forecast_stats.get('min', []),
            "max": forecast_stats.get('max', [])
        },
        "signals": {
            "bullish_count": consensus_result.get('bullish_count', 0),
            "bearish_count": consensus_result.get('bearish_count', 0),
            "strategies_executed": consensus_result.get('strategies_executed', 0)
        },
        "metrics": {
            "current_price": forecast_data.get('current_price'),
            "expected_return_bps": consensus_result.get('expected_return_bps'),
            "volatility": consensus_result.get('volatility')
        },
        "metadata": {
            "model_version": "ensemble-v1",
            "analysis_id": consensus_result.get('id'),
            "analyzed_at": _serialize_datetime(consensus_result.get('analyzed_at'))
        }
    }

    return export_data
