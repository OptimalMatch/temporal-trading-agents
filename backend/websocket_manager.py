"""
WebSocket connection manager for real-time progress updates.
"""
from typing import Dict, Set
from fastapi import WebSocket
import json
import asyncio
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        # Active connections by task_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Global connections (receive all updates)
        self.global_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, task_id: str = None):
        """Accept new WebSocket connection"""
        await websocket.accept()

        if task_id:
            if task_id not in self.active_connections:
                self.active_connections[task_id] = set()
            self.active_connections[task_id].add(websocket)
        else:
            # Global connection receives all updates
            self.global_connections.add(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str = None):
        """Remove WebSocket connection"""
        if task_id and task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        else:
            self.global_connections.discard(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")

    async def broadcast_to_task(self, message: dict, task_id: str):
        """Broadcast message to all connections listening to specific task"""
        # Send to task-specific connections
        if task_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error broadcasting to task connection: {e}")
                    disconnected.add(connection)

            # Clean up disconnected clients
            for connection in disconnected:
                self.active_connections[task_id].discard(connection)

        # Send to global connections
        await self.broadcast_global(message)

    async def broadcast_global(self, message: dict):
        """Broadcast message to all global connections"""
        disconnected = set()
        for connection in self.global_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to global connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.global_connections.discard(connection)

    async def send_progress(self, task_id: str, symbol: str, strategy_type: str,
                           status: str, progress: float, message: str,
                           details: dict = None):
        """Send progress update"""
        update = {
            "task_id": task_id,
            "symbol": symbol,
            "strategy_type": strategy_type,
            "status": status,
            "progress": progress,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        print(f"📡 Sending progress: task_id={task_id}, status={status}, progress={progress}%, msg={message}")
        print(f"   Task connections: {len(self.active_connections.get(task_id, set()))}, Global: {len(self.global_connections)}")
        await self.broadcast_to_task(update, task_id)


# Global instance
manager = ConnectionManager()
