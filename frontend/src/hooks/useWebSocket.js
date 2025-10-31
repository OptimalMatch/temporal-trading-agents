/**
 * React hook for WebSocket connection and real-time updates
 */
import { useEffect, useState, useCallback, useRef } from 'react';
import WebSocketService from '../services/websocket';

export function useWebSocket(taskId = null) {
  const [connected, setConnected] = useState(false);
  const [progress, setProgress] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    // Create WebSocket instance
    wsRef.current = new WebSocketService();
    wsRef.current.connect(taskId);

    // Subscribe to connection events
    const unsubConnected = wsRef.current.on('connected', () => setConnected(true));
    const unsubDisconnected = wsRef.current.on('disconnected', () => setConnected(false));

    // Subscribe to progress updates
    const unsubMessage = wsRef.current.on('message', (data) => {
      setProgress(prev => [...prev, data]);
    });

    // Cleanup on unmount
    return () => {
      unsubConnected();
      unsubDisconnected();
      unsubMessage();
      wsRef.current?.disconnect();
    };
  }, [taskId]);

  const subscribe = useCallback((event, callback) => {
    return wsRef.current?.on(event, callback);
  }, []);

  const clearProgress = useCallback(() => {
    setProgress([]);
  }, []);

  return {
    connected,
    progress,
    subscribe,
    clearProgress,
  };
}

export default useWebSocket;
