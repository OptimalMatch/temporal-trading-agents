/**
 * WebSocket service for real-time progress updates
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.listeners = new Map();
    this.reconnectTimeout = null;
    this.reconnectDelay = 3000;
    this.taskId = null;
  }

  connect(taskId = null) {
    this.taskId = taskId;
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host;
    const wsPath = taskId ? `/ws/progress/${taskId}` : '/ws/progress';
    const wsUrl = `${wsProtocol}//${wsHost}${wsPath}`;

    console.log('WebSocket: Connecting to', wsUrl);

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket: Connected');
      this.notifyListeners('connected', { connected: true });

      // Send ping every 30 seconds to keep connection alive
      this.pingInterval = setInterval(() => {
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Handle pong responses
        if (data.type === 'pong') {
          return;
        }

        // Log the message to console
        console.log('📨 WebSocket message:', data);

        // Notify all listeners
        this.notifyListeners('message', data);

        // Notify specific task listeners if task_id matches
        if (data.task_id) {
          this.notifyListeners(`task:${data.task_id}`, data);
        }

        // Notify status-specific listeners
        if (data.status) {
          this.notifyListeners(`status:${data.status}`, data);
        }
      } catch (error) {
        console.error('WebSocket: Error parsing message', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket: Error', error);
      this.notifyListeners('error', { error });
    };

    this.ws.onclose = () => {
      console.log('WebSocket: Disconnected');
      this.notifyListeners('disconnected', { connected: false });

      if (this.pingInterval) {
        clearInterval(this.pingInterval);
        this.pingInterval = null;
      }

      // Attempt to reconnect after delay
      this.reconnectTimeout = setTimeout(() => {
        console.log('WebSocket: Attempting to reconnect...');
        this.connect(this.taskId);
      }, this.reconnectDelay);
    };
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.listeners.clear();
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.listeners.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  off(event, callback) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  notifyListeners(event, data) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`WebSocket: Error in listener for ${event}`, error);
        }
      });
    }
  }
}

export default WebSocketService;
