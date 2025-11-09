/**
 * API service for interacting with the backend
 */

const API_BASE = '/api/v1';

class APIService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Request failed');
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  // Health check
  async getHealth() {
    return this.request('/health');
  }

  // Strategy Analysis
  async analyzeGradient(symbol) {
    return this.request(`/analyze/gradient?symbol=${symbol}`, { method: 'POST' });
  }

  async analyzeConfidence(symbol) {
    return this.request(`/analyze/confidence?symbol=${symbol}`, { method: 'POST' });
  }

  async analyzeConsensus(symbol, horizons = [3, 7, 14, 21], interval = '1d') {
    return this.request('/analyze/consensus', {
      method: 'POST',
      body: JSON.stringify({ symbol, horizons, interval }),
    });
  }

  // Get analysis status (for async operations)
  async getAnalysisStatus(analysisId) {
    return this.request(`/analysis/${analysisId}`);
  }

  // History
  async getAllRecentAnalyses(strategyType = null, limit = 100) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (strategyType) params.append('strategy_type', strategyType);
    return this.request(`/history/analyses?${params}`);
  }

  async getAnalysisHistory(symbol, strategyType = null, limit = 100) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (strategyType) params.append('strategy_type', strategyType);
    return this.request(`/history/analyses/${symbol}?${params}`);
  }

  async getAllConsensus(limit = 100) {
    return this.request(`/history/consensus?limit=${limit}`);
  }

  async getConsensusHistory(symbol, limit = 100) {
    return this.request(`/history/consensus/${symbol}?limit=${limit}`);
  }

  // Analytics
  async getSymbolAnalytics(symbol) {
    return this.request(`/analytics/${symbol}`);
  }

  // Tickers
  async getAvailableTickers(market = 'all') {
    return this.request(`/tickers?market=${market}`);
  }

  // Scheduled Tasks
  async getScheduledTasks(symbol = null, isActive = null) {
    const params = new URLSearchParams();
    if (symbol) params.append('symbol', symbol);
    if (isActive !== null) params.append('is_active', isActive.toString());
    const query = params.toString();
    return this.request(`/schedule${query ? '?' + query : ''}`);
  }

  async getScheduledTask(taskId) {
    return this.request(`/schedule/${taskId}`);
  }

  async createScheduledTask(task) {
    return this.request('/schedule', {
      method: 'POST',
      body: JSON.stringify(task),
    });
  }

  async updateScheduledTask(taskId, updates) {
    return this.request(`/schedule/${taskId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  }

  async deleteScheduledTask(taskId) {
    return this.request(`/schedule/${taskId}`, { method: 'DELETE' });
  }

  // Backtesting
  async createBacktest(config) {
    return this.request('/backtest', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getBacktests(symbol = null, limit = 50) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (symbol) params.append('symbol', symbol);
    return this.request(`/backtest?${params}`);
  }

  async getBacktest(runId) {
    return this.request(`/backtest/${runId}`);
  }

  async deleteBacktest(runId) {
    return this.request(`/backtest/${runId}`, { method: 'DELETE' });
  }

  // Paper Trading
  async createPaperTradingSession(config) {
    return this.request('/paper-trading', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getPaperTradingSessions(status = null, limit = 50) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (status) params.append('status', status);
    return this.request(`/paper-trading?${params}`);
  }

  async getPaperTradingSession(sessionId) {
    return this.request(`/paper-trading/${sessionId}`);
  }

  async pausePaperTradingSession(sessionId) {
    return this.request(`/paper-trading/${sessionId}/pause`, { method: 'POST' });
  }

  async resumePaperTradingSession(sessionId) {
    return this.request(`/paper-trading/${sessionId}/resume`, { method: 'POST' });
  }

  async stopPaperTradingSession(sessionId) {
    return this.request(`/paper-trading/${sessionId}/stop`, { method: 'POST' });
  }

  async deletePaperTradingSession(sessionId) {
    return this.request(`/paper-trading/${sessionId}`, { method: 'DELETE' });
  }

  // Parameter Optimization
  async createOptimization(request) {
    return this.request('/optimize/parameters', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getOptimizations(limit = 50) {
    return this.request(`/optimize?limit=${limit}`);
  }

  async getOptimization(optimizationId) {
    return this.request(`/optimize/${optimizationId}`);
  }

  async deleteOptimization(optimizationId) {
    return this.request(`/optimize/${optimizationId}`, { method: 'DELETE' });
  }

  async cancelOptimization(optimizationId) {
    return this.request(`/optimize/${optimizationId}/cancel`, { method: 'POST' });
  }

  // Generic HTTP methods for federation and other endpoints
  async get(endpoint) {
    return this.request(endpoint, { method: 'GET' });
  }

  async post(endpoint, data = null) {
    const options = { method: 'POST' };
    if (data) {
      options.body = JSON.stringify(data);
    }
    return this.request(endpoint, options);
  }

  async patch(endpoint, data) {
    return this.request(endpoint, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
}

export const api = new APIService();
export default api;
