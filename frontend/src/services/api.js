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

  async analyzeConsensus(symbol, horizons = [3, 7, 14, 21], interval = '1d', inferenceMode = false) {
    return this.request('/analyze/consensus', {
      method: 'POST',
      body: JSON.stringify({ symbol, horizons, interval, inference_mode: inferenceMode }),
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

  async getAllConsensus(limit = 100, includeImported = false) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (includeImported) params.append('include_imported', 'true');
    return this.request(`/history/consensus?${params}`);
  }

  async getConsensusHistory(symbol, limit = 100, includeImported = false) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (includeImported) params.append('include_imported', 'true');
    return this.request(`/history/consensus/${symbol}?${params}`);
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

  // Auto-Optimization (backend-orchestrated multi-stage workflow)
  async createAutoOptimize(name, config) {
    return this.request(`/auto-optimize?name=${encodeURIComponent(name)}`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getAutoOptimize(autoOptimizeId) {
    return this.request(`/auto-optimize/${autoOptimizeId}`);
  }

  async listAutoOptimizes(limit = 50) {
    return this.request(`/auto-optimize?limit=${limit}`);
  }

  async cancelAutoOptimize(autoOptimizeId) {
    return this.request(`/auto-optimize/${autoOptimizeId}/cancel`, {
      method: 'POST',
    });
  }

  // Model Cache Management
  async getCachedModels() {
    return this.request('/cache/models');
  }

  async deleteCachedModel(cacheKey) {
    return this.request(`/cache/models/${cacheKey}`, { method: 'DELETE' });
  }

  async clearCache(symbol = null, interval = null) {
    const params = new URLSearchParams();
    if (symbol) params.append('symbol', symbol);
    if (interval) params.append('interval', interval);
    const query = params.toString();
    return this.request(`/cache/clear${query ? '?' + query : ''}`, { method: 'POST' });
  }

  async exportCachedModel(cacheKey) {
    const url = `${API_BASE}/cache/export/${cacheKey}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to export cached model');
    }
    const blob = await response.blob();
    return blob;
  }

  async importCachedModel(file) {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${API_BASE}/cache/import`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Failed to import cached model');
    }

    return await response.json();
  }

  async getCacheStats() {
    return this.request('/cache/stats');
  }

  // Full System Backup/Restore
  async createFullBackup() {
    return this.request('/backup/full', { method: 'POST' });
  }

  async downloadBackup(backupId) {
    const url = `${API_BASE}/backup/download/${backupId}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to download backup');
    }
    const blob = await response.blob();
    return blob;
  }

  async restoreFullBackup(file) {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${API_BASE}/backup/restore`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Failed to restore backup');
    }

    return await response.json();
  }

  async listBackups() {
    return this.request('/backup/list');
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
