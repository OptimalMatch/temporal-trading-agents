import { useState, useEffect } from 'react';
import { Settings, Play, TrendingUp, Target, Award, AlertCircle, BarChart2, CheckCircle2, Clock, Loader2 } from 'lucide-react';
import api from '../services/api';

export default function OptimizationPage() {
  const [optimizations, setOptimizations] = useState([]);
  const [selectedOptimization, setSelectedOptimization] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(false);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'AAPL',
    startDate: '2020-01-01',
    endDate: '2023-12-31',
    initialCapital: 100000,
    optimizationMetric: 'sharpe_ratio',
    topN: 10,
    // Parameter grid
    positionSizes: [5, 10, 15, 20],
    minEdges: [30, 50, 70],
    strongBuyThresholds: [0.75, 0.80, 0.85],
    buyThresholds: [0.55, 0.60, 0.65],
    moderateBuyThresholds: [0.45, 0.50, 0.55],
  });

  useEffect(() => {
    loadOptimizations();

    // Poll for updates every 5 seconds
    const pollInterval = setInterval(() => {
      loadOptimizations();
    }, 5000);

    return () => clearInterval(pollInterval);
  }, []);

  const loadOptimizations = async () => {
    try {
      const response = await api.getOptimizations(50);
      setOptimizations(response.optimizations || []);
    } catch (error) {
      console.error('Failed to load optimizations:', error);
    }
  };

  const loadOptimizationDetails = async (optimizationId) => {
    try {
      const optimization = await api.getOptimization(optimizationId);
      setSelectedOptimization(optimization);
    } catch (error) {
      console.error('Failed to load optimization details:', error);
    }
  };

  const handleCreateOptimization = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const request = {
        name: formData.name || `${formData.symbol} Optimization`,
        base_config: {
          symbol: formData.symbol,
          start_date: formData.startDate,
          end_date: formData.endDate,
          initial_capital: formData.initialCapital,
          walk_forward: {
            enabled: true,
            train_window_days: 252,
            test_window_days: 63,
            retrain_frequency_days: 21,
          },
        },
        parameter_grid: {
          position_size_pct: formData.positionSizes,
          min_edge_bps: formData.minEdges,
          strong_buy_threshold: formData.strongBuyThresholds,
          buy_threshold: formData.buyThresholds,
          moderate_buy_threshold: formData.moderateBuyThresholds,
          sell_threshold: [0.60],  // Keep these fixed for now
          moderate_sell_threshold: [0.50],
        },
        optimization_metric: formData.optimizationMetric,
        top_n_results: formData.topN,
      };

      await api.createOptimization(request);
      setShowCreateForm(false);
      setFormData({ ...formData, name: '' });
      loadOptimizations();
    } catch (error) {
      console.error('Failed to create optimization:', error);
      alert('Failed to create optimization: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      pending: { color: 'bg-yellow-900 text-yellow-300 border-yellow-700', icon: Clock },
      running: { color: 'bg-blue-900 text-blue-300 border-blue-700', icon: Loader2, spin: true },
      completed: { color: 'bg-green-900 text-green-300 border-green-700', icon: CheckCircle2 },
      failed: { color: 'bg-red-900 text-red-300 border-red-700', icon: AlertCircle },
    };

    const config = statusConfig[status] || statusConfig.pending;
    const Icon = config.icon;

    return (
      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded border text-xs font-medium ${config.color}`}>
        <Icon className={`w-3 h-3 ${config.spin ? 'animate-spin' : ''}`} />
        {status}
      </span>
    );
  };

  const getMetricLabel = (metric) => {
    const labels = {
      sharpe_ratio: 'Sharpe Ratio',
      total_return: 'Total Return',
      profit_factor: 'Profit Factor',
      win_rate: 'Win Rate',
      max_drawdown: 'Max Drawdown',
    };
    return labels[metric] || metric;
  };

  const formatMetricValue = (value, metric) => {
    if (metric === 'total_return' || metric === 'win_rate' || metric === 'max_drawdown') {
      return `${(value * 100).toFixed(2)}%`;
    }
    return value.toFixed(2);
  };

  const calculateTotalCombinations = () => {
    return (
      formData.positionSizes.length *
      formData.minEdges.length *
      formData.strongBuyThresholds.length *
      formData.buyThresholds.length *
      formData.moderateBuyThresholds.length
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-100 flex items-center gap-3">
            <Settings className="w-8 h-8 text-brand-500" />
            Parameter Optimization
          </h1>
          <p className="text-gray-400 mt-1">
            Find optimal consensus thresholds and position sizing through grid search
          </p>
        </div>

        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          New Optimization
        </button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gray-100 mb-4">Configure Optimization</h2>

          <form onSubmit={handleCreateOptimization} className="space-y-6">
            {/* Basic Settings */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Name
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                  placeholder="AAPL Sharpe Optimization"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Symbol
                </label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Start Date
                </label>
                <input
                  type="date"
                  value={formData.startDate}
                  onChange={(e) => setFormData({ ...formData, startDate: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  End Date
                </label>
                <input
                  type="date"
                  value={formData.endDate}
                  onChange={(e) => setFormData({ ...formData, endDate: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Optimization Metric
                </label>
                <select
                  value={formData.optimizationMetric}
                  onChange={(e) => setFormData({ ...formData, optimizationMetric: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                >
                  <option value="sharpe_ratio">Sharpe Ratio</option>
                  <option value="total_return">Total Return</option>
                  <option value="profit_factor">Profit Factor</option>
                  <option value="win_rate">Win Rate</option>
                  <option value="max_drawdown">Max Drawdown (minimize)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Top N Results
                </label>
                <input
                  type="number"
                  value={formData.topN}
                  onChange={(e) => setFormData({ ...formData, topN: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                  min="1"
                  max="50"
                />
              </div>
            </div>

            {/* Parameter Grid */}
            <div className="border-t border-gray-700 pt-6">
              <h3 className="text-lg font-semibold text-gray-100 mb-4">Parameter Grid</h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Position Sizes (%)
                  </label>
                  <input
                    type="text"
                    value={formData.positionSizes.join(', ')}
                    onChange={(e) => setFormData({
                      ...formData,
                      positionSizes: e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
                    })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                    placeholder="5, 10, 15, 20"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Min Edge (bps)
                  </label>
                  <input
                    type="text"
                    value={formData.minEdges.join(', ')}
                    onChange={(e) => setFormData({
                      ...formData,
                      minEdges: e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
                    })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                    placeholder="30, 50, 70"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Strong Buy Thresholds
                  </label>
                  <input
                    type="text"
                    value={formData.strongBuyThresholds.join(', ')}
                    onChange={(e) => setFormData({
                      ...formData,
                      strongBuyThresholds: e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
                    })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                    placeholder="0.75, 0.80, 0.85"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Buy Thresholds
                  </label>
                  <input
                    type="text"
                    value={formData.buyThresholds.join(', ')}
                    onChange={(e) => setFormData({
                      ...formData,
                      buyThresholds: e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
                    })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                    placeholder="0.55, 0.60, 0.65"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Moderate Buy Thresholds
                  </label>
                  <input
                    type="text"
                    value={formData.moderateBuyThresholds.join(', ')}
                    onChange={(e) => setFormData({
                      ...formData,
                      moderateBuyThresholds: e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
                    })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-brand-500"
                    placeholder="0.45, 0.50, 0.55"
                  />
                </div>
              </div>

              <div className="mt-4 p-3 bg-gray-700 rounded border border-gray-600">
                <p className="text-sm text-gray-300">
                  <strong>Total Combinations:</strong> {calculateTotalCombinations()}
                  <span className="text-gray-400 ml-2">
                    (Estimated time: ~{Math.ceil(calculateTotalCombinations() / 20)} minutes)
                  </span>
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 justify-end">
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Start Optimization
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Optimization List */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-gray-100">Recent Optimizations</h2>
        </div>

        {optimizations.length === 0 ? (
          <div className="p-12 text-center text-gray-400">
            <BarChart2 className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No optimizations yet. Create one to get started!</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-700">
            {optimizations.map((opt) => (
              <div
                key={opt.optimization_id}
                onClick={() => loadOptimizationDetails(opt.optimization_id)}
                className="p-4 hover:bg-gray-750 cursor-pointer transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-gray-100">{opt.name}</h3>
                      {getStatusBadge(opt.status)}
                    </div>

                    <div className="flex items-center gap-4 text-sm text-gray-400">
                      <span className="flex items-center gap-1">
                        <Target className="w-4 h-4" />
                        {opt.base_config?.symbol}
                      </span>
                      <span className="flex items-center gap-1">
                        <TrendingUp className="w-4 h-4" />
                        {getMetricLabel(opt.optimization_metric)}
                      </span>
                      {opt.status === 'running' && (
                        <span>
                          Progress: {opt.completed_combinations}/{opt.total_combinations}
                        </span>
                      )}
                      {opt.status === 'completed' && opt.best_parameters && (
                        <span className="flex items-center gap-1 text-green-400">
                          <Award className="w-4 h-4" />
                          Best: Pos {opt.best_parameters.position_size_pct}% |
                          SB {(opt.best_parameters.strong_buy_threshold * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="text-right text-sm text-gray-400">
                    <div>{new Date(opt.created_at).toLocaleDateString()}</div>
                    <div>{new Date(opt.created_at).toLocaleTimeString()}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Optimization Details Modal */}
      {selectedOptimization && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedOptimization(null)}>
          <div className="bg-gray-800 border border-gray-700 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="sticky top-0 bg-gray-800 border-b border-gray-700 px-6 py-4 flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-gray-100">{selectedOptimization.name}</h2>
                <p className="text-sm text-gray-400 mt-1">
                  {selectedOptimization.base_config?.symbol} | {getMetricLabel(selectedOptimization.optimization_metric)}
                </p>
              </div>
              <button
                onClick={() => setSelectedOptimization(null)}
                className="text-gray-400 hover:text-gray-200"
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Status and Progress */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-gray-700 p-4 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Status</div>
                  <div>{getStatusBadge(selectedOptimization.status)}</div>
                </div>
                <div className="bg-gray-700 p-4 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Progress</div>
                  <div className="text-lg font-semibold text-gray-100">
                    {selectedOptimization.completed_combinations}/{selectedOptimization.total_combinations}
                  </div>
                </div>
                <div className="bg-gray-700 p-4 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Date Range</div>
                  <div className="text-sm text-gray-100">
                    {selectedOptimization.base_config?.start_date} to {selectedOptimization.base_config?.end_date}
                  </div>
                </div>
                <div className="bg-gray-700 p-4 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Execution Time</div>
                  <div className="text-lg font-semibold text-gray-100">
                    {selectedOptimization.execution_time_ms
                      ? `${(selectedOptimization.execution_time_ms / 1000).toFixed(1)}s`
                      : 'Running...'}
                  </div>
                </div>
              </div>

              {/* Best Parameters */}
              {selectedOptimization.best_parameters && (
                <div className="bg-green-900 bg-opacity-20 border border-green-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Award className="w-5 h-5 text-green-400" />
                    <h3 className="text-lg font-semibold text-green-400">Best Parameters</h3>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-sm text-gray-400">Position Size</div>
                      <div className="text-lg font-semibold text-gray-100">
                        {selectedOptimization.best_parameters.position_size_pct}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Min Edge</div>
                      <div className="text-lg font-semibold text-gray-100">
                        {selectedOptimization.best_parameters.min_edge_bps} bps
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Strong Buy Threshold</div>
                      <div className="text-lg font-semibold text-gray-100">
                        {(selectedOptimization.best_parameters.strong_buy_threshold * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Buy Threshold</div>
                      <div className="text-lg font-semibold text-gray-100">
                        {(selectedOptimization.best_parameters.buy_threshold * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Moderate Buy Threshold</div>
                      <div className="text-lg font-semibold text-gray-100">
                        {(selectedOptimization.best_parameters.moderate_buy_threshold * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Top Results */}
              {selectedOptimization.top_results && selectedOptimization.top_results.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-100 mb-3">Top {selectedOptimization.top_results.length} Results</h3>
                  <div className="space-y-2">
                    {selectedOptimization.top_results.map((result, idx) => (
                      <div
                        key={idx}
                        className={`p-4 rounded-lg border ${
                          idx === 0
                            ? 'bg-green-900 bg-opacity-20 border-green-700'
                            : 'bg-gray-700 border-gray-600'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <span className={`text-2xl font-bold ${idx === 0 ? 'text-green-400' : 'text-gray-400'}`}>
                              #{result.rank}
                            </span>
                            <div>
                              <div className="text-sm text-gray-400">
                                Pos: {result.parameters.position_size_pct}% |
                                Edge: {result.parameters.min_edge_bps}bps |
                                SB: {(result.parameters.strong_buy_threshold * 100).toFixed(0)}% |
                                B: {(result.parameters.buy_threshold * 100).toFixed(0)}% |
                                MB: {(result.parameters.moderate_buy_threshold * 100).toFixed(0)}%
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className={`text-2xl font-bold ${idx === 0 ? 'text-green-400' : 'text-gray-100'}`}>
                              {formatMetricValue(result.metric_value, selectedOptimization.optimization_metric)}
                            </div>
                            <div className="text-sm text-gray-400">
                              {getMetricLabel(selectedOptimization.optimization_metric)}
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-5 gap-4 mt-3 pt-3 border-t border-gray-600">
                          <div>
                            <div className="text-xs text-gray-400">Total Return</div>
                            <div className="text-sm font-semibold text-gray-100">
                              {(result.metrics.total_return * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-400">Sharpe</div>
                            <div className="text-sm font-semibold text-gray-100">
                              {result.metrics.sharpe_ratio.toFixed(2)}
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-400">Max DD</div>
                            <div className="text-sm font-semibold text-red-400">
                              {(result.metrics.max_drawdown * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-400">Win Rate</div>
                            <div className="text-sm font-semibold text-gray-100">
                              {(result.metrics.win_rate * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-400">Trades</div>
                            <div className="text-sm font-semibold text-gray-100">
                              {result.metrics.total_trades}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
