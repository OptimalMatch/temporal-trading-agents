import { useState, useEffect } from 'react';
import {
  Play, TrendingUp, TrendingDown, Activity, DollarSign,
  BarChart3, AlertCircle, CheckCircle, Loader2, Trash2, HelpCircle
} from 'lucide-react';
import { format } from 'date-fns';
import { api } from '../services/api';
import BacktestResults from '../components/BacktestResults';
import StrategyEducation from '../components/StrategyEducation';

export default function BacktestPage() {
  const [backtests, setBacktests] = useState([]);
  const [selectedBacktest, setSelectedBacktest] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showStrategyEducation, setShowStrategyEducation] = useState(false);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'BTC-USD',
    start_date: '2023-01-01',
    end_date: '2024-01-01',
    initial_capital: 100000,
    position_size_pct: 10,
    min_edge_bps: 55,
    walk_forward_enabled: true,
    train_window_days: 252,
    test_window_days: 63,
    retrain_frequency_days: 21,
    enabled_strategies: ['gradient', 'confidence', 'volatility', 'acceleration', 'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'],
  });

  useEffect(() => {
    // Initial load
    loadBacktests();

    // Poll for updates every 3 seconds
    const pollInterval = setInterval(() => {
      loadBacktests();
    }, 3000);

    return () => clearInterval(pollInterval);
  }, []); // Empty dependency array - only run once on mount

  const loadBacktests = async () => {
    try {
      const data = await api.getBacktests();
      setBacktests(data);
    } catch (error) {
      console.error('Failed to load backtests:', error);
    }
  };

  const handleCreateBacktest = async (e) => {
    e.preventDefault();

    const config = {
      symbol: formData.symbol,
      start_date: formData.start_date,
      end_date: formData.end_date,
      initial_capital: parseFloat(formData.initial_capital),
      position_size_pct: parseFloat(formData.position_size_pct),
      min_edge_bps: parseFloat(formData.min_edge_bps),
      transaction_costs: {
        taker_fee_bps: 5.0,
        maker_rebate_bps: 0.0,
        half_spread_bps: 2.0,
        slippage_coefficient: 0.1,
        adverse_selection_bps: 2.0,
        sec_fee_bps: 0.23,
      },
      walk_forward: {
        enabled: formData.walk_forward_enabled,
        train_window_days: parseInt(formData.train_window_days),
        test_window_days: parseInt(formData.test_window_days),
        retrain_frequency_days: parseInt(formData.retrain_frequency_days),
      },
      use_consensus: true,
      individual_strategies: [],
      enabled_strategies: formData.enabled_strategies,
    };

    // Close form immediately and fire off API call in background
    setShowCreateForm(false);

    // Reset form
    setFormData({
      ...formData,
      name: '',
    });

    // Fire off API call without blocking (don't await)
    api.createBacktest({
      name: formData.name || `Backtest ${formData.symbol}`,
      config,
    }).then(() => {
      // Reload backtests after successful creation
      loadBacktests();
    }).catch((error) => {
      console.error('Failed to create backtest:', error);
      alert('Failed to create backtest: ' + error.message);
      loadBacktests(); // Still reload to show any partial state
    });
  };

  const handleDeleteBacktest = async (runId) => {
    if (!confirm('Are you sure you want to delete this backtest?')) return;

    try {
      await api.deleteBacktest(runId);
      loadBacktests();
      if (selectedBacktest?.run_id === runId) {
        setSelectedBacktest(null);
      }
    } catch (error) {
      console.error('Failed to delete backtest:', error);
      alert('Failed to delete backtest: ' + error.message);
    }
  };

  const handleViewBacktest = async (backtest) => {
    try {
      const details = await api.getBacktest(backtest.run_id);
      setSelectedBacktest(details);
    } catch (error) {
      console.error('Failed to load backtest details:', error);
      alert('Failed to load backtest details: ' + error.message);
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      pending: 'bg-yellow-900/30 text-yellow-300 border-yellow-700',
      running: 'bg-blue-900/30 text-blue-300 border-blue-700',
      completed: 'bg-green-900/30 text-green-300 border-green-700',
      failed: 'bg-red-900/30 text-red-300 border-red-700',
    };

    const icons = {
      pending: AlertCircle,
      running: Loader2,
      completed: CheckCircle,
      failed: AlertCircle,
    };

    const Icon = icons[status] || AlertCircle;
    const isRunning = status === 'running';

    return (
      <span className={`inline-flex items-center space-x-1 px-2 py-1 rounded-lg border text-xs font-medium ${styles[status]}`}>
        <Icon className={`w-3 h-3 ${isRunning ? 'animate-spin' : ''}`} />
        <span>{status.toUpperCase()}</span>
      </span>
    );
  };

  const formatReturn = (value) => {
    if (value === null || value === undefined) return 'N/A';
    const pct = (value * 100).toFixed(2);
    const isPositive = value >= 0;
    return (
      <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
        {isPositive ? '+' : ''}{pct}%
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Backtesting</h1>
          <p className="text-gray-400 mt-1">
            Test consensus strategies on historical data with walk-forward validation
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg transition-colors"
        >
          <Play className="w-5 h-5" />
          <span>New Backtest</span>
        </button>
      </div>

      {/* Create Backtest Form Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-700">
              <h2 className="text-xl font-bold text-gray-100">Create New Backtest</h2>
            </div>

            <form onSubmit={handleCreateBacktest} className="p-6 space-y-6">
              {/* Basic Settings */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-200">Basic Settings</h3>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Backtest Name
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="e.g., BTC Consensus Q4 2023"
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Symbol
                  </label>
                  <input
                    type="text"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Date Range
                  </label>

                  {/* Quick Preset Buttons */}
                  <div className="flex flex-wrap gap-2 mb-3">
                    <button
                      type="button"
                      onClick={() => {
                        const end = new Date();
                        const start = new Date();
                        start.setMonth(start.getMonth() - 6);
                        setFormData({
                          ...formData,
                          start_date: start.toISOString().split('T')[0],
                          end_date: end.toISOString().split('T')[0],
                        });
                      }}
                      className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      Last 6 Months
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const end = new Date();
                        const start = new Date();
                        start.setFullYear(start.getFullYear() - 1);
                        setFormData({
                          ...formData,
                          start_date: start.toISOString().split('T')[0],
                          end_date: end.toISOString().split('T')[0],
                        });
                      }}
                      className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      Last Year
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const end = new Date();
                        const start = new Date();
                        start.setFullYear(start.getFullYear() - 2);
                        setFormData({
                          ...formData,
                          start_date: start.toISOString().split('T')[0],
                          end_date: end.toISOString().split('T')[0],
                        });
                      }}
                      className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      Last 2 Years
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setFormData({
                          ...formData,
                          start_date: '2024-01-01',
                          end_date: '2024-12-31',
                        });
                      }}
                      className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      2024
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setFormData({
                          ...formData,
                          start_date: '2023-01-01',
                          end_date: '2023-12-31',
                        });
                      }}
                      className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      2023
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        Start Date
                      </label>
                      <input
                        type="date"
                        value={formData.start_date}
                        onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        End Date
                      </label>
                      <input
                        type="date"
                        value={formData.end_date}
                        onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                        required
                      />
                    </div>
                  </div>

                  {/* Date Range Validation */}
                  {(() => {
                    const start = new Date(formData.start_date);
                    const end = new Date(formData.end_date);
                    const daysDiff = Math.floor((end - start) / (1000 * 60 * 60 * 24));
                    const minDays = formData.walk_forward_enabled ? 315 : 30;

                    if (daysDiff < minDays) {
                      return (
                        <div className="mt-2 p-2 bg-yellow-900/30 border border-yellow-700 rounded text-xs text-yellow-300">
                          <AlertCircle className="w-3 h-3 inline mr-1" />
                          {formData.walk_forward_enabled
                            ? `Walk-forward validation requires at least 315 days (currently ${daysDiff} days). Will run simple backtest instead.`
                            : `Date range: ${daysDiff} days`
                          }
                        </div>
                      );
                    }
                    return (
                      <div className="mt-2 text-xs text-gray-400">
                        Date range: {daysDiff} days
                        {formData.walk_forward_enabled && ` • ${Math.floor(daysDiff / 315)} walk-forward periods`}
                      </div>
                    );
                  })()}
                </div>
              </div>

              {/* Trading Parameters */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-200">Trading Parameters</h3>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Initial Capital ($)
                    </label>
                    <input
                      type="number"
                      value={formData.initial_capital}
                      onChange={(e) => setFormData({ ...formData, initial_capital: e.target.value })}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Position Size (%)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={formData.position_size_pct}
                      onChange={(e) => setFormData({ ...formData, position_size_pct: e.target.value })}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Minimum Edge (basis points)
                  </label>
                  <input
                    type="number"
                    step="1"
                    value={formData.min_edge_bps}
                    onChange={(e) => setFormData({ ...formData, min_edge_bps: e.target.value })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                    required
                  />
                  <p className="text-xs text-gray-400 mt-1">
                    Minimum predicted move to execute trade (default: 55 bps = 3x transaction costs)
                  </p>
                </div>
              </div>

              {/* Strategy Selection */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-200">Consensus Strategies</h3>
                  <div className="flex items-center space-x-2">
                    <button
                      type="button"
                      onClick={() => setShowStrategyEducation(true)}
                      className="px-3 py-1 text-xs bg-brand-900/30 hover:bg-brand-800/50 text-brand-400 rounded border border-brand-700 flex items-center space-x-1"
                    >
                      <HelpCircle className="w-3.5 h-3.5" />
                      <span>Learn About Strategies</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const allStrategies = ['gradient', 'confidence', 'volatility', 'acceleration', 'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'];
                        setFormData({ ...formData, enabled_strategies: allStrategies });
                      }}
                      className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      onClick={() => setFormData({ ...formData, enabled_strategies: [] })}
                      className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600"
                    >
                      Clear All
                    </button>
                  </div>
                </div>

                <p className="text-sm text-gray-400">
                  Select which strategies to include in the consensus voting system ({formData.enabled_strategies.length} selected)
                </p>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    { key: 'gradient', label: 'Forecast Gradient', description: 'Directional forecast changes' },
                    { key: 'confidence', label: 'Confidence-Weighted', description: 'High-confidence predictions' },
                    { key: 'volatility', label: 'Volatility Sizing', description: 'Volatility-adjusted positions' },
                    { key: 'acceleration', label: 'Acceleration', description: 'Forecast momentum' },
                    { key: 'swing', label: 'Swing Trading', description: 'Multi-day position holds' },
                    { key: 'risk_adjusted', label: 'Risk-Adjusted', description: 'Risk-normalized signals' },
                    { key: 'mean_reversion', label: 'Mean Reversion', description: 'Counter-trend opportunities' },
                    { key: 'multi_timeframe', label: 'Multi-Timeframe', description: 'Cross-timeframe alignment' },
                  ].map((strategy) => (
                    <label
                      key={strategy.key}
                      className={`flex items-start space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        formData.enabled_strategies.includes(strategy.key)
                          ? 'bg-brand-900/20 border-brand-700'
                          : 'bg-gray-700/30 border-gray-600 hover:border-gray-500'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={formData.enabled_strategies.includes(strategy.key)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setFormData({
                              ...formData,
                              enabled_strategies: [...formData.enabled_strategies, strategy.key],
                            });
                          } else {
                            setFormData({
                              ...formData,
                              enabled_strategies: formData.enabled_strategies.filter((s) => s !== strategy.key),
                            });
                          }
                        }}
                        className="mt-0.5 w-4 h-4 rounded border-gray-600 bg-gray-700"
                      />
                      <div className="flex-1">
                        <div className="text-sm font-medium text-gray-200">{strategy.label}</div>
                        <div className="text-xs text-gray-400">{strategy.description}</div>
                      </div>
                    </label>
                  ))}
                </div>

                {formData.enabled_strategies.length === 0 && (
                  <div className="p-2 bg-yellow-900/30 border border-yellow-700 rounded text-xs text-yellow-300">
                    <AlertCircle className="w-3 h-3 inline mr-1" />
                    Please select at least one strategy for the consensus system
                  </div>
                )}
              </div>

              {/* Walk-Forward Validation */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-200">Walk-Forward Validation</h3>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.walk_forward_enabled}
                      onChange={(e) => setFormData({ ...formData, walk_forward_enabled: e.target.checked })}
                      className="w-4 h-4 rounded border-gray-600 bg-gray-700"
                    />
                    <span className="text-sm text-gray-300">Enable</span>
                  </label>
                </div>

                {formData.walk_forward_enabled && (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Train Window (days)
                      </label>
                      <input
                        type="number"
                        value={formData.train_window_days}
                        onChange={(e) => setFormData({ ...formData, train_window_days: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Test Window (days)
                      </label>
                      <input
                        type="number"
                        value={formData.test_window_days}
                        onChange={(e) => setFormData({ ...formData, test_window_days: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Retrain Frequency (days)
                      </label>
                      <input
                        type="number"
                        value={formData.retrain_frequency_days}
                        onChange={(e) => setFormData({ ...formData, retrain_frequency_days: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-100"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading || formData.enabled_strategies.length === 0}
                  className="flex items-center space-x-2 px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Creating...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      <span>Run Backtest</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Backtest Results View */}
      {selectedBacktest ? (
        <BacktestResults
          backtest={selectedBacktest}
          onClose={() => setSelectedBacktest(null)}
        />
      ) : (
        /* Backtest List */
        <div className="grid gap-4">
          {backtests.length === 0 ? (
            <div className="bg-gray-800 rounded-lg p-12 text-center">
              <BarChart3 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-300 mb-2">
                No backtests yet
              </h3>
              <p className="text-gray-400 mb-4">
                Create your first backtest to test consensus strategies on historical data
              </p>
              <button
                onClick={() => setShowCreateForm(true)}
                className="inline-flex items-center space-x-2 px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg transition-colors"
              >
                <Play className="w-5 h-5" />
                <span>Create Backtest</span>
              </button>
            </div>
          ) : (
            backtests.map((backtest) => (
              <div
                key={backtest.run_id}
                className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-brand-700 transition-colors cursor-pointer"
                onClick={() => handleViewBacktest(backtest)}
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-100 mb-1">
                      {backtest.name}
                    </h3>
                    <p className="text-sm text-gray-400">
                      {backtest.symbol} • {backtest.start_date} to {backtest.end_date}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusBadge(backtest.status)}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteBacktest(backtest.run_id);
                      }}
                      className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4 text-gray-400 hover:text-red-400" />
                    </button>
                  </div>
                </div>

                {backtest.status === 'completed' && (
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Total Return</p>
                      <p className="text-lg font-semibold">{formatReturn(backtest.total_return)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Sharpe Ratio</p>
                      <p className="text-lg font-semibold text-gray-100">
                        {backtest.sharpe_ratio?.toFixed(2) || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Max Drawdown</p>
                      <p className="text-lg font-semibold text-red-400">
                        {backtest.max_drawdown ? (backtest.max_drawdown * 100).toFixed(2) + '%' : 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Total Trades</p>
                      <p className="text-lg font-semibold text-gray-100">
                        {backtest.total_trades || 0}
                      </p>
                    </div>
                  </div>
                )}

                {backtest.status === 'failed' && (
                  <div className="flex items-center space-x-2 text-red-400 text-sm">
                    <AlertCircle className="w-4 h-4" />
                    <span>Backtest failed</span>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Strategy Education Modal */}
      <StrategyEducation
        isOpen={showStrategyEducation}
        onClose={() => setShowStrategyEducation(false)}
      />
    </div>
  );
}
