import { useState, useEffect } from 'react';
import { Settings, Play, TrendingUp, Target, Award, AlertCircle, BarChart2, CheckCircle2, Clock, Loader2, HelpCircle, Info, Lightbulb, X, Trash2, XCircle } from 'lucide-react';
import api from '../services/api';

export default function OptimizationPage() {
  const [optimizations, setOptimizations] = useState([]);
  const [selectedOptimization, setSelectedOptimization] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [availableTickers, setAvailableTickers] = useState([]);
  const [tickersLoading, setTickersLoading] = useState(true);
  const [selectedMarket, setSelectedMarket] = useState('crypto');

  // Form state - store parameter grids as strings for easier editing
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'BTC-USD',
    startDate: '2020-01-01',
    endDate: '2023-12-31',
    initialCapital: 100000,
    optimizationMetric: 'sharpe_ratio',
    topN: 10,
    // Parameter grid - stored as strings to allow comma typing
    positionSizes: '5, 10, 15, 20',
    minEdges: '30, 50, 70',
    strongBuyThresholds: '0.75, 0.80, 0.85',
    buyThresholds: '0.55, 0.60, 0.65',
    moderateBuyThresholds: '0.45, 0.50, 0.55',
    enabled_strategies: ['gradient', 'confidence', 'volatility', 'acceleration', 'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'],
  });

  useEffect(() => {
    loadOptimizations();
    loadTickers();

    // Poll for updates every 5 seconds
    const pollInterval = setInterval(() => {
      loadOptimizations();
    }, 5000);

    return () => clearInterval(pollInterval);
  }, []);

  const loadTickers = async () => {
    try {
      const data = await api.getAvailableTickers('all');
      setAvailableTickers(data.tickers || []);
      setTickersLoading(false);
    } catch (err) {
      console.error('Error loading tickers:', err);
      setTickersLoading(false);
    }
  };

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

  const handleCancelOptimization = async (optimizationId, e) => {
    e.stopPropagation(); // Prevent triggering the row click
    if (!confirm('Cancel this optimization? It will be marked as failed.')) return;

    try {
      await api.cancelOptimization(optimizationId);
      loadOptimizations();
    } catch (error) {
      console.error('Failed to cancel optimization:', error);
      alert('Failed to cancel optimization: ' + error.message);
    }
  };

  const handleDeleteOptimization = async (optimizationId, e) => {
    e.stopPropagation(); // Prevent triggering the row click
    if (!confirm('Delete this optimization? This cannot be undone.')) return;

    try {
      await api.deleteOptimization(optimizationId);
      loadOptimizations();
    } catch (error) {
      console.error('Failed to delete optimization:', error);
      alert('Failed to delete optimization: ' + error.message);
    }
  };

  // Helper function to parse comma-separated values
  const parseValues = (str) => {
    return str.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
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
          enabled_strategies: formData.enabled_strategies,
          walk_forward: {
            enabled: true,
            train_window_days: 252,
            test_window_days: 63,
            retrain_frequency_days: 21,
          },
        },
        parameter_grid: {
          position_size_pct: parseValues(formData.positionSizes),
          min_edge_bps: parseValues(formData.minEdges),
          strong_buy_threshold: parseValues(formData.strongBuyThresholds),
          buy_threshold: parseValues(formData.buyThresholds),
          moderate_buy_threshold: parseValues(formData.moderateBuyThresholds),
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

  const calculateTotalCombinations = () => {
    return parseValues(formData.positionSizes).length *
           parseValues(formData.minEdges).length *
           parseValues(formData.strongBuyThresholds).length *
           parseValues(formData.buyThresholds).length *
           parseValues(formData.moderateBuyThresholds).length;
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      pending: { color: 'bg-yellow-900/50 text-yellow-400 border-yellow-700', icon: Clock, spin: false },
      running: { color: 'bg-blue-900/50 text-blue-400 border-blue-700', icon: Loader2, spin: true },
      completed: { color: 'bg-green-900/50 text-green-400 border-green-700', icon: CheckCircle2, spin: false },
      failed: { color: 'bg-red-900/50 text-red-400 border-red-700', icon: AlertCircle, spin: false },
    };

    const config = statusConfig[status] || statusConfig.pending;
    const Icon = config.icon;

    return (
      <span className={`px-3 py-1 rounded-lg text-sm font-medium border ${config.color} flex items-center space-x-2`}>
        <Icon className={`w-4 h-4 ${config.spin ? 'animate-spin' : ''}`} />
        <span className="capitalize">{status}</span>
      </span>
    );
  };

  const formatMetric = (metric, value) => {
    if (value === null || value === undefined) return 'N/A';

    switch (metric) {
      case 'sharpe_ratio':
      case 'profit_factor':
        return value.toFixed(2);
      case 'total_return':
      case 'max_drawdown':
      case 'win_rate':
        return (value * 100).toFixed(2) + '%';
      default:
        return value.toFixed(2);
    }
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Parameter Optimization</h1>
          <p className="text-gray-400 mt-1">Find the best trading strategy parameters for your portfolio</p>
        </div>
        {!showCreateForm && (
          <button
            onClick={() => setShowCreateForm(true)}
            className="bg-brand-600 hover:bg-brand-700 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <Play className="w-5 h-5" />
            <span>New Optimization</span>
          </button>
        )}
      </div>

      {/* Educational Info Section */}
      <div className="bg-gradient-to-br from-brand-900/30 to-blue-900/30 border border-brand-700/50 rounded-lg p-6">
        <div className="flex items-start space-x-4">
          <div className="bg-brand-500/20 rounded-lg p-3">
            <Info className="w-6 h-6 text-brand-400" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-bold text-gray-100 mb-2">What is Parameter Optimization?</h3>
            <p className="text-gray-300 mb-3">
              Parameter optimization systematically tests different combinations of trading strategy parameters to find the settings that would have performed best historically. This helps you discover optimal values for position sizing, risk management thresholds, and consensus voting rules.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2 mb-2">
                  <Target className="w-5 h-5 text-brand-400" />
                  <h4 className="font-bold text-gray-100">How It Works</h4>
                </div>
                <p className="text-sm text-gray-400">
                  We run backtests across all parameter combinations using your historical data, then rank results by your chosen metric (Sharpe Ratio, Total Return, etc.).
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2 mb-2">
                  <Award className="w-5 h-5 text-brand-400" />
                  <h4 className="font-bold text-gray-100">What You Get</h4>
                </div>
                <p className="text-sm text-gray-400">
                  The top-performing parameter sets are saved and can be reviewed in detail. You can then manually apply these settings to future backtests or paper trading sessions.
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center space-x-2 mb-2">
                  <Lightbulb className="w-5 h-5 text-brand-400" />
                  <h4 className="font-bold text-gray-100">Using Results</h4>
                </div>
                <p className="text-sm text-gray-400">
                  Review the optimized parameters below, then copy them to the Backtest or Paper Trading pages to test them on new data or trade with them live.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Create Optimization Form */}
      {showCreateForm && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-100">Configure Optimization</h2>
            <button
              onClick={() => setShowCreateForm(false)}
              className="text-gray-400 hover:text-gray-200"
            >
              Cancel
            </button>
          </div>

          <form onSubmit={handleCreateOptimization} className="space-y-6">
            {/* Basic Configuration */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-100 border-b border-gray-700 pb-2">Basic Configuration</h3>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Optimization Name (Optional)
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., BTC Conservative Settings"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">Leave blank to auto-generate from symbol</p>
              </div>

              {/* Market Type Toggle */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Market Type</label>
                <div className="inline-flex rounded-lg bg-gray-700 p-1">
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedMarket('crypto');
                      setFormData({ ...formData, symbol: 'BTC-USD' });
                    }}
                    className={`px-6 py-2 rounded-md font-medium transition-colors ${
                      selectedMarket === 'crypto'
                        ? 'bg-brand-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Crypto
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedMarket('stocks');
                      setFormData({ ...formData, symbol: 'AAPL' });
                    }}
                    className={`px-6 py-2 rounded-md font-medium transition-colors ${
                      selectedMarket === 'stocks'
                        ? 'bg-brand-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Stocks
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Symbol
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                    placeholder={
                      tickersLoading
                        ? "Loading symbols..."
                        : selectedMarket === 'crypto'
                        ? "Type or select a crypto (e.g., BTC-USD, ETH-USD)"
                        : "Type or select a stock (e.g., AAPL, TSLA)"
                    }
                    list="optimization-symbols"
                    disabled={tickersLoading}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500 disabled:opacity-50"
                  />
                  <datalist id="optimization-symbols">
                    {availableTickers
                      .filter(t => t.market === selectedMarket)
                      .map((ticker) => (
                        <option key={ticker.symbol} value={ticker.symbol}>
                          {ticker.name}
                        </option>
                      ))}
                  </datalist>
                  {!tickersLoading && availableTickers.length > 0 && (
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-gray-500">
                      {availableTickers.filter(t => t.market === selectedMarket).length} {selectedMarket === 'crypto' ? 'cryptos' : 'stocks'}
                    </div>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">The asset to optimize parameters for</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Start Date</label>
                  <input
                    type="date"
                    value={formData.startDate}
                    onChange={(e) => setFormData({ ...formData, startDate: e.target.value })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">End Date</label>
                  <input
                    type="date"
                    value={formData.endDate}
                    onChange={(e) => setFormData({ ...formData, endDate: e.target.value })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Initial Capital ($)
                </label>
                <input
                  type="number"
                  value={formData.initialCapital}
                  onChange={(e) => setFormData({ ...formData, initialCapital: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">Starting portfolio value for backtests</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Optimization Metric
                  </label>
                  <select
                    value={formData.optimizationMetric}
                    onChange={(e) => setFormData({ ...formData, optimizationMetric: e.target.value })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
                  >
                    <option value="sharpe_ratio">Sharpe Ratio (Risk-Adjusted Returns)</option>
                    <option value="total_return">Total Return (Profit %)</option>
                    <option value="profit_factor">Profit Factor (Win/Loss Ratio)</option>
                    <option value="win_rate">Win Rate (% Winning Trades)</option>
                    <option value="max_drawdown">Max Drawdown (Lowest Risk)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {formData.optimizationMetric === 'sharpe_ratio' && 'Recommended: Balances returns with risk'}
                    {formData.optimizationMetric === 'total_return' && 'Maximizes profit (may be riskier)'}
                    {formData.optimizationMetric === 'profit_factor' && 'Favors consistent winning trades'}
                    {formData.optimizationMetric === 'win_rate' && 'Maximizes % of profitable trades'}
                    {formData.optimizationMetric === 'max_drawdown' && 'Minimizes largest loss period'}
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Top Results to Save
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={formData.topN}
                    onChange={(e) => setFormData({ ...formData, topN: parseInt(e.target.value) })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">Number of best parameter sets to keep</p>
                </div>
              </div>
            </div>

            {/* Parameter Grid Configuration */}
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-gray-700 pb-2">
                <h3 className="text-lg font-semibold text-gray-100">Parameter Grid</h3>
                <div className="flex items-center space-x-2 text-sm text-gray-400">
                  <HelpCircle className="w-4 h-4" />
                  <span>Enter comma-separated values to test</span>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <Info className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm text-blue-200">
                    <p className="font-semibold mb-1">How to use parameter grids:</p>
                    <p>Each parameter can have multiple values to test. The optimizer will test every combination. Start with fewer values (2-3 per parameter) for faster results, then refine with more values.</p>
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Position Sizes (% of Portfolio)
                </label>
                <input
                  type="text"
                  value={formData.positionSizes}
                  onChange={(e) => setFormData({
                    ...formData,
                    positionSizes: e.target.value
                  })}
                  placeholder="5, 10, 15, 20"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  <strong>What it means:</strong> How much of your portfolio to risk per trade. <strong>Recommended:</strong> 5-15% for balanced risk. Higher = more profit potential but larger drawdowns.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Minimum Edge (Basis Points)
                </label>
                <input
                  type="text"
                  value={formData.minEdges}
                  onChange={(e) => setFormData({
                    ...formData,
                    minEdges: e.target.value
                  })}
                  placeholder="30, 50, 70"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  <strong>What it means:</strong> Minimum expected profit (in 0.01%) to take a trade. <strong>Recommended:</strong> 50-70 to cover transaction costs. Lower = more trades but higher costs. Higher = fewer, more selective trades.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Strong Buy Thresholds (Consensus %)
                </label>
                <input
                  type="text"
                  value={formData.strongBuyThresholds}
                  onChange={(e) => setFormData({
                    ...formData,
                    strongBuyThresholds: e.target.value
                  })}
                  placeholder="0.75, 0.80, 0.85"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  <strong>What it means:</strong> % of strategies that must agree for a "Strong Buy" signal. <strong>Recommended:</strong> 0.75-0.85. Higher = more agreement required = fewer but stronger signals.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Buy Thresholds (Consensus %)
                </label>
                <input
                  type="text"
                  value={formData.buyThresholds}
                  onChange={(e) => setFormData({
                    ...formData,
                    buyThresholds: e.target.value
                  })}
                  placeholder="0.55, 0.60, 0.65"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  <strong>What it means:</strong> % of strategies that must agree for a "Buy" signal. <strong>Recommended:</strong> 0.55-0.65. Moderate threshold for regular trading signals.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Moderate Buy Thresholds (Consensus %)
                </label>
                <input
                  type="text"
                  value={formData.moderateBuyThresholds}
                  onChange={(e) => setFormData({
                    ...formData,
                    moderateBuyThresholds: e.target.value
                  })}
                  placeholder="0.45, 0.50, 0.55"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  <strong>What it means:</strong> % of strategies for a "Moderate Buy" signal. <strong>Recommended:</strong> 0.45-0.55. Lower threshold for weaker but still valid signals.
                </p>
              </div>
            </div>

            {/* Strategy Selection */}
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-gray-700 pb-2">
                <h3 className="text-lg font-semibold text-gray-100">Consensus Strategies</h3>
                <div className="flex items-center space-x-2">
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
                <div className="p-2 bg-yellow-900/30 border border-yellow-700 rounded text-xs text-yellow-300 flex items-center space-x-2">
                  <AlertCircle className="w-3 h-3" />
                  <span>Please select at least one strategy for the consensus system</span>
                </div>
              )}
            </div>

            {/* Summary */}
            <div className="bg-brand-900/20 border border-brand-700 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-gray-300 mb-1">Total Combinations</div>
                  <div className="text-2xl font-bold text-brand-400">{calculateTotalCombinations()}</div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-300 mb-1">Estimated Time</div>
                  <div className="text-lg font-semibold text-gray-100">
                    ~{Math.ceil(calculateTotalCombinations() / 18)} minutes
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                Each combination runs a full backtest. Fewer combinations = faster results.
              </p>
            </div>

            {/* Submit Button */}
            <div className="flex space-x-4">
              <button
                type="submit"
                disabled={loading || formData.enabled_strategies.length === 0}
                className="flex-1 bg-brand-600 hover:bg-brand-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Starting Optimization...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Start Optimization</span>
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Optimizations List */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-gray-100 mb-4 flex items-center">
          <BarChart2 className="w-5 h-5 mr-2 text-brand-500" />
          Recent Optimizations
        </h2>

        {optimizations.length === 0 ? (
          <div className="text-center py-12">
            <Settings className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 mb-2">No optimizations yet</p>
            <p className="text-sm text-gray-500">Create your first optimization to find optimal trading parameters</p>
          </div>
        ) : (
          <div className="space-y-3">
            {optimizations.map((opt) => (
              <div
                key={opt.optimization_id}
                onClick={() => loadOptimizationDetails(opt.optimization_id)}
                className="bg-gray-700/50 hover:bg-gray-700 rounded-lg p-4 cursor-pointer transition-colors border border-gray-600 hover:border-brand-600"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-100">{opt.name}</h3>
                    <p className="text-sm text-gray-400">
                      {opt.base_config.symbol} • {opt.base_config.start_date} to {opt.base_config.end_date}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusBadge(opt.status)}
                    {(opt.status === 'running' || opt.status === 'pending') && (
                      <button
                        onClick={(e) => handleCancelOptimization(opt.optimization_id, e)}
                        className="p-2 hover:bg-red-900/50 rounded transition-colors"
                        title="Cancel optimization"
                      >
                        <XCircle className="w-5 h-5 text-red-400" />
                      </button>
                    )}
                    <button
                      onClick={(e) => handleDeleteOptimization(opt.optimization_id, e)}
                      className="p-2 hover:bg-red-900/50 rounded transition-colors"
                      title="Delete optimization"
                    >
                      <Trash2 className="w-5 h-5 text-gray-400 hover:text-red-400" />
                    </button>
                  </div>
                </div>

                {opt.status === 'running' && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between text-sm text-gray-400 mb-1">
                      <span>{opt.completed_combinations || 0} / {opt.total_combinations || 0} combinations</span>
                      <span>{Math.round((opt.completed_combinations || 0) / (opt.total_combinations || 1) * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div
                        className="bg-brand-500 h-2 rounded-full transition-all"
                        style={{ width: `${Math.min((opt.completed_combinations || 0) / (opt.total_combinations || 1) * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                )}

                {opt.status === 'completed' && opt.best_result && (
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="text-gray-400">Best {getMetricLabel(opt.optimization_metric)}</div>
                      <div className="text-gray-100 font-semibold">
                        {formatMetric(opt.optimization_metric, opt.best_result.metrics[opt.optimization_metric])}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">Position Size</div>
                      <div className="text-gray-100 font-semibold">
                        {opt.best_result.parameters.position_size_pct}%
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">Min Edge</div>
                      <div className="text-gray-100 font-semibold">
                        {opt.best_result.parameters.min_edge_bps} bps
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">Strong Buy</div>
                      <div className="text-gray-100 font-semibold">
                        {(opt.best_result.parameters.strong_buy_threshold * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                )}

                <div className="text-xs text-gray-500 mt-2">
                  Created {new Date(opt.created_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Results Modal */}
      {selectedOptimization && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedOptimization(null)}
        >
          <div
            className="bg-gray-800 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-gray-800 border-b border-gray-700 p-6 z-10">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-gray-100">{selectedOptimization.name}</h2>
                  <p className="text-gray-400 mt-1">
                    {selectedOptimization.base_config.symbol} • Optimized for {getMetricLabel(selectedOptimization.optimization_metric)}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedOptimization(null)}
                  className="text-gray-400 hover:text-gray-200"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>

            <div className="p-6 space-y-6">
              {/* Status */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Status</div>
                  {getStatusBadge(selectedOptimization.status)}
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Total Combinations</div>
                  <div className="text-2xl font-bold text-gray-100">{selectedOptimization.total_combinations || 0}</div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Completed</div>
                  <div className="text-2xl font-bold text-gray-100">{selectedOptimization.completed_combinations || 0}</div>
                </div>
              </div>

              {/* Enabled Strategies */}
              {selectedOptimization.base_config.enabled_strategies && selectedOptimization.base_config.enabled_strategies.length > 0 && (
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3">
                    Consensus Strategies ({selectedOptimization.base_config.enabled_strategies.length} of 8 enabled)
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {(() => {
                      const allStrategies = [
                        { key: 'gradient', label: 'Forecast Gradient' },
                        { key: 'confidence', label: 'Confidence-Weighted' },
                        { key: 'volatility', label: 'Volatility Sizing' },
                        { key: 'acceleration', label: 'Acceleration' },
                        { key: 'swing', label: 'Swing Trading' },
                        { key: 'risk_adjusted', label: 'Risk-Adjusted' },
                        { key: 'mean_reversion', label: 'Mean Reversion' },
                        { key: 'multi_timeframe', label: 'Multi-Timeframe' },
                      ];

                      const enabledStrategies = selectedOptimization.base_config.enabled_strategies || allStrategies.map(s => s.key);

                      return allStrategies.map((strategy) => {
                        const isEnabled = enabledStrategies.includes(strategy.key);
                        return (
                          <div
                            key={strategy.key}
                            className={`flex items-center space-x-2 text-xs rounded px-2 py-1 ${
                              isEnabled
                                ? 'bg-brand-900/20 border border-brand-700'
                                : 'bg-gray-700/30 border border-gray-600 opacity-50'
                            }`}
                          >
                            <div className={`w-1.5 h-1.5 rounded-full ${isEnabled ? 'bg-brand-500' : 'bg-gray-500'}`}></div>
                            <span className={isEnabled ? 'text-gray-200' : 'text-gray-500'}>{strategy.label}</span>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
              )}

              {/* How to Use Results */}
              {selectedOptimization.status === 'completed' && (
                <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <Lightbulb className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-green-200">
                      <p className="font-semibold mb-2">How to use these optimized parameters:</p>
                      <ol className="list-decimal list-inside space-y-1">
                        <li>Review the top-ranked parameter sets below</li>
                        <li>Note the best values for position size, min edge, and thresholds</li>
                        <li>Go to the <strong>Backtest</strong> page and manually enter these parameters to validate on new data</li>
                        <li>If results remain strong, use these settings in <strong>Paper Trading</strong> or live trading</li>
                      </ol>
                    </div>
                  </div>
                </div>
              )}

              {/* Results Table */}
              {selectedOptimization.results && selectedOptimization.results.length > 0 && (
                <div>
                  <h3 className="text-lg font-bold text-gray-100 mb-4">
                    Top {selectedOptimization.results.length} Results (Ranked by {getMetricLabel(selectedOptimization.optimization_metric)})
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Rank</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Position Size</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Min Edge</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Strong Buy</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Buy</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Moderate Buy</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Sharpe</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Total Return</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Max DD</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium">Win Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedOptimization.results.map((result, idx) => (
                          <tr
                            key={result.run_id}
                            className={`border-b border-gray-700/50 ${
                              idx === 0 ? 'bg-green-900/20' : 'hover:bg-gray-700/30'
                            }`}
                          >
                            <td className="py-3 px-4">
                              {idx === 0 ? (
                                <span className="bg-green-600 text-white px-2 py-1 rounded font-bold text-sm">
                                  #1
                                </span>
                              ) : (
                                <span className="text-gray-400">#{idx + 1}</span>
                              )}
                            </td>
                            <td className="py-3 px-4 text-gray-100 font-medium">
                              {result.parameters.position_size_pct}%
                            </td>
                            <td className="py-3 px-4 text-gray-100">{result.parameters.min_edge_bps}</td>
                            <td className="py-3 px-4 text-gray-100">
                              {(result.parameters.strong_buy_threshold * 100).toFixed(0)}%
                            </td>
                            <td className="py-3 px-4 text-gray-100">
                              {(result.parameters.buy_threshold * 100).toFixed(0)}%
                            </td>
                            <td className="py-3 px-4 text-gray-100">
                              {(result.parameters.moderate_buy_threshold * 100).toFixed(0)}%
                            </td>
                            <td className="py-3 px-4 text-gray-100">
                              {result.metrics.sharpe_ratio?.toFixed(2) || 'N/A'}
                            </td>
                            <td className="py-3 px-4 text-gray-100">
                              {result.metrics.total_return ? (result.metrics.total_return * 100).toFixed(2) + '%' : 'N/A'}
                            </td>
                            <td className="py-3 px-4 text-gray-100">
                              {result.metrics.max_drawdown ? (result.metrics.max_drawdown * 100).toFixed(2) + '%' : 'N/A'}
                            </td>
                            <td className="py-3 px-4 text-gray-100">
                              {result.metrics.win_rate ? (result.metrics.win_rate * 100).toFixed(2) + '%' : 'N/A'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
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
