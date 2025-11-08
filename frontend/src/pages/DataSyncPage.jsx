import { useState, useEffect } from 'react';
import { Download, Plus, Trash2, Pause, Play, X, Database, Clock, CheckCircle, AlertCircle, RefreshCw, ArrowUpCircle, Zap } from 'lucide-react';
import api from '../services/api';

const API_BASE = '/api/v1';

function DataSyncPage() {
  const [activeJobs, setActiveJobs] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [inventory, setInventory] = useState([]);
  const [newSymbol, setNewSymbol] = useState('');
  const [newPeriod, setNewPeriod] = useState('2y');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [availableTickers, setAvailableTickers] = useState([]);
  const [tickersLoading, setTickersLoading] = useState(true);
  const [selectedMarket, setSelectedMarket] = useState('crypto');

  // Load available tickers once on mount
  useEffect(() => {
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

    loadTickers();
  }, []);

  // Poll for active jobs every 2 seconds
  useEffect(() => {
    const loadData = async () => {
      try {
        const [jobsRes, watchlistRes, inventoryRes] = await Promise.all([
          fetch(`${API_BASE}/sync/jobs`),
          fetch(`${API_BASE}/watchlist`),
          fetch(`${API_BASE}/inventory`)
        ]);

        if (jobsRes.ok) {
          const data = await jobsRes.json();
          // Filter for active/pending/paused jobs
          setActiveJobs(data.jobs.filter(j =>
            ['running', 'pending', 'paused'].includes(j.status)
          ));
        }

        if (watchlistRes.ok) {
          const data = await watchlistRes.json();
          setWatchlist(data.watchlist || []);
        }

        if (inventoryRes.ok) {
          const data = await inventoryRes.json();
          setInventory(data.inventory || []);
        }

        setLoading(false);
      } catch (err) {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    loadData();
    const interval = setInterval(loadData, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const handleAddToWatchlist = async (e) => {
    e.preventDefault();
    if (!newSymbol.trim()) return;

    try {
      const res = await fetch(`${API_BASE}/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: newSymbol.toUpperCase(),
          period: newPeriod,
          interval: '1d',
          auto_sync: true
        })
      });

      if (res.ok) {
        const data = await res.json();
        setWatchlist([...watchlist, data.watchlist_item]);
        setNewSymbol('');
      }
    } catch (err) {
      console.error('Error adding to watchlist:', err);
    }
  };

  const handleRemoveFromWatchlist = async (symbol) => {
    try {
      const res = await fetch(`${API_BASE}/watchlist/${symbol}`, {
        method: 'DELETE'
      });

      if (res.ok) {
        setWatchlist(watchlist.filter(item => item.symbol !== symbol));
      }
    } catch (err) {
      console.error('Error removing from watchlist:', err);
    }
  };

  const handleStartSync = async (symbol, period = '2y') => {
    try {
      const res = await fetch(`${API_BASE}/sync/jobs?symbol=${symbol}&period=${period}&interval=1d`, {
        method: 'POST'
      });

      if (res.ok) {
        const data = await res.json();
        if (data.started) {
          // Job will show up in next poll
        }
      }
    } catch (err) {
      console.error('Error starting sync:', err);
    }
  };

  const handlePauseJob = async (jobId) => {
    try {
      await fetch(`${API_BASE}/sync/jobs/${jobId}/pause`, { method: 'POST' });
    } catch (err) {
      console.error('Error pausing job:', err);
    }
  };

  const handleResumeJob = async (jobId) => {
    try {
      await fetch(`${API_BASE}/sync/jobs/${jobId}/resume`, { method: 'POST' });
    } catch (err) {
      console.error('Error resuming job:', err);
    }
  };

  const handleCancelJob = async (jobId) => {
    try {
      await fetch(`${API_BASE}/sync/jobs/${jobId}/cancel`, { method: 'POST' });
    } catch (err) {
      console.error('Error cancelling job:', err);
    }
  };

  const handleDeleteCache = async (symbol, period, interval) => {
    if (!confirm(`Delete cached data for ${symbol} (${period}, ${interval})?`)) {
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/inventory/${symbol}/${period}/${interval}`, {
        method: 'DELETE'
      });

      if (res.ok) {
        setInventory(inventory.filter(item =>
          !(item.symbol === symbol && item.period === period && item.interval === interval)
        ));
      }
    } catch (err) {
      console.error('Error deleting cache:', err);
    }
  };

  const handleReDownload = async (symbol, period, interval) => {
    try {
      // Start a new sync job with the same parameters
      const res = await fetch(`${API_BASE}/sync/jobs?symbol=${symbol}&period=${period}&interval=${interval}`, {
        method: 'POST'
      });

      if (res.ok) {
        const data = await res.json();
        if (data.started) {
          // Job will show up in next poll
        }
      }
    } catch (err) {
      console.error('Error re-downloading:', err);
    }
  };

  const handleExtendRange = async (symbol, currentPeriod, interval) => {
    const newPeriod = prompt(
      `Extend data range for ${symbol}\n\nCurrent period: ${currentPeriod}\nEnter new period (must be wider, e.g., '5y', '10y'):`,
      '5y'
    );

    if (!newPeriod) return;

    try {
      const res = await fetch(`${API_BASE}/inventory/${symbol}/extend?new_period=${newPeriod}&interval=${interval}`, {
        method: 'POST'
      });

      const data = await res.json();

      if (res.ok) {
        if (data.started) {
          alert(`Delta download started!\n\nFetching missing data:\n${data.delta_ranges.map(r => `- ${r.type}: ${r.start.split('T')[0]} to ${r.end.split('T')[0]}`).join('\n')}`);
        } else {
          alert(data.message);
        }
      } else {
        alert(`Error: ${data.detail || 'Failed to extend data range'}`);
      }
    } catch (err) {
      console.error('Error extending range:', err);
      alert('Error extending data range');
    }
  };

  const handleScheduleDeltaSync = async (symbol, currentPeriod, interval) => {
    const newPeriod = prompt(
      `Schedule Delta Sync + Analysis for ${symbol}\n\nCurrent period: ${currentPeriod}\nEnter new period (must be wider, e.g., '5y', '10y'):\n\nThis will:\n1. Fetch only missing data (delta)\n2. Automatically run consensus analysis after sync`,
      '5y'
    );

    if (!newPeriod) return;

    try {
      const res = await fetch(`${API_BASE}/inventory/${symbol}/schedule-delta-sync?new_period=${newPeriod}&interval=${interval}&trigger_analysis=true`, {
        method: 'POST'
      });

      const data = await res.json();

      if (res.ok) {
        if (data.started) {
          alert(`✅ Delta sync scheduled with auto-analysis!\n\nSymbol: ${symbol}\nFetching missing data:\n${data.delta_ranges.map(r => `- ${r.type}: ${r.start.split('T')[0]} to ${r.end.split('T')[0]}`).join('\n')}\n\n🎯 Consensus analysis will run automatically after sync completes.`);
        } else {
          alert(data.message);
        }
      } else {
        alert(`Error: ${data.detail || 'Failed to schedule delta sync'}`);
      }
    } catch (err) {
      console.error('Error scheduling delta sync:', err);
      alert('Error scheduling delta sync');
    }
  };

  const handleToggleAutoSchedule = async (symbol, interval, currentlyEnabled) => {
    if (currentlyEnabled) {
      // Disable auto-schedule
      if (!confirm(`Disable auto-scheduling for ${symbol}?\n\nThis will stop automatic daily delta sync and analysis.`)) {
        return;
      }

      try {
        const res = await fetch(`${API_BASE}/inventory/${symbol}/auto-schedule/disable?interval=${interval}`, {
          method: 'POST'
        });

        if (res.ok) {
          alert(`✅ Auto-scheduling disabled for ${symbol}`);
          // Refresh inventory to show updated status
          window.location.reload();
        } else {
          const data = await res.json();
          alert(`Error: ${data.detail || 'Failed to disable auto-scheduling'}`);
        }
      } catch (err) {
        console.error('Error disabling auto-schedule:', err);
        alert('Error disabling auto-scheduling');
      }
    } else {
      // Enable auto-schedule
      const frequency = prompt(
        `Enable auto-scheduling for ${symbol}?\n\nSelect frequency:\n- daily (runs at 9 AM UTC)\n- 12h (every 12 hours)\n- 6h (every 6 hours)\n\nEnter frequency:`,
        'daily'
      );

      if (!frequency || !['daily', '12h', '6h'].includes(frequency)) {
        if (frequency) alert('Invalid frequency. Please enter: daily, 12h, or 6h');
        return;
      }

      try {
        const res = await fetch(`${API_BASE}/inventory/${symbol}/auto-schedule/enable?interval=${interval}&frequency=${frequency}`, {
          method: 'POST'
        });

        const data = await res.json();

        if (res.ok) {
          alert(`✅ Auto-scheduling enabled for ${symbol}!\n\nFrequency: ${frequency}\nNext run: ${data.next_scheduled_sync ? new Date(data.next_scheduled_sync).toLocaleString() : 'Calculating...'}`);
          // Refresh inventory to show updated status
          window.location.reload();
        } else {
          alert(`Error: ${data.detail || 'Failed to enable auto-scheduling'}`);
        }
      } catch (err) {
        console.error('Error enabling auto-schedule:', err);
        alert('Error enabling auto-scheduling');
      }
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Data Sync</h1>
          <p className="text-gray-400 mt-1">Manage market data downloads and caching</p>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Active Downloads */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-gray-100 mb-4 flex items-center">
          <Download className="w-5 h-5 mr-2 text-brand-500" />
          Active Downloads
        </h2>

        {activeJobs.length === 0 ? (
          <p className="text-gray-400 text-center py-8">No active downloads</p>
        ) : (
          <div className="space-y-4">
            {activeJobs.map(job => (
              <div key={job.job_id} className="bg-gray-700/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <span className="text-lg font-bold text-gray-100">{job.symbol}</span>
                    <span className="px-2 py-1 bg-gray-600 rounded text-xs text-gray-300">
                      {job.period} • {job.interval}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      job.status === 'running' ? 'bg-green-900/50 text-green-400' :
                      job.status === 'paused' ? 'bg-yellow-900/50 text-yellow-400' :
                      'bg-gray-600 text-gray-300'
                    }`}>
                      {job.status}
                    </span>
                  </div>

                  <div className="flex items-center space-x-2">
                    {job.status === 'running' && (
                      <button
                        onClick={() => handlePauseJob(job.job_id)}
                        className="p-2 hover:bg-gray-600 rounded transition-colors"
                        title="Pause"
                      >
                        <Pause className="w-4 h-4 text-gray-300" />
                      </button>
                    )}
                    {job.status === 'paused' && (
                      <button
                        onClick={() => handleResumeJob(job.job_id)}
                        className="p-2 hover:bg-gray-600 rounded transition-colors"
                        title="Resume"
                      >
                        <Play className="w-4 h-4 text-gray-300" />
                      </button>
                    )}
                    <button
                      onClick={() => handleCancelJob(job.job_id)}
                      className="p-2 hover:bg-red-900/50 rounded transition-colors"
                      title="Cancel"
                    >
                      <X className="w-4 h-4 text-red-400" />
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="mb-2">
                  <div className="flex items-center justify-between text-sm text-gray-400 mb-1">
                    <span>
                      {job.completed_files} / {job.total_files} files
                      {job.total_files > 0 && ` (${Math.round(job.progress_percent)}%)`}
                    </span>
                    {job.elapsed_seconds > 0 && (
                      <span className="flex items-center space-x-3">
                        <span>{formatDuration(job.elapsed_seconds)} elapsed</span>
                        {job.eta_seconds > 0 && (
                          <span className="flex items-center">
                            <Clock className="w-3 h-3 mr-1" />
                            {formatDuration(job.eta_seconds)} remaining
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                  <div className="w-full bg-gray-600 rounded-full h-2">
                    <div
                      className="bg-brand-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(job.progress_percent, 100)}%` }}
                    />
                  </div>
                </div>

                {job.error_message && (
                  <div className="mt-2 text-sm text-red-400 flex items-center">
                    <AlertCircle className="w-4 h-4 mr-1" />
                    {job.error_message}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Watchlist */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-gray-100 mb-4 flex items-center">
          <Database className="w-5 h-5 mr-2 text-brand-500" />
          Watchlist
        </h2>

        {/* Market Type Toggle */}
        <div className="mb-4 flex items-center justify-center">
          <div className="inline-flex rounded-lg bg-gray-700 p-1">
            <button
              type="button"
              onClick={() => setSelectedMarket('crypto')}
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
              onClick={() => setSelectedMarket('stocks')}
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

        {/* Add to Watchlist Form */}
        <form onSubmit={handleAddToWatchlist} className="mb-6">
          <div className="flex space-x-3">
            <div className="flex-1 relative">
              <input
                type="text"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                placeholder={
                  tickersLoading
                    ? "Loading symbols..."
                    : selectedMarket === 'crypto'
                    ? "Type or select a crypto (e.g., BTC-USD, ETH-USD)"
                    : "Type or select a stock (e.g., AAPL, TSLA)"
                }
                list="available-symbols"
                disabled={tickersLoading}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500 disabled:opacity-50"
              />
              <datalist id="available-symbols">
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
            <select
              value={newPeriod}
              onChange={(e) => setNewPeriod(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
            >
              <option value="30d">30 days</option>
              <option value="6mo">6 months</option>
              <option value="1y">1 year</option>
              <option value="2y">2 years</option>
              <option value="5y">5 years</option>
            </select>
            <button
              type="submit"
              disabled={tickersLoading}
              className="bg-brand-600 hover:bg-brand-700 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Plus className="w-4 h-4" />
              <span>Add</span>
            </button>
          </div>
        </form>

        {/* Watchlist Items */}
        {watchlist.length === 0 ? (
          <p className="text-gray-400 text-center py-8">No symbols in watchlist</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {watchlist.map(item => (
              <div key={item.symbol} className="bg-gray-700/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg font-bold text-gray-100">{item.symbol}</span>
                  <button
                    onClick={() => handleRemoveFromWatchlist(item.symbol)}
                    className="p-1 hover:bg-red-900/50 rounded transition-colors"
                    title="Remove"
                  >
                    <Trash2 className="w-4 h-4 text-red-400" />
                  </button>
                </div>
                <div className="text-sm text-gray-400 mb-3">
                  {item.period} • {item.interval}
                  {item.last_synced_at && (
                    <div className="mt-1 flex items-center">
                      <CheckCircle className="w-3 h-3 mr-1 text-green-400" />
                      Synced {new Date(item.last_synced_at).toLocaleDateString()}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => handleStartSync(item.symbol, item.period)}
                  className="w-full bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded text-sm font-medium transition-colors flex items-center justify-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Sync Now</span>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Data Inventory */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-gray-100 mb-4">Data Inventory</h2>

        {/* Info about delta sync for paper trading */}
        <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4 mb-4 flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="text-blue-300 font-medium mb-1">Paper Trading Requires Recent Data</p>
            <p className="text-gray-400 mb-2">
              For paper trading to generate signals, data must be recent (within last 24 hours).
              Use the <Zap className="w-4 h-4 inline text-green-400" /> <strong className="text-green-400">Delta Sync + Auto Analysis</strong> button
              to fetch only the missing recent data and automatically run consensus strategy analysis.
            </p>
            <p className="text-gray-500 text-xs">
              Alternatively, use <ArrowUpCircle className="w-3 h-3 inline text-blue-400" /> <strong>Get Delta</strong> to only fetch data without analysis.
            </p>
          </div>
        </div>

        {inventory.length === 0 ? (
          <p className="text-gray-400 text-center py-8">No cached data</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Symbol</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Period</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Data Points</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Date Range</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Size</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Auto-Schedule</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Last Synced</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Last Analyzed</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Next Sync</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {inventory.map((item, idx) => (
                  <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="py-3 px-4 text-gray-100 font-medium">{item.symbol}</td>
                    <td className="py-3 px-4 text-gray-300">{item.period}</td>
                    <td className="py-3 px-4 text-gray-300">{item.total_days.toLocaleString()}</td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {item.date_range_start && item.date_range_end ? (
                        <div>
                          <div>
                            {new Date(item.date_range_start).toLocaleDateString()} -
                            {new Date(item.date_range_end).toLocaleDateString()}
                          </div>
                          {(() => {
                            const endDate = new Date(item.date_range_end);
                            const now = new Date();
                            const daysBehind = Math.floor((now - endDate) / (1000 * 60 * 60 * 24));

                            if (daysBehind > 7) {
                              return (
                                <div className="flex items-center space-x-1 mt-1">
                                  <span className={`text-xs px-2 py-0.5 rounded ${
                                    daysBehind > 30 ? 'bg-red-900/30 text-red-400 border border-red-700' :
                                    daysBehind > 14 ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-700' :
                                    'bg-blue-900/30 text-blue-400 border border-blue-700'
                                  }`}>
                                    {daysBehind} days behind
                                  </span>
                                </div>
                              );
                            }
                            return null;
                          })()}
                        </div>
                      ) : '-'}
                    </td>
                    <td className="py-3 px-4 text-gray-300">{formatBytes(item.file_size_bytes)}</td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => handleToggleAutoSchedule(item.symbol, item.interval, item.auto_schedule_enabled)}
                        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                          item.auto_schedule_enabled
                            ? 'bg-green-900/50 text-green-400 border border-green-700 hover:bg-green-900/70'
                            : 'bg-gray-700 text-gray-400 border border-gray-600 hover:bg-gray-600'
                        }`}
                        title={item.auto_schedule_enabled ? `Auto-schedule ON (${item.schedule_frequency})` : 'Click to enable auto-schedule'}
                      >
                        {item.auto_schedule_enabled ? `✓ ${item.schedule_frequency}` : 'OFF'}
                      </button>
                    </td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {item.last_auto_sync_at ? (
                        <div className="flex flex-col">
                          <span>{new Date(item.last_auto_sync_at).toLocaleDateString()}</span>
                          <span className="text-xs text-gray-500">
                            {new Date(item.last_auto_sync_at).toLocaleTimeString()} {Intl.DateTimeFormat().resolvedOptions().timeZone}
                          </span>
                        </div>
                      ) : (
                        <span className="text-gray-500">Never</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {item.last_auto_analysis_at ? (
                        <div className="flex flex-col">
                          <span>{new Date(item.last_auto_analysis_at).toLocaleDateString()}</span>
                          <span className="text-xs text-gray-500">
                            {new Date(item.last_auto_analysis_at).toLocaleTimeString()} {Intl.DateTimeFormat().resolvedOptions().timeZone}
                          </span>
                        </div>
                      ) : (
                        <span className="text-gray-500">Never</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {item.next_scheduled_sync ? (
                        <div className="flex flex-col">
                          <span className="text-blue-400">{new Date(item.next_scheduled_sync).toLocaleDateString()}</span>
                          <span className="text-xs text-gray-500">
                            {new Date(item.next_scheduled_sync).toLocaleTimeString()} {Intl.DateTimeFormat().resolvedOptions().timeZone}
                          </span>
                        </div>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => handleScheduleDeltaSync(item.symbol, item.period, item.interval)}
                          className="p-2 hover:bg-green-900/50 rounded transition-colors border border-green-700/50"
                          title="Delta Sync + Auto Analysis - Fetch missing data and run strategy analysis automatically (recommended for paper trading)"
                        >
                          <Zap className="w-4 h-4 text-green-400" />
                        </button>
                        <button
                          onClick={() => handleExtendRange(item.symbol, item.period, item.interval)}
                          className="p-2 hover:bg-blue-900/50 rounded transition-colors"
                          title="Get Delta - Fetch only missing recent data (required for paper trading)"
                        >
                          <ArrowUpCircle className="w-4 h-4 text-blue-400" />
                        </button>
                        <button
                          onClick={() => handleReDownload(item.symbol, item.period, item.interval)}
                          className="p-2 hover:bg-gray-600 rounded transition-colors"
                          title="Re-download"
                        >
                          <RefreshCw className="w-4 h-4 text-gray-300" />
                        </button>
                        <button
                          onClick={() => handleDeleteCache(item.symbol, item.period, item.interval)}
                          className="p-2 hover:bg-red-900/50 rounded transition-colors"
                          title="Delete cache"
                        >
                          <Trash2 className="w-4 h-4 text-red-400" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default DataSyncPage;
