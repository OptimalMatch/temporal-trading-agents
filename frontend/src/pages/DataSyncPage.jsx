import { useState, useEffect } from 'react';
import { Download, Plus, Trash2, Pause, Play, X, Database, Clock, CheckCircle, AlertCircle } from 'lucide-react';

const API_BASE = 'http://localhost:10750/api/v1';

function DataSyncPage() {
  const [activeJobs, setActiveJobs] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [inventory, setInventory] = useState([]);
  const [newSymbol, setNewSymbol] = useState('');
  const [newPeriod, setNewPeriod] = useState('2y');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Poll for active jobs every 2 seconds
  useEffect(() => {
    const loadData = async () => {
      try {
        const [jobsRes, watchlistRes, inventoryRes] = await Promise.all([
          fetch(`${API_BASE}/sync/jobs`),
          fetch(`${API_BASE}/sync/watchlist`),
          fetch(`${API_BASE}/sync/inventory`)
        ]);

        if (jobsRes.ok) {
          const jobs = await jobsRes.json();
          // Filter for active/pending/paused jobs
          setActiveJobs(jobs.filter(j =>
            ['running', 'pending', 'paused'].includes(j.status)
          ));
        }

        if (watchlistRes.ok) setWatchlist(await watchlistRes.json());
        if (inventoryRes.ok) setInventory(await inventoryRes.json());

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
      const res = await fetch(`${API_BASE}/sync/watchlist`, {
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
        const item = await res.json();
        setWatchlist([...watchlist, item]);
        setNewSymbol('');
      }
    } catch (err) {
      console.error('Error adding to watchlist:', err);
    }
  };

  const handleRemoveFromWatchlist = async (symbol) => {
    try {
      const res = await fetch(`${API_BASE}/sync/watchlist/${symbol}`, {
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

        {/* Add to Watchlist Form */}
        <form onSubmit={handleAddToWatchlist} className="mb-6">
          <div className="flex space-x-3">
            <input
              type="text"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value)}
              placeholder="Symbol (e.g., BTC-USD, AAPL)"
              className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
            />
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
              className="bg-brand-600 hover:bg-brand-700 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
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
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Last Updated</th>
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
                        <>
                          {new Date(item.date_range_start).toLocaleDateString()} -
                          {new Date(item.date_range_end).toLocaleDateString()}
                        </>
                      ) : '-'}
                    </td>
                    <td className="py-3 px-4 text-gray-300">{formatBytes(item.file_size_bytes)}</td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {new Date(item.last_updated_at).toLocaleDateString()}
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
