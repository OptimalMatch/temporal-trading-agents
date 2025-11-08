import { useState, useEffect } from 'react';
import {
  Play, TrendingUp, TrendingDown, Activity, DollarSign,
  Pause, Square, AlertCircle, CheckCircle, Loader2, Trash2, Clock
} from 'lucide-react';
import { format } from 'date-fns';
import { api } from '../services/api';

export default function PaperTradingPage() {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);

  // Helper to format UTC timestamp to local timezone
  // Backend now sends timezone-aware timestamps (with 'Z' suffix)
  const formatLocalTime = (utcTimestamp, formatStr = 'short') => {
    const date = new Date(utcTimestamp);

    if (formatStr === 'short') {
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      });
    } else if (formatStr === 'full') {
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
    }
    return date.toLocaleString();
  };

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'BTC-USD',
    initial_capital: 100000,
    position_size_pct: 10,
    min_edge_bps: 55,
    check_interval_minutes: 1,
    auto_execute: true,
    use_consensus: true,
  });

  useEffect(() => {
    loadSessions();

    // Poll for updates every 5 seconds
    const pollInterval = setInterval(() => {
      loadSessions();
      if (selectedSession) {
        loadSessionDetails(selectedSession.session_id);
      }
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [selectedSession?.session_id]);

  const loadSessions = async () => {
    try {
      const data = await api.getPaperTradingSessions();
      setSessions(data);
    } catch (error) {
      console.error('Failed to load paper trading sessions:', error);
    }
  };

  const loadSessionDetails = async (sessionId) => {
    try {
      const details = await api.getPaperTradingSession(sessionId);
      setSelectedSession(details);
    } catch (error) {
      console.error('Failed to load session details:', error);
    }
  };

  const handleCreateSession = async (e) => {
    e.preventDefault();

    const config = {
      symbol: formData.symbol,
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
      use_consensus: formData.use_consensus,
      check_interval_minutes: parseInt(formData.check_interval_minutes),
      auto_execute: formData.auto_execute,
    };

    setShowCreateForm(false);

    try {
      await api.createPaperTradingSession({
        name: formData.name || `Paper Trading ${formData.symbol}`,
        config,
      });

      // Reset form
      setFormData({
        ...formData,
        name: '',
      });

      loadSessions();
    } catch (error) {
      console.error('Failed to create paper trading session:', error);
      alert('Failed to create session: ' + error.message);
    }
  };

  const handlePauseSession = async (sessionId) => {
    try {
      await api.pausePaperTradingSession(sessionId);
      loadSessions();
      if (selectedSession?.session_id === sessionId) {
        loadSessionDetails(sessionId);
      }
    } catch (error) {
      console.error('Failed to pause session:', error);
      alert('Failed to pause session: ' + error.message);
    }
  };

  const handleResumeSession = async (sessionId) => {
    try {
      await api.resumePaperTradingSession(sessionId);
      loadSessions();
      if (selectedSession?.session_id === sessionId) {
        loadSessionDetails(sessionId);
      }
    } catch (error) {
      console.error('Failed to resume session:', error);
      alert('Failed to resume session: ' + error.message);
    }
  };

  const handleStopSession = async (sessionId) => {
    if (!confirm('Are you sure you want to stop this session? This cannot be undone.')) return;

    try {
      await api.stopPaperTradingSession(sessionId);
      loadSessions();
      if (selectedSession?.session_id === sessionId) {
        setSelectedSession(null);
      }
    } catch (error) {
      console.error('Failed to stop session:', error);
      alert('Failed to stop session: ' + error.message);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!confirm('Are you sure you want to delete this session? This will permanently remove all data and cannot be undone.')) return;

    try {
      await api.deletePaperTradingSession(sessionId);
      loadSessions();
      if (selectedSession?.session_id === sessionId) {
        setSelectedSession(null);
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
      alert('Failed to delete session: ' + error.message);
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      active: 'bg-green-900/30 text-green-300 border-green-700',
      paused: 'bg-yellow-900/30 text-yellow-300 border-yellow-700',
      stopped: 'bg-gray-700/30 text-gray-400 border-gray-600',
      error: 'bg-red-900/30 text-red-300 border-red-700',
    };

    const icons = {
      active: Activity,
      paused: Pause,
      stopped: Square,
      error: AlertCircle,
    };

    const Icon = icons[status] || AlertCircle;

    return (
      <span className={`inline-flex items-center space-x-1 px-2 py-1 rounded text-xs font-medium border ${styles[status]}`}>
        <Icon className={`w-3 h-3 ${status === 'active' ? 'animate-pulse' : ''}`} />
        <span>{status.toUpperCase()}</span>
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Paper Trading</h1>
          <p className="text-gray-400 mt-1">
            Test strategies in real-time with simulated capital
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-lg transition-colors"
        >
          <Play className="w-5 h-5" />
          <span>New Session</span>
        </button>
      </div>

      {/* Create Session Form */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h2 className="text-2xl font-bold text-gray-100 mb-4">Create Paper Trading Session</h2>

              <form onSubmit={handleCreateSession} className="space-y-4">
                {/* Session Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Session Name
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="My Paper Trading Session"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500"
                  />
                </div>

                {/* Symbol */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Symbol
                  </label>
                  <input
                    type="text"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                    placeholder="BTC-USD"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500"
                    required
                  />
                </div>

                {/* Initial Capital */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Initial Capital ($)
                  </label>
                  <input
                    type="number"
                    value={formData.initial_capital}
                    onChange={(e) => setFormData({ ...formData, initial_capital: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
                    required
                  />
                </div>

                {/* Position Size */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Position Size (% of capital)
                  </label>
                  <input
                    type="number"
                    value={formData.position_size_pct}
                    onChange={(e) => setFormData({ ...formData, position_size_pct: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
                    min="1"
                    max="100"
                    required
                  />
                </div>

                {/* Min Edge */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Minimum Edge (bps)
                  </label>
                  <input
                    type="number"
                    value={formData.min_edge_bps}
                    onChange={(e) => setFormData({ ...formData, min_edge_bps: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Signals below this threshold will be ignored (recommended: 55 bps)
                  </p>
                </div>

                {/* Check Interval */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Check Interval (minutes)
                  </label>
                  <input
                    type="number"
                    value={formData.check_interval_minutes}
                    onChange={(e) => setFormData({ ...formData, check_interval_minutes: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
                    min="1"
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    How often to check for new signals
                  </p>
                </div>

                {/* Auto Execute */}
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={formData.auto_execute}
                    onChange={(e) => setFormData({ ...formData, auto_execute: e.target.checked })}
                    className="w-4 h-4 text-brand-500 bg-gray-700 border-gray-600 rounded focus:ring-brand-500"
                  />
                  <label className="text-sm text-gray-300">
                    Auto-execute signals (uncheck to review manually)
                  </label>
                </div>

                {/* Actions */}
                <div className="flex items-center justify-end space-x-3 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowCreateForm(false)}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-lg transition-colors"
                  >
                    Create Session
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-gray-100 mb-4">Active Sessions</h2>

          {sessions.length === 0 ? (
            <div className="text-center py-8">
              <Activity className="w-12 h-12 text-gray-600 mx-auto mb-2" />
              <p className="text-gray-400">No paper trading sessions yet</p>
              <p className="text-sm text-gray-500 mt-1">
                Create a session to start testing your strategies
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  onClick={() => loadSessionDetails(session.session_id)}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedSession?.session_id === session.session_id
                      ? 'bg-brand-900/20 border-brand-700'
                      : 'bg-gray-700/30 border-gray-600 hover:bg-gray-700/50'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-semibold text-gray-100">{session.name}</h3>
                      <p className="text-sm text-gray-400">{session.symbol}</p>
                    </div>
                    {getStatusBadge(session.status)}
                  </div>

                  <div className="grid grid-cols-2 gap-3 mt-3">
                    <div>
                      <p className="text-xs text-gray-500">Equity</p>
                      <p className="text-sm font-semibold text-gray-200">
                        ${session.current_equity.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">P&L</p>
                      <p className={`text-sm font-semibold ${session.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${session.total_pnl.toFixed(2)} ({session.total_pnl_pct.toFixed(2)}%)
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Trades</p>
                      <p className="text-sm font-semibold text-gray-200">
                        {session.total_trades}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Min Edge</p>
                      <p className="text-sm font-semibold text-blue-400">
                        {session.min_edge_bps} bps
                      </p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-xs text-gray-500">Started</p>
                      <p className="text-sm text-gray-400">
                        {formatLocalTime(session.started_at, 'short')}
                      </p>
                      <p className="text-xs text-gray-500">
                        {Intl.DateTimeFormat().resolvedOptions().timeZone}
                      </p>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center space-x-2 mt-3">
                    {session.status === 'active' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePauseSession(session.session_id);
                        }}
                        className="text-xs px-2 py-1 bg-yellow-900/30 hover:bg-yellow-800/50 text-yellow-400 rounded border border-yellow-700"
                      >
                        <Pause className="w-3 h-3" />
                      </button>
                    )}
                    {session.status === 'paused' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleResumeSession(session.session_id);
                        }}
                        className="text-xs px-2 py-1 bg-green-900/30 hover:bg-green-800/50 text-green-400 rounded border border-green-700"
                      >
                        <Play className="w-3 h-3" />
                      </button>
                    )}
                    {(session.status === 'active' || session.status === 'paused') && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStopSession(session.session_id);
                        }}
                        className="text-xs px-2 py-1 bg-red-900/30 hover:bg-red-800/50 text-red-400 rounded border border-red-700"
                      >
                        <Square className="w-3 h-3" />
                      </button>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteSession(session.session_id);
                      }}
                      className="text-xs px-2 py-1 bg-gray-700/30 hover:bg-gray-600/50 text-gray-400 rounded border border-gray-600"
                      title="Delete session"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Session Details */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-gray-100 mb-4">Session Details</h2>

          {!selectedSession ? (
            <div className="text-center py-8">
              <CheckCircle className="w-12 h-12 text-gray-600 mx-auto mb-2" />
              <p className="text-gray-400">Select a session to view details</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Current Positions */}
              <div>
                <h3 className="text-sm font-semibold text-gray-300 mb-2">Current Positions</h3>
                {selectedSession.positions && selectedSession.positions.length > 0 ? (
                  <div className="space-y-2">
                    {selectedSession.positions.map((pos, idx) => (
                      <div key={idx} className="p-3 bg-gray-700/30 rounded border border-gray-600">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-semibold text-gray-200">{pos.symbol}</span>
                          <span className="text-sm text-gray-400">{pos.shares.toFixed(4)} shares</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">Entry:</span>
                            <span className="text-gray-300 ml-1">${pos.entry_price.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Current:</span>
                            <span className="text-gray-300 ml-1">${pos.current_price.toFixed(2)}</span>
                          </div>
                          <div className="col-span-2">
                            <span className="text-gray-500">Unrealized P&L:</span>
                            <span className={`ml-1 font-semibold ${pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              ${pos.unrealized_pnl.toFixed(2)} ({pos.unrealized_pnl_pct.toFixed(2)}%)
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 py-2">No open positions</p>
                )}
              </div>

              {/* Recent Trades */}
              <div>
                <h3 className="text-sm font-semibold text-gray-300 mb-2">Recent Trades</h3>
                {selectedSession.trades && selectedSession.trades.length > 0 ? (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {selectedSession.trades.slice(-10).reverse().map((trade, idx) => (
                      <div key={idx} className="p-3 bg-gray-700/30 rounded border border-gray-600 text-xs">
                        <div className="flex items-center justify-between mb-1">
                          <span className={`font-semibold ${trade.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                            {trade.side.toUpperCase()} {trade.symbol}
                          </span>
                          <span className="text-gray-500">
                            {formatLocalTime(trade.timestamp, 'short')}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-400">
                            {trade.shares.toFixed(4)} @ ${trade.price.toFixed(2)}
                          </span>
                          {trade.metadata?.realized_pnl !== undefined && (
                            <span className={`font-semibold ${trade.metadata.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              ${trade.metadata.realized_pnl.toFixed(2)}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 py-2">No trades yet</p>
                )}
              </div>

              {/* Signal Activity Log */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-gray-300 flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    Signal Activity
                  </h3>
                  <span className="text-xs text-gray-500">
                    {Intl.DateTimeFormat().resolvedOptions().timeZone}
                  </span>
                </div>
                {selectedSession.signal_logs && selectedSession.signal_logs.length > 0 ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {selectedSession.signal_logs.slice(-20).reverse().map((log, idx) => (
                      <div key={idx} className={`p-2 rounded border text-xs ${
                        log.action_taken === 'executed' ? 'bg-green-900/20 border-green-700' :
                        log.action_taken === 'rejected' ? 'bg-red-900/20 border-red-700' :
                        log.action_taken === 'no_signal' ? 'bg-gray-700/20 border-gray-600' :
                        'bg-yellow-900/20 border-yellow-700'
                      }`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-gray-500">
                            {formatLocalTime(log.timestamp, 'full')}
                          </span>
                          {log.current_price && (
                            <span className="text-gray-400">
                              ${log.current_price.toFixed(2)}
                            </span>
                          )}
                        </div>
                        {log.signal && (
                          <div className="flex items-center space-x-2 mb-1">
                            <span className={`font-semibold ${
                              log.signal === 'BUY' ? 'text-green-400' :
                              log.signal === 'SELL' ? 'text-red-400' :
                              'text-gray-400'
                            }`}>
                              {log.signal}
                            </span>
                            {log.expected_return_bps && (
                              <span className="text-xs text-gray-500">
                                {log.expected_return_bps.toFixed(1)} bps
                              </span>
                            )}
                            {log.consensus_score && (
                              <span className="text-xs text-gray-500">
                                {(log.consensus_score * 100).toFixed(0)}% consensus
                              </span>
                            )}
                          </div>
                        )}
                        <div className="text-gray-400">{log.reason}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 py-2">No signal checks yet</p>
                )}
              </div>

              {/* Session Info */}
              <div className="pt-4 border-t border-gray-700">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-500">Cash:</span>
                    <span className="text-gray-300 ml-2">${selectedSession.cash.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Winning:</span>
                    <span className="text-gray-300 ml-2">{selectedSession.winning_trades}/{selectedSession.total_trades}</span>
                  </div>
                  {selectedSession.next_signal_check && (
                    <div className="col-span-2">
                      <span className="text-gray-500">Next check:</span>
                      <span className="text-gray-300 ml-2">
                        {formatLocalTime(selectedSession.next_signal_check, 'full')}
                      </span>
                      <span className="text-xs text-gray-500 ml-2">
                        ({Intl.DateTimeFormat().resolvedOptions().timeZone})
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
