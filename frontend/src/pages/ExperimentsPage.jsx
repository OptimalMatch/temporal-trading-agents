import { useState, useEffect } from 'react';
import { Target, Plus, TrendingUp, TrendingDown, Trash2, CheckCircle, Eye, BarChart3 } from 'lucide-react';

const API_BASE = '/api/v1';

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    symbol: 'BTC-USD',
    parameter_tested: 'min_edge_bps'
  });

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const response = await fetch(`${API_BASE}/experiments`);
      const data = await response.json();
      setExperiments(data);
    } catch (error) {
      console.error('Failed to load experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateExperiment = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch(
        `${API_BASE}/experiments?name=${encodeURIComponent(formData.name)}&description=${encodeURIComponent(formData.description)}&symbol=${formData.symbol}&parameter_tested=${formData.parameter_tested}`,
        { method: 'POST' }
      );

      if (response.ok) {
        setShowCreateForm(false);
        setFormData({
          name: '',
          description: '',
          symbol: 'BTC-USD',
          parameter_tested: 'min_edge_bps'
        });
        loadExperiments();
      } else {
        alert('Failed to create experiment');
      }
    } catch (error) {
      console.error('Error creating experiment:', error);
      alert('Error creating experiment');
    }
  };

  const completeExperiment = async (experimentId) => {
    if (!confirm('Mark this experiment as completed?')) return;

    try {
      const response = await fetch(`${API_BASE}/experiments/${experimentId}/complete`, {
        method: 'POST'
      });

      if (response.ok) {
        loadExperiments();
      }
    } catch (error) {
      console.error('Error completing experiment:', error);
    }
  };

  const deleteExperiment = async (experimentId) => {
    if (!confirm('Delete this experiment? (Sessions will not be deleted)')) return;

    try {
      const response = await fetch(`${API_BASE}/experiments/${experimentId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        loadExperiments();
        if (selectedExperiment?.experiment_id === experimentId) {
          setSelectedExperiment(null);
          setComparisonData(null);
        }
      }
    } catch (error) {
      console.error('Error deleting experiment:', error);
    }
  };

  const viewComparison = async (experiment) => {
    setSelectedExperiment(experiment);
    setLoadingComparison(true);

    try {
      const response = await fetch(`${API_BASE}/experiments/${experiment.experiment_id}/compare`);
      const data = await response.json();
      setComparisonData(data);
    } catch (error) {
      console.error('Error loading comparison:', error);
      alert('Failed to load comparison data');
    } finally {
      setLoadingComparison(false);
    }
  };

  const formatDateTime = (dateStr) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatNumber = (num, decimals = 2) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toFixed(decimals);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading experiments...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Target className="w-8 h-8 text-purple-400" />
          <h1 className="text-3xl font-bold text-gray-100">Experiments</h1>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          <span>New Experiment</span>
        </button>
      </div>

      {/* Create Experiment Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg max-w-2xl w-full border border-gray-700">
            <div className="p-6">
              <h2 className="text-2xl font-bold text-gray-100 mb-4">Create New Experiment</h2>

              <form onSubmit={handleCreateExperiment} className="space-y-4">
                {/* Experiment Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Experiment Name *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="BTC Min Edge Comparison"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  />
                </div>

                {/* Description */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Description *
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Compare different minimum edge thresholds for BTC-USD trading..."
                    rows="3"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  />
                </div>

                {/* Symbol */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Symbol *
                  </label>
                  <input
                    type="text"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                    placeholder="BTC-USD"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    The trading symbol this experiment will test
                  </p>
                </div>

                {/* Parameter Being Tested */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Parameter Being Tested *
                  </label>
                  <select
                    value={formData.parameter_tested}
                    onChange={(e) => setFormData({ ...formData, parameter_tested: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  >
                    <option value="min_edge_bps">Minimum Edge (bps)</option>
                    <option value="position_size_pct">Position Size (%)</option>
                    <option value="check_interval_minutes">Check Interval (minutes)</option>
                    <option value="other">Other</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    The strategy parameter you want to A/B test
                  </p>
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
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                  >
                    Create Experiment
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Experiments List */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-750">
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Parameter
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Sessions
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan="7" className="px-6 py-8 text-center text-gray-500">
                    No experiments yet. Create one to start A/B testing your strategies.
                  </td>
                </tr>
              ) : (
                experiments.map((exp) => (
                  <tr key={exp.experiment_id} className="hover:bg-gray-750">
                    <td className="px-6 py-4 text-sm text-gray-300">
                      <div>
                        <div className="font-semibold">{exp.name}</div>
                        <div className="text-xs text-gray-500">{exp.description}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-300">
                      <span className="font-mono">{exp.symbol}</span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-300">
                      {exp.parameter_tested}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-300">
                      <span className="font-semibold">{exp.session_ids.length}</span>
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          exp.status === 'active'
                            ? 'bg-green-900/50 text-green-400'
                            : 'bg-gray-700 text-gray-400'
                        }`}
                      >
                        {exp.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-400">
                      {formatDateTime(exp.created_at)}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => viewComparison(exp)}
                          className="p-2 hover:bg-purple-900/50 rounded border border-purple-700/50"
                          title="View Comparison"
                        >
                          <Eye className="w-4 h-4 text-purple-400" />
                        </button>
                        {exp.status === 'active' && (
                          <button
                            onClick={() => completeExperiment(exp.experiment_id)}
                            className="p-2 hover:bg-green-900/50 rounded border border-green-700/50"
                            title="Mark Complete"
                          >
                            <CheckCircle className="w-4 h-4 text-green-400" />
                          </button>
                        )}
                        <button
                          onClick={() => deleteExperiment(exp.experiment_id)}
                          className="p-2 hover:bg-red-900/50 rounded border border-red-700/50"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4 text-red-400" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison View */}
      {selectedExperiment && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-700 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <BarChart3 className="w-6 h-6 text-purple-400" />
              <div>
                <h2 className="text-xl font-bold text-gray-100">{selectedExperiment.name}</h2>
                <p className="text-sm text-gray-400">{selectedExperiment.description}</p>
              </div>
            </div>
            <button
              onClick={() => {
                setSelectedExperiment(null);
                setComparisonData(null);
              }}
              className="text-gray-400 hover:text-gray-300"
            >
              Close
            </button>
          </div>

          <div className="p-6">
            {loadingComparison ? (
              <div className="text-center text-gray-400 py-8">Loading comparison data...</div>
            ) : comparisonData ? (
              <div className="space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-750 rounded-lg p-4 border border-gray-700">
                    <p className="text-xs text-gray-400 mb-1">Total Sessions</p>
                    <p className="text-2xl font-bold text-gray-100">
                      {comparisonData.summary.total_sessions}
                    </p>
                  </div>
                  <div className="bg-gray-750 rounded-lg p-4 border border-gray-700">
                    <p className="text-xs text-gray-400 mb-1">Best P&L Session</p>
                    <p className="text-sm font-mono text-green-400">
                      {comparisonData.summary.best_pnl_session?.substring(0, 8) || 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Comparison Table */}
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gray-750">
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Session</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Min Edge (bps)</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">P&L</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">P&L %</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Total Trades</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Win/Loss</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Win Rate</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Max DD</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700">
                      {comparisonData.sessions.length === 0 ? (
                        <tr>
                          <td colSpan="9" className="px-4 py-8 text-center text-gray-500">
                            No sessions in this experiment yet.
                          </td>
                        </tr>
                      ) : (
                        comparisonData.sessions.map((session) => {
                          const isBestPnl = session.session_id === comparisonData.summary.best_pnl_session;

                          return (
                            <tr
                              key={session.session_id}
                              className={`hover:bg-gray-750 ${
                                isBestPnl ? 'bg-purple-900/20' : ''
                              }`}
                            >
                              <td className="px-4 py-3 text-xs">
                                <div>
                                  <div className="font-mono text-gray-300">
                                    {session.session_id.substring(0, 8)}
                                  </div>
                                  <div className="text-gray-500">{session.name}</div>
                                </div>
                              </td>
                              <td className="px-4 py-3 text-sm text-blue-400 font-semibold">
                                {session.min_edge_bps}
                              </td>
                              <td className="px-4 py-3 text-sm">
                                <span
                                  className={
                                    session.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                                  }
                                >
                                  ${formatNumber(session.total_pnl)}
                                  {isBestPnl && ' 🏆'}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-sm">
                                <span
                                  className={
                                    session.total_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'
                                  }
                                >
                                  {formatNumber(session.total_pnl_pct)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-300">
                                {session.total_trades}
                              </td>
                              <td className="px-4 py-3 text-sm">
                                <span className="text-green-400">{session.winning_trades || 0}</span>
                                <span className="text-gray-500"> / </span>
                                <span className="text-red-400">{session.losing_trades || 0}</span>
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-300">
                                {formatNumber(session.win_rate)}%
                              </td>
                              <td className="px-4 py-3 text-sm text-orange-400">
                                {formatNumber(session.max_drawdown)}%
                              </td>
                              <td className="px-4 py-3 text-xs">
                                <span
                                  className={`px-2 py-1 rounded ${
                                    session.status === 'active'
                                      ? 'bg-green-900/50 text-green-400'
                                      : 'bg-gray-700 text-gray-400'
                                  }`}
                                >
                                  {session.status}
                                </span>
                              </td>
                            </tr>
                          );
                        })
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">No comparison data available</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
