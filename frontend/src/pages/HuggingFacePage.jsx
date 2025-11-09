import { useState, useEffect } from 'react';
import { Upload, Plus, Trash2, Edit2, Save, X, Eye, EyeOff, Cloud, Download, CheckCircle, AlertCircle, Layers } from 'lucide-react';

const API_BASE = '/api/v1';

function HuggingFacePage() {
  const [configs, setConfigs] = useState([]);
  const [consensusAnalyses, setConsensusAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showTokenMap, setShowTokenMap] = useState({});

  // Modal states
  const [showExportModal, setShowExportModal] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);
  const [showEnsembleExportModal, setShowEnsembleExportModal] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState(null);

  // Form states
  const [formData, setFormData] = useState({
    symbol: '',
    interval: '1h',
    repo_id: '',
    token: '',
    private: false,
    auto_export: false,
    auto_import: false
  });

  const [exportFormData, setExportFormData] = useState({
    lookback: 45,
    focus: 'momentum',
    forecast_horizon: 24,
    commit_message: ''
  });

  const [importFormData, setImportFormData] = useState({
    repo_id: '',
    lookback: 45,
    focus: 'momentum',
    forecast_horizon: 24,
    force: false
  });

  const [ensembleExportFormData, setEnsembleExportFormData] = useState({
    consensus_id: '',
    commit_message: ''
  });

  useEffect(() => {
    loadConfigs();
    loadConsensusAnalyses();
  }, []);

  const loadConfigs = async () => {
    try {
      const res = await fetch(`${API_BASE}/huggingface/configs`);
      if (res.ok) {
        const data = await res.json();
        setConfigs(data.configs || []);
      }
      setLoading(false);
    } catch (err) {
      console.error('Error loading configs:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  const loadConsensusAnalyses = async () => {
    try {
      const res = await fetch(`${API_BASE}/history/consensus?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setConsensusAnalyses(data.results || []);
      }
    } catch (err) {
      console.error('Error loading consensus analyses:', err);
    }
  };

  const handleAddConfig = async (e) => {
    e.preventDefault();

    if (!formData.symbol.trim() || !formData.repo_id.trim()) {
      alert('Symbol and Repository ID are required');
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/huggingface/configs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: formData.symbol.toUpperCase(),
          interval: formData.interval,
          repo_id: formData.repo_id,
          token: formData.token || null,
          private: formData.private,
          auto_export: formData.auto_export,
          auto_import: formData.auto_import
        })
      });

      if (res.ok) {
        const newConfig = await res.json();
        setConfigs([...configs, newConfig]);
        setFormData({
          symbol: '',
          interval: '1h',
          repo_id: '',
          token: '',
          private: false,
          auto_export: false,
          auto_import: false
        });
        setShowAddForm(false);
      } else {
        const errorData = await res.json();
        alert(`Error: ${errorData.detail || 'Failed to create configuration'}`);
      }
    } catch (err) {
      console.error('Error adding config:', err);
      alert('Failed to add configuration');
    }
  };

  const handleUpdateConfig = async (configId, updates) => {
    try {
      const res = await fetch(`${API_BASE}/huggingface/configs/${configId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });

      if (res.ok) {
        const updatedConfig = await res.json();
        setConfigs(configs.map(c => c.id === configId ? updatedConfig : c));
        setEditingId(null);
      } else {
        const errorData = await res.json();
        alert(`Error: ${errorData.detail || 'Failed to update configuration'}`);
      }
    } catch (err) {
      console.error('Error updating config:', err);
      alert('Failed to update configuration');
    }
  };

  const handleDeleteConfig = async (configId, symbol, interval) => {
    if (!confirm(`Delete HuggingFace configuration for ${symbol} (${interval})?`)) {
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/huggingface/configs/${configId}`, {
        method: 'DELETE'
      });

      if (res.ok) {
        setConfigs(configs.filter(c => c.id !== configId));
      } else {
        const errorData = await res.json();
        alert(`Error: ${errorData.detail || 'Failed to delete configuration'}`);
      }
    } catch (err) {
      console.error('Error deleting config:', err);
      alert('Failed to delete configuration');
    }
  };

  const openExportModal = (config) => {
    setSelectedConfig(config);
    setExportFormData({
      lookback: 45,
      focus: 'momentum',
      forecast_horizon: 24,
      commit_message: `Export ${config.symbol} model for ${config.interval} trading`
    });
    setShowExportModal(true);
  };

  const handleExportModel = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch(`${API_BASE}/huggingface/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedConfig.symbol,
          interval: selectedConfig.interval,
          lookback: exportFormData.lookback,
          focus: exportFormData.focus,
          forecast_horizon: exportFormData.forecast_horizon,
          commit_message: exportFormData.commit_message || undefined
        })
      });

      const data = await res.json();

      if (res.ok) {
        alert(`Model exported successfully!\n\nRepository: ${data.repo_id}\nURL: ${data.url}`);
        loadConfigs(); // Reload to update last_export timestamp
        setShowExportModal(false);
      } else {
        alert(`Export failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error exporting model:', err);
      alert('Failed to export model');
    }
  };

  const openImportModal = (config) => {
    setSelectedConfig(config);
    setImportFormData({
      repo_id: config.repo_id,
      lookback: 45,
      focus: 'momentum',
      forecast_horizon: 24,
      force: false
    });
    setShowImportModal(true);
  };

  const handleImportModel = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch(`${API_BASE}/huggingface/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_id: importFormData.repo_id,
          symbol: selectedConfig.symbol,
          interval: selectedConfig.interval,
          lookback: importFormData.lookback,
          focus: importFormData.focus,
          forecast_horizon: importFormData.forecast_horizon,
          force: importFormData.force
        })
      });

      const data = await res.json();

      if (res.ok) {
        alert(`Model imported successfully!\n\nModel: ${data.model_path}`);
        loadConfigs(); // Reload to update last_import timestamp
        setShowImportModal(false);
      } else {
        alert(`Import failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error importing model:', err);
      alert('Failed to import model');
    }
  };

  const openEnsembleExportModal = (config) => {
    setSelectedConfig(config);
    setEnsembleExportFormData({
      consensus_id: '',
      commit_message: `Export ensemble models for ${config.symbol} (${config.interval})`
    });
    setShowEnsembleExportModal(true);
  };

  const handleExportEnsemble = async (e) => {
    e.preventDefault();

    if (!ensembleExportFormData.consensus_id) {
      alert('Please select a consensus analysis');
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/huggingface/export-ensemble`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          consensus_id: ensembleExportFormData.consensus_id,
          repo_id: selectedConfig.repo_id,
          commit_message: ensembleExportFormData.commit_message || undefined
        })
      });

      const data = await res.json();

      if (res.ok) {
        const successMsg = `${data.message}\n\nRepository: ${data.repo_id}\n\nExported models:\n${data.exported.map(m => `- ${m.focus} (lookback=${m.lookback})`).join('\n')}`;
        const failureMsg = data.failed.length > 0 ? `\n\nFailed models:\n${data.failed.map(m => `- ${m.focus} (lookback=${m.lookback}): ${m.error}`).join('\n')}` : '';
        alert(successMsg + failureMsg);
        loadConfigs(); // Reload to update last_export timestamp
        setShowEnsembleExportModal(false);
      } else {
        alert(`Ensemble export failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error exporting ensemble:', err);
      alert('Failed to export ensemble');
    }
  };

  const toggleTokenVisibility = (configId) => {
    setShowTokenMap(prev => ({
      ...prev,
      [configId]: !prev[configId]
    }));
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Never';
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-400">Loading configurations...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-100 flex items-center gap-2">
            <Upload className="w-8 h-8 text-brand-500" />
            HuggingFace Model Sharing
          </h1>
          <p className="text-gray-400 mt-1">
            Export and import trained models to/from HuggingFace Hub
          </p>
        </div>

        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          {showAddForm ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          {showAddForm ? 'Cancel' : 'Add Configuration'}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-700 text-red-400 px-4 py-3 rounded-lg">
          Error: {error}
        </div>
      )}

      {showAddForm && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-gray-100 mb-4">Add HuggingFace Configuration</h2>
          <form onSubmit={handleAddConfig} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Symbol *
                </label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  placeholder="BTC-USD"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Interval *
                </label>
                <select
                  value={formData.interval}
                  onChange={(e) => setFormData({ ...formData, interval: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                >
                  <option value="1m">1 minute</option>
                  <option value="5m">5 minutes</option>
                  <option value="15m">15 minutes</option>
                  <option value="1h">1 hour</option>
                  <option value="4h">4 hours</option>
                  <option value="1d">1 day</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                HuggingFace Repository ID *
              </label>
              <input
                type="text"
                value={formData.repo_id}
                onChange={(e) => setFormData({ ...formData, repo_id: e.target.value })}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                placeholder="username/repo-name"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                API Token (optional)
              </label>
              <input
                type="password"
                value={formData.token}
                onChange={(e) => setFormData({ ...formData, token: e.target.value })}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                placeholder="hf_..."
              />
            </div>

            <div className="flex items-center gap-6">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={formData.private}
                  onChange={(e) => setFormData({ ...formData, private: e.target.checked })}
                  className="w-4 h-4"
                />
                <span className="text-sm text-gray-300">Private Repository</span>
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={formData.auto_export}
                  onChange={(e) => setFormData({ ...formData, auto_export: e.target.checked })}
                  className="w-4 h-4"
                />
                <span className="text-sm text-gray-300">Auto-Export</span>
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={formData.auto_import}
                  onChange={(e) => setFormData({ ...formData, auto_import: e.target.checked })}
                  className="w-4 h-4"
                />
                <span className="text-sm text-gray-300">Auto-Import</span>
              </label>
            </div>

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg"
              >
                Create Configuration
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Symbol</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Interval</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Repository</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Token</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Status</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Last Export</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Last Import</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {configs.length === 0 ? (
                <tr>
                  <td colSpan="8" className="px-4 py-8 text-center text-gray-500">
                    No configurations yet. Add one to get started.
                  </td>
                </tr>
              ) : (
                configs.map((config) => (
                  <tr key={config.id} className="hover:bg-gray-700/50">
                    <td className="px-4 py-3 text-gray-100 font-medium">{config.symbol}</td>
                    <td className="px-4 py-3 text-gray-300">{config.interval}</td>
                    <td className="px-4 py-3 text-gray-300">
                      <a
                        href={`https://huggingface.co/${config.repo_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-brand-400 hover:text-brand-300"
                      >
                        {config.repo_id}
                      </a>
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      <div className="flex items-center gap-2">
                        {config.token ? (
                          <>
                            <code className="text-xs">
                              {showTokenMap[config.id] ? config.token : '••••••••'}
                            </code>
                            <button
                              onClick={() => toggleTokenVisibility(config.id)}
                              className="text-gray-500 hover:text-gray-300"
                            >
                              {showTokenMap[config.id] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </button>
                          </>
                        ) : (
                          <span className="text-gray-500 text-xs">No token</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-2">
                        {config.private && (
                          <span className="px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded">Private</span>
                        )}
                        {!config.enabled && (
                          <span className="px-2 py-1 text-xs bg-red-900/30 text-red-400 rounded">Disabled</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {formatTimestamp(config.last_export)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {formatTimestamp(config.last_import)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => openExportModal(config)}
                          className="p-1 text-green-400 hover:text-green-300"
                          title="Export Model"
                        >
                          <Cloud className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => openEnsembleExportModal(config)}
                          className="p-1 text-blue-400 hover:text-blue-300"
                          title="Export Ensemble"
                        >
                          <Layers className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => openImportModal(config)}
                          className="p-1 text-purple-400 hover:text-purple-300"
                          title="Import Model"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteConfig(config.id, config.symbol, config.interval)}
                          className="p-1 text-red-400 hover:text-red-300"
                          title="Delete Configuration"
                        >
                          <Trash2 className="w-4 h-4" />
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

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-gray-100 mb-4">
              Export Model to HuggingFace
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              {selectedConfig?.symbol} ({selectedConfig?.interval})
            </p>

            <form onSubmit={handleExportModel} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Lookback Period
                </label>
                <input
                  type="number"
                  value={exportFormData.lookback}
                  onChange={(e) => setExportFormData({ ...exportFormData, lookback: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Focus Strategy
                </label>
                <select
                  value={exportFormData.focus}
                  onChange={(e) => setExportFormData({ ...exportFormData, focus: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                >
                  <option value="momentum">Momentum</option>
                  <option value="balanced">Balanced</option>
                  <option value="mean_reversion">Mean Reversion</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Forecast Horizon (hours)
                </label>
                <input
                  type="number"
                  value={exportFormData.forecast_horizon}
                  onChange={(e) => setExportFormData({ ...exportFormData, forecast_horizon: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Commit Message (optional)
                </label>
                <input
                  type="text"
                  value={exportFormData.commit_message}
                  onChange={(e) => setExportFormData({ ...exportFormData, commit_message: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                />
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowExportModal(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg"
                >
                  Export Model
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-gray-100 mb-4">
              Import Model from HuggingFace
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              {selectedConfig?.symbol} ({selectedConfig?.interval})
            </p>

            <form onSubmit={handleImportModel} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Repository ID
                </label>
                <input
                  type="text"
                  value={importFormData.repo_id}
                  onChange={(e) => setImportFormData({ ...importFormData, repo_id: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Lookback Period
                </label>
                <input
                  type="number"
                  value={importFormData.lookback}
                  onChange={(e) => setImportFormData({ ...importFormData, lookback: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Focus Strategy
                </label>
                <select
                  value={importFormData.focus}
                  onChange={(e) => setImportFormData({ ...importFormData, focus: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                >
                  <option value="momentum">Momentum</option>
                  <option value="balanced">Balanced</option>
                  <option value="mean_reversion">Mean Reversion</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Forecast Horizon (hours)
                </label>
                <input
                  type="number"
                  value={importFormData.forecast_horizon}
                  onChange={(e) => setImportFormData({ ...importFormData, forecast_horizon: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                />
              </div>

              <div>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={importFormData.force}
                    onChange={(e) => setImportFormData({ ...importFormData, force: e.target.checked })}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-gray-300">Force re-download (ignore cache)</span>
                </label>
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowImportModal(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg"
                >
                  Import Model
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Ensemble Export Modal */}
      {showEnsembleExportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-gray-100 mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-blue-400" />
              Export Ensemble Models
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              Export all models from a consensus analysis to {selectedConfig?.repo_id}
            </p>

            <form onSubmit={handleExportEnsemble} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Consensus Analysis
                </label>
                <select
                  value={ensembleExportFormData.consensus_id}
                  onChange={(e) => setEnsembleExportFormData({ ...ensembleExportFormData, consensus_id: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                  required
                >
                  <option value="">Select consensus analysis...</option>
                  {consensusAnalyses
                    .filter(c => c.symbol === selectedConfig?.symbol)
                    .map((consensus) => (
                      <option key={consensus.consensus_id} value={consensus.consensus_id}>
                        {consensus.symbol} - {new Date(consensus.timestamp).toLocaleDateString()} ({consensus.model_info?.length || 0} models)
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Commit Message (optional)
                </label>
                <input
                  type="text"
                  value={ensembleExportFormData.commit_message}
                  onChange={(e) => setEnsembleExportFormData({ ...ensembleExportFormData, commit_message: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-100"
                />
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowEnsembleExportModal(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
                >
                  Export Ensemble
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default HuggingFacePage;
