import { useState, useEffect } from 'react';
import { Upload, Plus, Trash2, Edit2, Save, X, Eye, EyeOff, Cloud, Download, CheckCircle, AlertCircle } from 'lucide-react';

const API_BASE = '/api/v1';

function HuggingFacePage() {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showTokenMap, setShowTokenMap] = useState({});

  // Form state
  const [formData, setFormData] = useState({
    symbol: '',
    interval: '1h',
    repo_id: '',
    token: '',
    private: false,
    auto_export: false,
    auto_import: false
  });

  useEffect(() => {
    loadConfigs();
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

  const handleExportModel = async (config) => {
    const lookback = prompt('Enter lookback period (e.g., 45):', '45');
    if (!lookback) return;

    const focus = prompt('Enter focus (momentum/balanced/mean_reversion):', 'momentum');
    if (!focus) return;

    const horizon = prompt('Enter forecast horizon (e.g., 24):', '24');
    if (!horizon) return;

    try {
      const res = await fetch(`${API_BASE}/huggingface/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: config.symbol,
          interval: config.interval,
          lookback: parseInt(lookback),
          focus: focus,
          forecast_horizon: parseInt(horizon),
          commit_message: `Export ${config.symbol} ${focus} model for ${config.interval} trading`
        })
      });

      const data = await res.json();

      if (res.ok) {
        alert(`Model exported successfully!\n\nRepository: ${data.repo_id}\nURL: ${data.url}`);
        loadConfigs(); // Reload to update last_export timestamp
      } else {
        alert(`Export failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error exporting model:', err);
      alert('Failed to export model');
    }
  };

  const handleImportModel = async (config) => {
    const repoId = prompt('Enter HuggingFace repository ID:', config.repo_id);
    if (!repoId) return;

    const lookback = prompt('Enter lookback period (e.g., 45):', '45');
    if (!lookback) return;

    const focus = prompt('Enter focus (momentum/balanced/mean_reversion):', 'momentum');
    if (!focus) return;

    const horizon = prompt('Enter forecast horizon (e.g., 24):', '24');
    if (!horizon) return;

    try {
      const res = await fetch(`${API_BASE}/huggingface/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_id: repoId,
          symbol: config.symbol,
          interval: config.interval,
          lookback: parseInt(lookback),
          focus: focus,
          forecast_horizon: parseInt(horizon),
          force: false
        })
      });

      const data = await res.json();

      if (res.ok) {
        alert(`Model imported successfully!\n\nTraining date: ${data.metadata.training_timestamp}\nValidation loss: ${data.metadata.best_val_loss}`);
        loadConfigs(); // Reload to update last_import timestamp
      } else {
        alert(`Import failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error importing model:', err);
      alert('Failed to import model');
    }
  };

  const toggleShowToken = (configId) => {
    setShowTokenMap(prev => ({
      ...prev,
      [configId]: !prev[configId]
    }));
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    return date.toLocaleString();
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
          <h1 className="text-3xl font-bold text-gray-100">HuggingFace Integration</h1>
          <p className="text-gray-400 mt-1">Manage model export/import configurations</p>
        </div>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
        >
          {showAddForm ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          <span>{showAddForm ? 'Cancel' : 'Add Configuration'}</span>
        </button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Add Configuration Form */}
      {showAddForm && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-gray-100 mb-4">New HuggingFace Configuration</h2>
          <form onSubmit={handleAddConfig} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Symbol *
                </label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                  placeholder="e.g., BTC-USD"
                  required
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Interval *
                </label>
                <select
                  value={formData.interval}
                  onChange={(e) => setFormData({ ...formData, interval: e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:border-brand-500"
                >
                  <option value="1d">Daily (1d)</option>
                  <option value="1h">Hourly (1h)</option>
                  <option value="1m">Minute (1m)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                HuggingFace Repository ID *
              </label>
              <input
                type="text"
                value={formData.repo_id}
                onChange={(e) => setFormData({ ...formData, repo_id: e.target.value })}
                placeholder="e.g., username/btc-eur-momentum-1h"
                required
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                HuggingFace API Token (optional)
              </label>
              <input
                type="password"
                value={formData.token}
                onChange={(e) => setFormData({ ...formData, token: e.target.value })}
                placeholder="hf_..."
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brand-500"
              />
              <p className="text-xs text-gray-500 mt-1">Required for private repositories</p>
            </div>

            <div className="flex items-center space-x-6">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.private}
                  onChange={(e) => setFormData({ ...formData, private: e.target.checked })}
                  className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-brand-500"
                />
                <span className="text-sm text-gray-300">Private Repository</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.auto_export}
                  onChange={(e) => setFormData({ ...formData, auto_export: e.target.checked })}
                  className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-brand-500"
                />
                <span className="text-sm text-gray-300">Auto Export</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.auto_import}
                  onChange={(e) => setFormData({ ...formData, auto_import: e.target.checked })}
                  className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-brand-500"
                />
                <span className="text-sm text-gray-300">Auto Import</span>
              </label>
            </div>

            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => {
                  setShowAddForm(false);
                  setFormData({
                    symbol: '',
                    interval: '1h',
                    repo_id: '',
                    token: '',
                    private: false,
                    auto_export: false,
                    auto_import: false
                  });
                }}
                className="bg-gray-700 hover:bg-gray-600 text-gray-300 px-4 py-2 rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                Create Configuration
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Configurations List */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-gray-100 mb-4 flex items-center">
          <Cloud className="w-5 h-5 mr-2 text-brand-500" />
          Configurations
        </h2>

        {configs.length === 0 ? (
          <p className="text-gray-400 text-center py-8">
            No HuggingFace configurations. Add one to get started.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Symbol</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Interval</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Repository ID</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Token</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Private</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Last Export</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Last Import</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {configs.map((config) => (
                  <tr key={config.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="py-3 px-4 text-gray-100 font-medium">{config.symbol}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        config.interval === '1m' ? 'bg-purple-900/30 text-purple-400 border border-purple-700' :
                        config.interval === '1h' ? 'bg-blue-900/30 text-blue-400 border border-blue-700' :
                        'bg-green-900/30 text-green-400 border border-green-700'
                      }`}>
                        {config.interval}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-gray-300 font-mono text-sm">{config.repo_id}</td>
                    <td className="py-3 px-4">
                      {config.token ? (
                        <div className="flex items-center space-x-2">
                          <span className="text-gray-400 font-mono text-xs">
                            {showTokenMap[config.id] ? config.token : '••••••••'}
                          </span>
                          <button
                            onClick={() => toggleShowToken(config.id)}
                            className="p-1 hover:bg-gray-600 rounded transition-colors"
                          >
                            {showTokenMap[config.id] ? (
                              <EyeOff className="w-3 h-3 text-gray-400" />
                            ) : (
                              <Eye className="w-3 h-3 text-gray-400" />
                            )}
                          </button>
                        </div>
                      ) : (
                        <span className="text-gray-500 text-sm">None</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      {config.private ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : (
                        <X className="w-4 h-4 text-gray-500" />
                      )}
                    </td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {formatTimestamp(config.last_export)}
                    </td>
                    <td className="py-3 px-4 text-gray-300 text-sm">
                      {formatTimestamp(config.last_import)}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => handleExportModel(config)}
                          className="p-2 hover:bg-green-900/50 rounded transition-colors"
                          title="Export Model"
                        >
                          <Upload className="w-4 h-4 text-green-400" />
                        </button>
                        <button
                          onClick={() => handleImportModel(config)}
                          className="p-2 hover:bg-blue-900/50 rounded transition-colors"
                          title="Import Model"
                        >
                          <Download className="w-4 h-4 text-blue-400" />
                        </button>
                        <button
                          onClick={() => handleDeleteConfig(config.id, config.symbol, config.interval)}
                          className="p-2 hover:bg-red-900/50 rounded transition-colors"
                          title="Delete Configuration"
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

export default HuggingFacePage;
