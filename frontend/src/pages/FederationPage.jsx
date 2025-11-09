import { useState, useEffect } from 'react';
import { Plus, Server, Trash2, RefreshCw, Link2, Download, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import api from '../services/api';
import { format } from 'date-fns';
import LogsModal from '../components/LogsModal';

const STATUS_COLORS = {
  active: 'text-green-600 bg-green-50',
  inactive: 'text-gray-600 bg-gray-50',
  error: 'text-red-600 bg-red-50',
};

const STATUS_ICONS = {
  active: CheckCircle,
  inactive: AlertCircle,
  error: XCircle,
};

function FederationPage() {
  const [instances, setInstances] = useState([]);
  const [importedForecasts, setImportedForecasts] = useState([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState('instances'); // 'instances' or 'forecasts'
  const [selectedForecast, setSelectedForecast] = useState(null); // For forecast details modal

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    base_url: '',
    api_key: '',
    enabled: true,
  });

  // Webhook registration state
  const [webhookUrl, setWebhookUrl] = useState('');

  useEffect(() => {
    loadInstances();
    loadImportedForecasts();

    // Auto-detect local webhook URL
    const baseUrl = window.location.origin.replace(':10752', ':10750');
    setWebhookUrl(`${baseUrl}/api/v1/federation/webhook`);
  }, []);

  async function loadInstances() {
    setLoading(true);
    try {
      const data = await api.get('/federation/instances');
      setInstances(data.instances || []);
    } catch (error) {
      console.error('Failed to load instances:', error);
    } finally {
      setLoading(false);
    }
  }

  async function loadImportedForecasts() {
    try {
      const data = await api.get('/federation/forecasts');
      setImportedForecasts(data.forecasts || []);
    } catch (error) {
      console.error('Failed to load imported forecasts:', error);
    }
  }

  async function handleCreateInstance(e) {
    e.preventDefault();
    try {
      await api.post('/federation/instances', formData);
      setShowCreateForm(false);
      setFormData({ name: '', base_url: '', api_key: '', enabled: true });
      await loadInstances();
    } catch (error) {
      console.error('Failed to create instance:', error);
      alert('Failed to create instance: ' + error.message);
    }
  }

  async function handleDeleteInstance(instanceId) {
    if (!confirm('Are you sure you want to delete this remote instance? All imported forecasts from this instance will also be deleted.')) return;

    try {
      await api.delete(`/federation/instances/${instanceId}`);
      await loadInstances();
      await loadImportedForecasts();
    } catch (error) {
      console.error('Failed to delete instance:', error);
      alert('Failed to delete instance: ' + error.message);
    }
  }

  async function handleRegisterWebhook(instanceId) {
    const url = prompt('Enter your local webhook URL:', webhookUrl);
    if (!url) return;

    try {
      const result = await api.post(`/federation/instances/${instanceId}/register-webhook?local_webhook_url=${encodeURIComponent(url)}`);
      alert('Webhook registered successfully!');
      await loadInstances();
    } catch (error) {
      console.error('Failed to register webhook:', error);
      alert('Failed to register webhook: ' + error.message);
    }
  }

  async function handleCheckHealth(instanceId) {
    try {
      const health = await api.get(`/federation/instances/${instanceId}/health`);
      const message = health.healthy
        ? `✅ Instance is healthy!\n\nStatus: ${health.status_code}\nResponse time: ${Math.round(health.response_time_ms)}ms`
        : `❌ Instance is unhealthy\n\nError: ${health.error || 'Unknown error'}`;
      alert(message);
      await loadInstances();
    } catch (error) {
      console.error('Failed to check health:', error);
      alert('Failed to check health: ' + error.message);
    }
  }

  async function handleImportForecast(instanceId) {
    const symbol = prompt('Enter symbol to import (e.g., BTC-USD):');
    if (!symbol) return;

    const interval = prompt('Enter interval (1d or 1h):', '1d');
    if (!interval) return;

    try {
      await api.post(`/federation/instances/${instanceId}/import?symbol=${symbol}&interval=${interval}`);
      alert(`Successfully imported forecast for ${symbol} (${interval})`);
      await loadImportedForecasts();
    } catch (error) {
      console.error('Failed to import forecast:', error);
      alert('Failed to import forecast: ' + error.message);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Federation</h1>
          <p className="text-gray-600 mt-1">Connect with other trading instances and share forecasts</p>
        </div>
        {selectedTab === 'instances' && (
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="btn-primary flex items-center space-x-2"
          >
            <Plus className="w-5 h-5" />
            <span>Add Instance</span>
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex space-x-4 border-b border-gray-200">
        <button
          onClick={() => setSelectedTab('instances')}
          className={`px-4 py-2 font-medium ${
            selectedTab === 'instances'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Remote Instances ({instances.length})
        </button>
        <button
          onClick={() => setSelectedTab('forecasts')}
          className={`px-4 py-2 font-medium ${
            selectedTab === 'forecasts'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Imported Forecasts ({importedForecasts.length})
        </button>
      </div>

      {/* Create Instance Form */}
      {selectedTab === 'instances' && showCreateForm && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Add Remote Instance</h2>
          <form onSubmit={handleCreateInstance} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Instance Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="input"
                placeholder="e.g., Production Instance"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base URL
              </label>
              <input
                type="url"
                value={formData.base_url}
                onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                className="input"
                placeholder="http://hostname:10750"
                required
              />
              <p className="text-sm text-gray-500 mt-1">
                e.g., http://pop-os-1:10750 or http://192.168.1.100:10750
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                API Key (Optional)
              </label>
              <input
                type="password"
                value={formData.api_key}
                onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                className="input"
                placeholder="Optional authentication key"
              />
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="enabled"
                checked={formData.enabled}
                onChange={(e) => setFormData({ ...formData, enabled: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="enabled" className="text-sm text-gray-700">
                Enable this instance
              </label>
            </div>

            <div className="flex space-x-3 pt-4">
              <button type="submit" className="btn-primary">
                Add Instance
              </button>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Remote Instances Tab */}
      {selectedTab === 'instances' && (
        <div className="space-y-4">
          {loading ? (
            <div className="card text-center py-8 text-gray-500">Loading...</div>
          ) : instances.length === 0 ? (
            <div className="card text-center py-8">
              <Server className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-500">No remote instances configured</p>
              <p className="text-sm text-gray-400 mt-1">Add an instance to start sharing forecasts</p>
            </div>
          ) : (
            instances.map((instance) => {
              const StatusIcon = STATUS_ICONS[instance.status] || AlertCircle;
              return (
                <div key={instance.id} className="card">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="text-lg font-semibold text-gray-900">{instance.name}</h3>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${STATUS_COLORS[instance.status]}`}>
                          <StatusIcon className="w-3 h-3 mr-1" />
                          {instance.status.charAt(0).toUpperCase() + instance.status.slice(1)}
                        </span>
                        {instance.webhook_registered && (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700">
                            <Link2 className="w-3 h-3 mr-1" />
                            Webhook Active
                          </span>
                        )}
                      </div>

                      <p className="text-sm text-gray-600 mt-1">{instance.base_url}</p>

                      <div className="grid grid-cols-2 gap-4 mt-3 text-sm text-gray-600">
                        {instance.last_sync && (
                          <div>
                            <span className="font-medium">Last Sync:</span>{' '}
                            {format(new Date(instance.last_sync), 'MMM d, h:mm a')}
                          </div>
                        )}
                        {instance.last_health_check && (
                          <div>
                            <span className="font-medium">Last Health Check:</span>{' '}
                            {format(new Date(instance.last_health_check), 'MMM d, h:mm a')}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex space-x-2 ml-4">
                      <button
                        onClick={() => handleCheckHealth(instance.id)}
                        className="btn-secondary text-sm flex items-center space-x-1"
                        title="Check health"
                      >
                        <RefreshCw className="w-4 h-4" />
                      </button>

                      {!instance.webhook_registered && (
                        <button
                          onClick={() => handleRegisterWebhook(instance.id)}
                          className="btn-primary text-sm flex items-center space-x-1"
                          title="Register webhook"
                        >
                          <Link2 className="w-4 h-4" />
                          <span>Register</span>
                        </button>
                      )}

                      <button
                        onClick={() => handleImportForecast(instance.id)}
                        className="btn-primary text-sm flex items-center space-x-1"
                        title="Import forecast"
                      >
                        <Download className="w-4 h-4" />
                        <span>Import</span>
                      </button>

                      <button
                        onClick={() => handleDeleteInstance(instance.id)}
                        className="btn-danger text-sm flex items-center space-x-1"
                        title="Delete instance"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Imported Forecasts Tab */}
      {selectedTab === 'forecasts' && (
        <div className="space-y-4">
          {importedForecasts.length === 0 ? (
            <div className="card text-center py-8">
              <Download className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-500">No imported forecasts</p>
              <p className="text-sm text-gray-400 mt-1">Import forecasts from remote instances to view them here</p>
            </div>
          ) : (
            <div className="grid gap-4">
              {importedForecasts.map((forecast) => (
                <div
                  key={forecast.id}
                  className="card cursor-pointer hover:border-brand-500 transition-colors"
                  onClick={() => setSelectedForecast(forecast)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="text-lg font-semibold text-gray-900">{forecast.symbol}</h3>
                        <span className="text-sm text-gray-500">{forecast.interval}</span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          forecast.consensus.includes('BUY') ? 'bg-green-100 text-green-800' :
                          forecast.consensus.includes('SELL') ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {forecast.consensus}
                        </span>
                      </div>

                      <p className="text-sm text-gray-600 mt-1">
                        From: <span className="font-medium">{forecast.remote_instance_name}</span>
                      </p>

                      <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
                        <div>
                          <span className="text-gray-500">Current Price:</span>
                          <p className="font-semibold">${forecast.current_price.toFixed(2)}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Bullish/Bearish:</span>
                          <p className="font-semibold">
                            {forecast.signals.bullish_count} / {forecast.signals.bearish_count}
                          </p>
                        </div>
                        <div>
                          <span className="text-gray-500">Imported:</span>
                          <p className="font-semibold">
                            {format(new Date(forecast.imported_at), 'MMM d, h:mm a')}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Forecast Details Modal */}
      {selectedForecast && (
        <LogsModal
          analysis={{
            symbol: selectedForecast.symbol,
            strategy_type: `Imported from ${selectedForecast.remote_instance_name}`,
            created_at: selectedForecast.remote_created_at,
            current_price: selectedForecast.current_price,
            forecast_data: selectedForecast.forecast_data,
            signal: {
              signal: selectedForecast.consensus,
              position_size_pct: 0
            },
            status: 'imported',
            execution_time_ms: null,
            logs: [
              `Forecast imported from remote instance: ${selectedForecast.remote_instance_name}`,
              `Original forecast ID: ${selectedForecast.original_forecast_id}`,
              `Imported at: ${new Date(selectedForecast.imported_at).toLocaleString()}`,
              `Interval: ${selectedForecast.interval}`,
              `Consensus: ${selectedForecast.consensus}`,
              `Confidence: ${selectedForecast.confidence}%`,
              `Signal breakdown: ${selectedForecast.signals.bullish_count} bullish, ${selectedForecast.signals.bearish_count} bearish`,
              `Forecast horizon: ${selectedForecast.forecast_data.horizon_days} days`,
              `Current price: $${selectedForecast.current_price.toFixed(2)}`,
            ]
          }}
          onClose={() => setSelectedForecast(null)}
        />
      )}
    </div>
  );
}

export default FederationPage;
