import { useState, useEffect } from 'react';
import { Trash2, Download, Upload, RefreshCw, Database, HardDrive, Calendar, Activity, Archive, FileUp } from 'lucide-react';
import api from '../services/api';

function CacheManagement() {
  const [cachedModels, setCachedModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [stats, setStats] = useState(null);
  const [filterSymbol, setFilterSymbol] = useState('');
  const [filterInterval, setFilterInterval] = useState('');
  const [importFile, setImportFile] = useState(null);
  const [restoreFile, setRestoreFile] = useState(null);
  const [showConfirm, setShowConfirm] = useState(null);
  const [backupInProgress, setBackupInProgress] = useState(false);
  const [restoreInProgress, setRestoreInProgress] = useState(false);

  useEffect(() => {
    loadCachedModels();
  }, []);

  async function loadCachedModels() {
    setLoading(true);
    setError(null);
    try {
      const [modelsData, statsData] = await Promise.all([
        api.getCachedModels(),
        api.getCacheStats()
      ]);
      setCachedModels(modelsData.models || []);
      setStats(statsData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteModel(cacheKey) {
    if (!showConfirm || showConfirm !== cacheKey) {
      setShowConfirm(cacheKey);
      return;
    }

    try {
      await api.deleteCachedModel(cacheKey);
      await loadCachedModels();
      setShowConfirm(null);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleClearCache() {
    if (!showConfirm || showConfirm !== 'clear-all') {
      setShowConfirm('clear-all');
      return;
    }

    try {
      await api.clearCache(filterSymbol || null, filterInterval || null);
      await loadCachedModels();
      setShowConfirm(null);
      setFilterSymbol('');
      setFilterInterval('');
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleExportModel(cacheKey) {
    try {
      const blob = await api.exportCachedModel(cacheKey);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${cacheKey}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleImportModel() {
    if (!importFile) return;

    try {
      await api.importCachedModel(importFile);
      await loadCachedModels();
      setImportFile(null);
      // Reset file input
      document.getElementById('cache-import-file').value = '';
      setSuccess('Model imported successfully');
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleCreateBackup() {
    setBackupInProgress(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await api.createFullBackup();

      // Auto-download the backup
      const blob = await api.downloadBackup(result.backup_id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${result.backup_id}.tar.gz`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setSuccess(`Full backup created successfully (${result.size_mb.toFixed(2)} MB)`);
      setTimeout(() => setSuccess(null), 10000);
    } catch (err) {
      setError(err.message);
    } finally {
      setBackupInProgress(false);
    }
  }

  async function handleRestoreBackup() {
    if (!restoreFile) return;

    if (!showConfirm || showConfirm !== 'restore-full') {
      setShowConfirm('restore-full');
      return;
    }

    setRestoreInProgress(true);
    setError(null);
    setSuccess(null);

    try {
      await api.restoreFullBackup(restoreFile);
      await loadCachedModels();
      setRestoreFile(null);
      document.getElementById('backup-restore-file').value = '';
      setShowConfirm(null);
      setSuccess('Full system restore completed successfully! The page will reload in 3 seconds.');

      // Reload page after 3 seconds to reflect restored state
      setTimeout(() => window.location.reload(), 3000);
    } catch (err) {
      setError(err.message);
      setShowConfirm(null);
    } finally {
      setRestoreInProgress(false);
    }
  }

  function formatBytes(bytes) {
    return `${bytes.toFixed(2)} MB`;
  }

  function formatDate(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString();
  }

  // Filter models based on filters
  const filteredModels = cachedModels.filter(model => {
    if (filterSymbol && model.symbol !== filterSymbol) return false;
    if (filterInterval && model.interval !== filterInterval) return false;
    return true;
  });

  // Get unique symbols and intervals for filter dropdowns
  const symbols = [...new Set(cachedModels.map(m => m.symbol))].sort();
  const intervals = [...new Set(cachedModels.map(m => m.interval))].sort();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
            <Database className="w-6 h-6" />
            Model Cache Management
          </h2>
          <p className="text-gray-400 mt-1">
            Manage cached models, view metadata, and import/export models
          </p>
        </div>
        <button
          onClick={loadCachedModels}
          disabled={loading}
          className="btn-secondary flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <Database className="w-8 h-8 text-brand-500" />
              <div>
                <p className="text-sm text-gray-400">Total Models</p>
                <p className="text-2xl font-bold text-gray-100">{stats.num_models}</p>
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <HardDrive className="w-8 h-8 text-blue-500" />
              <div>
                <p className="text-sm text-gray-400">Cache Size</p>
                <p className="text-2xl font-bold text-gray-100">{formatBytes(stats.total_size_mb)}</p>
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-green-500" />
              <div>
                <p className="text-sm text-gray-400">Symbols</p>
                <p className="text-2xl font-bold text-gray-100">{Object.keys(stats.by_symbol || {}).length}</p>
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <Calendar className="w-8 h-8 text-purple-500" />
              <div>
                <p className="text-sm text-gray-400">Intervals</p>
                <p className="text-2xl font-bold text-gray-100">{Object.keys(stats.by_interval || {}).length}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters and Actions */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Symbol Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Filter by Symbol
            </label>
            <select
              value={filterSymbol}
              onChange={(e) => setFilterSymbol(e.target.value)}
              className="input w-full"
            >
              <option value="">All Symbols</option>
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>

          {/* Interval Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Filter by Interval
            </label>
            <select
              value={filterInterval}
              onChange={(e) => setFilterInterval(e.target.value)}
              className="input w-full"
            >
              <option value="">All Intervals</option>
              {intervals.map(interval => (
                <option key={interval} value={interval}>{interval}</option>
              ))}
            </select>
          </div>

          {/* Import */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Import Model
            </label>
            <div className="flex gap-2">
              <input
                id="cache-import-file"
                type="file"
                accept=".zip"
                onChange={(e) => setImportFile(e.target.files[0])}
                className="hidden"
              />
              <label
                htmlFor="cache-import-file"
                className="btn-secondary flex-1 flex items-center justify-center gap-2 cursor-pointer"
              >
                <Upload className="w-4 h-4" />
                Choose File
              </label>
              {importFile && (
                <button
                  onClick={handleImportModel}
                  className="btn-primary px-3"
                >
                  Import
                </button>
              )}
            </div>
            {importFile && (
              <p className="text-xs text-gray-400 mt-1">{importFile.name}</p>
            )}
          </div>

          {/* Clear Cache */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Clear Cache
            </label>
            <button
              onClick={handleClearCache}
              className={`w-full flex items-center justify-center gap-2 ${
                showConfirm === 'clear-all'
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'btn-secondary'
              }`}
            >
              <Trash2 className="w-4 h-4" />
              {showConfirm === 'clear-all' ? 'Confirm Clear' : 'Clear'}
            </button>
          </div>
        </div>
      </div>

      {/* Full System Backup & Restore */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Archive className="w-5 h-5" />
          Full System Backup & Restore
        </h3>
        <p className="text-sm text-gray-400 mb-4">
          Create complete system backups including model cache and MongoDB data.
          Restore from previous backups to recover your entire system state.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Create Backup */}
          <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
            <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
              <Download className="w-4 h-4" />
              Create Full Backup
            </h4>
            <p className="text-xs text-gray-400 mb-3">
              Creates a tar.gz archive containing all cached models and MongoDB records.
            </p>
            <button
              onClick={handleCreateBackup}
              disabled={backupInProgress}
              className="w-full btn-primary flex items-center justify-center gap-2"
            >
              {backupInProgress ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Creating Backup...
                </>
              ) : (
                <>
                  <Archive className="w-4 h-4" />
                  Create & Download Backup
                </>
              )}
            </button>
          </div>

          {/* Restore Backup */}
          <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
            <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
              <FileUp className="w-4 h-4" />
              Restore from Backup
            </h4>
            <p className="text-xs text-gray-400 mb-3">
              Upload a backup file to restore model cache and MongoDB data.
            </p>
            <div className="space-y-2">
              <input
                id="backup-restore-file"
                type="file"
                accept=".tar.gz,.tgz"
                onChange={(e) => {
                  setRestoreFile(e.target.files[0]);
                  setShowConfirm(null);
                }}
                className="hidden"
              />
              <label
                htmlFor="backup-restore-file"
                className="btn-secondary w-full flex items-center justify-center gap-2 cursor-pointer"
              >
                <Upload className="w-4 h-4" />
                Choose Backup File
              </label>
              {restoreFile && (
                <>
                  <p className="text-xs text-gray-400 truncate">{restoreFile.name}</p>
                  <button
                    onClick={handleRestoreBackup}
                    disabled={restoreInProgress}
                    className={`w-full flex items-center justify-center gap-2 ${
                      showConfirm === 'restore-full'
                        ? 'bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors'
                        : 'bg-brand-600 hover:bg-brand-700 text-white font-medium py-2 px-4 rounded-lg transition-colors'
                    }`}
                  >
                    {restoreInProgress ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Restoring...
                      </>
                    ) : showConfirm === 'restore-full' ? (
                      <>
                        <Upload className="w-4 h-4" />
                        Click Again to Confirm Restore
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4" />
                        Restore from Backup
                      </>
                    )}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {showConfirm === 'restore-full' && (
          <div className="mt-4 p-3 bg-yellow-900 border border-yellow-700 rounded-lg">
            <p className="text-sm text-yellow-200">
              ⚠️ <strong>Warning:</strong> This will replace all existing model cache and MongoDB data.
              This action cannot be undone. Click the restore button again to confirm.
            </p>
          </div>
        )}
      </div>

      {/* Success Display */}
      {success && (
        <div className="p-4 bg-green-900 border border-green-700 rounded-lg text-green-200">
          <p className="font-medium">Success</p>
          <p className="text-sm mt-1">{success}</p>
          <button
            onClick={() => setSuccess(null)}
            className="text-sm underline mt-2"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-900 border border-red-700 rounded-lg text-red-200">
          <p className="font-medium">Error</p>
          <p className="text-sm mt-1">{error}</p>
          <button
            onClick={() => setError(null)}
            className="text-sm underline mt-2"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Models Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800 border-b border-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Interval
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Focus
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Horizon
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Lookback
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Size
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Last Modified
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Fine-tuned
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {loading ? (
                <tr>
                  <td colSpan="9" className="px-4 py-8 text-center text-gray-400">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                    Loading cached models...
                  </td>
                </tr>
              ) : filteredModels.length === 0 ? (
                <tr>
                  <td colSpan="9" className="px-4 py-8 text-center text-gray-400">
                    {cachedModels.length === 0
                      ? 'No cached models found'
                      : 'No models match the current filters'}
                  </td>
                </tr>
              ) : (
                filteredModels.map((model) => (
                  <tr key={model.cache_key} className="hover:bg-gray-800 transition-colors">
                    <td className="px-4 py-3 text-sm font-medium text-gray-200">
                      {model.symbol}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {model.interval}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      <span className="px-2 py-1 bg-brand-900 text-brand-300 rounded-md text-xs">
                        {model.focus}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {model.forecast_horizon}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {model.lookback}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {formatBytes(model.total_size_mb)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {formatDate(model.last_modified)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {model.fine_tuned ? (
                        <span className="text-xs px-2 py-1 bg-green-900 text-green-300 rounded-md">
                          Yes ({model.fine_tune_count}x)
                        </span>
                      ) : (
                        <span className="text-xs px-2 py-1 bg-gray-700 text-gray-400 rounded-md">
                          No
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => handleExportModel(model.cache_key)}
                          className="p-2 text-blue-400 hover:text-blue-300 hover:bg-gray-700 rounded transition-colors"
                          title="Export model"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteModel(model.cache_key)}
                          className={`p-2 rounded transition-colors ${
                            showConfirm === model.cache_key
                              ? 'bg-red-600 text-white hover:bg-red-700'
                              : 'text-red-400 hover:text-red-300 hover:bg-gray-700'
                          }`}
                          title={showConfirm === model.cache_key ? 'Confirm delete' : 'Delete model'}
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

        {/* Footer with count */}
        {filteredModels.length > 0 && (
          <div className="px-4 py-3 bg-gray-800 border-t border-gray-700 text-sm text-gray-400">
            Showing {filteredModels.length} of {cachedModels.length} cached models
          </div>
        )}
      </div>
    </div>
  );
}

export default CacheManagement;
