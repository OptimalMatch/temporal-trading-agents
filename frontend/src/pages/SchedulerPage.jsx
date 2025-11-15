import { useState, useEffect } from 'react';
import { Plus, Play, Pause, Trash2, Clock, Zap, BarChart } from 'lucide-react';
import api from '../services/api';
import { format } from 'date-fns';

const FREQUENCIES = [
  { value: 'hourly', label: 'Hourly' },
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'one_time', label: 'One-Time' },
];

const TASK_TYPES = [
  { value: 'analysis', label: 'Analysis', icon: BarChart },
  { value: 'auto_optimize', label: 'Auto-Optimize', icon: Zap },
];

const STRATEGY_OPTIONS = [
  { value: 'gradient', label: 'Forecast Gradient' },
  { value: 'confidence', label: 'Confidence-Weighted' },
  { value: 'timeframe', label: 'Multi-Timeframe' },
  { value: 'volatility', label: 'Volatility Sizing' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
  { value: 'acceleration', label: 'Acceleration' },
  { value: 'swing', label: 'Swing Trading' },
  { value: 'risk_adjusted', label: 'Risk-Adjusted' },
  { value: 'all', label: 'All Strategies (Consensus)' },
];

const AUTO_OPTIMIZE_STRATEGIES = [
  'gradient', 'confidence', 'volatility', 'acceleration',
  'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'
];

// Helper function to parse datetime ensuring it's treated as UTC
function parseUTCDate(dateString) {
  if (!dateString) return null;
  // If the datetime string doesn't have timezone info, append 'Z' to treat it as UTC
  const hasTimezone = dateString.endsWith('Z') || dateString.includes('+') ||
                      (dateString.includes('T') && dateString.split('T')[1].includes('-'));
  const utcString = hasTimezone ? dateString : dateString + 'Z';
  return new Date(utcString);
}

function SchedulerPage() {
  const [tasks, setTasks] = useState([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [availableTickers, setAvailableTickers] = useState([]);
  const [tickersLoading, setTickersLoading] = useState(true);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    task_type: 'analysis',
    symbol: 'BTC-USD',
    frequency: 'daily',
    scheduled_datetime: '',
    // Analysis fields
    strategy_type: 'all',
    interval: '1d',
    horizons: [3, 7, 14, 21],
    inference_mode: false,
    // Auto-optimize fields
    start_date: '2023-01-01',
    end_date: new Date().toISOString().split('T')[0],
    initial_capital: 100000,
    enabled_strategies: AUTO_OPTIMIZE_STRATEGIES,
  });

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

  useEffect(() => {
    loadTasks();
  }, []);

  async function loadTasks() {
    setLoading(true);
    try {
      const data = await api.getScheduledTasks();
      setTasks(data.tasks || []);
    } catch (error) {
      console.error('Failed to load tasks:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateTask(e) {
    e.preventDefault();
    try {
      // Build request based on task type
      const request = {
        name: formData.name,
        task_type: formData.task_type,
        symbol: formData.symbol,
        frequency: formData.frequency,
      };

      // Add one-time scheduling datetime if applicable
      if (formData.frequency === 'one_time') {
        if (!formData.scheduled_datetime) {
          alert('Please select a date and time for one-time scheduling');
          return;
        }
        request.scheduled_datetime = new Date(formData.scheduled_datetime).toISOString();
      }

      // Add task-specific fields
      if (formData.task_type === 'analysis') {
        request.strategy_type = formData.strategy_type;
        request.interval = formData.interval;
        request.horizons = formData.horizons;
        request.inference_mode = formData.inference_mode;
      } else if (formData.task_type === 'auto_optimize') {
        request.start_date = formData.start_date;
        request.end_date = formData.end_date;
        request.initial_capital = formData.initial_capital;
        request.enabled_strategies = formData.enabled_strategies;
      }

      await api.createScheduledTask(request);
      setShowCreateForm(false);
      resetForm();
      await loadTasks();
    } catch (error) {
      console.error('Failed to create task:', error);
      alert('Failed to create task: ' + error.message);
    }
  }

  function resetForm() {
    setFormData({
      name: '',
      task_type: 'analysis',
      symbol: 'BTC-USD',
      frequency: 'daily',
      scheduled_datetime: '',
      strategy_type: 'all',
      interval: '1d',
      horizons: [3, 7, 14, 21],
      inference_mode: false,
      start_date: '2023-01-01',
      end_date: new Date().toISOString().split('T')[0],
      initial_capital: 100000,
      enabled_strategies: AUTO_OPTIMIZE_STRATEGIES,
    });
  }

  async function handleToggleTask(taskId, currentStatus) {
    try {
      await api.updateScheduledTask(taskId, { is_active: !currentStatus });
      await loadTasks();
    } catch (error) {
      console.error('Failed to toggle task:', error);
    }
  }

  async function handleDeleteTask(taskId) {
    if (!confirm('Are you sure you want to delete this scheduled task?')) return;

    try {
      await api.deleteScheduledTask(taskId);
      await loadTasks();
    } catch (error) {
      console.error('Failed to delete task:', error);
    }
  }

  const toggleHorizon = (horizon) => {
    if (formData.horizons.includes(horizon)) {
      setFormData({ ...formData, horizons: formData.horizons.filter(h => h !== horizon) });
    } else {
      setFormData({ ...formData, horizons: [...formData.horizons, horizon].sort((a, b) => a - b) });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Scheduled Tasks</h1>
          <p className="text-gray-600 mt-1">Automate analysis and optimization workflows</p>
        </div>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>New Task</span>
        </button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Create Scheduled Task</h2>
          <form onSubmit={handleCreateTask} className="space-y-4">
            {/* Task Type Selector */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Task Type</label>
              <div className="grid grid-cols-2 gap-3">
                {TASK_TYPES.map(type => {
                  const Icon = type.icon;
                  return (
                    <button
                      key={type.value}
                      type="button"
                      onClick={() => setFormData({ ...formData, task_type: type.value })}
                      className={`p-4 rounded-lg border-2 transition-all flex items-center gap-3 ${
                        formData.task_type === type.value
                          ? 'border-brand-600 bg-brand-50'
                          : 'border-gray-300 hover:border-gray-400'
                      }`}
                    >
                      <Icon className={`w-5 h-5 ${formData.task_type === type.value ? 'text-brand-600' : 'text-gray-500'}`} />
                      <span className={`font-medium ${formData.task_type === type.value ? 'text-brand-600' : 'text-gray-700'}`}>
                        {type.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Common Fields */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Task Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="input w-full"
                  placeholder={formData.task_type === 'analysis' ? 'Daily BTC Analysis' : 'BTC Auto-Optimize'}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Symbol</label>
                <div className="relative">
                  <input
                    type="text"
                    value={formData.symbol}
                    onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                    placeholder={
                      tickersLoading
                        ? "Loading symbols..."
                        : "Type or select symbol (e.g., BTC-USD, AAPL, ETH-USD)"
                    }
                    list="scheduler-available-symbols"
                    disabled={tickersLoading}
                    className="input w-full disabled:opacity-50"
                    required
                  />
                  <datalist id="scheduler-available-symbols">
                    {availableTickers.map((ticker) => (
                      <option key={ticker.symbol} value={ticker.symbol}>
                        {ticker.name}
                      </option>
                    ))}
                  </datalist>
                  {!tickersLoading && availableTickers.length > 0 && (
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-gray-500">
                      {availableTickers.length} symbols
                    </div>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Frequency</label>
                <select
                  value={formData.frequency}
                  onChange={(e) => setFormData({ ...formData, frequency: e.target.value })}
                  className="input w-full"
                >
                  {FREQUENCIES.map(freq => (
                    <option key={freq.value} value={freq.value}>{freq.label}</option>
                  ))}
                </select>
              </div>

              {/* One-Time Date/Time Picker */}
              {formData.frequency === 'one_time' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Scheduled Date & Time</label>
                  <input
                    type="datetime-local"
                    value={formData.scheduled_datetime}
                    onChange={(e) => setFormData({ ...formData, scheduled_datetime: e.target.value })}
                    className="input w-full"
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Your timezone: {Intl.DateTimeFormat().resolvedOptions().timeZone} (UTC{new Date().getTimezoneOffset() > 0 ? '-' : '+'}
                    {Math.abs(Math.floor(new Date().getTimezoneOffset() / 60)).toString().padStart(2, '0')}:
                    {Math.abs(new Date().getTimezoneOffset() % 60).toString().padStart(2, '0')})
                  </p>
                </div>
              )}
            </div>

            {/* Analysis-Specific Fields */}
            {formData.task_type === 'analysis' && (
              <div className="space-y-4 border-t pt-4">
                <h3 className="text-lg font-semibold text-gray-800">Analysis Configuration</h3>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Strategy</label>
                  <select
                    value={formData.strategy_type}
                    onChange={(e) => setFormData({ ...formData, strategy_type: e.target.value })}
                    className="input w-full"
                  >
                    {STRATEGY_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>

                {formData.strategy_type === 'all' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Data Interval</label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setFormData({ ...formData, interval: '1d', horizons: [3, 7, 14, 21] })}
                          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                            formData.interval === '1d'
                              ? 'bg-brand-600 text-white'
                              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                          }`}
                        >
                          Daily (1d)
                        </button>
                        <button
                          type="button"
                          onClick={() => setFormData({ ...formData, interval: '1h', horizons: [6, 12, 24, 72] })}
                          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                            formData.interval === '1h'
                              ? 'bg-brand-600 text-white'
                              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                          }`}
                        >
                          Hourly (1h)
                        </button>
                      </div>
                    </div>

                    {formData.interval === '1d' && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Forecast Horizons (days)
                        </label>
                        <div className="flex flex-wrap gap-2">
                          {[3, 7, 14, 21, 30].map(days => (
                            <button
                              key={days}
                              type="button"
                              onClick={() => toggleHorizon(days)}
                              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                                formData.horizons.includes(days)
                                  ? 'bg-brand-600 text-white'
                                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                              }`}
                            >
                              {days}d
                            </button>
                          ))}
                        </div>
                        <p className="text-sm text-gray-500 mt-1">
                          Selected: {formData.horizons.join(', ')} days
                        </p>
                      </div>
                    )}

                    <div>
                      <label className="flex items-center gap-3 p-3 bg-gray-50 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                        <input
                          type="checkbox"
                          checked={formData.inference_mode}
                          onChange={(e) => setFormData({ ...formData, inference_mode: e.target.checked })}
                          className="w-4 h-4 text-brand-600"
                        />
                        <div>
                          <span className="block text-sm font-medium text-gray-800">Inference Mode (Fast)</span>
                          <span className="block text-xs text-gray-500">Use cached models without training</span>
                        </div>
                      </label>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Auto-Optimize-Specific Fields */}
            {formData.task_type === 'auto_optimize' && (
              <div className="space-y-4 border-t pt-4">
                <h3 className="text-lg font-semibold text-gray-800">Auto-Optimize Configuration</h3>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
                    <input
                      type="date"
                      value={formData.start_date}
                      onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
                      className="input w-full"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
                    <input
                      type="date"
                      value={formData.end_date}
                      onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
                      className="input w-full"
                      required
                    />
                  </div>

                  <div className="col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Initial Capital</label>
                    <input
                      type="number"
                      value={formData.initial_capital}
                      onChange={(e) => setFormData({ ...formData, initial_capital: Number(e.target.value) })}
                      className="input w-full"
                      min="1000"
                      step="1000"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Enabled Strategies</label>
                  <div className="flex flex-wrap gap-2">
                    {AUTO_OPTIMIZE_STRATEGIES.map(strategy => (
                      <button
                        key={strategy}
                        type="button"
                        onClick={() => {
                          if (formData.enabled_strategies.includes(strategy)) {
                            setFormData({
                              ...formData,
                              enabled_strategies: formData.enabled_strategies.filter(s => s !== strategy)
                            });
                          } else {
                            setFormData({
                              ...formData,
                              enabled_strategies: [...formData.enabled_strategies, strategy]
                            });
                          }
                        }}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          formData.enabled_strategies.includes(strategy)
                            ? 'bg-brand-600 text-white'
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        {strategy.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </button>
                    ))}
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    Selected: {formData.enabled_strategies.length} strategies
                  </p>
                </div>
              </div>
            )}

            <div className="flex space-x-2">
              <button type="submit" className="btn-primary">Create Task</button>
              <button
                type="button"
                onClick={() => {
                  setShowCreateForm(false);
                  resetForm();
                }}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Tasks List */}
      <div className="card">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-600"></div>
          </div>
        ) : tasks.length === 0 ? (
          <div className="text-center py-12">
            <Clock className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No scheduled tasks yet</p>
            <p className="text-sm text-gray-500 mt-1">Create one to automate your workflows</p>
          </div>
        ) : (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">{tasks.length} Scheduled Tasks</h2>
            <div className="space-y-3">
              {tasks.map(task => {
                const TaskIcon = task.task_type === 'auto_optimize' ? Zap : BarChart;
                return (
                  <div key={task.id} className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <TaskIcon className="w-5 h-5 text-gray-500" />
                          <h3 className="font-semibold text-gray-900">{task.name}</h3>
                          {task.is_running && (
                            <span className="badge" style={{ backgroundColor: '#10b981', color: 'white', animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' }}>
                              ▶ Running
                            </span>
                          )}
                          <span className={`badge ${task.is_active ? 'badge-success' : 'badge-warning'}`}>
                            {task.is_active ? 'Active' : 'Paused'}
                          </span>
                          <span className="badge badge-info">{task.frequency}</span>
                          <span className="badge" style={{ backgroundColor: task.task_type === 'auto_optimize' ? '#8b5cf6' : '#3b82f6', color: 'white' }}>
                            {task.task_type === 'auto_optimize' ? 'Auto-Optimize' : 'Analysis'}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">
                          {task.symbol} • {task.task_type === 'analysis' ? task.strategy_type : 'Optimization'} • Run count: {task.run_count}
                        </p>
                        <div className="text-xs text-gray-500 mt-1 space-y-1">
                          <div>
                            {task.frequency === 'one_time' ? (
                              <>
                                Scheduled for: {task.next_run ? format(parseUTCDate(task.next_run), 'MMM d, yyyy h:mm a') : 'Not scheduled'}
                                {' '}({Intl.DateTimeFormat().resolvedOptions().timeZone})
                              </>
                            ) : (
                              <>Next run: {task.next_run ? format(parseUTCDate(task.next_run), 'MMM d, yyyy h:mm a') : 'Not scheduled'}</>
                            )}
                          </div>
                          {task.last_started && (
                            <div className="text-blue-600">
                              Last started: {format(parseUTCDate(task.last_started), 'MMM d, yyyy h:mm a')}
                              {task.is_running && ' (currently running)'}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => handleToggleTask(task.id, task.is_active)}
                          className={`p-2 rounded-lg transition-colors ${
                            task.is_active
                              ? 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200'
                              : 'bg-green-100 text-green-700 hover:bg-green-200'
                          }`}
                          title={task.is_active ? 'Pause' : 'Activate'}
                        >
                          {task.is_active ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                        </button>

                        <button
                          onClick={() => handleDeleteTask(task.id)}
                          className="p-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SchedulerPage;
