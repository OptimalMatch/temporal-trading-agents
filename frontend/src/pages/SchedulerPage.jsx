import { useState, useEffect } from 'react';
import { Plus, Play, Pause, Trash2, Clock } from 'lucide-react';
import api from '../services/api';
import { format } from 'date-fns';

const FREQUENCIES = [
  { value: 'hourly', label: 'Hourly' },
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
];

const STRATEGY_OPTIONS = [
  { value: 'gradient', label: 'Gradient' },
  { value: 'confidence', label: 'Confidence' },
  { value: 'all', label: 'All Strategies (Consensus)' },
];

function SchedulerPage() {
  const [tasks, setTasks] = useState([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(false);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'BTC-USD',
    strategy_type: 'all',
    frequency: 'daily',
  });

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
      await api.createScheduledTask(formData);
      setShowCreateForm(false);
      setFormData({ name: '', symbol: 'BTC-USD', strategy_type: 'all', frequency: 'daily' });
      await loadTasks();
    } catch (error) {
      console.error('Failed to create task:', error);
      alert('Failed to create task: ' + error.message);
    }
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Scheduled Tasks</h1>
          <p className="text-gray-600 mt-1">Automate strategy analysis on a schedule</p>
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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Task Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="input w-full"
                  placeholder="Daily BTC Analysis"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Symbol</label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                  className="input w-full"
                  placeholder="BTC-USD"
                  required
                />
              </div>

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
            </div>

            <div className="flex space-x-2">
              <button type="submit" className="btn-primary">Create Task</button>
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
            <p className="text-sm text-gray-500 mt-1">Create one to automate your analysis</p>
          </div>
        ) : (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">{tasks.length} Scheduled Tasks</h2>
            <div className="space-y-3">
              {tasks.map(task => (
                <div key={task.id} className="p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="font-semibold text-gray-900">{task.name}</h3>
                        <span className={`badge ${task.is_active ? 'badge-success' : 'badge-warning'}`}>
                          {task.is_active ? 'Active' : 'Paused'}
                        </span>
                        <span className="badge badge-info">{task.frequency}</span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {task.symbol} • {task.strategy_type} • Run count: {task.run_count}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Next run: {task.next_run ? format(new Date(task.next_run), 'MMM d, yyyy h:mm a') : 'Not scheduled'}
                      </p>
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
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SchedulerPage;
