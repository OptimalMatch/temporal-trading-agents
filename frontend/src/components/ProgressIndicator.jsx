import { Loader, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

function ProgressIndicator({ progress }) {
  if (!progress) return null;

  const statusConfig = {
    starting: { icon: Loader, color: 'text-blue-400', bg: 'bg-blue-900 border-blue-700', label: 'Starting' },
    training: { icon: Loader, color: 'text-yellow-400', bg: 'bg-yellow-900 border-yellow-700', label: 'Training Models' },
    analyzing: { icon: Loader, color: 'text-purple-400', bg: 'bg-purple-900 border-purple-700', label: 'Analyzing' },
    finalizing: { icon: Loader, color: 'text-indigo-400', bg: 'bg-indigo-900 border-indigo-700', label: 'Finalizing' },
    completed: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-900 border-green-700', label: 'Completed' },
    error: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-900 border-red-700', label: 'Error' },
  };

  const config = statusConfig[progress.status] || statusConfig.analyzing;
  const Icon = config.icon;
  const isAnimating = !['completed', 'error'].includes(progress.status);

  return (
    <div className="card bg-gradient-to-r from-brand-900 to-blue-900 border-brand-700">
      <div className="flex items-start space-x-4">
        <div className={`p-3 rounded-lg border ${config.bg} ${isAnimating ? 'animate-pulse' : ''}`}>
          <Icon className={`w-6 h-6 ${config.color} ${isAnimating ? 'animate-spin' : ''}`} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h3 className="text-lg font-semibold text-gray-100">
                {progress.symbol} - {progress.strategy_type}
              </h3>
              <p className="text-sm text-gray-400">{progress.message}</p>
            </div>
            <span className={`badge border ${config.bg} ${config.color}`}>
              {config.label}
            </span>
          </div>

          {/* Progress Bar */}
          {progress.progress > 0 && progress.status !== 'completed' && (
            <div className="mt-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-300">Progress</span>
                <span className="text-sm font-medium text-gray-300">{progress.progress.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-brand-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress.progress}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Details */}
          {progress.details && Object.keys(progress.details).length > 0 && (
            <div className="mt-3 p-3 bg-gray-900 bg-opacity-50 rounded-lg border border-gray-700">
              <p className="text-xs font-medium text-gray-400 mb-1">Details:</p>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(progress.details).map(([key, value]) => (
                  <div key={key} className="text-sm">
                    <span className="text-gray-400">{key}:</span>{' '}
                    <span className="font-medium text-gray-200">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Timestamp */}
          <p className="text-xs text-gray-500 mt-2">
            {new Date(progress.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ProgressIndicator;
