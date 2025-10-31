import { X } from 'lucide-react';

function LogsModal({ analysis, onClose }) {
  if (!analysis) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
         onClick={onClose}>
      <div className="bg-gray-800 rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] flex flex-col border border-gray-700"
           onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-2xl font-bold text-gray-100">
              Analysis Logs: {analysis.symbol}
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              {analysis.strategy_type} • {new Date(analysis.created_at).toLocaleString()}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-gray-400" />
          </button>
        </div>

        {/* Logs Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {analysis.logs && analysis.logs.length > 0 ? (
            <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm space-y-1">
              {analysis.logs.map((log, idx) => (
                <div key={idx} className={`${
                  log.includes('ERROR') || log.includes('Failed') ? 'text-red-400' :
                  log.includes('SUCCESS') || log.includes('completed') || log.includes('Completed') ? 'text-green-400' :
                  log.includes('WARNING') || log.includes('Warning') ? 'text-yellow-400' :
                  'text-gray-300'
                }`}>
                  {log}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-400 mb-2">No logs available for this analysis</p>
              <p className="text-sm text-gray-500">
                Logs will be captured in future analysis runs.
                See IMPLEMENTATION_GUIDE_LOGS.md for details.
              </p>
            </div>
          )}
        </div>

        {/* Footer with metrics */}
        <div className="p-6 border-t border-gray-700 bg-gray-800">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Status:</span>
              <span className="ml-2 text-gray-100 font-medium capitalize">{analysis.status}</span>
            </div>
            <div>
              <span className="text-gray-400">Execution Time:</span>
              <span className="ml-2 text-gray-100 font-medium">
                {analysis.execution_time_ms ? `${(analysis.execution_time_ms / 1000).toFixed(2)}s` : 'N/A'}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Signal:</span>
              <span className="ml-2 text-gray-100 font-medium">{analysis.signal?.signal || 'N/A'}</span>
            </div>
          </div>
          {analysis.error && (
            <div className="mt-4 p-3 bg-red-900 border border-red-700 rounded-lg">
              <p className="text-sm text-red-200">
                <span className="font-semibold">Error:</span> {analysis.error}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default LogsModal;
