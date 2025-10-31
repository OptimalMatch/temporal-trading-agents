# Implementation Guide: Analysis Logs Feature

## Overview
Add clickable analysis history with console logs display to help debug issues during analysis runs.

## Status
- ✅ Models updated with `logs` field (StrategyAnalysis, ConsensusResult)
- ✅ Log capture utility created (`backend/log_capture.py`)
- ⏳ Backend integration needed
- ⏳ Frontend modal component needed

## Backend Implementation

### 1. Add logs collection to analysis functions

In `backend/main.py`, modify `run_gradient_analysis_background` and `run_consensus_analysis_background`:

```python
async def run_gradient_analysis_background(analysis_id: str, symbol: str, database: Database):
    """Background task to run gradient analysis"""
    logs = []  # Initialize logs list

    try:
        logs.append(f"[{datetime.utcnow().isoformat()}] Starting gradient analysis for {symbol}")

        # Send WebSocket update: Starting
        await ws_manager.send_progress(...)
        logs.append(f"[{datetime.utcnow().isoformat()}] Progress: Starting gradient analysis...")

        # ... existing code ...

        # When saving analysis, include logs:
        analysis.logs = logs

        await database.db.strategy_analyses.update_one(
            {"id": analysis_id},
            {"$set": analysis.dict()}
        )

    except Exception as e:
        logs.append(f"[{datetime.utcnow().isoformat()}] ERROR: {str(e)}")
        # ... existing error handling ...
```

### 2. Helper function for log + progress

Add to `backend/main.py`:

```python
async def log_and_send_progress(logs: List[str], ws_manager, task_id: str, symbol: str,
                                 strategy_type: str, status: str, progress: float, message: str):
    """Helper to both log and send WebSocket progress"""
    timestamp = datetime.utcnow().isoformat()
    logs.append(f"[{timestamp}] {message} (Progress: {progress}%)")

    await ws_manager.send_progress(
        task_id=task_id,
        symbol=symbol,
        strategy_type=strategy_type,
        status=status,
        progress=progress,
        message=message
    )
```

## Frontend Implementation

### 1. Create LogsModal Component

Create `frontend/src/components/LogsModal.jsx`:

```jsx
import { X } from 'lucide-react';

function LogsModal({ analysis, onClose }) {
  if (!analysis) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] flex flex-col border border-gray-700">
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
                  log.includes('ERROR') ? 'text-red-400' :
                  log.includes('SUCCESS') || log.includes('completed') ? 'text-green-400' :
                  log.includes('WARNING') ? 'text-yellow-400' :
                  'text-gray-300'
                }`}>
                  {log}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-400">
              No logs available for this analysis
            </div>
          )}
        </div>

        {/* Footer with metrics */}
        <div className="p-6 border-t border-gray-700 bg-gray-750">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Status:</span>
              <span className="ml-2 text-gray-100 font-medium">{analysis.status}</span>
            </div>
            <div>
              <span className="text-gray-400">Execution Time:</span>
              <span className="ml-2 text-gray-100 font-medium">
                {analysis.execution_time_ms ? `${(analysis.execution_time_ms / 1000).toFixed(2)}s` : 'N/A'}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Signal:</span>
              <span className="ml-2 text-gray-100 font-medium">{analysis.signal?.signal}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LogsModal;
```

### 2. Update HistoryPage to use LogsModal

Modify `frontend/src/pages/HistoryPage.jsx`:

```jsx
import LogsModal from '../components/LogsModal';

function HistoryPage() {
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  // ... existing code ...

  return (
    <div className="space-y-6">
      {/* ... existing JSX ... */}

      {/* Make analysis cards clickable */}
      <div key={idx}
           className="p-4 border border-gray-700 rounded-lg hover:border-brand-500 transition-colors cursor-pointer"
           onClick={() => setSelectedAnalysis(analysis)}>
        {/* ... existing card content ... */}
      </div>

      {/* Logs Modal */}
      {selectedAnalysis && (
        <LogsModal
          analysis={selectedAnalysis}
          onClose={() => setSelectedAnalysis(null)}
        />
      )}
    </div>
  );
}
```

## Testing

1. Start a new analysis: `curl -X POST http://localhost:10750/api/v1/analyze/gradient?symbol=BTC-USD`
2. Check database to verify logs are being stored:
   ```bash
   docker exec -it temporal-trading-mongodb mongosh temporal_trading --eval "db.strategy_analyses.findOne({}, {logs: 1})"
   ```
3. Click on an analysis in the History page to see the logs modal

## Next Steps

1. Apply the same logging pattern to `run_consensus_analysis_background`
2. Add more detailed log entries at key points (data loading, model training start/end, strategy execution)
3. Consider adding log levels (INFO, WARNING, ERROR) for better filtering
4. Add a download logs button to the modal
5. Add search/filter functionality for logs

## Files Modified

- ✅ `backend/models.py` - Added `logs` field
- ✅ `backend/log_capture.py` - Created utility (can be used for future enhancements)
- ⏳ `backend/main.py` - Need to add logs collection
- ⏳ `frontend/src/components/LogsModal.jsx` - Need to create
- ⏳ `frontend/src/pages/HistoryPage.jsx` - Need to add modal integration
