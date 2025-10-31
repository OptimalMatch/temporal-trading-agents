import { useState, useEffect } from 'react';
import { Play, Loader } from 'lucide-react';
import api from '../services/api';
import useWebSocket from '../hooks/useWebSocket';
import ProgressIndicator from '../components/ProgressIndicator';
import StrategyCard from '../components/StrategyCard';

const STRATEGY_OPTIONS = [
  { value: 'gradient', label: 'Forecast Gradient', description: 'Analyzes the shape of the forecast curve' },
  { value: 'confidence', label: 'Confidence-Weighted', description: 'Uses model agreement for position sizing' },
  { value: 'timeframe', label: 'Multi-Timeframe', description: 'Compares multiple forecast horizons' },
  { value: 'volatility', label: 'Volatility Sizing', description: 'Adjusts position based on forecast uncertainty' },
  { value: 'mean_reversion', label: 'Mean Reversion', description: 'Trades mean reversion with forecast confirmation' },
  { value: 'acceleration', label: 'Acceleration', description: 'Identifies momentum changes in forecast' },
  { value: 'swing', label: 'Swing Trading', description: 'Finds swing opportunities within forecast' },
  { value: 'risk_adjusted', label: 'Risk-Adjusted', description: 'Comprehensive risk metrics analysis' },
  { value: 'all', label: 'All Strategies (Consensus)', description: 'Run all 8 strategies and calculate consensus' },
];

const POPULAR_SYMBOLS = [
  'BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'SPY'
];

function AnalyzePage() {
  const [symbol, setSymbol] = useState('BTC-USD');
  const [selectedStrategy, setSelectedStrategy] = useState('all');
  const [horizons, setHorizons] = useState([3, 7, 14, 21]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);

  const { connected, progress, subscribe } = useWebSocket(analysisId);

  const latestProgress = progress[progress.length - 1];

  // Listen for analysis completion via WebSocket
  useEffect(() => {
    if (!analysisId) return;

    const unsubscribe = subscribe('status:completed', async (data) => {
      if (data.task_id === analysisId) {
        console.log('Analysis completed via WebSocket:', data);

        // Fetch final results
        try {
          const finalResult = await api.getAnalysisStatus(analysisId);
          setResult(finalResult);
          setLoading(false);
        } catch (err) {
          console.error('Failed to fetch final results:', err);
          setError(err.message);
          setLoading(false);
        }
      }
    });

    return () => unsubscribe?.();
  }, [analysisId, subscribe]);

  async function handleAnalyze() {
    setLoading(true);
    setError(null);
    setResult(null);
    setAnalysisId(null);

    try {
      let response;

      if (selectedStrategy === 'all') {
        response = await api.analyzeConsensus(symbol, horizons);
      } else if (selectedStrategy === 'gradient') {
        response = await api.analyzeGradient(symbol);
      } else if (selectedStrategy === 'confidence') {
        response = await api.analyzeConfidence(symbol);
      } else {
        throw new Error('Selected strategy not yet implemented in API');
      }

      // Check if response is async (contains analysis_id)
      if (response.analysis_id && response.status === 'pending') {
        // Async operation - store analysis_id for WebSocket tracking
        console.log('Analysis started:', response.analysis_id);
        setAnalysisId(response.analysis_id);
        setLoading(false); // Button returns to normal immediately
        // Progress will be shown via WebSocket updates in ProgressIndicator
      } else {
        // Sync operation - display results immediately
        setResult(response);
        setLoading(false);
      }
    } catch (err) {
      setError(err.message);
      console.error('Analysis failed:', err);
      setLoading(false);
    }
  }

  const selectedStrategyInfo = STRATEGY_OPTIONS.find(s => s.value === selectedStrategy);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analyze Trading Strategy</h1>
        <p className="text-gray-600 mt-1">
          Run trading strategy analysis on any symbol with real-time progress updates
        </p>
      </div>

      {/* Live Progress */}
      {latestProgress && latestProgress.status !== 'completed' && (
        <ProgressIndicator progress={latestProgress} />
      )}

      {/* Analysis Form */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-6">Configuration</h2>

        <div className="space-y-6">
          {/* Symbol Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Trading Symbol
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="input flex-1"
                placeholder="e.g., BTC-USD, AAPL"
              />
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {POPULAR_SYMBOLS.map(sym => (
                <button
                  key={sym}
                  onClick={() => setSymbol(sym)}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    symbol === sym
                      ? 'bg-brand-100 text-brand-700'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {sym}
                </button>
              ))}
            </div>
          </div>

          {/* Strategy Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Strategy
            </label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="input w-full"
            >
              {STRATEGY_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {selectedStrategyInfo && (
              <p className="text-sm text-gray-600 mt-2">
                {selectedStrategyInfo.description}
              </p>
            )}
          </div>

          {/* Horizons (for consensus only) */}
          {selectedStrategy === 'all' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Forecast Horizons (days)
              </label>
              <div className="flex flex-wrap gap-2">
                {[3, 7, 14, 21, 30].map(days => (
                  <button
                    key={days}
                    onClick={() => {
                      if (horizons.includes(days)) {
                        setHorizons(horizons.filter(h => h !== days));
                      } else {
                        setHorizons([...horizons, days].sort((a, b) => a - b));
                      }
                    }}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      horizons.includes(days)
                        ? 'bg-brand-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {days}d
                  </button>
                ))}
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Selected: {horizons.join(', ')} days
              </p>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex items-center space-x-4">
            <button
              onClick={handleAnalyze}
              disabled={loading || !symbol}
              className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Run Analysis</span>
                </>
              )}
            </button>

            {connected && (
              <div className="flex items-center space-x-2 text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium">Real-time updates active</span>
              </div>
            )}
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              <p className="font-medium">Analysis Failed</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-6">Analysis Results</h2>

          {selectedStrategy === 'all' ? (
            <StrategyCard consensus={result} />
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Signal</p>
                  <p className="text-2xl font-bold text-gray-900">{result.signal?.signal}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Position Size</p>
                  <p className="text-2xl font-bold text-gray-900">{result.signal?.position_size_pct}%</p>
                </div>
              </div>

              {result.signal?.rationale && (
                <div className="p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-gray-700 mb-1">Rationale</p>
                  <p className="text-gray-900">{result.signal.rationale}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default AnalyzePage;
