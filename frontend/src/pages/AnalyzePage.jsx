import { useState, useEffect } from 'react';
import { Play, Loader } from 'lucide-react';
import api from '../services/api';
import useWebSocket from '../hooks/useWebSocket';
import ProgressIndicator from '../components/ProgressIndicator';
import StrategyCard from '../components/StrategyCard';
import ForecastChart from '../components/ForecastChart';

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
  const [interval, setInterval] = useState('1d'); // '1d' for daily, '1h' for hourly
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
        response = await api.analyzeConsensus(symbol, horizons, interval);
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
        <h1 className="text-3xl font-bold text-gray-100">Analyze Trading Strategy</h1>
        <p className="text-gray-400 mt-1">
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
            <label className="block text-sm font-medium text-gray-300 mb-2">
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
                      ? 'bg-brand-900 text-brand-300 border border-brand-700'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {sym}
                </button>
              ))}
            </div>
          </div>

          {/* Strategy Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
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
              <p className="text-sm text-gray-400 mt-2">
                {selectedStrategyInfo.description}
              </p>
            )}
          </div>

          {/* Data Interval (for consensus only) */}
          {selectedStrategy === 'all' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Data Interval
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setInterval('1d');
                    setHorizons([3, 7, 14, 21]); // Reset to daily defaults
                  }}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    interval === '1d'
                      ? 'bg-brand-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Daily (1d)
                </button>
                <button
                  onClick={() => {
                    setInterval('1h');
                    setHorizons([6, 12, 24, 72]); // Reset to hourly defaults (6h, 12h, 24h, 72h)
                  }}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    interval === '1h'
                      ? 'bg-brand-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Hourly (1h)
                </button>
              </div>
              <p className="text-sm text-gray-400 mt-2">
                {interval === '1d'
                  ? 'Forecast based on daily data (good for long-term trends)'
                  : 'Forecast based on hourly data (fresher signals, good for 24/7 crypto markets)'}
              </p>
            </div>
          )}

          {/* Horizons (for consensus only) */}
          {selectedStrategy === 'all' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Forecast Horizons ({interval === '1h' ? 'hours' : 'days'})
              </label>
              <div className="flex flex-wrap gap-2">
                {(interval === '1h' ? [3, 6, 12, 24, 48, 72] : [3, 7, 14, 21, 30]).map(periods => (
                  <button
                    key={periods}
                    onClick={() => {
                      if (horizons.includes(periods)) {
                        setHorizons(horizons.filter(h => h !== periods));
                      } else {
                        setHorizons([...horizons, periods].sort((a, b) => a - b));
                      }
                    }}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      horizons.includes(periods)
                        ? 'bg-brand-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {periods}{interval === '1h' ? 'h' : 'd'}
                  </button>
                ))}
              </div>
              <p className="text-sm text-gray-400 mt-2">
                Selected: {horizons.join(', ')} {interval === '1h' ? 'hours' : 'days'}
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
            <div className="p-4 bg-red-900 border border-red-700 rounded-lg text-red-200">
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
            <div className="space-y-6">
              <StrategyCard consensus={result} />
              {result.forecast_data && (
                <ForecastChart
                  forecastData={result.forecast_data}
                  symbol={result.symbol}
                  interval={result.interval || '1d'}
                />
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400">Signal</p>
                  <p className="text-2xl font-bold text-gray-100">{result.signal?.signal}</p>
                </div>
                <div className="p-4 bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400">Position Size</p>
                  <p className="text-2xl font-bold text-gray-100">{result.signal?.position_size_pct}%</p>
                </div>
              </div>

              {result.signal?.rationale && (
                <div className="p-4 bg-blue-900 rounded-lg border border-blue-700">
                  <p className="text-sm font-medium text-gray-300 mb-1">Rationale</p>
                  <p className="text-gray-100">{result.signal.rationale}</p>
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
