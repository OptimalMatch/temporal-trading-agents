import { useState, useEffect } from 'react';
import { Search, Filter, Calendar, Network } from 'lucide-react';
import api from '../services/api';
import { format } from 'date-fns';
import LogsModal from '../components/LogsModal';

const STRATEGY_TYPES = [
  { value: '', label: 'All Strategies' },
  { value: 'gradient', label: 'Gradient' },
  { value: 'confidence', label: 'Confidence' },
  { value: 'timeframe', label: 'Multi-Timeframe' },
  { value: 'volatility', label: 'Volatility' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
  { value: 'acceleration', label: 'Acceleration' },
  { value: 'swing', label: 'Swing Trading' },
  { value: 'risk_adjusted', label: 'Risk-Adjusted' },
];

function HistoryPage() {
  const [symbol, setSymbol] = useState('');
  const [strategyFilter, setStrategyFilter] = useState('');
  const [analyses, setAnalyses] = useState([]);
  const [consensusHistory, setConsensusHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState('consensus'); // 'analyses' or 'consensus'
  const [selectedAnalysis, setSelectedAnalysis] = useState(null); // For logs modal

  useEffect(() => {
    loadHistory();
  }, [symbol, strategyFilter, viewMode]);

  async function loadHistory() {
    setLoading(true);
    try {
      if (viewMode === 'analyses') {
        // If symbol is empty, fetch all analyses; otherwise fetch for specific symbol
        const data = symbol
          ? await api.getAnalysisHistory(symbol, strategyFilter || null, 50)
          : await api.getAllRecentAnalyses(strategyFilter || null, 50);
        setAnalyses(data.analyses || []);
      } else {
        // If symbol is empty, fetch all consensus; otherwise fetch for specific symbol
        // Include imported forecasts in consensus view
        const data = symbol
          ? await api.getConsensusHistory(symbol, 50, true)
          : await api.getAllConsensus(50, true);
        setConsensusHistory(data.results || []);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-100">Analysis History</h1>
        <p className="text-gray-400 mt-1">View past strategy analyses and consensus results</p>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Symbol {symbol === '' && <span className="text-xs text-gray-500">(showing all)</span>}
            </label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="input w-full"
              placeholder="Leave empty for all symbols, or enter BTC-USD, TSLA, etc."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">View Type</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value)}
              className="input w-full"
            >
              <option value="analyses">Individual Analyses</option>
              <option value="consensus">Consensus Results</option>
            </select>
          </div>

          {viewMode === 'analyses' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Strategy</label>
              <select
                value={strategyFilter}
                onChange={(e) => setStrategyFilter(e.target.value)}
                className="input w-full"
              >
                {STRATEGY_TYPES.map(type => (
                  <option key={type.value} value={type.value}>{type.label}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      <div className="card">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-600"></div>
          </div>
        ) : viewMode === 'analyses' ? (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">
              {analyses.length} Analyses Found
            </h2>
            <div className="space-y-3">
              {analyses.map((analysis, idx) => (
                <div key={idx}
                     className="p-4 border border-gray-700 rounded-lg hover:border-brand-500 transition-colors cursor-pointer"
                     onClick={() => setSelectedAnalysis(analysis)}>
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className="font-semibold text-gray-100">{analysis.symbol}</span>
                        <span className="badge badge-info">{analysis.strategy_type}</span>
                        <span className={`badge ${
                          analysis.signal?.signal?.includes('BUY') ? 'badge-success' :
                          analysis.signal?.signal?.includes('SELL') ? 'badge-danger' :
                          'badge-warning'
                        }`}>
                          {analysis.signal?.signal}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mt-1">
                        ${analysis.current_price?.toFixed(2)} • Position: {analysis.signal?.position_size_pct || 0}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-400">
                        {format(new Date(analysis.created_at), 'MMM d, yyyy')}
                      </p>
                      <p className="text-xs text-gray-500">
                        {format(new Date(analysis.created_at), 'h:mm a')}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">
              {consensusHistory.length} Consensus Results
            </h2>
            <div className="space-y-3">
              {consensusHistory.map((consensus, idx) => (
                <div key={idx}
                     className="p-4 border border-gray-700 rounded-lg hover:border-brand-500 transition-colors cursor-pointer"
                     onClick={() => setSelectedAnalysis(consensus)}>
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className="font-semibold text-gray-100">{consensus.symbol}</span>
                        <span className={`badge ${
                          consensus.consensus?.includes('BUY') ? 'badge-success' :
                          consensus.consensus?.includes('SELL') ? 'badge-danger' :
                          'badge-warning'
                        }`}>
                          {consensus.consensus}
                        </span>
                        {consensus.source === 'imported' && (
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-900 text-blue-300 border border-blue-700">
                            <Network className="w-3 h-3 mr-1" />
                            Remote
                          </span>
                        )}
                        <span className="text-sm text-gray-400">
                          {consensus.source === 'imported' && consensus.remote_instance_name
                            ? `From: ${consensus.remote_instance_name}`
                            : consensus.strength}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 mt-1">
                        ${consensus.current_price?.toFixed(2)} • Bullish: {consensus.bullish_count}/{consensus.total_count}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-400">
                        {format(new Date(consensus.created_at), 'MMM d, yyyy')}
                      </p>
                      <p className="text-xs text-gray-500">
                        {format(new Date(consensus.created_at), 'h:mm a')}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Logs Modal */}
      {selectedAnalysis && (
        <LogsModal
          analysis={
            selectedAnalysis.source === 'imported'
              ? {
                  symbol: selectedAnalysis.symbol,
                  strategy_type: `Imported from ${selectedAnalysis.remote_instance_name}`,
                  created_at: selectedAnalysis.remote_created_at || selectedAnalysis.created_at,
                  current_price: selectedAnalysis.current_price,
                  forecast_data: selectedAnalysis.forecast_data,
                  signal: {
                    signal: selectedAnalysis.consensus,
                    position_size_pct: 0
                  },
                  status: 'imported',
                  execution_time_ms: null,
                  logs: [
                    `Forecast imported from remote instance: ${selectedAnalysis.remote_instance_name}`,
                    `Original forecast ID: ${selectedAnalysis.original_forecast_id || selectedAnalysis.id}`,
                    `Imported at: ${new Date(selectedAnalysis.imported_at || selectedAnalysis.created_at).toLocaleString()}`,
                    `Interval: ${selectedAnalysis.interval}`,
                    `Consensus: ${selectedAnalysis.consensus}`,
                    `Confidence: ${selectedAnalysis.confidence || 0}%`,
                    `Signal breakdown: ${selectedAnalysis.signals?.bullish_count || selectedAnalysis.bullish_count} bullish, ${selectedAnalysis.signals?.bearish_count || selectedAnalysis.bearish_count} bearish`,
                    `Forecast horizon: ${selectedAnalysis.forecast_data?.horizon_days || 'N/A'} days`,
                    `Current price: $${selectedAnalysis.current_price?.toFixed(2)}`,
                  ]
                }
              : selectedAnalysis
          }
          onClose={() => setSelectedAnalysis(null)}
        />
      )}
    </div>
  );
}

export default HistoryPage;
