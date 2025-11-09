import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, Clock, AlertCircle } from 'lucide-react';
import api from '../services/api';
import StrategyCard from '../components/StrategyCard';
import ForecastChart from '../components/ForecastChart';
import ProgressIndicator from '../components/ProgressIndicator';
import useWebSocket from '../hooks/useWebSocket';

function StatCard({ title, value, icon: Icon, trend, color = 'blue' }) {
  const colors = {
    blue: 'bg-blue-900 text-blue-400 border border-blue-700',
    green: 'bg-green-900 text-green-400 border border-green-700',
    yellow: 'bg-yellow-900 text-yellow-400 border border-yellow-700',
    red: 'bg-red-900 text-red-400 border border-red-700',
  };

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="mt-2 text-3xl font-semibold text-gray-100">{value}</p>
          {trend && (
            <p className="mt-2 text-sm text-gray-400">
              <span className={trend > 0 ? 'text-green-400' : 'text-red-400'}>
                {trend > 0 ? '+' : ''}{trend}%
              </span>{' '}
              from last period
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colors[color]}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
}

function RecentAnalysis({ analysis }) {
  const signalColors = {
    BUY: 'badge-success',
    SELL: 'badge-danger',
    HOLD: 'badge-warning',
    'NO_SIGNAL': 'badge-info',
  };

  const signal = analysis.signal?.signal || 'UNKNOWN';
  const signalType = signal.includes('BUY') ? 'BUY' :
                     signal.includes('SELL') ? 'SELL' :
                     signal.includes('HOLD') ? 'HOLD' : 'NO_SIGNAL';

  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-700 last:border-0">
      <div className="flex-1">
        <div className="flex items-center space-x-3">
          <span className="font-semibold text-gray-100">{analysis.symbol}</span>
          <span className={`badge ${signalColors[signalType]}`}>{signal}</span>
        </div>
        <p className="text-sm text-gray-400 mt-1">
          {analysis.strategy_type} • ${analysis.current_price?.toFixed(2)}
        </p>
      </div>
      <div className="text-right">
        <p className="text-sm font-medium text-gray-100">
          {analysis.signal?.position_size_pct || 0}% position
        </p>
        <p className="text-xs text-gray-400">
          {new Date(analysis.created_at).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { connected, progress } = useWebSocket();

  useEffect(() => {
    loadDashboardData();
    // Refresh every 30 seconds
    const interval = setInterval(loadDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  async function loadDashboardData() {
    try {
      setLoading(true);

      // First, get the most recent consensus from any symbol (including imported)
      const allConsensus = await api.getAllConsensus(1, true);
      const latestConsensus = allConsensus.results?.[0];

      // Determine which symbol to show analytics for
      const symbolToShow = latestConsensus?.symbol || 'BTC-USD';

      // Load analytics for the symbol with latest consensus and all recent analyses
      const [symbolAnalytics, allRecentAnalyses] = await Promise.all([
        api.getSymbolAnalytics(symbolToShow),
        api.getAllRecentAnalyses(null, 20),  // Get recent analyses from all symbols
      ]);

      setStats(symbolAnalytics);
      setRecentAnalyses(allRecentAnalyses.analyses || []);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  }

  const latestProgress = progress[progress.length - 1];

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card flex items-center space-x-3 text-red-400 bg-red-900 border-red-700">
        <AlertCircle className="w-5 h-5" />
        <span>Error loading dashboard: {error}</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Dashboard</h1>
          <p className="text-gray-400 mt-1">
            Real-time trading strategy analysis and monitoring
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm text-gray-400">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Live Progress */}
      {latestProgress && latestProgress.status !== 'completed' && (
        <ProgressIndicator progress={latestProgress} />
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Analyses"
          value={stats?.total_analyses || 0}
          icon={Activity}
          color="blue"
        />
        <StatCard
          title="Consensus Analyses"
          value={stats?.consensus_count || 0}
          icon={TrendingUp}
          color="green"
        />
        <StatCard
          title="Last Analysis"
          value={stats?.latest_consensus ? new Date(stats.latest_consensus.created_at).toLocaleDateString() : 'N/A'}
          icon={Clock}
          color="yellow"
        />
        <StatCard
          title="Latest Signal"
          value={stats?.latest_consensus?.consensus || 'N/A'}
          icon={stats?.latest_consensus?.consensus?.includes('BUY') ? TrendingUp : TrendingDown}
          color={stats?.latest_consensus?.consensus?.includes('BUY') ? 'green' : 'red'}
        />
      </div>

      {/* Strategy Breakdown */}
      {stats?.strategy_breakdown && Object.keys(stats.strategy_breakdown).length > 0 && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Strategy Breakdown</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(stats.strategy_breakdown).map(([strategy, count]) => (
              <div key={strategy} className="text-center p-4 bg-gray-700 rounded-lg">
                <p className="text-2xl font-bold text-gray-100">{count}</p>
                <p className="text-sm text-gray-400 capitalize">{strategy}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Latest Consensus */}
      {stats?.latest_consensus && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Latest Consensus Analysis</h2>
          <div className="space-y-6">
            <StrategyCard consensus={stats.latest_consensus} />
            {stats.latest_consensus.forecast_data && (
              <ForecastChart forecastData={stats.latest_consensus.forecast_data} symbol={stats.latest_consensus.symbol} />
            )}
          </div>
        </div>
      )}

      {/* Recent Analyses */}
      {recentAnalyses.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Recent Analyses</h2>
          <div className="space-y-0">
            {recentAnalyses.map((analysis, index) => (
              <RecentAnalysis key={index} analysis={analysis} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
