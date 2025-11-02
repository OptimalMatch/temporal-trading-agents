import { X, TrendingUp, TrendingDown, DollarSign, Activity, BarChart3, ArrowLeft, Cloud, CloudRain, CloudSnow, Sun, Wind, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

export default function BacktestResults({ backtest, onClose }) {
  if (!backtest || !backtest.metrics) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <Activity className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <p className="text-gray-400">No results available yet</p>
        <button
          onClick={onClose}
          className="mt-4 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
        >
          Back to List
        </button>
      </div>
    );
  }

  const metrics = backtest.metrics;

  // Prepare equity curve data for chart
  const equityCurveData = backtest.equity_curve?.map((point) => ({
    timestamp: new Date(point.timestamp).toLocaleDateString(),
    equity: point.equity,
    drawdown: Math.abs(point.drawdown * 100),
  })) || [];

  const formatPercent = (value) => {
    const pct = (value * 100).toFixed(2);
    const isPositive = value >= 0;
    return (
      <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
        {isPositive ? '+' : ''}{pct}%
      </span>
    );
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <button
            onClick={onClose}
            className="flex items-center space-x-2 text-gray-400 hover:text-gray-300 mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to List</span>
          </button>
          <h2 className="text-2xl font-bold text-gray-100">{backtest.name}</h2>
          <p className="text-gray-400 mt-1">
            {backtest.config.symbol} • {backtest.config.start_date} to {backtest.config.end_date}
          </p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="w-5 h-5 text-brand-500" />
            <p className="text-sm text-gray-400">Total Return</p>
          </div>
          <p className="text-2xl font-bold">{formatPercent(metrics.total_return)}</p>
          <p className="text-xs text-gray-400 mt-1">
            Annualized: {formatPercent(metrics.annualized_return)}
          </p>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="w-5 h-5 text-blue-500" />
            <p className="text-sm text-gray-400">Sharpe Ratio</p>
          </div>
          <p className="text-2xl font-bold text-gray-100">
            {metrics.sharpe_ratio.toFixed(2)}
          </p>
          {metrics.median_period_sharpe && (
            <p className="text-xs text-gray-400 mt-1">
              Median Period: {metrics.median_period_sharpe.toFixed(2)}
            </p>
          )}
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingDown className="w-5 h-5 text-red-500" />
            <p className="text-sm text-gray-400">Max Drawdown</p>
          </div>
          <p className="text-2xl font-bold text-red-400">
            {(metrics.max_drawdown * 100).toFixed(2)}%
          </p>
          {metrics.worst_period_drawdown && (
            <p className="text-xs text-gray-400 mt-1">
              Worst Period: {(metrics.worst_period_drawdown * 100).toFixed(2)}%
            </p>
          )}
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart3 className="w-5 h-5 text-purple-500" />
            <p className="text-sm text-gray-400">Total Trades</p>
          </div>
          <p className="text-2xl font-bold text-gray-100">{metrics.total_trades}</p>
          <p className="text-xs text-gray-400 mt-1">
            Win Rate: {(metrics.win_rate * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Equity Curve Chart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-100 mb-4">Equity Curve</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={equityCurveData}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="timestamp"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af' }}
            />
            <YAxis stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '0.5rem',
              }}
              labelStyle={{ color: '#f3f4f6' }}
            />
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              fillOpacity={1}
              fill="url(#equityGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown Chart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-100 mb-4">Drawdown</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={equityCurveData}>
            <defs>
              <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="timestamp"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af' }}
            />
            <YAxis stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '0.5rem',
              }}
              labelStyle={{ color: '#f3f4f6' }}
            />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke="#ef4444"
              fillOpacity={1}
              fill="url(#drawdownGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Metrics */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Performance Metrics */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Performance Metrics</h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-400">Total Return</dt>
              <dd className="text-gray-100 font-semibold">{formatPercent(metrics.total_return)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Annualized Return</dt>
              <dd className="text-gray-100 font-semibold">{formatPercent(metrics.annualized_return)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Sharpe Ratio</dt>
              <dd className="text-gray-100 font-semibold">{metrics.sharpe_ratio.toFixed(2)}</dd>
            </div>
            {metrics.sortino_ratio && (
              <div className="flex justify-between">
                <dt className="text-gray-400">Sortino Ratio</dt>
                <dd className="text-gray-100 font-semibold">{metrics.sortino_ratio.toFixed(2)}</dd>
              </div>
            )}
            <div className="flex justify-between">
              <dt className="text-gray-400">Profit Factor</dt>
              <dd className="text-gray-100 font-semibold">{metrics.profit_factor.toFixed(2)}</dd>
            </div>
          </dl>
        </div>

        {/* Risk Metrics */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Risk Metrics</h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-400">Max Drawdown</dt>
              <dd className="text-red-400 font-semibold">{(metrics.max_drawdown * 100).toFixed(2)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Avg Drawdown</dt>
              <dd className="text-gray-100 font-semibold">{(metrics.avg_drawdown * 100).toFixed(2)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Volatility</dt>
              <dd className="text-gray-100 font-semibold">
                {metrics.annualized_volatility ? (metrics.annualized_volatility * 100).toFixed(2) + '%' : 'N/A'}
              </dd>
            </div>
            {metrics.worst_period_drawdown && (
              <div className="flex justify-between">
                <dt className="text-gray-400">Worst Period Drawdown</dt>
                <dd className="text-red-400 font-semibold">{(metrics.worst_period_drawdown * 100).toFixed(2)}%</dd>
              </div>
            )}
          </dl>
        </div>

        {/* Trade Statistics */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Trade Statistics</h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-400">Total Trades</dt>
              <dd className="text-gray-100 font-semibold">{metrics.total_trades}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Winning Trades</dt>
              <dd className="text-green-400 font-semibold">{metrics.winning_trades}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Losing Trades</dt>
              <dd className="text-red-400 font-semibold">{metrics.losing_trades}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Win Rate</dt>
              <dd className="text-gray-100 font-semibold">{(metrics.win_rate * 100).toFixed(1)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Avg Win</dt>
              <dd className="text-green-400 font-semibold">{formatCurrency(metrics.avg_win)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Avg Loss</dt>
              <dd className="text-red-400 font-semibold">{formatCurrency(metrics.avg_loss)}</dd>
            </div>
          </dl>
        </div>

        {/* Cost Analysis */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Cost Analysis</h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-400">Total Costs</dt>
              <dd className="text-gray-100 font-semibold">{formatCurrency(metrics.total_costs)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Costs % of Capital</dt>
              <dd className="text-gray-100 font-semibold">{(metrics.costs_pct_of_capital * 100).toFixed(2)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Avg Cost per Trade</dt>
              <dd className="text-gray-100 font-semibold">
                {formatCurrency(metrics.total_costs / metrics.total_trades)}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Walk-Forward Periods */}
      {backtest.period_metrics && backtest.period_metrics.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Walk-Forward Periods</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-4 text-sm font-medium text-gray-400">Period</th>
                  <th className="text-left py-2 px-4 text-sm font-medium text-gray-400">Test Period</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Return</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Sharpe</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Max DD</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Trades</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Avg Capital</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Peak Capital</th>
                  <th className="text-right py-2 px-4 text-sm font-medium text-gray-400">Utilization</th>
                </tr>
              </thead>
              <tbody>
                {backtest.period_metrics.map((period) => (
                  <tr key={period.period_id} className="border-b border-gray-700/50">
                    <td className="py-2 px-4 text-sm text-gray-300">Period {period.period_number}</td>
                    <td className="py-2 px-4 text-sm text-gray-400">
                      {new Date(period.test_start).toLocaleDateString()} - {new Date(period.test_end).toLocaleDateString()}
                    </td>
                    <td className="py-2 px-4 text-sm text-right">
                      {formatPercent(period.total_return)}
                    </td>
                    <td className="py-2 px-4 text-sm text-right text-gray-100">
                      {period.sharpe_ratio.toFixed(2)}
                    </td>
                    <td className="py-2 px-4 text-sm text-right text-red-400">
                      {(period.max_drawdown * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 px-4 text-sm text-right text-gray-100">
                      {period.total_trades}
                    </td>
                    <td className="py-2 px-4 text-sm text-right text-gray-100">
                      {period.avg_capital_deployed ? formatCurrency(period.avg_capital_deployed) : 'N/A'}
                    </td>
                    <td className="py-2 px-4 text-sm text-right text-gray-100">
                      {period.peak_capital_deployed ? formatCurrency(period.peak_capital_deployed) : 'N/A'}
                    </td>
                    <td className="py-2 px-4 text-sm text-right text-blue-400">
                      {period.capital_utilization ? (period.capital_utilization * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Walk-Forward Summary */}
          <div className="mt-4 p-4 bg-gray-700/50 rounded-lg">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-400 mb-1">Median Period Sharpe</p>
                <p className="text-lg font-semibold text-gray-100">
                  {metrics.median_period_sharpe?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 mb-1">Period Win Rate</p>
                <p className="text-lg font-semibold text-gray-100">
                  {metrics.period_win_rate ? (metrics.period_win_rate * 100).toFixed(0) + '%' : 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 mb-1">Total Periods</p>
                <p className="text-lg font-semibold text-gray-100">
                  {backtest.period_metrics.length}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Regime Analysis */}
      {backtest.regime_analysis && backtest.regime_analysis.regime_statistics && Object.keys(backtest.regime_analysis.regime_statistics).length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Cloud className="w-6 h-6 text-brand-500" />
              <h3 className="text-lg font-semibold text-gray-100">Market Regime Analysis</h3>
            </div>
            <div className="text-sm text-gray-400">
              {backtest.regime_analysis.total_regime_changes} regime changes
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left border-b border-gray-700">
                  <th className="py-2 px-4 text-sm font-medium text-gray-400">Regime</th>
                  <th className="py-2 px-4 text-sm font-medium text-gray-400 text-right">Time %</th>
                  <th className="py-2 px-4 text-sm font-medium text-gray-400 text-right">Trades</th>
                  <th className="py-2 px-4 text-sm font-medium text-gray-400 text-right">Win Rate</th>
                  <th className="py-2 px-4 text-sm font-medium text-gray-400 text-right">Avg Return</th>
                  <th className="py-2 px-4 text-sm font-medium text-gray-400 text-right">Total P&L</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(backtest.regime_analysis.regime_statistics)
                  .sort((a, b) => b[1].regime_pct - a[1].regime_pct)
                  .map(([regimeName, stats]) => {
                    // Determine icon and color based on regime
                    let Icon = Cloud;
                    let volColor = 'text-gray-400';
                    let trendColor = 'text-gray-400';

                    if (regimeName.includes('high_vol')) {
                      Icon = Zap;
                      volColor = 'text-orange-400';
                    } else if (regimeName.includes('low_vol')) {
                      Icon = Sun;
                      volColor = 'text-yellow-400';
                    } else if (regimeName.includes('med_vol')) {
                      Icon = Wind;
                      volColor = 'text-blue-400';
                    }

                    if (regimeName.includes('uptrend')) {
                      trendColor = 'text-green-400';
                    } else if (regimeName.includes('downtrend')) {
                      trendColor = 'text-red-400';
                    }

                    return (
                      <tr key={regimeName} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                        <td className="py-3 px-4">
                          <div className="flex items-center space-x-2">
                            <Icon className={`w-4 h-4 ${volColor}`} />
                            <div>
                              <div className={`text-sm font-medium ${trendColor}`}>
                                {stats.description.split(' - ')[0]}
                              </div>
                              <div className="text-xs text-gray-500">
                                {stats.description.split(' - ')[1]}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-sm text-right text-gray-300">
                          {stats.regime_pct.toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-sm text-right text-gray-300">
                          {stats.total_trades}
                          {stats.total_trades > 0 && (
                            <span className="text-xs text-gray-500 ml-1">
                              ({stats.winning_trades}W/{stats.losing_trades}L)
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-sm text-right">
                          {stats.total_trades > 0 ? (
                            <span className={stats.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}>
                              {stats.win_rate.toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-gray-500">-</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-sm text-right">
                          {stats.avg_return !== 0 ? (
                            <span className={stats.avg_return > 0 ? 'text-green-400' : 'text-red-400'}>
                              {(stats.avg_return * 100).toFixed(3)}%
                            </span>
                          ) : (
                            <span className="text-gray-500">-</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-sm text-right font-medium">
                          {(() => {
                            const pnlPct = (stats.total_pnl * 100);
                            const rounded = parseFloat(pnlPct.toFixed(2));

                            if (rounded === 0) {
                              return <span className="text-gray-500">0.00%</span>;
                            }

                            return (
                              <span className={rounded > 0 ? 'text-green-400' : 'text-red-400'}>
                                {rounded > 0 ? '+' : ''}{rounded.toFixed(2)}%
                              </span>
                            );
                          })()}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>

          <div className="mt-4 p-3 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-4 text-xs text-gray-400">
              <div className="flex items-center space-x-1">
                <Zap className="w-3 h-3 text-orange-400" />
                <span>High Vol</span>
              </div>
              <div className="flex items-center space-x-1">
                <Wind className="w-3 h-3 text-blue-400" />
                <span>Med Vol</span>
              </div>
              <div className="flex items-center space-x-1">
                <Sun className="w-3 h-3 text-yellow-400" />
                <span>Low Vol</span>
              </div>
              <div className="border-l border-gray-600 pl-4 flex items-center space-x-2">
                <TrendingUp className="w-3 h-3 text-green-400" />
                <span className="text-green-400">Uptrend</span>
                <TrendingDown className="w-3 h-3 text-red-400 ml-2" />
                <span className="text-red-400">Downtrend</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Configuration */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-100 mb-4">Configuration</h3>
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-gray-400 mb-1">Initial Capital</dt>
            <dd className="text-gray-100 font-semibold">{formatCurrency(backtest.config.initial_capital)}</dd>
          </div>
          <div>
            <dt className="text-gray-400 mb-1">Position Size</dt>
            <dd className="text-gray-100 font-semibold">{backtest.config.position_size_pct}%</dd>
          </div>
          <div>
            <dt className="text-gray-400 mb-1">Min Edge (bps)</dt>
            <dd className="text-gray-100 font-semibold">{backtest.config.min_edge_bps}</dd>
          </div>
          <div>
            <dt className="text-gray-400 mb-1">Walk-Forward</dt>
            <dd className="text-gray-100 font-semibold">
              {backtest.config.walk_forward.enabled ? 'Enabled' : 'Disabled'}
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
