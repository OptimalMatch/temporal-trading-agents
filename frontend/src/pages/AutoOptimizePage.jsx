import { useState, useEffect } from 'react';
import { Zap, Play, CheckCircle2, Clock, AlertCircle, TrendingUp, Settings, ArrowRight, Loader2, Award, Target, BarChart2 } from 'lucide-react';
import api from '../services/api';

export default function AutoOptimizePage() {
  const [isRunning, setIsRunning] = useState(false);
  const [autoOptimizeRun, setAutoOptimizeRun] = useState(null);
  const [error, setError] = useState(null);
  const [pollInterval, setPollInterval] = useState(null);

  // Configuration
  const [config, setConfig] = useState({
    symbol: 'BTC-USD',
    start_date: '2023-01-01',
    end_date: new Date().toISOString().split('T')[0],
    initial_capital: 100000,
    enabled_strategies: ['gradient', 'confidence', 'volatility', 'acceleration', 'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'],
  });

  const optimizationStages = [
    {
      id: 1,
      name: 'Min Edge Discovery',
      description: 'Finding optimal minimum edge threshold',
      icon: Target,
      estimatedTime: '45-60 min',
    },
    {
      id: 2,
      name: 'Position Sizing',
      description: 'Optimizing position size using best min_edge from Stage 1',
      icon: BarChart2,
      estimatedTime: '10-15 min',
    },
    {
      id: 3,
      name: 'Strategy Selection',
      description: 'Testing each strategy individually to find best performer',
      icon: Award,
      estimatedTime: '20-30 min',
    },
    {
      id: 4,
      name: 'Fine Tuning',
      description: 'Narrowing parameter ranges around winners',
      icon: Settings,
      estimatedTime: '15-20 min',
    },
  ];

  // Cleanup poll interval on unmount
  useEffect(() => {
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [pollInterval]);

  // Check for existing running workflow on mount
  useEffect(() => {
    const checkExistingRun = async () => {
      try {
        const response = await api.listAutoOptimizes(1); // Get the most recent run
        const runs = response.auto_optimizes || [];
        if (runs.length > 0) {
          const latestRun = runs[0];
          // If the latest run is still running or pending, attach to it
          if (latestRun.status === 'running' || latestRun.status === 'pending') {
            setAutoOptimizeRun(latestRun);
            setIsRunning(true);

            // Start polling for this run
            const interval = setInterval(async () => {
              try {
                const updatedRun = await api.getAutoOptimize(latestRun.auto_optimize_id);
                setAutoOptimizeRun(updatedRun);

                // Stop polling if completed, failed, or cancelled
                if (updatedRun.status === 'completed' || updatedRun.status === 'failed' || updatedRun.status === 'cancelled') {
                  clearInterval(interval);
                  setIsRunning(false);

                  if (updatedRun.status === 'failed') {
                    setError(updatedRun.error_message || 'Auto-optimization failed');
                  } else if (updatedRun.status === 'cancelled') {
                    setError('Auto-optimization was cancelled');
                  }
                }
              } catch (err) {
                console.error('Error polling auto-optimize status:', err);
              }
            }, 5000);

            setPollInterval(interval);
          }
        }
      } catch (err) {
        console.error('Error checking for existing run:', err);
      }
    };

    checkExistingRun();
  }, []); // Run once on mount

  const startAutoOptimization = async () => {
    setIsRunning(true);
    setError(null);
    setAutoOptimizeRun(null);

    try {
      // Create auto-optimize run (backend will orchestrate all stages)
      const run = await api.createAutoOptimize('Auto-Optimize Run', config);
      setAutoOptimizeRun(run);

      // Start polling for status updates every 5 seconds
      const interval = setInterval(async () => {
        try {
          const updatedRun = await api.getAutoOptimize(run.auto_optimize_id);
          setAutoOptimizeRun(updatedRun);

          // Stop polling if completed, failed, or cancelled
          if (updatedRun.status === 'completed' || updatedRun.status === 'failed' || updatedRun.status === 'cancelled') {
            clearInterval(interval);
            setIsRunning(false);

            if (updatedRun.status === 'failed') {
              setError(updatedRun.error_message || 'Auto-optimization failed');
            } else if (updatedRun.status === 'cancelled') {
              setError('Auto-optimization was cancelled');
            }
          }
        } catch (err) {
          console.error('Error polling auto-optimize status:', err);
        }
      }, 5000);

      setPollInterval(interval);

    } catch (err) {
      setError(err.message);
      setIsRunning(false);
      console.error('Auto-optimization failed:', err);
    }
  };

  const handleCancel = async () => {
    if (!autoOptimizeRun?.auto_optimize_id) return;

    if (!confirm('Are you sure you want to cancel this auto-optimization workflow?')) return;

    try {
      await api.cancelAutoOptimize(autoOptimizeRun.auto_optimize_id);
      // The polling will pick up the cancelled status
    } catch (err) {
      alert(`Failed to cancel: ${err.message}`);
    }
  };

  const applyOptimalParameters = async () => {
    if (!autoOptimizeRun?.optimal_params) return;

    if (!confirm('Apply these optimal parameters to a new paper trading session?')) return;

    try {
      const request = {
        name: `Auto-Optimized ${config.symbol} (${new Date().toLocaleDateString()})`,
        config: {
          symbol: config.symbol,
          initial_capital: config.initial_capital,
          position_size_pct: autoOptimizeRun.optimal_params.position_size_pct,
          min_edge_bps: autoOptimizeRun.optimal_params.min_edge_bps,
          strong_buy_threshold: autoOptimizeRun.optimal_params.strong_buy_threshold,
          buy_threshold: autoOptimizeRun.optimal_params.buy_threshold,
          moderate_buy_threshold: autoOptimizeRun.optimal_params.moderate_buy_threshold,
          sell_threshold: 0.60,
          moderate_sell_threshold: 0.50,
        },
      };

      await api.createPaperTradingSession(request);
      alert('New paper trading session created with optimal parameters!');
    } catch (err) {
      alert(`Failed to create session: ${err.message}`);
    }
  };

  const resetAndRunAnother = () => {
    setAutoOptimizeRun(null);
    setError(null);
    setIsRunning(false);
    if (pollInterval) {
      clearInterval(pollInterval);
      setPollInterval(null);
    }
  };

  const getStageStatus = (stageId) => {
    if (!autoOptimizeRun) return 'pending';

    if (autoOptimizeRun.current_stage > stageId) {
      return 'completed';
    } else if (autoOptimizeRun.current_stage === stageId) {
      return 'running';
    } else {
      return 'pending';
    }
  };

  const formatDuration = (ms) => {
    if (!ms || ms < 0) return '-';
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const isCompleted = autoOptimizeRun?.status === 'completed';
  const isFailed = autoOptimizeRun?.status === 'failed';
  const isCancelled = autoOptimizeRun?.status === 'cancelled';

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Zap className="w-8 h-8 text-yellow-400" />
          <h1 className="text-3xl font-bold text-gray-100">Auto-Optimizer</h1>
        </div>
        <p className="text-gray-400">
          Automatically runs a multi-stage optimization workflow to find the best trading parameters
        </p>
      </div>

      {/* Configuration */}
      {!isRunning && !autoOptimizeRun && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-100 mb-4">Configuration</h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Symbol</label>
              <input
                type="text"
                value={config.symbol}
                onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Initial Capital</label>
              <input
                type="number"
                value={config.initial_capital}
                onChange={(e) => setConfig({ ...config, initial_capital: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Start Date</label>
              <input
                type="date"
                value={config.start_date}
                onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">End Date</label>
              <input
                type="date"
                value={config.end_date}
                onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>
          </div>
        </div>
      )}

      {/* Optimization Stages Overview */}
      {!isRunning && !autoOptimizeRun && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-100 mb-4">Optimization Workflow</h2>

          <div className="space-y-4">
            {optimizationStages.map((stage, index) => {
              const Icon = stage.icon;
              return (
                <div key={stage.id} className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
                      <Icon className="w-5 h-5 text-brand-400" />
                    </div>
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <h3 className="font-semibold text-gray-100">
                        Stage {stage.id}: {stage.name}
                      </h3>
                      <span className="text-sm text-gray-400">{stage.estimatedTime}</span>
                    </div>
                    <p className="text-sm text-gray-400">{stage.description}</p>
                  </div>

                  {index < optimizationStages.length - 1 && (
                    <ArrowRight className="w-5 h-5 text-gray-600 mt-2" />
                  )}
                </div>
              );
            })}
          </div>

          <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <p className="text-sm text-blue-400">
              <strong>Total estimated time:</strong> 90-125 minutes
              <br />
              The optimizer will run all 4 stages automatically on the backend. You can close this page and return later - the workflow will continue running.
            </p>
          </div>
        </div>
      )}

      {/* Start Button */}
      {!isRunning && !autoOptimizeRun && (
        <button
          onClick={startAutoOptimization}
          className="w-full py-4 bg-gradient-to-r from-brand-500 to-purple-500 hover:from-brand-600 hover:to-purple-600 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all"
        >
          <Play className="w-5 h-5" />
          Start Auto-Optimization
        </button>
      )}

      {/* Progress */}
      {(isRunning || autoOptimizeRun) && !isCompleted && !isFailed && !isCancelled && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-100 mb-4 flex items-center gap-2">
            <Loader2 className="w-5 h-5 animate-spin text-brand-400" />
            Optimization in Progress
          </h2>

          <div className="space-y-4">
            {optimizationStages.map((stage) => {
              const stageStatus = getStageStatus(stage.id);
              const Icon = stage.icon;
              const stageResult = autoOptimizeRun?.stages?.find(s => s.stage_id === stage.id);

              let statusIcon;
              let statusColor;

              if (stageStatus === 'completed') {
                statusIcon = <CheckCircle2 className="w-5 h-5 text-green-400" />;
                statusColor = 'text-green-400';
              } else if (stageStatus === 'running') {
                statusIcon = <Loader2 className="w-5 h-5 animate-spin text-brand-400" />;
                statusColor = 'text-brand-400';
              } else {
                statusIcon = <Clock className="w-5 h-5 text-gray-600" />;
                statusColor = 'text-gray-600';
              }

              return (
                <div key={stage.id} className="border border-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-shrink-0">
                      {statusIcon}
                    </div>

                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className={`font-medium ${statusColor}`}>
                          {stage.name}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Show winners for completed stages */}
                  {stageStatus === 'completed' && stageResult && (
                    <div className="mt-3 pt-3 border-t border-gray-700 text-sm">
                      <div className="grid grid-cols-2 gap-3">
                        {/* Key Metrics */}
                        <div>
                          <div className="text-gray-400 mb-1">Best Result:</div>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Sharpe:</span>
                              <span className="text-green-400 font-mono">
                                {stageResult.best_metrics?.sharpe_ratio?.toFixed(2) || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Return:</span>
                              <span className="text-blue-400 font-mono">
                                {stageResult.best_metrics?.total_return_pct?.toFixed(1) || 'N/A'}%
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Winning Parameters */}
                        <div>
                          <div className="text-gray-400 mb-1">Winners:</div>
                          <div className="space-y-1">
                            {stageResult.best_params?.min_edge_bps && (
                              <div className="flex justify-between">
                                <span className="text-gray-400">Min Edge:</span>
                                <span className="text-brand-400 font-mono">
                                  {stageResult.best_params.min_edge_bps} bps
                                </span>
                              </div>
                            )}
                            {stageResult.best_params?.position_size_pct && (
                              <div className="flex justify-between">
                                <span className="text-gray-400">Position:</span>
                                <span className="text-brand-400 font-mono">
                                  {stageResult.best_params.position_size_pct}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {autoOptimizeRun?.started_at && (
            <div className="mt-4 pt-4 border-t border-gray-700 text-sm text-gray-400">
              Running for: {formatDuration(Date.now() - new Date(autoOptimizeRun.started_at + 'Z').getTime())}
            </div>
          )}

          {/* Cancel Button */}
          <button
            onClick={handleCancel}
            className="mt-4 w-full py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all"
          >
            <AlertCircle className="w-5 h-5" />
            Cancel Optimization
          </button>
        </div>
      )}

      {/* Final Results */}
      {isCompleted && autoOptimizeRun && (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-gray-100 mb-4 flex items-center gap-2">
              <Award className="w-6 h-6 text-yellow-400" />
              Optimization Complete!
            </h2>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
                <div className="text-2xl font-bold text-green-400">
                  {autoOptimizeRun.optimal_metrics?.sharpe_ratio?.toFixed(2) || 'N/A'}
                </div>
                {autoOptimizeRun.improvement_pct && (
                  <div className="text-sm text-gray-400">
                    +{autoOptimizeRun.improvement_pct.toFixed(1)}% improvement
                  </div>
                )}
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">Total Return</div>
                <div className="text-2xl font-bold text-blue-400">
                  {autoOptimizeRun.optimal_metrics?.total_return_pct?.toFixed(1) || 'N/A'}%
                </div>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">Execution Time</div>
                <div className="text-2xl font-bold text-purple-400">
                  {formatDuration(autoOptimizeRun.execution_time_ms)}
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 mb-4">
              <h3 className="font-semibold text-gray-100 mb-3">Optimal Parameters</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-400">Min Edge:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {autoOptimizeRun.optimal_params?.min_edge_bps || 'N/A'} bps
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Position Size:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {autoOptimizeRun.optimal_params?.position_size_pct || 'N/A'}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Strong Buy:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {autoOptimizeRun.optimal_params?.strong_buy_threshold || 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Buy:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {autoOptimizeRun.optimal_params?.buy_threshold || 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={applyOptimalParameters}
              className="w-full py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all"
            >
              <TrendingUp className="w-5 h-5" />
              Apply to Paper Trading
            </button>
          </div>

          <button
            onClick={resetAndRunAnother}
            className="w-full py-3 bg-gray-700 hover:bg-gray-600 text-gray-100 font-semibold rounded-lg transition-all"
          >
            Run Another Optimization
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-red-400">Optimization Failed</div>
            <div className="text-sm text-red-300 mt-1">{error}</div>
          </div>
        </div>
      )}
    </div>
  );
}
