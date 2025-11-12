import { useState, useEffect } from 'react';
import { Zap, Play, CheckCircle2, Clock, AlertCircle, TrendingUp, Settings, ArrowRight, Loader2, Award, Target, BarChart2 } from 'lucide-react';
import api from '../services/api';

export default function AutoOptimizePage() {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [stages, setStages] = useState([]);
  const [finalResults, setFinalResults] = useState(null);
  const [error, setError] = useState(null);

  // Configuration
  const [config, setConfig] = useState({
    symbol: 'BTC-USD',
    startDate: '2023-01-01',
    endDate: new Date().toISOString().split('T')[0],
    initialCapital: 100000,
    enabledStrategies: ['gradient', 'confidence', 'volatility', 'acceleration', 'swing', 'risk_adjusted', 'mean_reversion', 'multi_timeframe'],
  });

  const optimizationStages = [
    {
      id: 1,
      name: 'Min Edge Discovery',
      description: 'Finding optimal minimum edge threshold',
      params: {
        position_size_pct: [15],
        min_edge_bps: [25, 50, 75, 100, 125, 150, 175, 200],
        strong_buy_threshold: [0.75, 0.80],
        buy_threshold: [0.60, 0.65],
        moderate_buy_threshold: [0.50, 0.55],
      },
      icon: Target,
      estimatedTime: '45-60 min',
    },
    {
      id: 2,
      name: 'Position Sizing',
      description: 'Optimizing position size using best min_edge from Stage 1',
      params: {
        position_size_pct: [5, 10, 15, 20, 25, 30],
        min_edge_bps: null, // Will be filled from stage 1 winner
        strong_buy_threshold: null, // Will be filled from stage 1
        buy_threshold: null,
        moderate_buy_threshold: null,
      },
      icon: BarChart2,
      estimatedTime: '10-15 min',
    },
    {
      id: 3,
      name: 'Strategy Selection',
      description: 'Testing each strategy individually to find best performer',
      params: {
        // Will test each strategy one at a time
        position_size_pct: null, // From stage 2
        min_edge_bps: null, // From stage 1
      },
      icon: Award,
      estimatedTime: '20-30 min',
    },
    {
      id: 4,
      name: 'Fine Tuning',
      description: 'Narrowing parameter ranges around winners',
      params: {
        // Narrow ranges around winners from previous stages
      },
      icon: Settings,
      estimatedTime: '15-20 min',
    },
  ];

  const startAutoOptimization = async () => {
    setIsRunning(true);
    setError(null);
    setStages([]);
    setFinalResults(null);

    try {
      // Stage 1: Min Edge Discovery
      const stage1Result = await runOptimizationStage(1);
      updateStageStatus(1, 'completed', stage1Result);

      // Stage 2: Position Sizing (using stage 1 winner)
      const stage2Params = {
        ...optimizationStages[1].params,
        min_edge_bps: [stage1Result.best_params.min_edge_bps],
        strong_buy_threshold: [stage1Result.best_params.strong_buy_threshold],
        buy_threshold: [stage1Result.best_params.buy_threshold],
        moderate_buy_threshold: [stage1Result.best_params.moderate_buy_threshold],
      };
      const stage2Result = await runOptimizationStage(2, stage2Params);
      updateStageStatus(2, 'completed', stage2Result);

      // Stage 3: Strategy Selection (using winners from 1 & 2)
      const stage3Results = await runStrategyComparison(
        stage1Result.best_params.min_edge_bps,
        stage2Result.best_params.position_size_pct
      );
      updateStageStatus(3, 'completed', stage3Results);

      // Stage 4: Fine Tuning
      const stage4Result = await runFineTuning(stage1Result, stage2Result, stage3Results);
      updateStageStatus(4, 'completed', stage4Result);

      // Compile final results
      setFinalResults({
        optimal_params: stage4Result.best_params,
        baseline_sharpe: stage1Result.baseline_sharpe || 0,
        optimized_sharpe: stage4Result.best_metrics.sharpe_ratio,
        improvement: {
          sharpe: ((stage4Result.best_metrics.sharpe_ratio / (stage1Result.baseline_sharpe || 1) - 1) * 100),
          total_return: stage4Result.best_metrics.total_return_pct,
          win_rate: stage4Result.best_metrics.win_rate,
        },
        stages: stages,
      });

    } catch (err) {
      setError(err.message);
      console.error('Auto-optimization failed:', err);
    } finally {
      setIsRunning(false);
    }
  };

  const runOptimizationStage = async (stageId, customParams = null) => {
    updateStageStatus(stageId, 'running');

    const stage = optimizationStages[stageId - 1];
    const params = customParams || stage.params;

    const request = {
      name: `Auto-Opt Stage ${stageId}: ${stage.name}`,
      base_config: {
        symbol: config.symbol,
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: config.initialCapital,
        enabled_strategies: config.enabledStrategies,
        walk_forward: {
          enabled: true,
          train_window_days: 365,
          test_window_days: 63,
          retrain_frequency_days: 21,
        },
      },
      parameter_grid: {
        position_size_pct: params.position_size_pct,
        min_edge_bps: params.min_edge_bps,
        strong_buy_threshold: params.strong_buy_threshold,
        buy_threshold: params.buy_threshold,
        moderate_buy_threshold: params.moderate_buy_threshold,
        sell_threshold: [0.60],
        moderate_sell_threshold: [0.50],
      },
      optimization_metric: 'sharpe_ratio',
      top_n_results: 5,
    };

    const optimization = await api.createOptimization(request);

    // Poll for completion, passing stageId explicitly
    return await pollOptimizationCompletion(optimization.optimization_id, stageId);
  };

  const runStrategyComparison = async (bestMinEdge, bestPositionSize) => {
    updateStageStatus(3, 'running');

    const strategies = config.enabledStrategies;
    const results = [];
    const totalStrategies = strategies.length;

    for (let i = 0; i < strategies.length; i++) {
      const strategy = strategies[i];

      // Update progress for Stage 3 showing current strategy
      updateStageProgress(3, {
        completed: i,
        total: totalStrategies,
        percentage: Math.round((i / totalStrategies) * 100),
        currentStrategy: strategy
      });

      const request = {
        name: `Auto-Opt Stage 3: ${strategy}`,
        base_config: {
          symbol: config.symbol,
          start_date: config.startDate,
          end_date: config.endDate,
          initial_capital: config.initialCapital,
          enabled_strategies: [strategy], // Only this strategy
          walk_forward: {
            enabled: true,
            train_window_days: 365,
            test_window_days: 63,
            retrain_frequency_days: 21,
          },
        },
        parameter_grid: {
          position_size_pct: [bestPositionSize],
          min_edge_bps: [bestMinEdge],
          strong_buy_threshold: [0.75],
          buy_threshold: [0.60],
          moderate_buy_threshold: [0.50],
          sell_threshold: [0.60],
          moderate_sell_threshold: [0.50],
        },
        optimization_metric: 'sharpe_ratio',
        top_n_results: 1,
      };

      const optimization = await api.createOptimization(request);
      const result = await pollOptimizationCompletion(optimization.optimization_id, 3);
      results.push({
        strategy: strategy,
        sharpe: result.best_metrics.sharpe_ratio,
        return: result.best_metrics.total_return_pct,
      });
    }

    // Final update showing all strategies completed
    updateStageProgress(3, {
      completed: totalStrategies,
      total: totalStrategies,
      percentage: 100
    });

    // Sort by Sharpe ratio
    results.sort((a, b) => b.sharpe - a.sharpe);

    return {
      best_strategies: results.slice(0, 3).map(r => r.strategy),
      all_results: results,
    };
  };

  const runFineTuning = async (stage1Result, stage2Result, stage3Results) => {
    updateStageStatus(4, 'running');

    // Narrow ranges around winners
    const bestMinEdge = stage1Result.best_params.min_edge_bps;
    const bestPositionSize = stage2Result.best_params.position_size_pct;

    const request = {
      name: `Auto-Opt Stage 4: Fine Tuning`,
      base_config: {
        symbol: config.symbol,
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: config.initialCapital,
        enabled_strategies: stage3Results.best_strategies,
        walk_forward: {
          enabled: true,
          train_window_days: 365,
          test_window_days: 63,
          retrain_frequency_days: 21,
        },
      },
      parameter_grid: {
        position_size_pct: [
          Math.max(5, bestPositionSize - 5),
          Math.max(5, bestPositionSize - 2),
          bestPositionSize,
          bestPositionSize + 2,
          Math.min(30, bestPositionSize + 5),
        ],
        min_edge_bps: [
          Math.max(10, bestMinEdge - 20),
          Math.max(10, bestMinEdge - 10),
          bestMinEdge,
          bestMinEdge + 10,
          bestMinEdge + 20,
        ],
        strong_buy_threshold: [0.70, 0.75, 0.80],
        buy_threshold: [0.55, 0.60, 0.65],
        moderate_buy_threshold: [0.45, 0.50, 0.55],
        sell_threshold: [0.60],
        moderate_sell_threshold: [0.50],
      },
      optimization_metric: 'sharpe_ratio',
      top_n_results: 10,
    };

    const optimization = await api.createOptimization(request);
    return await pollOptimizationCompletion(optimization.optimization_id, 4);
  };

  const pollOptimizationCompletion = async (optimizationId, stageId) => {
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 5000)); // Poll every 5 seconds

      const optimization = await api.getOptimization(optimizationId);

      if (optimization.status === 'completed') {
        return {
          best_params: optimization.top_results[0].parameters,
          best_metrics: optimization.top_results[0].metrics,
          baseline_sharpe: optimization.baseline_metrics?.sharpe_ratio,
        };
      }

      if (optimization.status === 'failed') {
        throw new Error(`Optimization ${optimizationId} failed`);
      }

      // Update progress with completed/total combinations
      if (optimization.total_combinations > 0) {
        updateStageProgress(stageId, {
          completed: optimization.completed_combinations || 0,
          total: optimization.total_combinations,
          percentage: Math.round((optimization.completed_combinations || 0) / optimization.total_combinations * 100)
        });
      }
    }
  };

  const updateStageStatus = (stageId, status, result = null) => {
    setStages(prev => {
      const updated = [...prev];
      const stageIndex = updated.findIndex(s => s.id === stageId);

      if (stageIndex >= 0) {
        updated[stageIndex] = { ...updated[stageIndex], status, result };
      } else {
        updated.push({
          id: stageId,
          name: optimizationStages[stageId - 1].name,
          status,
          result,
        });
      }

      return updated;
    });

    if (status === 'running') {
      setCurrentStage(stageId);
    }
  };

  const updateStageProgress = (stageId, progress) => {
    console.log('updateStageProgress called:', { stageId, progress });
    // Update specified stage progress
    setStages(prev => {
      const updated = [...prev];
      const stageIndex = updated.findIndex(s => s.id === stageId);
      console.log('Found stage at index:', stageIndex, 'stages:', updated.length);
      if (stageIndex >= 0) {
        updated[stageIndex] = { ...updated[stageIndex], progress };
        console.log('Updated stage:', updated[stageIndex]);
      }
      return updated;
    });
  };

  const applyOptimalParameters = async () => {
    if (!finalResults) return;

    if (!confirm('Apply these optimal parameters to a new paper trading session?')) return;

    try {
      const request = {
        config: {
          symbol: config.symbol,
          initial_capital: config.initialCapital,
          position_size_pct: finalResults.optimal_params.position_size_pct,
          min_edge_bps: finalResults.optimal_params.min_edge_bps,
          strong_buy_threshold: finalResults.optimal_params.strong_buy_threshold,
          buy_threshold: finalResults.optimal_params.buy_threshold,
          moderate_buy_threshold: finalResults.optimal_params.moderate_buy_threshold,
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
      {!isRunning && !finalResults && (
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
                value={config.initialCapital}
                onChange={(e) => setConfig({ ...config, initialCapital: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Start Date</label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">End Date</label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-gray-100 rounded border border-gray-600"
              />
            </div>
          </div>
        </div>
      )}

      {/* Optimization Stages Overview */}
      {!isRunning && !finalResults && (
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
              The optimizer will run all 4 stages automatically and find the optimal parameters for maximizing Sharpe ratio.
            </p>
          </div>
        </div>
      )}

      {/* Start Button */}
      {!isRunning && !finalResults && (
        <button
          onClick={startAutoOptimization}
          className="w-full py-4 bg-gradient-to-r from-brand-500 to-purple-500 hover:from-brand-600 hover:to-purple-600 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all"
        >
          <Play className="w-5 h-5" />
          Start Auto-Optimization
        </button>
      )}

      {/* Progress */}
      {isRunning && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-100 mb-4 flex items-center gap-2">
            <Loader2 className="w-5 h-5 animate-spin text-brand-400" />
            Optimization in Progress
          </h2>

          <div className="space-y-4">
            {optimizationStages.map((stage) => {
              const stageStatus = stages.find(s => s.id === stage.id);
              const Icon = stage.icon;

              let statusIcon;
              let statusColor;

              if (stageStatus?.status === 'completed') {
                statusIcon = <CheckCircle2 className="w-5 h-5 text-green-400" />;
                statusColor = 'text-green-400';
              } else if (stageStatus?.status === 'running') {
                statusIcon = <Loader2 className="w-5 h-5 animate-spin text-brand-400" />;
                statusColor = 'text-brand-400';
              } else {
                statusIcon = <Clock className="w-5 h-5 text-gray-600" />;
                statusColor = 'text-gray-600';
              }

              return (
                <div key={stage.id} className="flex items-center gap-4">
                  <div className="flex-shrink-0">
                    {statusIcon}
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className={`font-medium ${statusColor}`}>
                        {stage.name}
                      </span>
                      {stageStatus?.progress && (
                        <span className="text-sm text-gray-400">
                          {stage.id === 3 && stageStatus.progress.currentStrategy ? (
                            <>Testing {stageStatus.progress.currentStrategy} ({stageStatus.progress.completed}/{stageStatus.progress.total} strategies)</>
                          ) : (
                            <>{stageStatus.progress.completed}/{stageStatus.progress.total} combinations ({stageStatus.progress.percentage}%)</>
                          )}
                        </span>
                      )}
                    </div>

                    {stageStatus?.status === 'running' && stageStatus?.progress && (
                      <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-brand-500 h-2 rounded-full transition-all"
                          style={{ width: `${stageStatus.progress.percentage}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Final Results */}
      {finalResults && (
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
                  {finalResults.optimized_sharpe.toFixed(2)}
                </div>
                <div className="text-sm text-gray-400">
                  +{finalResults.improvement.sharpe.toFixed(1)}% improvement
                </div>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">Total Return</div>
                <div className="text-2xl font-bold text-blue-400">
                  {finalResults.improvement.total_return.toFixed(1)}%
                </div>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">Win Rate</div>
                <div className="text-2xl font-bold text-purple-400">
                  {finalResults.improvement.win_rate.toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 mb-4">
              <h3 className="font-semibold text-gray-100 mb-3">Optimal Parameters</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-400">Min Edge:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {finalResults.optimal_params.min_edge_bps} bps
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Position Size:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {finalResults.optimal_params.position_size_pct}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Strong Buy:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {finalResults.optimal_params.strong_buy_threshold}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Buy:</span>{' '}
                  <span className="text-gray-100 font-mono">
                    {finalResults.optimal_params.buy_threshold}
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
            onClick={() => {
              setFinalResults(null);
              setStages([]);
            }}
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
