import { Brain, TrendingUp, Activity, Zap, Target, Shield, RotateCcw, Layers, Info, CheckCircle2, Users } from 'lucide-react';

export default function StrategyEducation({ isOpen, onClose }) {
  if (!isOpen) return null;

  const strategies = [
    {
      key: 'gradient',
      name: 'Forecast Gradient',
      icon: TrendingUp,
      color: 'blue',
      description: 'Analyzes the SHAPE of the temporal forecast curve rather than just endpoints',
      howItWorks: 'Identifies patterns like U-shaped (dip then recovery), inverted-U (peak then decline), steep rises, and gradual trends to optimize entry and exit timing',
      usesModel: 'Single 14-day temporal forecast',
      example: 'If the forecast shows a U-shape (price dips 5% then recovers 8%), it buys at the predicted trough and sells at recovery'
    },
    {
      key: 'confidence',
      name: 'Confidence-Weighted',
      icon: Target,
      color: 'green',
      description: 'Weights trading signals by the temporal model\'s confidence scores',
      howItWorks: 'Only takes positions when the model has high confidence in its predictions. Position size scales with confidence level',
      usesModel: 'Single 14-day temporal forecast + confidence metrics',
      example: 'A bullish forecast with 85% confidence gets larger position than same forecast with 60% confidence'
    },
    {
      key: 'volatility',
      name: 'Volatility Sizing',
      icon: Activity,
      color: 'purple',
      description: 'Dynamically adjusts position sizes based on forecast volatility and uncertainty',
      howItWorks: 'Reduces position size in high-volatility periods, increases in low-volatility periods for optimal risk management',
      usesModel: 'Single 14-day temporal forecast + volatility metrics',
      example: 'During volatile market: smaller 5% position. During calm market: larger 15% position for same signal'
    },
    {
      key: 'acceleration',
      name: 'Acceleration',
      icon: Zap,
      color: 'yellow',
      description: 'Looks at forecast momentum and rate of change',
      howItWorks: 'Identifies when the temporal model predicts accelerating or decelerating trends to catch momentum early',
      usesModel: 'Single 14-day temporal forecast - analyzing derivatives',
      example: 'Forecast shows price gaining 2%/day initially, then 5%/day - signals strong acceleration, early entry'
    },
    {
      key: 'swing',
      name: 'Swing Trading',
      icon: RotateCcw,
      color: 'orange',
      description: 'Uses 14-day temporal forecasts for multi-day position holds',
      howItWorks: 'Enters positions based on forecast direction and holds for several days as predicted trends unfold',
      usesModel: 'Single 14-day temporal forecast - full horizon',
      example: 'Forecast shows steady 10% gain over 14 days - enters position and holds for swing duration'
    },
    {
      key: 'risk_adjusted',
      name: 'Risk-Adjusted',
      icon: Shield,
      color: 'red',
      description: 'Normalizes signals by forecast risk metrics for optimal risk-reward',
      howItWorks: 'Calculates risk-adjusted returns using forecast uncertainty to ensure trades meet minimum risk/reward thresholds',
      usesModel: 'Single 14-day temporal forecast + uncertainty bands',
      example: 'Forecast +8% with ±3% uncertainty = 2.67 risk/reward ratio. Only trades if ratio > 2.0'
    },
    {
      key: 'mean_reversion',
      name: 'Mean Reversion',
      icon: TrendingUp,
      color: 'cyan',
      description: 'Identifies overbought/oversold conditions and trades reversions when temporal forecast confirms',
      howItWorks: 'Uses technical indicators (moving averages) to detect deviations, but ONLY trades when the temporal forecast agrees the price will revert',
      usesModel: 'Single 14-day temporal forecast + technical analysis (SMAs)',
      example: 'Price 12% below 50-day MA (oversold), but only buys if temporal forecast shows recovery to mean',
      hybrid: true
    },
    {
      key: 'multi_timeframe',
      name: 'Multi-Timeframe',
      icon: Layers,
      color: 'indigo',
      description: 'Trains temporal models at multiple horizons and looks for cross-timeframe alignment',
      howItWorks: 'Generates forecasts at 3, 7, 14, and 21 day horizons. Strong signals occur when all timeframes agree on direction',
      usesModel: 'Multiple temporal forecasts (3, 7, 14, 21 days)',
      example: 'All 4 timeframes predict gains (3d: +2%, 7d: +5%, 14d: +8%, 21d: +12%) = strong aligned buy signal',
      multiHorizon: true
    }
  ];

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-900/20 border-blue-700 text-blue-400',
      green: 'bg-green-900/20 border-green-700 text-green-400',
      purple: 'bg-purple-900/20 border-purple-700 text-purple-400',
      yellow: 'bg-yellow-900/20 border-yellow-700 text-yellow-400',
      orange: 'bg-orange-900/20 border-orange-700 text-orange-400',
      red: 'bg-red-900/20 border-red-700 text-red-400',
      cyan: 'bg-cyan-900/20 border-cyan-700 text-cyan-400',
      indigo: 'bg-indigo-900/20 border-indigo-700 text-indigo-400',
    };
    return colors[color] || colors.blue;
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-gray-800 border-b border-gray-700 p-6 z-10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-100 flex items-center space-x-3">
                <Brain className="w-8 h-8 text-brand-500" />
                <span>Consensus Strategy Education</span>
              </h2>
              <p className="text-gray-400 mt-2">
                Understanding how each strategy leverages temporal forecasting
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-200"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Consensus Voting Strength */}
          <div className="bg-brand-900/20 border border-brand-700 rounded-lg p-6">
            <div className="flex items-start space-x-4">
              <Users className="w-8 h-8 text-brand-500 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-xl font-bold text-gray-100 mb-3">
                  The Power of Consensus Voting
                </h3>
                <p className="text-gray-300 mb-4">
                  All 8 strategies use the same temporal forecasting model, but analyze it in fundamentally different ways.
                  This diversity of interpretation creates a robust consensus system:
                </p>
                <div className="grid md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                      <div className="font-semibold text-gray-200">6 Single-Horizon Strategies</div>
                    </div>
                    <p className="text-sm text-gray-400">
                      Use the 14-day temporal forecast from different angles: shape, confidence, volatility, momentum, duration, and risk
                    </p>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <CheckCircle2 className="w-5 h-5 text-cyan-400" />
                      <div className="font-semibold text-gray-200">1 Hybrid Strategy</div>
                    </div>
                    <p className="text-sm text-gray-400">
                      Combines temporal forecasts with technical analysis (Mean Reversion) for confirmation
                    </p>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <CheckCircle2 className="w-5 h-5 text-indigo-400" />
                      <div className="font-semibold text-gray-200">1 Multi-Horizon Strategy</div>
                    </div>
                    <p className="text-sm text-gray-400">
                      Uses temporal forecasts at 4 different timeframes (3, 7, 14, 21 days) for cross-timeframe alignment
                    </p>
                  </div>
                </div>
                <div className="bg-gray-700/30 border-l-4 border-brand-500 p-4 rounded">
                  <p className="text-sm text-gray-300">
                    <strong>Why this matters:</strong> When multiple strategies agree despite analyzing the forecast differently,
                    it signals high-probability opportunities. Disagreement often indicates uncertainty or conflicting signals that are best avoided.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Individual Strategies */}
          <div>
            <h3 className="text-xl font-bold text-gray-100 mb-4">Individual Strategy Breakdown</h3>
            <div className="space-y-4">
              {strategies.map((strategy) => {
                const Icon = strategy.icon;
                return (
                  <div key={strategy.key} className={`border rounded-lg p-5 ${getColorClasses(strategy.color)}`}>
                    <div className="flex items-start space-x-4">
                      <Icon className="w-8 h-8 flex-shrink-0 mt-1" />
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="text-lg font-bold text-gray-100">{strategy.name}</h4>
                          <div className="flex items-center space-x-2">
                            {strategy.hybrid && (
                              <span className="px-2 py-1 bg-cyan-900/50 border border-cyan-700 rounded text-xs font-medium text-cyan-300">
                                Hybrid
                              </span>
                            )}
                            {strategy.multiHorizon && (
                              <span className="px-2 py-1 bg-indigo-900/50 border border-indigo-700 rounded text-xs font-medium text-indigo-300">
                                Multi-Horizon
                              </span>
                            )}
                          </div>
                        </div>

                        <p className="text-gray-200 mb-3">{strategy.description}</p>

                        <div className="grid md:grid-cols-2 gap-3 text-sm">
                          <div>
                            <div className="font-semibold text-gray-300 mb-1">How It Works:</div>
                            <p className="text-gray-400">{strategy.howItWorks}</p>
                          </div>
                          <div>
                            <div className="font-semibold text-gray-300 mb-1">Temporal Model Usage:</div>
                            <p className="text-gray-400">{strategy.usesModel}</p>
                          </div>
                        </div>

                        <div className="mt-3 p-3 bg-gray-900/50 rounded border border-gray-600">
                          <div className="font-semibold text-gray-300 text-xs mb-1">Example:</div>
                          <p className="text-gray-400 text-sm">{strategy.example}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Key Takeaways */}
          <div className="bg-green-900/20 border border-green-700 rounded-lg p-5">
            <div className="flex items-start space-x-3">
              <Info className="w-6 h-6 text-green-400 flex-shrink-0 mt-1" />
              <div>
                <h4 className="font-bold text-gray-100 mb-2">Key Takeaways</h4>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400 mt-1">•</span>
                    <span>Every strategy is powered by temporal forecasting - none are purely technical indicators</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400 mt-1">•</span>
                    <span>Each strategy interprets the forecast from a unique angle, creating diverse viewpoints</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400 mt-1">•</span>
                    <span>Consensus voting combines these viewpoints - agreement = confidence, disagreement = caution</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400 mt-1">•</span>
                    <span>You can test different strategy combinations to find what works best for your market and timeframe</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
