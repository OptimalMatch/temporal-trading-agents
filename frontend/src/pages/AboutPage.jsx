import { Brain, TrendingUp, Layers, Target, Zap, CheckCircle2, BarChart3, Users, GitBranch, Activity } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-brand-900/30 to-blue-900/30 border border-brand-700/50 rounded-lg p-8">
        <div className="flex items-start space-x-6">
          <Brain className="w-16 h-16 text-brand-500 flex-shrink-0" />
          <div>
            <h1 className="text-4xl font-bold text-gray-100 mb-3">
              Temporal Forecasting for Trading
            </h1>
            <p className="text-xl text-gray-300 leading-relaxed">
              A next-generation trading system that combines deep learning time-series forecasting
              with ensemble methods and multi-strategy consensus voting to predict market movements
              and generate high-probability trading signals.
            </p>
          </div>
        </div>
      </div>

      {/* What is Temporal Forecasting */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-4">
          <TrendingUp className="w-8 h-8 text-brand-500" />
          <h2 className="text-2xl font-bold text-gray-100">What is Temporal Forecasting?</h2>
        </div>
        <div className="space-y-4 text-gray-300">
          <p>
            <strong className="text-gray-100">Temporal forecasting</strong> is a deep learning approach that predicts
            future values of time-series data by learning patterns from historical sequences. Unlike traditional technical
            indicators that look at simple price patterns, temporal models capture complex, non-linear relationships across
            multiple time scales.
          </p>
          <p>
            Our system uses <strong className="text-gray-100">Temporal Fusion Transformers (TFT)</strong>, a state-of-the-art
            architecture developed by Google Research that combines:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li><strong className="text-gray-100">Attention mechanisms</strong> - Automatically identifies which historical periods are most relevant</li>
            <li><strong className="text-gray-100">Multi-horizon forecasting</strong> - Predicts multiple time steps into the future (3, 7, 14, 21 days)</li>
            <li><strong className="text-gray-100">Quantile predictions</strong> - Provides uncertainty estimates, not just point forecasts</li>
            <li><strong className="text-gray-100">Feature importance</strong> - Shows which inputs drive predictions</li>
          </ul>
        </div>
      </div>

      {/* The Ensemble Approach */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Layers className="w-8 h-8 text-purple-500" />
          <h2 className="text-2xl font-bold text-gray-100">Ensemble Learning Architecture</h2>
        </div>
        <div className="space-y-4 text-gray-300">
          <p>
            Rather than relying on a single model, our system uses an <strong className="text-gray-100">ensemble of multiple
            temporal models</strong>, each trained with different configurations:
          </p>

          <div className="grid md:grid-cols-3 gap-4 my-6">
            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <GitBranch className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-gray-100">Different Architectures</h3>
              </div>
              <p className="text-sm text-gray-400">
                Multiple model types (TFT, N-BEATS, DeepAR) capture different aspects of market behavior
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Activity className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-gray-100">Varied Hyperparameters</h3>
              </div>
              <p className="text-sm text-gray-400">
                Different learning rates, hidden dimensions, and attention heads prevent overfitting
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart3 className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-gray-100">Multiple Horizons</h3>
              </div>
              <p className="text-sm text-gray-400">
                Separate ensembles for 3-day, 7-day, 14-day, and 21-day forecasts
              </p>
            </div>
          </div>

          <p>
            Each ensemble combines 5-8 individual models using <strong className="text-gray-100">weighted averaging</strong> based
            on recent performance. This reduces model-specific biases and improves generalization to new market conditions.
          </p>
        </div>
      </div>

      {/* How Forecasts Become Trades */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Target className="w-8 h-8 text-green-500" />
          <h2 className="text-2xl font-bold text-gray-100">From Forecasts to Trading Signals</h2>
        </div>
        <div className="space-y-4 text-gray-300">
          <p>
            The raw temporal forecasts are transformed into actionable trading signals through an
            <strong className="text-gray-100"> 8-strategy consensus system</strong>. Each strategy analyzes
            the forecast from a unique perspective:
          </p>

          <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-4 my-4">
            <h3 className="font-semibold text-gray-100 mb-3">The 8 Consensus Strategies:</h3>
            <div className="grid md:grid-cols-2 gap-3 text-sm">
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Forecast Gradient</strong> - Analyzes the shape of the forecast curve (U-shaped, inverted-U, steep rise)
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Confidence-Weighted</strong> - Scales position size by model confidence scores
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Volatility Sizing</strong> - Adjusts positions based on forecast volatility
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Acceleration</strong> - Detects momentum in forecast trends
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Swing Trading</strong> - Multi-day positions based on forecast duration
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Risk-Adjusted</strong> - Normalizes by forecast uncertainty bands
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Mean Reversion</strong> - Combines forecasts with technical indicators (SMAs)
                </div>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-brand-500 mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="text-gray-200">Multi-Timeframe</strong> - Cross-validates across 3, 7, 14, 21-day horizons
                </div>
              </div>
            </div>
          </div>

          <p>
            Each strategy votes on every trading opportunity. A trade is only executed when
            <strong className="text-gray-100"> multiple strategies agree</strong>. This consensus approach filters out
            false signals and focuses on high-probability setups where different analytical perspectives align.
          </p>
        </div>
      </div>

      {/* Consensus Voting Power */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Users className="w-8 h-8 text-blue-500" />
          <h2 className="text-2xl font-bold text-gray-100">Why Consensus Voting Works</h2>
        </div>
        <div className="space-y-4 text-gray-300">
          <p>
            The power of our system comes from <strong className="text-gray-100">diversity of interpretation</strong>.
            All 8 strategies use the same temporal forecasts, but analyze them in fundamentally different ways:
          </p>

          <div className="grid md:grid-cols-3 gap-4 my-4">
            <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
              <h3 className="font-semibold text-gray-100 mb-2">When Strategies Agree</h3>
              <p className="text-sm text-gray-400">
                High consensus (6-8 strategies aligned) signals strong conviction. These are the highest-probability trades.
              </p>
            </div>

            <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
              <h3 className="font-semibold text-gray-100 mb-2">When Strategies Disagree</h3>
              <p className="text-sm text-gray-400">
                Low consensus (3-4 strategies) indicates conflicting signals. The system avoids these uncertain situations.
              </p>
            </div>

            <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
              <h3 className="font-semibold text-gray-100 mb-2">Adaptive Learning</h3>
              <p className="text-sm text-gray-400">
                Each strategy's weight can be adjusted based on recent performance, creating a meta-learning system.
              </p>
            </div>
          </div>

          <div className="bg-brand-900/20 border border-brand-700 rounded-lg p-4">
            <p className="text-sm text-gray-300">
              <strong className="text-brand-400">Key Insight:</strong> This isn't just voting - it's combining
              different lenses on the same prediction. Like having 8 expert traders analyzing the same forecast,
              each bringing their unique perspective. When they all agree, confidence is high. When they disagree,
              it's a signal to stay cautious.
            </p>
          </div>
        </div>
      </div>

      {/* Backtesting & Optimization */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="w-8 h-8 text-yellow-500" />
          <h2 className="text-2xl font-bold text-gray-100">Rigorous Testing Framework</h2>
        </div>
        <div className="space-y-4 text-gray-300">
          <p>
            Every strategy and parameter combination can be tested using our comprehensive backtesting engine:
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold text-gray-100 mb-2">Walk-Forward Validation</h3>
              <ul className="text-sm space-y-1 ml-4 list-disc">
                <li>Simulates real-world trading with train/test splits</li>
                <li>Prevents overfitting by testing on unseen data</li>
                <li>Models are retrained periodically as in production</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-100 mb-2">Realistic Cost Modeling</h3>
              <ul className="text-sm space-y-1 ml-4 list-disc">
                <li>Exchange fees (maker/taker)</li>
                <li>Bid-ask spread costs</li>
                <li>Slippage and market impact</li>
                <li>SEC fees for equity trading</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-100 mb-2">Parameter Optimization</h3>
              <ul className="text-sm space-y-1 ml-4 list-disc">
                <li>Grid search across parameter combinations</li>
                <li>Strategy selection optimization</li>
                <li>Multi-objective ranking (Sharpe, return, drawdown)</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-100 mb-2">Market Regime Analysis</h3>
              <ul className="text-sm space-y-1 ml-4 list-disc">
                <li>Tracks performance across 9 market regimes</li>
                <li>Identifies which conditions favor each strategy</li>
                <li>Helps adapt to changing market dynamics</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Stack */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="w-8 h-8 text-indigo-500" />
          <h2 className="text-2xl font-bold text-gray-100">Technology Stack</h2>
        </div>
        <div className="grid md:grid-cols-2 gap-6 text-gray-300">
          <div>
            <h3 className="font-semibold text-gray-100 mb-3">Forecasting Engine</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>PyTorch Forecasting</strong> - Production-grade temporal models</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>PyTorch Lightning</strong> - Distributed training infrastructure</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>Polygon.io/Massive.com</strong> - Market data feeds</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>MongoDB</strong> - Time-series data storage</span>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-100 mb-3">Trading System</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>FastAPI</strong> - High-performance API backend</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>React</strong> - Interactive dashboard</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>Docker</strong> - Containerized deployment</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-brand-500 mt-1">•</span>
                <span><strong>NumPy/Pandas</strong> - Quantitative analysis</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="bg-gradient-to-r from-green-900/20 to-brand-900/20 border border-green-700/50 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-gray-100 mb-4">Getting Started</h2>
        <div className="space-y-3 text-gray-300">
          <div className="flex items-start space-x-3">
            <span className="bg-green-900/50 text-green-400 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5">1</span>
            <p><strong className="text-gray-100">Dashboard</strong> - View live forecasts and consensus signals for your tracked symbols</p>
          </div>
          <div className="flex items-start space-x-3">
            <span className="bg-green-900/50 text-green-400 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5">2</span>
            <p><strong className="text-gray-100">Backtest</strong> - Test different strategy combinations and parameters on historical data</p>
          </div>
          <div className="flex items-start space-x-3">
            <span className="bg-green-900/50 text-green-400 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5">3</span>
            <p><strong className="text-gray-100">Optimize</strong> - Find the best parameter sets for your chosen symbols and timeframes</p>
          </div>
          <div className="flex items-start space-x-3">
            <span className="bg-green-900/50 text-green-400 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5">4</span>
            <p><strong className="text-gray-100">Paper Trade</strong> - Validate your strategy with simulated live trading before risking capital</p>
          </div>
        </div>
      </div>
    </div>
  );
}
