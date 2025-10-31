import { TrendingUp, TrendingDown, Minus, Target, Shield } from 'lucide-react';

function StrategyCard({ consensus }) {
  if (!consensus) return null;

  const isBullish = consensus.consensus?.includes('BUY') || consensus.consensus?.includes('BULLISH');
  const isBearish = consensus.consensus?.includes('SELL') || consensus.consensus?.includes('AVOID');

  return (
    <div className="space-y-4">
      {/* Main Consensus */}
      <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
        <div className="flex items-center space-x-4">
          {isBullish ? (
            <div className="p-3 bg-green-900 border border-green-700 rounded-lg">
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          ) : isBearish ? (
            <div className="p-3 bg-red-900 border border-red-700 rounded-lg">
              <TrendingDown className="w-8 h-8 text-red-400" />
            </div>
          ) : (
            <div className="p-3 bg-yellow-900 border border-yellow-700 rounded-lg">
              <Minus className="w-8 h-8 text-yellow-400" />
            </div>
          )}
          <div>
            <h3 className="text-2xl font-bold text-gray-100">{consensus.consensus}</h3>
            <p className="text-gray-400">Strength: {consensus.strength}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-400">Average Position</p>
          <p className="text-3xl font-bold text-gray-100">{consensus.avg_position?.toFixed(0)}%</p>
        </div>
      </div>

      {/* Strategy Counts */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-4 bg-green-900 border border-green-700 rounded-lg">
          <p className="text-3xl font-bold text-green-400">{consensus.bullish_count}</p>
          <p className="text-sm text-gray-400">Bullish</p>
        </div>
        <div className="text-center p-4 bg-red-900 border border-red-700 rounded-lg">
          <p className="text-3xl font-bold text-red-400">{consensus.bearish_count}</p>
          <p className="text-sm text-gray-400">Bearish</p>
        </div>
        <div className="text-center p-4 bg-gray-700 rounded-lg">
          <p className="text-3xl font-bold text-gray-400">{consensus.neutral_count}</p>
          <p className="text-sm text-gray-400">Neutral</p>
        </div>
      </div>

      {/* Strategy Lists */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {consensus.bullish_strategies && consensus.bullish_strategies.length > 0 && (
          <div className="p-4 border border-green-700 bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-green-400 mb-2">Bullish Strategies</h4>
            <ul className="space-y-1">
              {consensus.bullish_strategies.map((strategy, idx) => (
                <li key={idx} className="text-sm text-gray-300 flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  {strategy}
                </li>
              ))}
            </ul>
          </div>
        )}
        {consensus.bearish_strategies && consensus.bearish_strategies.length > 0 && (
          <div className="p-4 border border-red-700 bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-red-400 mb-2">Bearish/Neutral Strategies</h4>
            <ul className="space-y-1">
              {consensus.bearish_strategies.map((strategy, idx) => (
                <li key={idx} className="text-sm text-gray-300 flex items-center">
                  <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                  {strategy}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Current Price */}
      <div className="flex items-center justify-between p-4 bg-blue-900 border border-blue-700 rounded-lg">
        <span className="text-gray-300 font-medium">Current Price</span>
        <span className="text-2xl font-bold text-blue-400">
          ${consensus.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>
    </div>
  );
}

export default StrategyCard;
