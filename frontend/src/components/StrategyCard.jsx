import { TrendingUp, TrendingDown, Minus, Target, Shield } from 'lucide-react';

function StrategyCard({ consensus }) {
  if (!consensus) return null;

  const isBullish = consensus.consensus?.includes('BUY') || consensus.consensus?.includes('BULLISH');
  const isBearish = consensus.consensus?.includes('SELL') || consensus.consensus?.includes('AVOID');

  return (
    <div className="space-y-4">
      {/* Main Consensus */}
      <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-4">
          {isBullish ? (
            <div className="p-3 bg-green-100 rounded-lg">
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
          ) : isBearish ? (
            <div className="p-3 bg-red-100 rounded-lg">
              <TrendingDown className="w-8 h-8 text-red-600" />
            </div>
          ) : (
            <div className="p-3 bg-yellow-100 rounded-lg">
              <Minus className="w-8 h-8 text-yellow-600" />
            </div>
          )}
          <div>
            <h3 className="text-2xl font-bold text-gray-900">{consensus.consensus}</h3>
            <p className="text-gray-600">Strength: {consensus.strength}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-600">Average Position</p>
          <p className="text-3xl font-bold text-gray-900">{consensus.avg_position?.toFixed(0)}%</p>
        </div>
      </div>

      {/* Strategy Counts */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-4 bg-green-50 rounded-lg">
          <p className="text-3xl font-bold text-green-600">{consensus.bullish_count}</p>
          <p className="text-sm text-gray-600">Bullish</p>
        </div>
        <div className="text-center p-4 bg-red-50 rounded-lg">
          <p className="text-3xl font-bold text-red-600">{consensus.bearish_count}</p>
          <p className="text-sm text-gray-600">Bearish</p>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-3xl font-bold text-gray-600">{consensus.neutral_count}</p>
          <p className="text-sm text-gray-600">Neutral</p>
        </div>
      </div>

      {/* Strategy Lists */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {consensus.bullish_strategies && consensus.bullish_strategies.length > 0 && (
          <div className="p-4 border border-green-200 rounded-lg">
            <h4 className="font-semibold text-green-700 mb-2">Bullish Strategies</h4>
            <ul className="space-y-1">
              {consensus.bullish_strategies.map((strategy, idx) => (
                <li key={idx} className="text-sm text-gray-700 flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  {strategy}
                </li>
              ))}
            </ul>
          </div>
        )}
        {consensus.bearish_strategies && consensus.bearish_strategies.length > 0 && (
          <div className="p-4 border border-red-200 rounded-lg">
            <h4 className="font-semibold text-red-700 mb-2">Bearish/Neutral Strategies</h4>
            <ul className="space-y-1">
              {consensus.bearish_strategies.map((strategy, idx) => (
                <li key={idx} className="text-sm text-gray-700 flex items-center">
                  <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                  {strategy}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Current Price */}
      <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
        <span className="text-gray-700 font-medium">Current Price</span>
        <span className="text-2xl font-bold text-blue-600">
          ${consensus.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>
    </div>
  );
}

export default StrategyCard;
