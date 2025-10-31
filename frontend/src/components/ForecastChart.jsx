import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

function ForecastChart({ forecastData, symbol }) {
  const [showIndividualModels, setShowIndividualModels] = useState(true);
  const [historicalDays, setHistoricalDays] = useState(120);
  const [extendedHistoricalData, setExtendedHistoricalData] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  // Fetch additional historical data when user selects longer timeframes
  useEffect(() => {
    if (!symbol || !forecastData) return;

    const availableDays = forecastData.historical_days;

    // Only fetch if user selected more days than what's in forecastData
    if (historicalDays > availableDays) {
      const fetchHistoricalData = async () => {
        setLoadingHistory(true);
        try {
          // Use relative path to go through nginx proxy
          const response = await fetch(`/api/v1/history/prices/${symbol}`);
          if (response.ok) {
            const data = await response.json();
            // Extract close prices from the historical data
            const closePrices = data.prices.map(p => p.close);
            setExtendedHistoricalData(closePrices);
          } else {
            console.error('Failed to fetch historical data:', response.status, response.statusText);
          }
        } catch (error) {
          console.error('Failed to fetch extended historical data:', error);
        } finally {
          setLoadingHistory(false);
        }
      };

      fetchHistoricalData();
    } else {
      // Don't need extended data, clear it
      setExtendedHistoricalData(null);
    }
  }, [symbol, historicalDays, forecastData]);

  if (!forecastData) {
    return (
      <div className="p-8 text-center text-gray-400">
        No forecast data available for visualization
      </div>
    );
  }

  // Build chart data combining historical and forecast
  const chartData = [];

  // Determine which historical data to use
  let historicalPrices = forecastData.historical_prices;
  let availableDays = forecastData.historical_days;

  // Use extended historical data if available and needed
  if (extendedHistoricalData && historicalDays > forecastData.historical_days) {
    historicalPrices = extendedHistoricalData;
    availableDays = extendedHistoricalData.length;
  }

  // Historical data (negative days) - filter based on user selection
  const displayDays = Math.min(historicalDays, availableDays);
  const startIndex = availableDays - displayDays;
  const histStart = -displayDays;

  historicalPrices.slice(startIndex).forEach((price, idx) => {
    chartData.push({
      day: histStart + idx,
      historical: price,
      isHistorical: true
    });
  });

  // Add current price marker (day 0)
  chartData.push({
    day: 0,
    historical: forecastData.current_price,
    forecast_median: forecastData.current_price,
    isHistorical: false
  });

  // Forecast data (positive days)
  forecastData.forecast_days.forEach((day, idx) => {
    const dataPoint = {
      day: day,
      forecast_median: forecastData.ensemble_median[idx],
      forecast_q25: forecastData.ensemble_q25[idx],
      forecast_q75: forecastData.ensemble_q75[idx],
      forecast_min: forecastData.ensemble_min[idx],
      forecast_max: forecastData.ensemble_max[idx],
      isHistorical: false
    };

    // Add individual model predictions if enabled
    if (showIndividualModels) {
      forecastData.individual_models.forEach((model) => {
        dataPoint[`model_${model.name}`] = model.prices[idx];
      });
    }

    chartData.push(dataPoint);
  });

  // Format price for display
  const formatPrice = (value) => {
    if (value === null || value === undefined) return '';
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || payload.length === 0) return null;

    const data = payload[0].payload;
    const isHist = data.isHistorical;

    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl">
        <p className="text-gray-300 font-semibold mb-2">
          {data.day === 0 ? 'Today' : data.day < 0 ? `${Math.abs(data.day)} days ago` : `+${data.day} days`}
        </p>
        {isHist && data.historical && (
          <p className="text-blue-400">Historical: {formatPrice(data.historical)}</p>
        )}
        {!isHist && data.forecast_median && (
          <>
            <p className="text-green-400 font-bold">Forecast: {formatPrice(data.forecast_median)}</p>
            {data.forecast_q25 && data.forecast_q75 && (
              <p className="text-yellow-400 text-sm">
                Range: {formatPrice(data.forecast_q25)} - {formatPrice(data.forecast_q75)}
              </p>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-100">Price Forecast Visualization</h3>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-400">Historical:</label>
            <select
              value={historicalDays}
              onChange={(e) => setHistoricalDays(Number(e.target.value))}
              className="px-2 py-1 text-sm bg-gray-700 text-gray-300 rounded border border-gray-600 focus:outline-none focus:border-brand-500"
              disabled={loadingHistory}
            >
              <option value={60}>60 days</option>
              <option value={120}>120 days</option>
              <option value={365}>1 year</option>
              <option value={730}>2 years</option>
            </select>
            {loadingHistory && (
              <span className="text-xs text-gray-400 animate-pulse">Loading...</span>
            )}
          </div>
          <button
            onClick={() => setShowIndividualModels(!showIndividualModels)}
            className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
          >
            {showIndividualModels ? 'Hide' : 'Show'} Individual Models
          </button>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="day"
            stroke="#9CA3AF"
            label={{ value: 'Days', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
          />
          <YAxis
            stroke="#9CA3AF"
            label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Current price reference line */}
          <ReferenceLine x={0} stroke="#6B7280" strokeDasharray="3 3" label="Today" />

          {/* Historical prices */}
          <Line
            type="monotone"
            dataKey="historical"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
            name="Historical"
            connectNulls
          />

          {/* Forecast median */}
          <Line
            type="monotone"
            dataKey="forecast_median"
            stroke="#10B981"
            strokeWidth={3}
            dot={{ fill: '#10B981', r: 4 }}
            name="Forecast (Median)"
            connectNulls
          />

          {/* Confidence band (25th-75th percentile) */}
          <Area
            type="monotone"
            dataKey="forecast_q75"
            stroke="none"
            fill="#F59E0B"
            fillOpacity={0.2}
            name="75th Percentile"
          />
          <Area
            type="monotone"
            dataKey="forecast_q25"
            stroke="none"
            fill="#F59E0B"
            fillOpacity={0.2}
            name="25th Percentile"
          />

          {/* Individual models (if enabled) */}
          {showIndividualModels && forecastData.individual_models.map((model, idx) => (
            <Line
              key={model.name}
              type="monotone"
              dataKey={`model_${model.name}`}
              stroke={`hsl(${(idx * 360) / forecastData.individual_models.length}, 70%, 60%)`}
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              name={model.name}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Model Summary */}
      {showIndividualModels && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-gray-700">
          {forecastData.individual_models.map((model, idx) => (
            <div key={model.name} className="p-2 bg-gray-800 rounded border border-gray-700">
              <div className="flex items-center space-x-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{
                    backgroundColor: `hsl(${(idx * 360) / forecastData.individual_models.length}, 70%, 60%)`
                  }}
                ></div>
                <div>
                  <p className="text-xs font-medium text-gray-300">{model.name}</p>
                  <p className={`text-xs font-semibold ${
                    model.final_change_pct > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {model.final_change_pct > 0 ? '+' : ''}{model.final_change_pct.toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ForecastChart;
