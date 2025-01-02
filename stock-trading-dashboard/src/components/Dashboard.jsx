import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ImprovedDashboard = ({ modelResults }) => {
  if (!modelResults) return null;

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <h1 className="text-2xl font-bold">DASHBOARD</h1>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-2">
            <p className="text-gray-400">Current Price</p>
            <p className="text-4xl">₹{modelResults.latest_price?.toFixed(2) || 'N/A'}</p>
          </div>
          <div className="space-y-2">
            <p className="text-gray-400">Total Return</p>
            <p className="text-4xl">{modelResults.total_return?.toFixed(2) || 0}%</p>
            <p className="text-green-500">↑ {Math.abs(modelResults.total_return || 0).toFixed(2)}%</p>
          </div>
          <div className="space-y-2">
            <p className="text-gray-400">Win Rate</p>
            <p className="text-4xl">{modelResults.win_rate?.toFixed(2) || 0}%</p>
          </div>
        </div>

        {/* Price Chart */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Price Prediction</h2>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={modelResults.chart_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis 
                  dataKey="date" 
                  stroke="#888"
                  tickFormatter={(date) => {
                    const d = new Date(date);
                    return `${d.toLocaleString('default', { month: 'short' })} ${d.getDate()}, ${d.getFullYear().toString().substr(-2)}`;
                  }}
                />
                <YAxis stroke="#888" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#333', border: 'none' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  name="Actual Price" 
                  stroke="#fff" 
                  dot={{ fill: '#fff' }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  name="Predicted Price" 
                  stroke="#ff0000" 
                  dot={{ fill: '#ff0000' }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImprovedDashboard;