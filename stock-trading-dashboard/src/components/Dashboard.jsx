import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign, Award, Clock } from 'lucide-react';

const Dashboard = ({ modelResults }) => {
  if (!modelResults) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <p className="text-gray-500">Loading trading data...</p>
      </div>
    );
  }

  const StatCard = ({ title, value, icon: Icon, trend, color }) => (
    <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-gray-500 text-sm mb-1">{title}</p>
          <h3 className="text-2xl font-bold text-gray-900">{value}</h3>
          {trend && (
            <p className={`text-sm mt-2 ${trend >= 0 ? 'text-green-500' : 'text-red-500'} flex items-center`}>
              {trend >= 0 ? <TrendingUp size={16} className="mr-1" /> : <TrendingDown size={16} className="mr-1" />}
              {Math.abs(trend)}%
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon size={20} className="text-white" />
        </div>
      </div>
    </div>
  );

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">NABIL Trading Dashboard</h1>
            <p className="text-gray-500">AI-Powered Stock Analysis & Predictions</p>
          </div>
          <div className="flex items-center space-x-2">
            <Clock className="text-gray-400" size={16} />
            <span className="text-sm text-gray-500">
              Last updated: {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Current Price"
            value={`₹${modelResults.latest_price?.toFixed(2)}`}
            icon={DollarSign}
            color="bg-blue-500"
          />
          <StatCard
            title="Total Return"
            value={`${modelResults.total_return?.toFixed(2)}%`}
            icon={Activity}
            trend={modelResults.total_return}
            color="bg-green-500"
          />
          <StatCard
            title="Win Rate"
            value={`${modelResults.win_rate?.toFixed(1)}%`}
            icon={Award}
            color="bg-purple-500"
          />
          <StatCard
            title="Total Trades"
            value={modelResults.total_trades}
            icon={TrendingUp}
            color="bg-orange-500"
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Price Chart */}
          <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-lg font-semibold mb-4">Price Prediction</h2>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={modelResults.chart_data}>
                <defs>
                  <linearGradient id="actualGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#82ca9d" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                  style={{ fontSize: '12px' }}
                />
                <YAxis style={{ fontSize: '12px' }} />
                <Tooltip 
                  labelFormatter={formatDate}
                  contentStyle={{ 
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    border: 'none',
                    borderRadius: '8px',
                    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#8884d8" 
                  fillOpacity={1}
                  fill="url(#actualGradient)"
                  name="Actual Price"
                />
                <Area 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#82ca9d" 
                  fillOpacity={1}
                  fill="url(#predictedGradient)"
                  name="Predicted Price"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Trading Signals */}
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-lg font-semibold mb-4">Recent Trading Signals</h2>
            <div className="space-y-4">
              {modelResults.recent_signals?.map((signal, index) => (
                <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-gray-50">
                  <div>
                    <p className="text-sm text-gray-500">{formatDate(signal.date)}</p>
                    <p className="font-medium">₹{signal.price.toFixed(2)}</p>
                  </div>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    signal.action === 'BUY' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {signal.action}
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">Confidence</p>
                    <p className="font-medium">{signal.confidence.toFixed(1)}%</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;