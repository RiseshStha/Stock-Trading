from datetime import datetime, timedelta
import numpy as np

def process_model_results(results):
    """Process model results into a format suitable for the dashboard"""
    if not results or not isinstance(results, dict):
        raise ValueError("Invalid results format")
    
    # Create chart data
    chart_data = []
    base_date = datetime.now()
    
    for i in range(len(results['actual_prices'])):
        current_date = (base_date - timedelta(days=len(results['actual_prices'])-i-1)).isoformat()
        chart_data.append({
            'date': current_date,
            'actual': float(results['actual_prices'][i]),
            'predicted': float(results['predicted_prices'][i])
        })
    
    # Create trading signals
    recent_signals = []
    for i in range(len(results['signals'])):
        if results['signals'][i] != 0:  # Only include buy/sell signals
            signal_date = (base_date - timedelta(days=len(results['signals'])-i-1)).isoformat()
            signal = {
                'date': signal_date,
                'price': float(results['actual_prices'][i]),
                'action': 'BUY' if results['signals'][i] == 1 else 'SELL',
                'confidence': float(100 - (abs(results['predicted_prices'][i] - results['actual_prices'][i]) / 
                                   results['actual_prices'][i] * 100))
            }
            recent_signals.append(signal)
    
    # Get latest 10 signals
    recent_signals = recent_signals[-10:]
    
    # Calculate metrics
    dashboard_data = {
        'latest_price': float(results['actual_prices'][-1]),
        'total_return': float(results['cumulative_returns'][-1] * 100),
        'win_rate': float(results['metrics']['win_rate']),
        'total_trades': int(results['metrics']['total_trades']),
        'chart_data': chart_data,
        'recent_signals': recent_signals
    }
    
    return dashboard_data