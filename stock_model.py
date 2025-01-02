import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import requests


# Base URL for the API
base_url = "https://the-value-crew.github.io/nepse-api"

# Fetch the list of companies
companies_url = f"{base_url}/data/companies.json"
response = requests.get(companies_url)

if response.status_code == 200:
    # Parse the companies list
    companies = response.json()

    # Check if 'NABIL' exists
    if "NABIL" in companies:
        # Fetch data for the NABIL company
        company_data_url = f"{base_url}/data/company/NABIL.json"
        company_response = requests.get(company_data_url)

        if company_response.status_code == 200:
            nabil_data = company_response.json()

            # Prepare data for the table
            rows = []
            for date, data in nabil_data.items():
                price_data = data.get("price", {})
                row = {
                    "Date": date,
                    "Max Price": price_data.get("max", "N/A"),
                    "Min Price": price_data.get("min", "N/A"),
                    "Close Price": price_data.get("close", "N/A"),
                    "Previous Close": price_data.get("prevClose", "N/A"),
                    "Price Difference": price_data.get("diff", "N/A"),
                    "Number of Transactions": data.get("numTrans", "N/A"),
                    "Traded Shares": data.get("tradedShares", "N/A"),
                    "Traded Amount": data.get("amount", "N/A"),
                }
                rows.append(row)

            # Create a DataFrame
            df = pd.DataFrame(rows)

            # Convert "Date" column to datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Sort by "Date" column in descending order
            df = df.sort_values(by='Date', ascending=False)

            # Display the sorted table
            print("\nNABIL Stock Data (Latest Date on Top):\n")
            print(df.to_string(index=False))
        else:
            print(f"Failed to fetch data for NABIL. Status Code: {company_response.status_code}")
    else:
        print("NABIL is not found in the companies list.")
else:
    print(f"Failed to fetch companies list. Status Code: {response.status_code}")



df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# # Extract the day name and month name
df['Day'] = df['Date'].dt.day_name()    # Day of the week, e.g., 'Sunday'
df['Month'] = df['Date'].dt.month_name()

# sorting data according to date
df = df.sort_values(by='Date', ascending=False)
df.head()


columns = ['Date','Close Price','Previous Close','Traded Shares','Day','Month']
nabil = pd.DataFrame(df, columns=columns)
nabil.set_index('Date', inplace=True)
nabil


#model

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class StockPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_data(self, df):
        # Create features
        df['Returns'] = df['Close Price'].pct_change()
        df['Price_Momentum'] = df['Close Price'].diff(periods=5)
        df['Volume_Momentum'] = df['Traded Shares'].diff(periods=5)
        df['Price_Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        # Create features and target arrays
        features = ['Close Price', 'Returns', 'Price_Momentum', 
                   'Volume_Momentum', 'Price_Volatility', 'Traded Shares']
        
        # Scale features and target separately
        X_scaled = self.scaler_x.fit_transform(df[features])
        y_scaled = self.scaler_y.fit_transform(df[['Close Price']])
        
        X, y = [], []
        for i in range(self.lookback, len(X_scaled)):
            X.append(X_scaled[i-self.lookback:i])
            y.append(y_scaled[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        if self.model is None:
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X):
        scaled_predictions = self.model.predict(X)
        return self.scaler_y.inverse_transform(scaled_predictions.reshape(-1, 1))[:, 0]
    
class TradingStrategy:
    def __init__(self, threshold=0.02, confidence_threshold=0.9):
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        
    def generate_signals(self, actual_prices, predicted_prices):
        signals = [0]  # Initialize with no position
        position = 0  # 0: no position, 1: long, -1: short
        
        # Ensure both arrays have the same length
        min_len = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:min_len]
        predicted_prices = predicted_prices[:min_len]
        
        for i in range(1, len(actual_prices)):
            predicted_return = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            if position == 0:  # No position
                if predicted_return > self.threshold:
                    signals.append(1)  # Buy signal
                    position = 1
                elif predicted_return < -self.threshold:
                    signals.append(-1)  # Sell signal
                    position = -1
                else:
                    signals.append(0)  # Hold
            else:
                if (position == 1 and predicted_return < 0) or \
                   (position == -1 and predicted_return > 0):
                    signals.append(0)  # Close position
                    position = 0
                else:
                    signals.append(position)  # Maintain position
                    
        return np.array(signals)
    
    def calculate_returns(self, prices, signals):
        # Calculate returns
        daily_returns = np.diff(prices) / prices[:-1]
        
        # Ensure signals array matches daily_returns length
        signals = signals[:len(daily_returns)]
        
        # Calculate strategy returns and cumulative returns
        strategy_returns = daily_returns * signals
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        return strategy_returns, cumulative_returns

def main(nabil_df):
    # Convert string prices to float
    nabil_df['Close Price'] = pd.to_numeric(nabil_df['Close Price'], errors='coerce')
    nabil_df['Traded Shares'] = pd.to_numeric(nabil_df['Traded Shares'], errors='coerce')
    
    # Initialize predictor and prepare data
    predictor = StockPredictor(lookback=60)
    X, y = predictor.prepare_data(nabil_df)
    
    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train the model
    history = predictor.train(X_train, y_train, epochs=50)
    
    # Make predictions
    predicted_prices = predictor.predict(X_test)
    actual_prices = predictor.scaler_y.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    
    # Generate trading signals
    strategy = TradingStrategy(threshold=0.02)
    signals = strategy.generate_signals(actual_prices, predicted_prices)
    
    # Calculate returns
    strategy_returns, cumulative_returns = strategy.calculate_returns(actual_prices, signals)
    
    # Print trading strategy analysis
    print("\n=== NABIL Stock Trading Strategy Analysis ===")
    print("\nHistorical Trading Signals:")
    for i in range(len(actual_prices)):
        price = actual_prices[i]
        pred_price = predicted_prices[i]
        signal = signals[i]
        confidence = 100 - (abs(pred_price - price) / price * 100)
        
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        if signal != 0:  # Only print when there's a buy or sell signal
            print(f"Day {i+1}:")
            print(f"  Actual Price: {price:.2f}")
            print(f"  Predicted Next Price: {pred_price:.2f}")
            print(f"  Signal: {signal_text}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Expected Return: {((pred_price - price) / price * 100):.2f}%\n")
    
    # Calculate strategy metrics
    total_trades = np.sum(np.abs(np.diff(signals)) != 0)
    winning_trades = np.sum(strategy_returns > 0)
    losing_trades = np.sum(strategy_returns < 0)
    
    print("\n=== Strategy Performance Metrics ===")
    print(f"Total Number of Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {(winning_trades/total_trades*100):.2f}% if trades > 0 else 0")
    print(f"Final Cumulative Return: {cumulative_returns[-1]*100:.2f}%")
    
    # Latest prediction
    print("\n=== Latest Trading Signal ===")
    latest_actual = actual_prices[-1]
    latest_pred = predicted_prices[-1]
    latest_signal = signals[-1]
    latest_confidence = 100 - (abs(latest_pred - latest_actual) / latest_actual * 100)
    
    print(f"Current Price: {latest_actual:.2f}")
    print(f"Predicted Next Price: {latest_pred:.2f}")
    print(f"Signal: {'BUY' if latest_signal == 1 else 'SELL' if latest_signal == -1 else 'HOLD'}")
    print(f"Confidence: {latest_confidence:.2f}%")
    print(f"Expected Return: {((latest_pred - latest_actual) / latest_actual * 100):.2f}%")
    
    # Create results dictionary
    results = {
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'history': history.history,
        'metrics': {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades/total_trades*100) if total_trades > 0 else 0,
            'final_return': cumulative_returns[-1]*100
        }
    }
    
    return results

# Visualization function
def plot_results(results):
    import matplotlib.pyplot as plt
    
    # Create figure with 3 subplots
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Predictions vs Actual with Buy/Sell signals
    plt.subplot(3, 1, 1)
    plt.plot(results['actual_prices'], label='Actual Prices', alpha=0.7)
    plt.plot(results['predicted_prices'], label='Predicted Prices', alpha=0.7)
    
    # Add buy/sell signals
    for i in range(len(results['signals'])):
        if results['signals'][i] == 1:  # Buy signal
            plt.scatter(i, results['actual_prices'][i], color='green', marker='^', s=100, label='Buy' if i == 0 else "")
        elif results['signals'][i] == -1:  # Sell signal
            plt.scatter(i, results['actual_prices'][i], color='red', marker='v', s=100, label='Sell' if i == 0 else "")
    
    plt.title('NABIL Stock Price Prediction with Trading Signals')
    plt.legend()
    
    # Plot 2: Trading Confidence
    plt.subplot(3, 1, 2)
    confidence = np.abs((results['predicted_prices'] - results['actual_prices']) / results['actual_prices'] * 100)
    plt.plot(100 - confidence, label='Prediction Accuracy (%)', color='purple')
    plt.axhline(y=90, color='g', linestyle='--', label='90% Accuracy Threshold')
    plt.title('Prediction Accuracy')
    plt.legend()
    plt.ylim(0, 100)
    
    # Plot 3: Cumulative Returns with Trade Points
    plt.subplot(3, 1, 3)
    plt.plot(results['cumulative_returns'] * 100, label='Cumulative Returns (%)', color='blue')
    
    # Mark profitable and unprofitable trades
    trade_returns = results['strategy_returns'] * 100
    plt.scatter(np.where(trade_returns > 0)[0], 
               results['cumulative_returns'][trade_returns > 0] * 100, 
               color='green', marker='o', label='Profitable Trade')
    plt.scatter(np.where(trade_returns < 0)[0], 
               results['cumulative_returns'][trade_returns < 0] * 100, 
               color='red', marker='o', label='Unprofitable Trade')
    
    plt.title('Strategy Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    total_trades = np.sum(np.abs(np.diff(results['signals']) != 0))
    profitable_trades = np.sum(trade_returns > 0)
    success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    print("\nTrading Performance Metrics:")
    print(f"Total number of trades: {total_trades}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Final cumulative return: {results['cumulative_returns'][-1]*100:.2f}%")
    print(f"Average prediction accuracy: {np.mean(100-confidence):.2f}%")
    
    # Print latest trading signal
    latest_signal = results['signals'][-1]
    signal_text = "BUY" if latest_signal == 1 else "SELL" if latest_signal == -1 else "HOLD"
    latest_confidence = 100 - confidence[-1]
    
    print(f"\nLatest Trading Signal: {signal_text}")
    print(f"Signal confidence: {latest_confidence:.2f}%")

# To use the model:
results = main(nabil)
plot_results(results)