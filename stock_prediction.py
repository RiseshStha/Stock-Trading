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








# Model
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
    def __init__(self, threshold=0.02):
        self.threshold = threshold
        
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
    
    # Create results dictionary
    results = {
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'history': history.history
    }
    
    return results

# Visualization function
def plot_results(results):
    import matplotlib.pyplot as plt
    
    # Plot predictions vs actual prices
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['actual_prices'], label='Actual Prices')
    plt.plot(results['predicted_prices'], label='Predicted Prices')
    plt.title('NABIL Stock Price Prediction')
    plt.legend()
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    plt.plot(results['cumulative_returns'], label='Cumulative Returns')
    plt.title('Strategy Cumulative Returns')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# To use the model:
results = main(nabil)
plot_results(results)
print(f"Final cumulative return: {results['cumulative_returns'][-1]:.2%}")