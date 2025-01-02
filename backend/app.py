from flask import Flask, jsonify
from flask_cors import CORS
from models.stock_model import StockPredictor, TradingStrategy
from utils.modelProcessor import process_model_results
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
import threading
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global cache for model results
cache = {
    'last_update': None,
    'model_results': None,
    'update_interval': timedelta(hours=1)  # Update every hour
}

def should_update_cache():
    return (
        cache['last_update'] is None or
        cache['model_results'] is None or
        datetime.now() - cache['last_update'] > cache['update_interval']
    )

def update_model_results():
    try:
        logger.info("Updating model results...")
        df = fetch_stock_data()
        if df is not None:
            predictor = StockPredictor(lookback=30)
            results = predictor.train_and_predict(df)
            processed_results = process_model_results(results)
            
            cache['model_results'] = processed_results
            cache['last_update'] = datetime.now()
            
            logger.info("Model results updated successfully")
            return processed_results
    except Exception as e:
        logger.error(f"Error updating model results: {str(e)}")
    return None

def background_update():
    while True:
        if should_update_cache():
            update_model_results()
        time.sleep(300)  # Check every 5 minutes

# Start background update thread
update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()

def fetch_stock_data():
    try:
        logger.debug("Fetching stock data...")
        base_url = "https://the-value-crew.github.io/nepse-api"
        companies_url = f"{base_url}/data/companies.json"
        response = requests.get(companies_url)
        
        if response.status_code == 200:
            companies = response.json()
            if "NABIL" in companies:
                company_data_url = f"{base_url}/data/company/NABIL.json"
                company_response = requests.get(company_data_url)
                
                if company_response.status_code == 200:
                    nabil_data = company_response.json()
                    rows = []
                    for date, data in nabil_data.items():
                        price_data = data.get("price", {})
                        row = {
                            "Date": date,
                            "Close Price": price_data.get("close", "N/A"),
                            "Previous Close": price_data.get("prevClose", "N/A"),
                            "Traded Shares": data.get("tradedShares", "N/A"),
                        }
                        rows.append(row)
                    
                    df = pd.DataFrame(rows)
                    df['Close Price'] = pd.to_numeric(df['Close Price'], errors='coerce')
                    df['Traded Shares'] = pd.to_numeric(df['Traded Shares'], errors='coerce')
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values(by='Date', ascending=False)
                    
                    logger.debug(f"Successfully fetched data. Shape: {df.shape}")
                    return df
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
    return None

@app.route('/api/model-results', methods=['GET'])
def get_model_results():
    try:
        if should_update_cache():
            logger.info("Cache expired, updating model results...")
            update_model_results()
        
        if cache['model_results'] is None:
            return jsonify({"error": "No model results available"}), 500
            
        return jsonify(cache['model_results'])
    
    except Exception as e:
        logger.error(f"Error in get_model_results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/force-update', methods=['POST'])
def force_update():
    """Endpoint to force model update"""
    try:
        results = update_model_results()
        if results:
            return jsonify({"message": "Model updated successfully"})
        return jsonify({"error": "Failed to update model"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache-status', methods=['GET'])
def get_cache_status():
    """Get the current status of the cache"""
    return jsonify({
        "last_update": cache['last_update'].isoformat() if cache['last_update'] else None,
        "has_data": cache['model_results'] is not None,
        "next_update": (cache['last_update'] + cache['update_interval']).isoformat() if cache['last_update'] else None
    })

if __name__ == '__main__':
    # Initial model update
    update_model_results()
    app.run(debug=True, host='0.0.0.0', port=5000)