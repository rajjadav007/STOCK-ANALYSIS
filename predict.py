#!/usr/bin/env python3
"""
Stock Price Predictor
Make predictions using the trained ML model
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def load_trained_model():
    """Load the trained model"""
    model_path = 'models/best_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        return model
    else:
        print("❌ No trained model found. Run main.py first to train a model.")
        return None

def predict_stock_price(open_price, high, low, volume, prev_close=None):
    """Make a stock price prediction"""
    model = load_trained_model()
    if model is None:
        return None
    
    # Set default previous close if not provided
    if prev_close is None:
        prev_close = open_price
    
    # Create feature data
    prediction_data = {
        'Open': open_price,
        'High': high,
        'Low': low,
        'Volume': volume,
        'Price_Change': high - open_price,  # Approximate
        'Price_Range': high - low,
        'Returns': (open_price - prev_close) / prev_close if prev_close > 0 else 0,
        'SMA_5': prev_close,  # Approximate with previous close
        'SMA_10': prev_close,  # Approximate with previous close
        'Volatility': 0.02,  # Default 2% volatility
        'Close_lag_1': prev_close,
        'Volume_lag_1': volume,
        'Year': 2024,
        'Month': 12,
        'DayOfWeek': 2,  # Tuesday
        'Quarter': 4
    }
    
    # Convert to DataFrame
    X = pd.DataFrame([prediction_data])
    
    # Make prediction
    predicted_price = model.predict(X)[0]
    
    return predicted_price

def demo_prediction():
    """Run a demonstration prediction"""
    print("🔮 STOCK PRICE PREDICTION DEMO")
    print("=" * 40)
    
    # Sample stock data
    open_price = 150.00
    high_price = 155.50
    low_price = 148.75
    volume = 1250000
    prev_close = 149.50
    
    print(f"📊 Input Data:")
    print(f"   Open: ${open_price:.2f}")
    print(f"   High: ${high_price:.2f}")
    print(f"   Low: ${low_price:.2f}")
    print(f"   Volume: {volume:,}")
    print(f"   Previous Close: ${prev_close:.2f}")
    
    # Make prediction
    predicted_price = predict_stock_price(open_price, high_price, low_price, volume, prev_close)
    
    if predicted_price is not None:
        expected_return = ((predicted_price - open_price) / open_price) * 100
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Predicted Close: ${predicted_price:.2f}")
        print(f"   Expected Return: {expected_return:+.2f}%")
        
        # Show model performance if available
        if os.path.exists('results/model_performance.csv'):
            results = pd.read_csv('results/model_performance.csv')
            best_model = results.loc[results['R²'].idxmax()]
            print(f"\n📈 Model Performance:")
            print(f"   Best Model: {best_model['Model']}")
            print(f"   R² Score: {best_model['R²']:.4f}")
            print(f"   RMSE: {best_model['RMSE']:.2f}")

if __name__ == "__main__":
    demo_prediction()