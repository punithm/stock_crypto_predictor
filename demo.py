#!/usr/bin/env python3
"""
Stock Price Predictor - Demo with Synthetic Data

This script demonstrates the stock prediction system using synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stock_crypto_predictor.model import LSTMModel, RandomForestModel, SimpleLinearModel
from stock_crypto_predictor.evaluator import ModelEvaluator
from stock_crypto_predictor.visualizer import StockVisualizer


def generate_synthetic_stock_data(ticker: str, num_days: int = 1260, start_price: float = 150):
    """
    Generate synthetic stock price data.
    
    Args:
        ticker: Stock symbol
        num_days: Number of days to generate
        start_price: Starting price
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=num_days, freq='D')
    
    # Generate realistic price movement with trend and volatility
    returns = np.random.normal(0.001, 0.02, num_days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    volume = np.random.uniform(50000000, 100000000, num_days)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, num_days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, num_days)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, num_days)),
        'Close': prices,
        'Volume': volume
    }, index=dates)
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data


def prepare_features(data: pd.DataFrame, lookback_days: int = 60):
    """Prepare features and labels for model training."""
    prices = data['Close'].values.reshape(-1, 1)
    
    X, y = [], []
    for i in range(len(prices) - lookback_days):
        X.append(prices[i:(i + lookback_days)])
        y.append(prices[i + lookback_days])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Prepared {len(X)} training samples with lookback={lookback_days} days")
    return X, y


def split_data(X, y, train_split: float = 0.8):
    """Split data into training and testing sets."""
    split_idx = int(len(X) * train_split)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def run_prediction(ticker: str, model_type: str = "lstm"):
    """Run stock price prediction for a given ticker using synthetic data."""
    print(f"\n{'='*50}")
    print(f"Stock Price Predictor - {ticker} (DEMO)")
    print(f"{'='*50}\n")

    # Step 1: Generate synthetic data
    print(f"[1/5] Generating synthetic data for {ticker}...")
    data = generate_synthetic_stock_data(ticker)
    print(f"Generated {len(data)} days of data")
    print(f"Price range: €{data['Close'].min():.2f} - €{data['Close'].max():.2f}")
    
    X, y = prepare_features(data, lookback_days=60)
    
    # Step 2: Split data
    print("\n[2/5] Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_split=0.8)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 3: Create and train model
    print(f"\n[3/5] Creating and training {model_type.upper()} model...")
    if model_type.lower() == "lstm":
        model = LSTMModel(lookback_days=60, epochs=20, batch_size=32)
    elif model_type.lower() == "rf":
        model = RandomForestModel(n_estimators=50, max_depth=20)
    elif model_type.lower() == "linear":
        model = SimpleLinearModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.train(X_train, y_train)
    
    # Step 4: Make predictions
    print("\n[4/5] Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating model performance...")
    metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
    ModelEvaluator.print_metrics(metrics)
    
    # Calculate directional accuracy
    dir_accuracy = ModelEvaluator.directional_accuracy(y_test, y_pred)
    print(f"Directional Accuracy: {dir_accuracy:.2f}%")
    
    # Make a future prediction
    print("\n" + "="*50)
    print("Making Future Prediction")
    print("="*50)
    
    current_price = data['Close'].iloc[-1]
    latest_prices = data['Close'].values[-60:]
    latest_sequence = latest_prices.reshape(1, 60, 1)
    future_pred = model.predict(latest_sequence)
    
    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"Predicted Next Price: ${future_pred[0][0]:.2f}")
    
    price_change = future_pred[0][0] - current_price
    percent_change = (price_change / current_price) * 100
    direction = "📈 UP" if price_change > 0 else "📉 DOWN"
    
    print(f"Expected Change: {direction} ${abs(price_change):.2f} ({abs(percent_change):.2f}%)")
    
    # Visualization
    print("\nGenerating visualizations...")
    StockVisualizer.plot_price_history(data, ticker)
    StockVisualizer.plot_predictions(y_test, y_pred, 
                                    title=f"{ticker} - {model_type.upper()} Model Predictions")
    
    return model, data, metrics


def compare_models(ticker: str = "AAPL"):
    """Compare performance of all three models."""
    print(f"\n{'='*50}")
    print(f"Model Comparison - {ticker} (DEMO)")
    print(f"{'='*50}\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_stock_data(ticker)
    X, y = prepare_features(data, lookback_days=60)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = {
        "LSTM": LSTMModel(lookback_days=60, epochs=20, batch_size=32),
        "Random Forest": RandomForestModel(n_estimators=50),
        "Linear": SimpleLinearModel()
    }
    
    predictions = {}
    metrics_all = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
        metrics_all[name] = metrics
        
        print(f"\n{name} Results:")
        ModelEvaluator.print_metrics(metrics)
    
    # Comparison summary
    print("\n" + "="*50)
    print("RMSE Comparison (Lower is Better)")
    print("="*50)
    for name in sorted(metrics_all.keys(), key=lambda x: metrics_all[x]['RMSE']):
        rmse = metrics_all[name]['RMSE']
        print(f"{name:15s}: ${rmse:.2f}")
    
    # Visualize comparison
    print("\nGenerating comparison visualization...")
    pred_list = [predictions[name] for name in models.keys()]
    StockVisualizer.plot_comparison(y_test, *pred_list, titles=list(models.keys()))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stock Price Predictor (Demo with Synthetic Data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict AAPL stock price using LSTM
  python demo.py AAPL --model lstm

  # Predict GOOGL stock price using Random Forest
  python demo.py GOOGL --model rf

  # Compare all models for MSFT
  python demo.py MSFT --compare

  # Predict with no visualization
  python demo.py AAPL --no-viz
        """
    )
    
    parser.add_argument("ticker", nargs="?", default="AAPL",
                       help="Stock symbol (default: AAPL)")
    parser.add_argument("--model", "-m", choices=["lstm", "rf", "linear"],
                       default="lstm", help="Model type to use (default: lstm)")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Compare all models")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization")
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_models(args.ticker)
        else:
            run_prediction(args.ticker, args.model)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
