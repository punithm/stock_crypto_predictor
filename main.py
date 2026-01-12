#!/usr/bin/env python3
"""
Stock Price Predictor - Main Script

This script demonstrates how to use the stock prediction system with different models.
"""

import sys
import argparse
import numpy as np
from stock_predictor.data_fetcher import StockDataFetcher
from stock_predictor.model import LSTMModel, RandomForestModel, SimpleLinearModel
from stock_predictor.evaluator import ModelEvaluator
from stock_predictor.visualizer import StockVisualizer


def split_data(X, y, train_split: float = 0.8):
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Labels
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_split)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def run_prediction(ticker: str, model_type: str = "lstm", visualize: bool = True):
    """
    Run stock price prediction for a given ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        model_type: Type of model ('lstm', 'rf', 'linear')
        visualize: Whether to display plots
    """
    print(f"\n{'='*50}")
    print(f"Stock Price Predictor - {ticker}")
    print(f"{'='*50}\n")

    # Step 1: Fetch and prepare data
    print(f"[1/5] Fetching data for {ticker}...")
    fetcher = StockDataFetcher(ticker, period="5y")
    fetcher.fetch_data()
    X, y = fetcher.prepare_features(lookback_days=60)
    
    # Step 2: Split data
    print("\n[2/5] Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_split=0.8)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 3: Create and train model
    print(f"\n[3/5] Creating and training {model_type.upper()} model...")
    if model_type.lower() == "lstm":
        model = LSTMModel(lookback_days=60, epochs=50, batch_size=32)
    elif model_type.lower() == "rf":
        model = RandomForestModel(n_estimators=100, max_depth=20)
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
    
    current_price = fetcher.get_current_price()
    latest_sequence = fetcher.get_latest_sequence(lookback_days=60)
    future_pred = model.predict(latest_sequence)
    
    print(f"\nCurrent Price: €{current_price:.2f}")
    print(f"Predicted Next Price: €{future_pred[0][0]:.2f}")
    
    price_change = future_pred[0][0] - current_price
    percent_change = (price_change / current_price) * 100
    direction = "📈 UP" if price_change > 0 else "📉 DOWN"
    
    print(f"Expected Change: {direction} {abs(price_change):.2f} ({abs(percent_change):.2f}%)")
    
    # Visualization
    if visualize:
        print("\nGenerating visualizations...")
        StockVisualizer.plot_price_history(fetcher.data, ticker)
        StockVisualizer.plot_predictions(y_test, y_pred, 
                                        title=f"{ticker} - {model_type.upper()} Model Predictions")
    
    return model, fetcher, metrics


def compare_models(ticker: str = "AAPL"):
    """
    Compare performance of all three models.
    
    Args:
        ticker: Stock symbol
    """
    print(f"\n{'='*50}")
    print(f"Model Comparison - {ticker}")
    print(f"{'='*50}\n")
    
    # Fetch data
    print("Fetching data...")
    fetcher = StockDataFetcher(ticker, period="5y")
    fetcher.fetch_data()
    X, y = fetcher.prepare_features(lookback_days=60)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = {
        "LSTM": LSTMModel(lookback_days=60, epochs=50, batch_size=32),
        "Random Forest": RandomForestModel(),
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
        print(f"{name:15s}: €{rmse:.2f}")
    
    # Visualize comparison
    print("\nGenerating comparison visualization...")
    pred_list = [predictions[name] for name in models.keys()]
    StockVisualizer.plot_comparison(y_test, *pred_list, titles=list(models.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Price Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict AAPL stock price using LSTM
  python main.py AAPL --model lstm

  # Predict GOOGL stock price using Random Forest
  python main.py GOOGL --model rf

  # Compare all models for MSFT
  python main.py MSFT --compare

  # Predict with no visualization
  python main.py AAPL --no-viz
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
            run_prediction(args.ticker, args.model, visualize=not args.no_viz)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
