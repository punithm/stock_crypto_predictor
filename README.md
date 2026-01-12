# 📈 Stock & Cryptocurrency Price Predictor with Web UI

A comprehensive, production-ready ML system for predicting **stock and cryptocurrency prices** with an interactive web interface and multiple machine learning models.

## Overview

This project implements a complete stock and cryptocurrency price prediction system with:
- **Interactive Web UI** built with Streamlit
- **Support for 8 stocks + 8 cryptocurrencies**
- **Multiple data sources** (Yahoo Finance, Alpha Vantage, CoinGecko)
- **Three different ML models** (LSTM, Random Forest, Linear Regression)
- **Real-time training visualization**
- **Comprehensive performance metrics**
- **EUR currency support** (especially for crypto)
- **Both synthetic and real data** support

## Features

🌐 **Interactive Web UI**
- Beautiful Streamlit interface
- Data source selector (Yahoo Finance, Alpha Vantage, CoinGecko)
- Real-time model training visualization
- Interactive Plotly charts
- 4 main tabs (Price History, Training, Predictions, Comparison)

📊 **Multiple Assets & Data Sources**
- **Stocks**: AAPL, GOOGL, MSFT, AMZN, META, TSLA, NFLX, NVDA
  - Yahoo Finance (default)
  - Alpha Vantage ⭐ (more reliable, free key available)
- **Cryptocurrencies**: BTC, ETH, ADA, SOL, DOT, XRP, LTC, DOGE
  - Yahoo Finance (default)
  - CoinGecko ⭐ (free, no API key needed, EUR support)

🤖 **Three ML Models**
- LSTM Neural Network (Best accuracy)
- Random Forest (Good balance)
- Linear Regression (Fast baseline)

📈 **Comprehensive Metrics**
- RMSE, MAE, MSE, R² Score
- MAPE (percentage error)
- Directional Accuracy (up/down prediction)
- Model comparison and ensemble support

💱 **Currency Support**
- EUR (€) display for prices
- CoinGecko supports 150+ currencies natively

💾 **Flexible Data**
- Live data from multiple sources
- Synthetic data for demos (no internet needed)
- Adjustable lookback window (30-120 days)
- Graceful fallback to synthetic data on fetch failure

✨ **Easy to Use**
- Simple web interface
- Data source selection dropdown
- Optional API key configuration
- Well-documented code

## Project Structure

```
stock_predictor/
├── __init__.py                  # Package initialization
├── data_fetcher.py             # Yahoo Finance stock data fetcher
├── crypto_fetcher.py           # Yahoo Finance crypto fetcher
├── alphavantage_fetcher.py     # Alpha Vantage stock data source
├── coingecko_fetcher.py        # CoinGecko crypto data source (free, no key)
├── model.py                    # ML model implementations
├── evaluator.py                # Model performance metrics
├── indicators.py               # Technical indicators (SMA, EMA, RSI, MACD)
├── news_sentiment.py           # News sentiment analysis
└── visualizer.py               # Plotting utilities

app.py                          # Streamlit web UI
main.py                         # CLI with real data
demo.py                         # Demo with synthetic data
requirements.txt                # Package dependencies
```

## Installation

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Setup

1. **Clone/Download the repository**
   ```bash
   cd py_project_1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 🌐 Web UI (Recommended)

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Then open `http://localhost:8503` in your browser.

**Features in Web UI:**
- ✅ Select between **Yahoo Finance**, **Alpha Vantage** (stocks), or **CoinGecko** (crypto)
- ✅ Paste optional API keys (Alpha Vantage free key, NewsAPI for sentiment)
- ✅ Choose synthetic or live data
- ✅ Train models with one click
- ✅ Compare multiple models side-by-side
- ✅ View predictions and performance metrics
- ✅ All prices displayed in EUR (€)

### CLI Scripts (Demo with Synthetic Data)

The demo script uses synthetic data and doesn't require internet connection:

```bash
# Use LSTM (default, best for time-series)
python demo.py AAPL --model lstm

# Use Random Forest
python demo.py GOOGL --model rf

# Use Simple Linear Model
python demo.py MSFT --model linear

# Compare all models
python demo.py AAPL --compare
```

### Real Data (Live Stock Prices)

The `main.py` script fetches real historical stock data:

```bash
python main.py AAPL --model lstm
python main.py GOOGL --compare
```

## Data Sources

### Yahoo Finance (Default)
- **For:** Stocks and cryptocurrencies
- **Pros:** No API key needed
- **Cons:** Occasionally rate limits
- **Best for:** Quick testing

### Alpha Vantage (Better for Stocks)
- **For:** Stocks only
- **Pros:** More reliable, real-time/intraday data
- **Cons:** Requires free API key
- **Setup:** 
  1. Get free key: https://www.alphavantage.co/
  2. In web UI, paste key under "Alpha Vantage API Key"
  3. Select "Alpha Vantage (Stocks)" from data source dropdown
- **Best for:** Production stock predictions

### CoinGecko (Best for Crypto) ⭐
- **For:** Cryptocurrencies only
- **Pros:** FREE, no API key needed, native EUR support, 150+ currencies
- **Cons:** None!
- **Setup:** Just select "CoinGecko (Crypto)" from dropdown
- **Best for:** Cryptocurrency predictions with EUR pricing

## Output Explanation

### Metrics Displayed

**MSE (Mean Squared Error)** - Average of squared errors (lower is better)

**RMSE (Root Mean Squared Error)** - Square root of MSE (same units as price)

**MAE (Mean Absolute Error)** - Average absolute deviation from actual prices

**R² Score** - Proportion of variance explained (1.0 is perfect, <0 is worse than baseline)

**MAPE (Mean Absolute Percentage Error)** - Percentage error relative to actual price

**Directional Accuracy** - % of correct up/down predictions

### Example Output

```
Stock Price Predictor - AAPL (DEMO)
==================================================

[1/5] Generating synthetic data for AAPL...
Generated 1260 days of data
Price range: €124.39 - €1503.52

[2/5] Splitting data into train/test sets (80/20)...
Training samples: 960
Testing samples: 240

[3/5] Creating and training LSTM model...
Model training completed

[4/5] Making predictions on test set...

[5/5] Evaluating model performance...
=== Model Performance Metrics ===
MSE   : 5275.41
RMSE  : €72.63
MAE   : €56.87
R2    : 0.8866
MAPE  : 4.97%
===================================
Directional Accuracy: 51.05%

==================================================
Making Future Prediction
==================================================

Current Price: €1381.92
Predicted Next Price: €1357.60
Expected Change: 📉 DOWN €24.31 (1.76%)
```

## Model Descriptions

### LSTM (Long Short-Term Memory) Neural Network
- **Best for:** Time-series data with long-term dependencies
- **Pros:** Captures temporal patterns, handles sequences well
- **Cons:** Slower training, requires more data
- **When to use:** For more accurate predictions with sufficient historical data

### Random Forest
- **Best for:** Non-linear relationships with feature interactions
- **Pros:** Fast training, handles non-linear patterns well
- **Cons:** Can overfit, less interpretable
- **When to use:** For quick predictions with mixed data quality

### Linear Regression
- **Best for:** Simple baseline and interpretability
- **Pros:** Fast, interpretable, good baseline
- **Cons:** Assumes linear relationship
- **When to use:** As a baseline for comparison

## How It Works

### Data Preparation

1. **Fetch/Generate Data** - Get historical OHLCV (Open, High, Low, Close, Volume) data from chosen source
2. **Feature Engineering** - Create sequences using a lookback window (default: 60 days)
3. **Normalization** - Scale data to 0-1 range for neural networks
4. **Train/Test Split** - 80% training, 20% testing

### Training Process

1. **Model Creation** - Initialize selected model architecture
2. **Training** - Fit model to training data
3. **Validation** - Evaluate on test data
4. **Prediction** - Make future price prediction

### Lookback Window

The model uses past N days to predict the next day's price:
```
Input (60 days):  [Day1, Day2, ... Day60]
Output (1 day):   [Day61]
```

## Customization

### Modify Model Parameters

Edit `demo.py` or `main.py` to change:

```python
# Lookback window
X, y = prepare_features(data, lookback_days=90)  # Instead of 60

# LSTM configuration
model = LSTMModel(lookback_days=60, epochs=100, batch_size=16)

# Random Forest configuration
model = RandomForestModel(n_estimators=200, max_depth=30)
```

### Create Custom Models

Add to `stock_predictor/model.py`:

```python
class MyCustomModel(StockPredictionModel):
    def __init__(self):
        super().__init__()
        # Your model here
    
    def train(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
```

## Important Notes

⚠️ **Disclaimer**: Stock prices are influenced by many factors. This predictor provides technical analysis based on historical patterns, but is NOT a financial advice tool. Use at your own risk and always conduct proper due diligence.

### Limitations

- Predictions are based solely on historical price data
- Real markets are affected by news, earnings, economic factors, etc.
- Past performance doesn't guarantee future results
- Works best with volatile stocks and sufficient historical data
- Data source reliability varies (CoinGecko most reliable for crypto)

### Best Practices

1. **Choose Right Data Source** - CoinGecko for crypto, Alpha Vantage for stocks
2. **Longer History** - Use 5+ years of data for better patterns
3. **Multiple Models** - Compare different models before deciding
4. **Market Context** - Always consider current market conditions
5. **Portfolio Approach** - Use predictions as one signal among many
6. **Regular Retraining** - Update models with new data periodically

## Troubleshooting

### "Could not fetch live data" error
- Check that the ticker is valid (e.g., AAPL, GOOGL, BTC)
- For Alpha Vantage: ensure API key is correct (get free from alphavantage.co)
- For CoinGecko: check internet connection (no key needed)
- App will automatically fall back to synthetic data
- Try with demo data first: check "Use Synthetic Data" box

### Data source not working
- **CoinGecko fails:** Check internet, ticker support, try another source
- **Alpha Vantage fails:** Invalid/missing API key, API rate limit reached
- **Yahoo Finance fails:** Rate limited, try again later or use another source

### Slow training
- Reduce number of epochs: `epochs=10`
- Reduce lookback window: `lookback_days=30`
- Use faster model: `--model linear`

### Poor predictions
- Use more historical data (5y instead of 1y)
- Increase training epochs
- Try different models with comparison feature
- Check if asset is volatile enough for patterns
- Try different data sources

## Performance Tips

### For Fastest Results
```bash
python demo.py AAPL --model linear --no-viz
```

### For Best Accuracy
```bash
# Use LSTM with CoinGecko (for crypto) or Alpha Vantage (for stocks)
# Select data source in web UI
streamlit run app.py
```

### For Comparison
```bash
python demo.py AAPL --compare
```

## License

This project is provided as-is for educational purposes.

## Author

Created as a comprehensive stock/crypto prediction system demonstrating:
- Machine learning model implementations
- Time-series data handling
- Model evaluation and comparison
- Data visualization

## Future Enhancements

- [ ] Add more ML models (SVM, XGBoost, Prophet)
- [ ] Implement ensemble predictions
- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Portfolio optimization
- [ ] Real-time prediction API
- [ ] Web dashboard
- [ ] GPU support for faster training

## References

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

---

**Questions?** Check the code comments or create an issue.
