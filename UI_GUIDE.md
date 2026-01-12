# 🚀 Web UI Quick Start Guide

## Using the Stock & Crypto Predictor Web Interface

### Starting the Application

1. **Open Terminal** in the project directory:
   ```bash
   cd /Users/punith/py_project_1
   ```

2. **Launch the Web UI**:
   ```bash
   python -m streamlit run app.py
   ```

3. **Open in Browser**:
   - The app will automatically open at: `http://localhost:8501`
   - Or manually navigate to that URL

### Features

#### 🎯 **Asset Selection** (Left Sidebar)
- **📊 Stock** - Choose from popular stocks:
  - Apple (AAPL)
  - Google (GOOGL)
  - Microsoft (MSFT)
  - Amazon (AMZN)
  - Meta (META)
  - Tesla (TSLA)
  - Netflix (NFLX)
  - Nvidia (NVDA)

- **🪙 Cryptocurrency** - Choose from popular cryptos:
  - Bitcoin (BTC-USD)
  - Ethereum (ETH-USD)
  - Cardano (ADA-USD)
  - Solana (SOL-USD)
  - Polkadot (DOT-USD)
  - Ripple (XRP-USD)
  - Litecoin (LTC-USD)
  - Dogecoin (DOGE-USD)

#### 🤖 **Model Selection** (Left Sidebar)
- **LSTM** - Best for time-series data (slower but more accurate)
- **Random Forest** - Good for pattern recognition (faster)
- **Linear Regression** - Simple baseline (fastest)

#### ⚙️ **Data Settings** (Left Sidebar)
- **Use Synthetic Data** - Check this to use demo data (no internet needed)
- **Lookback Window** - Adjust how many days to use for prediction (30-120 days)

### Main Interface Tabs

#### **Tab 1: 📊 Price History**
- View historical price chart
- See price statistics:
  - Current price
  - Min/Max prices
  - Average price
  - 7-day and 30-day changes
- Interactive chart with hover information

#### **Tab 2: 🤖 Model Training**
- Watch real-time training progress
- See training/testing split
- View detailed metrics:
  - **RMSE** - Prediction error (lower is better)
  - **MAE** - Average absolute error
  - **R² Score** - Goodness of fit
  - **MAPE** - Percentage error
  - **Directional Accuracy** - % of correct up/down predictions
- See prediction vs actual price comparison chart

#### **Tab 3: 🎯 Predictions**
- Get the next day's price prediction
- See expected price change (up/down)
- View recent prices with prediction
- Interactive chart showing historical and predicted prices

#### **Tab 4: 📈 Comparison**
- Train all 3 models simultaneously
- Compare performance metrics
- Visual comparison of RMSE and R² Score
- See all model predictions on the same chart
- Get recommendation for the best model

## Workflow Example

### Step 1: Select Asset
1. In left sidebar, choose **"📊 Stock"**
2. Select **"Apple"** from dropdown

### Step 2: Configure Settings
1. Choose **LSTM** model
2. Check **"Use Synthetic Data"** (for quick demo)
3. Keep **Lookback Window** at 60 days

### Step 3: View Price History
1. Click **"📊 Price History"** tab
2. See Apple's historical prices
3. Check current price and trends

### Step 4: Train Model
1. Click **"🤖 Model Training"** tab
2. Watch progress bar
3. Wait for training to complete
4. Review performance metrics

### Step 5: Make Prediction
1. Click **"🎯 Predictions"** tab
2. See predicted next price
3. Check if price is expected to go UP or DOWN
4. View confidence with directional accuracy

### Step 6: Compare Models (Optional)
1. Click **"📈 Comparison"** tab
2. Click **"Train & Compare All 3 Models"**
3. Wait for all 3 models to train
4. See which model performs best
5. Get recommendation

## Data Sources

### Synthetic Data (Demo Mode)
- Generated realistic price movements
- No internet required
- Perfect for testing and learning

### Live Data (Real Mode)
- Fetched from Yahoo Finance via yfinance
- Requires internet connection
- Real historical prices
- Better for actual predictions

## Tips & Tricks

### For Fastest Results
1. Use **"Use Synthetic Data"** option
2. Select **"Linear Regression"** model
3. Reduce lookback window to 30 days

### For Best Accuracy
1. Don't use synthetic data (use real data)
2. Select **"LSTM"** model
3. Increase lookback window to 90-120 days

### For Comparing Models
1. Always use **"Compare All 3 Models"** tab
2. Check RMSE (lower is better)
3. Check R² Score (higher is better)

### Understanding Metrics

**RMSE (Root Mean Squared Error)**
- Measures average prediction error
- Same units as price ($)
- Lower = Better (closer to 0)

**MAE (Mean Absolute Error)**
- Average absolute error
- Easy to interpret
- Lower = Better

**R² Score**
- How well model explains price movements
- Range: -∞ to 1.0
- 1.0 = Perfect fit
- 0.8+ = Good fit
- <0 = Worse than baseline

**MAPE (Mean Absolute Percentage Error)**
- Error as percentage of actual price
- Shows relative accuracy
- Lower % = Better

**Directional Accuracy**
- % of correct up/down predictions
- >50% = Better than random
- 70%+ = Very good

## Troubleshooting

### App Won't Start
```bash
# Check if port is already in use
lsof -i :8501

# Kill any existing Streamlit process
pkill -f streamlit

# Try again
python -m streamlit run app.py
```

### "Cannot fetch data" error
- Check your internet connection
- Uncheck "Use Synthetic Data" to use real data
- Try a different ticker symbol

### Slow training
- Check "Use Synthetic Data"
- Select "Linear Regression" model
- Reduce lookback window

### Browser won't load
- Wait 10 seconds for Streamlit to initialize
- Try refreshing the page (F5)
- Check console for errors

## Customization

### Add More Stocks
Edit `app.py` and modify the `popular_stocks` dictionary:
```python
popular_stocks = {
    "Apple": "AAPL",
    "Your Stock": "TICKER",  # Add here
}
```

### Add More Cryptos
Edit `app.py` and modify the `popular_cryptos` dictionary:
```python
popular_cryptos = {
    "Bitcoin": "BTC-USD",
    "Your Crypto": "XXX-USD",  # Add here
}
```

## Common Questions

**Q: Is this a financial advisor?**
A: No. This is for educational purposes and technical analysis only. Always do your own research.

**Q: How accurate are the predictions?**
A: Depends on market volatility and data quality. LSTM typically gets 60-85% directional accuracy on test data.

**Q: Can I use this for real trading?**
A: Not recommended. Use alongside other analysis tools and expert advice.

**Q: What's the difference between stocks and crypto predictions?**
A: Crypto is more volatile, so lookback windows might need adjustment. Both use the same ML models.

**Q: Which model should I use?**
A: LSTM for best accuracy, Random Forest for patterns, Linear for baseline comparison.

## Next Steps

1. Try different assets to see how models perform
2. Experiment with different lookback windows
3. Compare results with actual prices the next day
4. Learn more about the underlying models in the README.md

---

**Need help?** Check the README.md for detailed documentation.
