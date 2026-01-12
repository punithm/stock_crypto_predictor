# 🎯 Stock & Crypto Predictor - Complete System

## 🎉 What's New: Web UI & Crypto Support

### ✨ Added Features

#### 1. **Interactive Web Application** (`app.py`)
- Beautiful Streamlit interface
- Real-time model training visualization
- Interactive price charts with Plotly
- Side-by-side model comparison
- Easy asset and model selection

#### 2. **Cryptocurrency Support** (`crypto_fetcher.py`)
- Bitcoin, Ethereum, and 6 other major cryptos
- Same ML models as stocks
- Higher volatility handling
- Synthetic and real data support

#### 3. **Four Main Tabs in Web UI**
- **📊 Price History** - Historical price charts and statistics
- **🤖 Model Training** - Real-time training with metrics
- **🎯 Predictions** - Next-day price forecast
- **📈 Comparison** - Train and compare all 3 models

#### 4. **Enhanced Configuration**
- 8 popular stocks to choose from
- 8 major cryptocurrencies
- Adjustable lookback window (30-120 days)
- Toggle between synthetic and real data
- Model selection (LSTM, Random Forest, Linear)

## 📊 Project Structure

```
py_project_1/
├── 🌐 app.py                      # Main Streamlit web app
│
├── 📁 stock_predictor/            # ML library
│   ├── __init__.py
│   ├── data_fetcher.py           # Stock data + synthetic generator
│   ├── crypto_fetcher.py         # Crypto data + synthetic generator
│   ├── model.py                  # LSTM, RF, Linear models
│   ├── evaluator.py              # Metrics calculation
│   └── visualizer.py             # Plotting utilities
│
├── 🎬 CLI Scripts
│   ├── main.py                   # With live data
│   └── demo.py                   # With synthetic data
│
├── 📖 Documentation
│   ├── README.md                 # Detailed technical docs
│   ├── UI_GUIDE.md              # Web UI user guide
│   └── COMPLETE_GUIDE.md        # Complete reference
│
├── ⚙️ Setup
│   ├── requirements.txt          # Python dependencies
│   └── .venv/                    # Virtual environment
│
└── 🐍 Python 3.9 Virtual Environment
```

## 🚀 Quick Start

### Option 1: Web UI (Recommended)
```bash
cd /Users/punith/py_project_1
streamlit run app.py
```
Then open: **http://localhost:8501**

### Option 2: Command Line
```bash
# With synthetic data (demo)
python demo.py AAPL --model lstm --compare

# With real data
python main.py GOOGL --model rf
```

## 🎯 Available Assets

### Stocks (8 options)
- Apple (AAPL)
- Google (GOOGL)
- Microsoft (MSFT)
- Amazon (AMZN)
- Meta (META)
- Tesla (TSLA)
- Netflix (NFLX)
- Nvidia (NVDA)

### Cryptocurrencies (8 options)
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Cardano (ADA-USD)
- Solana (SOL-USD)
- Polkadot (DOT-USD)
- Ripple (XRP-USD)
- Litecoin (LTC-USD)
- Dogecoin (DOGE-USD)

## 🤖 Machine Learning Models

| Model | Best For | Speed | Accuracy |
|-------|----------|-------|----------|
| **LSTM** | Time-series patterns | Slow | Very High |
| **Random Forest** | Non-linear patterns | Medium | High |
| **Linear** | Baseline comparison | Fast | Medium |

## 📊 Web UI Interface

### Left Sidebar Controls
1. **Asset Type** - Stock or Cryptocurrency
2. **Asset Selection** - Choose from 8 options each
3. **Model Selection** - LSTM, RF, or Linear
4. **Data Settings** - Synthetic/Real toggle, Lookback window

### Main Content Tabs

#### Tab 1: Price History
- Interactive line chart
- Price statistics (current, min, max, avg)
- 7-day and 30-day change metrics

#### Tab 2: Model Training
- Real-time progress bar
- Training configuration display
- Comprehensive metrics:
  - RMSE, MAE, MSE
  - R² Score
  - MAPE
  - Directional Accuracy
- Actual vs Predicted price chart

#### Tab 3: Predictions
- Current price display
- Next-day prediction
- Price change ($ and %)
- Direction indicator (📈 UP / 📉 DOWN)
- Recent prices + prediction chart

#### Tab 4: Model Comparison
- Train all 3 models at once
- Performance comparison table
- RMSE comparison chart
- R² Score comparison chart
- All predictions on single chart
- Best model recommendation

## 📈 Performance Metrics Explained

**RMSE** (Root Mean Squared Error)
- Prediction error in same units as price
- Lower is better
- Good baseline for comparison

**R² Score** (Coefficient of Determination)
- How well model explains price movements
- Range: -∞ to 1.0
- 0.8+ = Excellent
- 0.5-0.8 = Good
- <0.5 = Poor

**MAPE** (Mean Absolute Percentage Error)
- Error as percentage of actual price
- Shows relative accuracy
- Lower % is better

**Directional Accuracy**
- Percentage of correct up/down predictions
- >50% = Better than random
- 70%+ = Very good
- 80%+ = Excellent

## 💡 Usage Tips

### For Best Accuracy
✅ Use real data (uncheck synthetic)
✅ Select LSTM model
✅ Use 60-90 day lookback
✅ Have 5+ years of history

### For Fast Demo
✅ Use synthetic data
✅ Select Linear model
✅ Use 30 day lookback

### For Fair Comparison
✅ Use "Comparison" tab
✅ Compare all 3 models
✅ Same data for all
✅ Check RMSE and R²

## 🔧 Technical Specifications

**Frontend**
- Streamlit 1.25+
- Plotly for interactive charts
- Python 3.9+

**Backend**
- TensorFlow/Keras (LSTM)
- Scikit-learn (Random Forest, Linear)
- Pandas/Numpy (Data processing)
- yfinance (Live data)

**Data Sources**
- Yahoo Finance (yfinance)
- Synthetic data generation

## 📚 Documentation Files

1. **COMPLETE_GUIDE.md** - This comprehensive overview
2. **UI_GUIDE.md** - Web interface user guide
3. **README.md** - Technical documentation
4. **requirements.txt** - Package dependencies

## ⚡ Installation & Setup

### First Time Setup
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
# Make sure you're in the project directory
cd /Users/punith/py_project_1

# Activate virtual environment (if not already)
source .venv/bin/activate

# Start the app
streamlit run app.py
```

## 🎓 Learning Outcomes

By using this system, you'll learn:

1. **ML Fundamentals**
   - Time-series prediction
   - Model evaluation
   - Hyperparameter tuning

2. **Financial Analysis**
   - Price patterns
   - Technical indicators
   - Risk assessment

3. **Web Development**
   - Interactive UI with Streamlit
   - Data visualization with Plotly
   - Real-time updates

4. **Data Science**
   - Data preprocessing
   - Feature engineering
   - Model comparison

## ⚠️ Important Disclaimers

⚠️ **Educational Purpose Only**
- NOT financial advice
- NOT investment recommendation
- Past performance ≠ Future results
- Use with expert consultation

🔍 **Limitations**
- Based only on historical price data
- Doesn't account for external factors
- Market conditions change
- Always verify independently

## 🐛 Troubleshooting

### App won't start
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Kill existing process
pkill -f streamlit

# Restart
streamlit run app.py
```

### Slow training
- Uncheck "Use Synthetic Data" if slow
- Select "Linear" model
- Reduce lookback window to 30

### Data fetch errors
- Check internet connection
- Verify ticker symbol
- Use synthetic data as fallback

## 🎉 You're Ready!

Everything is set up and ready to use:

1. **Open Web UI**: `streamlit run app.py`
2. **Select Asset**: Stock or crypto
3. **Train Model**: Select model and watch training
4. **Get Prediction**: See next-day price forecast
5. **Compare Models**: Try all 3 models side-by-side

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Start Web UI | `streamlit run app.py` |
| Demo with CLI | `python demo.py AAPL` |
| Real data CLI | `python main.py GOOGL` |
| Compare models | `python demo.py --compare` |
| Check requirements | `cat requirements.txt` |

## 🚀 Next Features to Add

- [ ] More stocks/cryptos
- [ ] Technical indicators
- [ ] Portfolio optimization
- [ ] Price alerts
- [ ] REST API
- [ ] Database storage
- [ ] Backtesting
- [ ] Advanced charts

---

## 📌 Summary

✅ **Complete Stock & Crypto Predictor**
✅ **Beautiful Web UI with Streamlit**
✅ **Multiple ML Models (LSTM, RF, Linear)**
✅ **Real & Synthetic Data Support**
✅ **8 Stocks + 8 Cryptocurrencies**
✅ **Comprehensive Metrics & Analysis**
✅ **Interactive Charts & Visualizations**
✅ **Model Comparison & Recommendations**

**Start predicting prices now:** `streamlit run app.py`
