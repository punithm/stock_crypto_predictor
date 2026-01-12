# 🎉 Stock & Crypto Predictor with Web UI - Complete Guide

## 📋 What You Have Built

A full-featured web application for predicting stock and cryptocurrency prices using machine learning.

### Components

```
Stock & Crypto Predictor Project
│
├── 📁 stock_predictor/          # Core ML library
│   ├── data_fetcher.py         # Stock data fetching & preprocessing
│   ├── crypto_fetcher.py       # Crypto data fetching & preprocessing
│   ├── model.py                # ML models (LSTM, RF, Linear)
│   ├── evaluator.py            # Model evaluation metrics
│   └── visualizer.py           # Plotting & visualization
│
├── 🌐 app.py                   # Streamlit Web UI (Main Entry Point)
├── 📊 main.py                  # CLI script with live data
├── 🎬 demo.py                  # CLI script with synthetic data
├── 📖 README.md                # Detailed documentation
├── 🚀 UI_GUIDE.md              # Web UI user guide
└── 🐍 Python 3.9 + ML Libraries
```

## 🚀 How to Use

### Launch the Web App

```bash
cd /Users/punith/py_project_1
streamlit run app.py
```

Then open: **http://localhost:8501**

### Or Use Command Line

**With synthetic data (no internet):**
```bash
python demo.py AAPL --model lstm --compare
```

**With live data (needs internet):**
```bash
python main.py GOOGL --model rf
```

## ✨ Key Features

### 1. **Multi-Asset Support**
- 📊 **8 Popular Stocks**: AAPL, GOOGL, MSFT, AMZN, META, TSLA, NFLX, NVDA
- 🪙 **8 Popular Cryptos**: BTC, ETH, ADA, SOL, DOT, XRP, LTC, DOGE

### 2. **Three ML Models**
- **LSTM Neural Network** - Best accuracy for time-series
- **Random Forest** - Good for pattern recognition
- **Linear Regression** - Fast baseline model

### 3. **Interactive Web Interface**
- 📊 **Price History Tab** - View historical prices and statistics
- 🤖 **Model Training Tab** - Train models and see metrics
- 🎯 **Predictions Tab** - Get next-day price predictions
- 📈 **Comparison Tab** - Compare all 3 models side-by-side

### 4. **Comprehensive Metrics**
- RMSE, MAE, MSE, R² Score, MAPE
- Directional Accuracy (up/down prediction accuracy)
- Visual comparisons and recommendations

### 5. **Data Flexibility**
- **Synthetic Data** - Demo mode, no internet needed
- **Live Data** - Real prices from Yahoo Finance

## 📊 Quick Usage Examples

### Example 1: Predict Apple Stock (Quick Demo)
1. Open app: `streamlit run app.py`
2. Keep default settings (Synthetic Data ON)
3. Select "Apple (AAPL)"
4. Go to "Model Training" tab
5. See real-time metrics
6. Go to "Predictions" tab to see next price prediction

### Example 2: Compare Models for Bitcoin
1. Select "Cryptocurrency" → "Bitcoin"
2. Go to "Comparison" tab
3. Click "Train & Compare All 3 Models"
4. Get recommendation for best model

### Example 3: Real Data with LSTM
1. Uncheck "Use Synthetic Data"
2. Select "LSTM" model
3. Adjust lookback window (60-90 days recommended)
4. Click "Model Training" tab
5. Wait for live data to fetch and model to train

## 📈 Understanding the Interface

### Left Sidebar
- **Asset Selection**: Choose stock or crypto
- **Model Selection**: Choose ML algorithm
- **Data Settings**: Toggle synthetic/real data, adjust lookback window

### Main Tabs
1. **Price History** - Charts and statistics
2. **Model Training** - Training progress and metrics
3. **Predictions** - Next-day price forecast
4. **Comparison** - All models side-by-side

### Metrics Explained
- **RMSE**: Lower = Better (prediction error in $)
- **R² Score**: Higher = Better (0.8+ is good)
- **Directional Accuracy**: Higher = Better (% of correct up/down calls)

## 💡 Best Practices

### For Accurate Predictions
✅ Use real data (uncheck synthetic)
✅ Use LSTM model
✅ Use 60-90 day lookback window
✅ Train on 5+ years of data

### For Fast Results
✅ Use synthetic data
✅ Use Linear model
✅ Use 30 day lookback window

### For Fair Comparisons
✅ Use comparison tab
✅ Same data for all models
✅ Check both RMSE and R² metrics

## 🔧 Technical Stack

```
Frontend:
- Streamlit (Web UI framework)
- Plotly (Interactive charts)

Backend:
- Python 3.9
- TensorFlow/Keras (LSTM)
- Scikit-learn (RF, Linear)
- Pandas/Numpy (Data processing)
- yfinance (Live data)
```

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application |
| `main.py` | CLI with live stock/crypto data |
| `demo.py` | CLI with synthetic data |
| `data_fetcher.py` | Stock data module |
| `crypto_fetcher.py` | Crypto data module |
| `model.py` | ML models implementation |
| `evaluator.py` | Model evaluation metrics |
| `visualizer.py` | Plotting utilities |

## ⚠️ Important Notes

**This is an educational tool for learning ML and technical analysis.**

- ❌ NOT financial advice
- ❌ NOT a replacement for expert analysis
- ⚠️ Past performance ≠ Future results
- 📊 Use with other analysis tools
- 🔍 Always do your own research

## 🎓 Learning Outcomes

After using this project, you'll understand:

1. **Data Fetching & Preprocessing** - How to get and prepare financial data
2. **Time Series Modeling** - LSTM networks for sequential data
3. **ML Pipelines** - Training, evaluation, and prediction workflows
4. **Model Comparison** - How to evaluate and compare models
5. **Web Development** - Building interactive data apps with Streamlit
6. **Financial Analysis** - Technical indicators and price prediction

## 🚀 Next Steps / Enhancements

Ideas for extending the project:

1. **More Models**: Add SVM, XGBoost, Prophet
2. **Technical Indicators**: RSI, MACD, Bollinger Bands
3. **Ensemble Predictions**: Combine multiple models
4. **Portfolio Optimization**: Suggest asset allocation
5. **Alerts**: Price target notifications
6. **API**: Rest API for programmatic access
7. **Database**: Store predictions for backtesting
8. **GPU Support**: Faster training with CUDA

## 🆘 Troubleshooting

### App won't start
```bash
# Check port
lsof -i :8501

# Kill and restart
pkill -f streamlit
streamlit run app.py
```

### Slow training
- Use synthetic data
- Select Linear model
- Reduce lookback window

### "Cannot fetch data" error
- Check internet connection
- Use synthetic data
- Verify ticker symbol is correct

## 📞 Getting Help

1. **UI Guide**: Read `UI_GUIDE.md`
2. **Documentation**: Read `README.md`
3. **Code Comments**: Check code in `stock_predictor/`
4. **Examples**: Run `demo.py` or `main.py`

## 🎯 Project Summary

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.9 |
| **Framework** | Streamlit |
| **Models** | LSTM, Random Forest, Linear Regression |
| **Assets** | 8 Stocks + 8 Cryptocurrencies |
| **Data Source** | Yahoo Finance (yfinance) |
| **Interface** | Interactive Web UI |
| **Metrics** | RMSE, MAE, R², MAPE, Directional Accuracy |
| **Mode** | Synthetic Demo + Real Data |

## ✅ Checklist

- ✅ Web UI with Streamlit
- ✅ Stock price prediction
- ✅ Crypto price prediction
- ✅ Multiple ML models
- ✅ Model comparison
- ✅ Interactive charts
- ✅ Real & synthetic data
- ✅ Comprehensive metrics
- ✅ Command-line tools
- ✅ Full documentation

## 🎉 You're All Set!

Your Stock & Crypto Predictor is ready to use:

```bash
streamlit run app.py
```

Then visit: **http://localhost:8501**

Enjoy exploring the intersection of finance and machine learning! 📈

---

**Remember**: This tool is for learning and analysis, not financial advice.
Always consult professionals before making investment decisions.
