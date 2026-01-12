# Quick Reference Card

## 🚀 Launch Commands

```bash
# Web UI (Recommended) - with data source selector
streamlit run app.py

# Demo with CLI (synthetic data)
python demo.py AAPL --model lstm

# Real data with CLI (auto fallback to synthetic)
python main.py GOOGL --compare
```

## 📊 Available Assets

### Stocks (8)
AAPL, GOOGL, MSFT, AMZN, META, TSLA, NFLX, NVDA

### Cryptocurrencies (8)
BTC, ETH, ADA, SOL, DOT, XRP, LTC, DOGE
(Displayed as EUR with CoinGecko or Alpha Vantage)

## 📡 Data Sources

| Source | Type | Key Needed | Best For |
|--------|------|-----------|----------|
| Yahoo Finance | Stock/Crypto | ❌ No | Default, quick |
| **Alpha Vantage** ⭐ | Stock | ✅ Free* | Reliable stocks |
| **CoinGecko** ⭐⭐ | Crypto | ❌ No | Free crypto, EUR native |

*Get free Alpha Vantage key: https://www.alphavantage.co/

## 🤖 Models

| Name | Command | Speed | Accuracy |
|------|---------|-------|----------|
| LSTM | `--model lstm` | Slow | ⭐⭐⭐⭐⭐ |
| Random Forest | `--model rf` | Medium | ⭐⭐⭐⭐ |
| Linear | `--model linear` | Fast | ⭐⭐⭐ |

## 📈 Metrics Reference

| Metric | What It Means | Good Value |
|--------|---------------|-----------|
| RMSE | Prediction error (€) | Lower is better |
| MAE | Average error (€) | Lower is better |
| R² | How well it fits | 0.8+ is good |
| MAPE | Error (%) | <5% is excellent |
| Dir. Acc. | Up/Down accuracy | >60% is good |

## 🎯 Web UI Workflow

1. **Select Asset Type** → Stock or Crypto
2. **Select Asset** → Choose specific ticker
3. **Select Data Source** → Yahoo/Alpha Vantage/CoinGecko
4. **Optional: Add API Keys** → Alpha Vantage, NewsAPI
5. **View Price History** → See past prices (€)
6. **Train Model** → Click to train selected model
7. **Get Prediction** → See next day's prediction
8. **Compare Models** → Train all 3 models side-by-side

## ⚡ Quick Tips

✨ **Fastest Demo**: Synthetic data + Linear model
🎯 **Best Accuracy**: CoinGecko (crypto) + LSTM model
📊 **Fair Comparison**: Use comparison tab to see all 3
💱 **EUR Pricing**: All prices shown in €, especially good with CoinGecko
🔄 **Fallback**: If live data fails, auto-uses synthetic data

## 🔧 File Locations

```
app.py                      - Web UI (Streamlit)
demo.py                     - CLI with demo data
main.py                     - CLI with real data
README.md                   - Full documentation
UI_GUIDE.md                 - Web UI detailed guide
QUICK_REFERENCE.md          - This file
GETTING_STARTED.md          - Beginner guide
ARCHITECTURE.md             - Technical architecture
```

## 📦 Key Files

```
stock_predictor/
  ├── data_fetcher.py          - Yahoo Finance stocks
  ├── crypto_fetcher.py        - Yahoo Finance crypto
  ├── alphavantage_fetcher.py  - Alpha Vantage stocks ⭐
  ├── coingecko_fetcher.py     - CoinGecko crypto ⭐
  ├── model.py                 - LSTM, RF, Linear models
  ├── evaluator.py             - Performance metrics
  ├── indicators.py            - Technical indicators
  ├── news_sentiment.py        - Sentiment analysis
  └── visualizer.py            - Plotting utilities
```

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't start | `pkill -f streamlit` then `streamlit run app.py` |
| Port 8503+ in use | Kill: `lsof -ti:8503 \| xargs kill -9` |
| Data won't fetch | Check internet, try different data source, use synthetic |
| Slow training | Use synthetic data, reduce epochs, try Linear model |
| Can't find ticker | Verify ticker symbol (e.g., AAPL not APPLE) |
| Alpha Vantage error | Get free key from alphavantage.co, check if valid |
| CoinGecko error | Check internet connection, try different crypto |

## 💡 Web UI Tips

- **Data Source Dropdown**: Switch between Yahoo/Alpha Vantage/CoinGecko
- **Synthetic Data Checkbox**: Toggle between demo and live data
- **API Key Fields**: Optional, enables premium features
- **Lookback Slider**: Adjust historical window (30-120 days)
- **Button-Triggered**: Training and comparison run on-click (not auto)
- **Charts**: Interactive Plotly charts - hover, zoom, click legend to toggle

## 🌐 Currency

- **Default Display**: EUR (€)
- **Best for EUR**: CoinGecko (native support)
- **Data Fetched As**: USD internally, displayed as EUR
- **All Prices**: Shown in € throughout UI
  ├── crypto_fetcher.py (crypto)
  ├── model.py (LSTM, RF, Linear)
  ├── evaluator.py (metrics)
  └── visualizer.py (charts)
```

## 🎓 Learning Path

1. Start with demo (synthetic data)
2. Try different models
3. Compare model performance
4. Switch to real data
5. Experiment with lookback window
6. Explore code and understand ML

## 💡 Key Concepts

- **Lookback Window** = Days of history used for prediction
- **LSTM** = Neural network for time-series
- **R²** = Goodness of fit (0.8+ is good)
- **Directional Accuracy** = % correct up/down calls

## ✅ Checklist

- [x] Web UI built
- [x] Stock predictor
- [x] Crypto predictor
- [x] Multiple models
- [x] Real + synthetic data
- [x] Comprehensive docs
- [x] Ready to use!

## 🎉 Ready to Start?

```bash
streamlit run app.py
```

Then visit: **http://localhost:8501**

---

**Remember**: For learning & analysis only, not financial advice!
