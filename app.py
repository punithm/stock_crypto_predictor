"""
Streamlit Web UI for Stock and Crypto Price Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os

# Optimize TensorFlow startup (must be before any TensorFlow imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Don't allocate all GPU memory
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU (faster for small models than GPU overhead)
warnings.filterwarnings('ignore')

from stock_crypto_predictor.data_fetcher import StockDataFetcher, generate_synthetic_stock_data
from stock_crypto_predictor.crypto_fetcher import CryptoDataFetcher, generate_synthetic_crypto_data
from stock_crypto_predictor.alphavantage_fetcher import AlphaVantageFetcher
from stock_crypto_predictor.coingecko_fetcher import CoinGeckoFetcher
from stock_crypto_predictor.evaluator import ModelEvaluator

# Configure page (FIRST call - must be at top)
st.set_page_config(
    page_title="Stock & Crypto Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy-load model imports
@st.cache_resource
def load_models():
    # Default loader kept for backward compatibility; prefer calling with model_type below.
    from stock_crypto_predictor.model import RandomForestModel, SimpleLinearModel
    return RandomForestModel, SimpleLinearModel

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 0rem; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📈 Stock & Crypto Price Predictor")
st.markdown("""
    Predict stock and cryptocurrency prices using advanced ML models.
    Choose your asset, select a model, and get instant predictions!
""")

# Add important disclaimer at the top
st.info("""
    ⚠️ **IMPORTANT DISCLAIMER:**
    - **Prices are DELAYED**: Data sources are 15-30 minutes delayed (not real-time)
    - **Not for Trading**: This app is for learning & predictions only, NOT for real trading
    - **Verify Before Trading**: Always check live prices on your broker (Trade Republic, Scalable Capital, etc.)
    - **Predictions are Estimates**: ML models can be wrong. Past performance ≠ future results
    - **Use at Your Own Risk**: Do NOT make financial decisions based solely on this app
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Asset Type Selection
asset_type = st.sidebar.radio(
    "Select Asset Type",
    options=["📊 Stock", "🪙 Cryptocurrency"],
    horizontal=False
)

# Get ticker based on asset type
if asset_type == "📊 Stock":
    st.sidebar.subheader("Stock Settings")
    
    # Popular stocks
    popular_stocks = {
        "Apple": "AAPL",
        "Google": "GOOGL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Meta": "META",
        "Tesla": "TSLA",
        "Netflix": "NFLX",
        "Nvidia": "NVDA"
    }
    
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        options=list(popular_stocks.keys()),
        index=0
    )
    ticker = popular_stocks[selected_stock]
    asset_name = f"{selected_stock} ({ticker})"
    
else:  # Cryptocurrency
    st.sidebar.subheader("Cryptocurrency Settings")
    
    # Popular cryptos (keep USD for yfinance, display as EUR)
    popular_cryptos = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Cardano": "ADA-USD",
        "Solana": "SOL-USD",
        "Polkadot": "DOT-USD",
        "Ripple": "XRP-USD",
        "Litecoin": "LTC-USD",
        "Dogecoin": "DOGE-USD"
    }
    
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(popular_cryptos.keys()),
        index=0
    )
    ticker = popular_cryptos[selected_crypto]
    asset_name = f"{selected_crypto} ({ticker})"

# Model selection
st.sidebar.subheader("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose ML Model",
    options=["LSTM", "Random Forest", "Linear Regression"],
    help="LSTM: Best for time-series (slower)\nRF: Good for patterns (faster)\nLinear: Baseline model (fastest)"
)

# Ensemble option
enable_ensemble = st.sidebar.checkbox(
    "Enable Adaptive Ensemble",
    value=False,
    help="Combine multiple models with adaptive weights based on recent validation performance"
)

# Data settings
st.sidebar.subheader("Data Settings")

# Performance options
st.sidebar.subheader("⚡ Performance")
if st.sidebar.button("🗑️ Clear Cache", help="Clear cached data to refresh from source"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Cache cleared! Refresh will load fresh data.")


# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source",
    options=["Yahoo Finance", "Alpha Vantage (Stocks)", "CoinGecko (Crypto)"],
    help="""
    **Price Source & Delays:**
    
    ⚡ **FASTEST →** Yahoo Finance: 15-30 min delayed, USD→EUR conversion
    🐌 **SLOW** Alpha Vantage: 15 min delayed, API rate limits (⏳ 10-30s wait)
    ⚡ **FAST** CoinGecko: 5-10 min delayed, EUR direct (best for crypto)
    
    **Recommendation:** Use Yahoo Finance or CoinGecko for faster loading.
    Alpha Vantage can timeout - switch to synthetic data if it's slow.
    
    ⚠️ All prices are DELAYED. Never use for real trading decisions.
    Always verify on your broker before executing trades.
    """
)

# Alpha Vantage API key (if selected)
av_api_key = ""
if "Alpha Vantage" in data_source:
    av_api_key = st.sidebar.text_input(
        "Alpha Vantage API Key",
        value="",
        type="password",
        help="📌 SETUP REQUIRED:\n1. Get FREE API key from https://www.alphavantage.co/\n2. Paste the key here (NOT 'demo')\n3. Demo key is too rate-limited for predictions\n\n❌ Error 'Information': Usually means invalid/missing API key\n✅ Solution: Get real key from alphavantage.co (it's free!)\n✅ For crypto: Use CoinGecko instead (no key needed)"
    )

use_synthetic = st.sidebar.checkbox(
    "Use Synthetic Data",
    value=True,
    help="✅ Check to use demo data (fast, no internet needed)\n❌ Uncheck for live data (slow, requires internet)"
)

lookback_days = st.sidebar.slider(
    "Lookback Window (days)",
    min_value=30,
    max_value=120,
    value=60,
    step=10
)

# News API key for sentiment (optional)
news_api_key = st.sidebar.text_input(
    "NewsAPI Key (optional)",
    value="",
    type="password",
    help="Provide a NewsAPI.org API key to enable news sentiment features (optional)."
)

# Function to fetch and prepare data
@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours instead of 1 hour
def fetch_and_prepare_data(ticker, asset_type, use_synthetic, lookback_days, data_source, av_api_key=None, news_api_key=None):
    """Fetch data and prepare features. Returns (data, X, y, sentiment_series).
    sentiment_series is a pandas Series indexed by datetime (may be empty).
    """
    sentiment_series = pd.Series(dtype=float)

    if use_synthetic:
        if asset_type == "📊 Stock":
            data = generate_synthetic_stock_data(ticker.split('-')[0] if '-' in ticker else ticker, 
                                               num_days=1260, start_price=150)
            dates = data.index
            np.random.seed(42)
            sentiment_series = pd.Series(np.random.normal(0.0, 0.05, len(dates)), index=dates)
        else:
            data = generate_synthetic_crypto_data(ticker.split('-')[0] if '-' in ticker else ticker, 
                                                 num_days=1260, start_price=40000)
            dates = data.index
            np.random.seed(42)
            sentiment_series = pd.Series(np.random.normal(0.0, 0.08, len(dates)), index=dates)
    else:
        try:
            if asset_type == "📊 Stock":
                # Choose stock fetcher based on data_source
                if "Alpha Vantage" in data_source:
                    st.warning("⏳ **Alpha Vantage can be slow** (API rate limits). Consider using Yahoo Finance or CoinGecko instead.")
                    fetcher = AlphaVantageFetcher(ticker, api_key=av_api_key or "demo")
                else:  # Yahoo Finance default
                    fetcher = StockDataFetcher(ticker, period="5y")
                
                if news_api_key:
                    try:
                        fetcher.set_news_api_key(news_api_key)
                    except Exception:
                        pass
                data = fetcher.fetch_data()
            else:
                # Choose crypto fetcher based on data_source
                if "CoinGecko" in data_source:
                    fetcher = CoinGeckoFetcher(ticker, currency="eur")
                else:  # Yahoo Finance default
                    fetcher = CryptoDataFetcher(ticker, period="5y")
                
                if news_api_key:
                    try:
                        fetcher.set_news_api_key(news_api_key)
                    except Exception:
                        pass
                data = fetcher.fetch_data()
            if news_api_key:
                try:
                    from stock_crypto_predictor.news_sentiment import get_sentiment_series_for_range
                    start = data.index[0].to_pydatetime()
                    end = data.index[-1].to_pydatetime()
                    print(f"Fetching sentiment for {ticker} from {start.date()} to {end.date()}...")
                    sentiment_series = get_sentiment_series_for_range(news_api_key, ticker, start, end)
                    if not sentiment_series.empty:
                        print(f"✅ Got sentiment data: {len(sentiment_series)} records")
                        sentiment_series.index = pd.to_datetime(sentiment_series.index)
                        sentiment_series = sentiment_series.reindex(data.index, method='ffill').fillna(0)
                    else:
                        print("⚠️ Sentiment query returned no data")
                        sentiment_series = pd.Series(dtype=float)
                except Exception as e:
                    print(f"❌ Sentiment fetch error: {str(e)}")
                    sentiment_series = pd.Series(dtype=float)

        except Exception as e:
            error_msg = str(e)
            # Show full error for debugging, expand display length for important messages
            display_error = error_msg[:400] if len(error_msg) > 400 else error_msg
            
            # Special handling for Alpha Vantage slow/timeout errors
            if "Alpha Vantage" in data_source or "timeout" in error_msg.lower():
                st.warning(f"⏱️ **Data source too slow or timeout** ({data_source}):\n{display_error}\n\n✅ **Switching to synthetic data for instant results.**")
            else:
                st.warning(f"⚠️ Could not fetch live data ({data_source}):\n{display_error}\n\nUsing synthetic data instead.")
            
            print(f"DEBUG: Full error from {data_source}: {error_msg}")
            if asset_type == "📊 Stock":
                data = generate_synthetic_stock_data(ticker.split('-')[0] if '-' in ticker else ticker)
            else:
                data = generate_synthetic_crypto_data(ticker.split('-')[0] if '-' in ticker else ticker)

    # Prepare features
    prices = data['Close'].values.reshape(-1, 1)
    X, y = [], []
    for i in range(len(prices) - lookback_days):
        X.append(prices[i:(i + lookback_days)])
        y.append(prices[i + lookback_days])

    X = np.array(X)
    y = np.array(y)

    return data, X, y, sentiment_series

# Main content
def main():
    # Fetch data (with status indicator)
    if use_synthetic:
        status_msg = "⚡ Loading synthetic demo data..."
    else:
        status_msg = f"📡 Loading live data from {data_source} (this may take 10-30 seconds)..."
    
    with st.spinner(status_msg):
        data, X, y, sentiment_series = fetch_and_prepare_data(ticker, asset_type, use_synthetic, lookback_days, data_source, av_api_key, news_api_key=news_api_key)

    # Show sentiment UI feedback
    if sentiment_series is not None and not sentiment_series.empty:
        st.sidebar.success("📰 Sentiment fetched")
        latest_sent = float(sentiment_series.iloc[-1])
        st.sidebar.metric("Latest Sentiment", f"{latest_sent:.3f}")
        try:
            recent_sent = sentiment_series.dropna()[-60:]
            if not recent_sent.empty:
                st.sidebar.line_chart(recent_sent)
        except Exception:
            pass
    else:
        st.sidebar.info("📰 No sentiment available (provide NewsAPI key)")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price History", "🤖 Model Training", "🎯 Predictions", "📈 Comparison"])
    
    with tab1:
        st.subheader(f"{asset_name} - Historical Price Data")
        
        # Add source attribution
        source_info = {
            "Yahoo Finance": "📊 Yahoo Finance (15-30 min delayed, USD→EUR)",
            "Alpha Vantage (Stocks)": "📊 Alpha Vantage (15 min delayed, USD→EUR)",
            "CoinGecko (Crypto)": "💰 CoinGecko (5-10 min delayed, native EUR)"
        }
        st.caption(f"📍 Data Source: {source_info.get(data_source, 'Unknown')}")
        st.caption("⚠️ These are NOT real-time prices. Always verify on your broker before trading.")
        
        # Display price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add min/max bands
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['High'],
            mode='lines',
            name='High',
            line=dict(color='rgba(0,200,0,0.2)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Low'],
            mode='lines',
            name='Low',
            line=dict(color='rgba(200,0,0,0.2)', width=0),
            fill='tonexty',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{asset_name} - Historical Close Prices",
            xaxis_title="Date",
            yaxis_title="Price (EUR)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"€{current_price:.2f}")
        
        with col2:
            min_price = data['Close'].min()
            st.metric("Min Price", f"€{min_price:.2f}")
        
        with col3:
            max_price = data['Close'].max()
            st.metric("Max Price", f"€{max_price:.2f}")
        
        with col4:
            avg_price = data['Close'].mean()
            st.metric("Avg Price", f"€{avg_price:.2f}")
        
        # Calculate changes
        if len(data) > 7:
            change_7d = ((data['Close'].iloc[-1] - data['Close'].iloc[-8]) / data['Close'].iloc[-8] * 100)
        else:
            change_7d = 0
            
        if len(data) > 30:
            change_30d = ((data['Close'].iloc[-1] - data['Close'].iloc[-31]) / data['Close'].iloc[-31] * 100)
        else:
            change_30d = 0
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("7-Day Change", f"{change_7d:.2f}%", 
                     delta=change_7d, delta_color="normal")
        with col6:
            st.metric("30-Day Change", f"{change_30d:.2f}%", 
                     delta=change_30d, delta_color="normal")
    
    with tab2:
        st.subheader(f"🤖 Training {model_type} Model")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Testing Samples", len(X_test))
        
        if st.button("▶️ Train Model", key="train_btn"):
            with st.status("🚀 Training in progress...", expanded=True) as status:
                st.write("⏳ Initializing model...")
                
                # Load models with minimal overhead - direct imports only
                if model_type == "LSTM":
                    st.write("Loading TensorFlow/Keras... (this may take 10-20 seconds on first load)")
                    from stock_crypto_predictor.model import LSTMModel
                    st.write("⏳ Building model architecture...")
                    model = LSTMModel(lookback_days=lookback_days, epochs=3, batch_size=16)  # Reduced from 5 to 3 epochs
                elif model_type == "Random Forest":
                    st.write("Loading Random Forest...")
                    from stock_crypto_predictor.model import RandomForestModel
                    model = RandomForestModel(n_estimators=50, max_depth=20)
                else:
                    st.write("Loading Linear Regression...")
                    from stock_crypto_predictor.model import SimpleLinearModel
                    model = SimpleLinearModel()
                
                st.write(f"🔧 Training {model_type} with {len(X_train)} samples...")
                
                # Train (this is where it takes time)
                model.train(X_train, y_train)
                
                st.write("📊 Making predictions on test set...")
                
                # Predict
                y_pred = model.predict(X_test)
                
                st.write("📈 Calculating metrics...")
                
                # Evaluate
                metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
                dir_accuracy = ModelEvaluator.directional_accuracy(y_test, y_pred)
                
                status.update(label="✅ Training complete!", state="complete")
            
            # Display metrics
            st.subheader("📊 Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RMSE", f"€{metrics['RMSE']:.2f}")
                st.metric("MAE", f"€{metrics['MAE']:.2f}")
            
            with col2:
                st.metric("R² Score", f"{metrics['R2']:.4f}")
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            with col3:
                st.metric("Directional Accuracy", f"{dir_accuracy:.2f}%")
            
            # Plot
            st.subheader("Predictions vs Actual")
            
            fig = go.Figure()
            x_axis = np.arange(len(y_test))
            
            fig.add_trace(go.Scatter(x=x_axis, y=y_test.flatten(), name='Actual', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=y_pred.flatten(), name='Predicted', line=dict(color='red', width=2, dash='dash')))
            
            fig.update_layout(title=f"{asset_name} - {model_type} Predictions", xaxis_title="Sample", yaxis_title="Price", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Store in session
            st.session_state.trained_model = model
            st.session_state.model_metrics = metrics
            st.session_state.data = data
            st.session_state.selected_model = model_type
    
    with tab3:
        st.subheader("🎯 Price Predictions")
        
        # Add prediction disclaimer
        st.warning("""
            ⚠️ **IMPORTANT - READ BEFORE USING PREDICTIONS:**
            - **Predictions are ESTIMATES ONLY** - they can be wrong
            - **Not for Real Trading** - Use only for learning & backtesting
            - **Past Performance ≠ Future Results** - Market conditions change
            - **Always Verify on Your Broker** - Never trade based only on this
            - **Use at Your Own Risk** - You are responsible for your decisions
        """)
        
        if 'trained_model' not in st.session_state:
            st.warning("⚠️ Please train a model first in the 'Model Training' tab")
        else:
            model = st.session_state.trained_model
            data = st.session_state.data
            # Allow using Ensemble if available
            model_choice = 'Trained Model'
            if 'ensemble_model' in st.session_state:
                model_choice = st.selectbox('Model for Prediction', options=['Trained Model', 'Ensemble'])
            else:
                st.write(f"Using trained model: {st.session_state.get('selected_model', 'Unknown')}")
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Make prediction
            latest_prices = data['Close'].values[-lookback_days:]
            latest_sequence = latest_prices.reshape(1, lookback_days, 1)
            if model_choice == 'Ensemble' and 'ensemble_model' in st.session_state:
                ensemble = st.session_state.ensemble_model
                future_pred = ensemble.predict(latest_sequence)
            else:
                future_pred = model.predict(latest_sequence)
            
            next_price = future_pred[0][0]
            price_change = next_price - current_price
            percent_change = (price_change / current_price) * 100
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"€{current_price:.2f}")
            
            with col2:
                st.metric("Predicted Next Price", f"€{next_price:.2f}")
            
            with col3:
                direction = "📈 UP" if price_change > 0 else "📉 DOWN"
                st.metric(
                    "Expected Change",
                    f"{direction}\n€{abs(price_change):.2f}",
                    delta=percent_change
                )
            
            # Detailed prediction info
            st.subheader("📋 Prediction Details")
            
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.info(f"""
                **Current Price:** €{current_price:.2f}
                
                **Model Used:** {st.session_state.get('selected_model', 'Unknown')}
                
                **Lookback Window:** {lookback_days} days
                """)
            
            with pred_col2:
                st.success(f"""
                **Predicted Price:** €{next_price:.2f}
                
                **Price Change:** €{abs(price_change):.2f}
                
                **Percent Change:** {percent_change:.2f}%
                """)
            
            # Show last 60 days with prediction
            st.subheader("📊 Recent Prices & Prediction")
            
            recent_dates = data.index[-lookback_days:].tolist()
            recent_prices = data['Close'].values[-lookback_days:].tolist()
            
            # Add predicted point
            next_date = data.index[-1] + pd.Timedelta(days=1)
            all_dates = recent_dates + [next_date]
            all_prices = recent_prices + [next_price]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=all_dates[:-1],
                y=all_prices[:-1],
                name='Historical Prices',
                mode='lines+markers',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[all_dates[-2], all_dates[-1]],
                y=[all_prices[-2], all_prices[-1]],
                name='Prediction',
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title=f"{asset_name} - Next Day Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔄 Compare Models")
        
        if st.button("Train All 3 Models", key="compare_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Load models
            LSTMModel, RandomForestModel, SimpleLinearModel = load_models()
            
            models_dict = {
                "LSTM": LSTMModel(lookback_days=lookback_days, epochs=5, batch_size=16),
                "Random Forest": RandomForestModel(n_estimators=50, max_depth=20),
                "Linear": SimpleLinearModel()
            }
            
            results = {}
            predictions_dict = {}
            
            for idx, (model_name, model) in enumerate(models_dict.items()):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((idx + 1) / 3)
                
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
                results[model_name] = metrics
                predictions_dict[model_name] = y_pred
            
            progress_bar.progress(1.0)
            status_text.text("✅ Done!")

            # Ensemble
            if enable_ensemble:
                from stock_crypto_predictor.model import EnsembleModel
                ensemble = EnsembleModel(models=models_dict)
                weights = ensemble.fit_weights(y_test, predictions_dict)
                st.subheader("Ensemble Weights")
                for k, v in weights.items():
                    st.write(f"{k}: {v:.3f}")
                st.session_state.ensemble_model = ensemble
            
            # Results table
            st.subheader("Results")
            comparison_data = {
                'Model': list(results.keys()),
                'RMSE': [results[m]['RMSE'] for m in results.keys()],
                'MAE': [results[m]['MAE'] for m in results.keys()],
                'R²': [results[m]['R2'] for m in results.keys()],
                'MAPE': [results[m]['MAPE'] for m in results.keys()]
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_rmse = px.bar(pd.DataFrame(comparison_data), x='Model', y='RMSE', title='RMSE (Lower=Better)', color='RMSE', color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig_rmse, use_container_width=True)
            with col2:
                fig_r2 = px.bar(pd.DataFrame(comparison_data), x='Model', y='R²', title='R² (Higher=Better)', color='R²', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_r2, use_container_width=True)
            
            # Predictions overlay
            st.subheader("Predictions")
            fig = go.Figure()
            x_axis = np.arange(len(y_test))
            fig.add_trace(go.Scatter(x=x_axis, y=y_test.flatten(), name='Actual', mode='lines', line=dict(color='black', width=3)))
            
            colors = ['blue', 'red', 'green']
            for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
                fig.add_trace(go.Scatter(x=x_axis, y=predictions.flatten(), name=f'{model_name}', mode='lines', line=dict(color=colors[idx], width=2, dash='dash')))
            
            fig.update_layout(title=f"{asset_name} - Model Comparison", xaxis_title="Sample", yaxis_title="Price", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
            st.success(f"🏆 Best: {best_model} (RMSE: ${results[best_model]['RMSE']:.2f})")

# Run the app
if __name__ == "__main__":
    main()
