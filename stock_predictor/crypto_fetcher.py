"""
Module for fetching cryptocurrency price data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .indicators import prepare_indicators


class CryptoDataFetcher:
    """Fetches and preprocesses historical cryptocurrency data."""

    def __init__(self, ticker: str, period: str = "5y"):
        """
        Initialize the crypto data fetcher.
        
        Args:
            ticker: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Time period to fetch ('5y', '1y', '6mo', etc.)
        """
        self.ticker = ticker.upper()
        self.period = period
        self.data = None
        self.news_api_key = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data using yfinance.
        
        Returns:
            DataFrame with OHLCV data
        """
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching {self.period} of historical data for {self.ticker}... (Attempt {attempt + 1}/{max_retries})")
                self.data = yf.download(self.ticker, period=self.period, progress=False, timeout=30)
                
                if self.data is None or self.data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Could not fetch data for ticker: {self.ticker}")
                
                print(f"Successfully fetched {len(self.data)} days of data")
                return self.data
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)[:50]}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise ValueError(f"Failed to fetch data after {max_retries} attempts: {str(e)[:100]}")

    def set_news_api_key(self, api_key: str):
        """Set News API key to enable sentiment feature fetching."""
        self.news_api_key = api_key

    def prepare_features(self, lookback_days: int = 60) -> tuple:
        """
        Prepare features and labels for model training.
        
        Args:
            lookback_days: Number of previous days to use for prediction
            
        Returns:
            Tuple of (X, y) - features and labels
        """
        if self.data is None:
            self.fetch_data()

        # Prepare multivariate features using technical indicators
        sentiment = None
        try:
            if getattr(self, 'news_api_key', None):
                from .news_sentiment import get_sentiment_series_for_range
                start = self.data.index[0].to_pydatetime()
                end = self.data.index[-1].to_pydatetime()
                sentiment = get_sentiment_series_for_range(self.news_api_key, self.ticker, start, end)
        except Exception:
            sentiment = None

        ind = prepare_indicators(self.data, lookback=lookback_days, sentiment_series=sentiment)

        features = ind.values  # shape (n_samples, n_features)
        targets = self.data['Close'].values

        X, y = [], []
        for i in range(len(features) - lookback_days):
            X.append(features[i:(i + lookback_days)])
            y.append(targets[i + lookback_days])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        print(f"Prepared {len(X)} training samples with lookback={lookback_days} days and {X.shape[2]} features")
        return X, y

    def get_latest_sequence(self, lookback_days: int = 60) -> np.ndarray:
        """
        Get the latest sequence for making predictions.
        
        Args:
            lookback_days: Number of previous days to use
            
        Returns:
            Array of shape (1, lookback_days, 1)
        """
        if self.data is None:
            self.fetch_data()

        ind = prepare_indicators(self.data, lookback=lookback_days)
        features = ind.values[-lookback_days:]
        return features.reshape(1, lookback_days, features.shape[1])

    def get_current_price(self) -> float:
        """Get the most recent closing price."""
        if self.data is None:
            self.fetch_data()
        return self.data['Close'].iloc[-1]

    def get_price_stats(self) -> dict:
        """Get price statistics."""
        if self.data is None:
            self.fetch_data()
        
        prices = self.data['Close']
        return {
            'current': prices.iloc[-1],
            'min': prices.min(),
            'max': prices.max(),
            'mean': prices.mean(),
            '7d_change': ((prices.iloc[-1] - prices.iloc[-8]) / prices.iloc[-8] * 100) if len(prices) > 7 else 0,
            '30d_change': ((prices.iloc[-1] - prices.iloc[-31]) / prices.iloc[-31] * 100) if len(prices) > 30 else 0,
        }


def generate_synthetic_crypto_data(ticker: str, num_days: int = 1260, start_price: float = 40000):
    """
    Generate synthetic cryptocurrency price data.
    
    Args:
        ticker: Crypto symbol
        num_days: Number of days to generate
        start_price: Starting price
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=num_days, freq='D')
    
    # Generate realistic crypto price movement with higher volatility
    returns = np.random.normal(0.002, 0.04, num_days)  # Higher volatility for crypto
    prices = start_price * np.exp(np.cumsum(returns))
    
    volume = np.random.uniform(10000000, 50000000, num_days)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.02, 0.02, num_days)),
        'High': prices * (1 + np.random.uniform(0, 0.03, num_days)),
        'Low': prices * (1 - np.random.uniform(0, 0.03, num_days)),
        'Close': prices,
        'Volume': volume
    }, index=dates)
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data
