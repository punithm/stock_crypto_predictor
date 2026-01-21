"""
Module for fetching stock price data using Alpha Vantage API.
Requires free API key from https://www.alphavantage.co/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .indicators import prepare_indicators

try:
    import requests
except ImportError:
    requests = None


class AlphaVantageFetcher:
    """Fetches stock data from Alpha Vantage API (free tier available)."""

    def __init__(self, ticker: str, api_key: str = None, period: str = "5y"):
        """
        Initialize Alpha Vantage fetcher.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
            api_key: Alpha Vantage API key (get free from alphavantage.co)
            period: Time period ('5y', '1y', etc.) - note: AV returns last 100 data points
        """
        self.ticker = ticker.upper()
        self.api_key = api_key or "demo"  # Demo key has limited calls
        self.period = period
        self.data = None
        self.news_api_key = None
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data from Alpha Vantage.
        
        Returns:
            DataFrame with OHLCV data
        """
        if not requests:
            raise ImportError("requests library not installed")

        import time
        max_retries = 1  # Only 1 retry to fail faster

        for attempt in range(max_retries):
            try:
                api_key_display = self.api_key[:4] + "****" if len(self.api_key) > 4 else "****"
                print(f"Fetching {self.period} of {self.ticker} from Alpha Vantage (key: {api_key_display}) - Attempt {attempt + 1}/{max_retries}")

                # Use outputsize=compact for free tier (outputsize=full is premium-only)
                # Compact returns ~100 days, which is usually enough
                output_size = 'compact' if attempt == 0 else 'compact'
                
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': self.ticker,
                    'apikey': self.api_key,
                    'outputsize': output_size
                }

                response = requests.get(self.base_url, params=params, timeout=10)  # Reduced timeout from 30s to 10s
                response.raise_for_status()
                data_json = response.json()

                # Standard error checks
                if 'Error Message' in data_json:
                    error_msg = data_json['Error Message']
                    if "demo" in self.api_key.lower():
                        raise ValueError(f"Alpha Vantage error (using DEMO key): {error_msg}\n→ Get a free key: https://www.alphavantage.co/")
                    raise ValueError(f"API Error: {error_msg}")

                if 'Note' in data_json:
                    api_note = data_json['Note']
                    if "demo" in self.api_key.lower():
                        raise ValueError(f"Alpha Vantage API limit (using DEMO key): {api_note}\n→ Get a free key: https://www.alphavantage.co/")
                    raise ValueError(f"API Limit: {api_note}")

                # Handle informational responses that sometimes replace the time series
                if 'Information' in data_json:
                    info_msg = data_json['Information']
                    # Most common: outputsize=full is premium feature; we already use compact, so this is unexpected
                    print(f"Alpha Vantage Info: {info_msg[:150]}")
                    raise ValueError(
                        f"Alpha Vantage: {info_msg[:200]}\n"
                        f"→ API Key may be invalid or account restricted.\n"
                        f"→ Verify your key at: https://www.alphavantage.co/\n"
                        f"→ Or try CoinGecko for crypto (free, no key needed)"
                    )

                if 'Time Series (Daily)' not in data_json:
                    response_keys = list(data_json.keys()) if isinstance(data_json, dict) else []
                    if "demo" in self.api_key.lower():
                        raise ValueError(f"No data for ticker '{self.ticker}' with DEMO key.\n→ Get a free API key from: https://www.alphavantage.co/\n→ Response keys: {response_keys}")
                    raise ValueError(f"No data returned. Check ticker '{self.ticker}' or API key.\nResponse: {response_keys}")

                time_series = data_json['Time Series (Daily)']

                # Parse data
                dates = []
                opens = []
                highs = []
                lows = []
                closes = []
                volumes = []

                for date_str in sorted(time_series.keys()):
                    ts = time_series[date_str]
                    dates.append(pd.to_datetime(date_str))
                    opens.append(float(ts['1. open']))
                    highs.append(float(ts['2. high']))
                    lows.append(float(ts['3. low']))
                    closes.append(float(ts['4. close']))
                    volumes.append(float(ts['5. volume']))

                # Create DataFrame
                df = pd.DataFrame({
                    'Open': opens,
                    'High': highs,
                    'Low': lows,
                    'Close': closes,
                    'Volume': volumes
                }, index=pd.DatetimeIndex(dates))

                self.data = df
                print(f"Successfully fetched {len(self.data)} days of {self.ticker} data from Alpha Vantage")
                return self.data

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)[:120]}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise ValueError(f"Failed to fetch data after {max_retries} attempts: {str(e)}")

    def set_news_api_key(self, api_key: str):
        """Set News API key (for compatibility)."""
        self.news_api_key = api_key

    def prepare_features(self, lookback_days: int = 60) -> tuple:
        """
        Prepare features and labels for model training.
        
        Args:
            lookback_days: Number of days to use as context
            
        Returns:
            Tuple of (X, y) arrays
        """
        if self.data is None:
            raise ValueError("Must fetch data first")

        prices = self.data['Close'].values.reshape(-1, 1)
        X, y = [], []
        
        for i in range(len(prices) - lookback_days):
            X.append(prices[i:(i + lookback_days)])
            y.append(prices[i + lookback_days])

        return np.array(X), np.array(y)

    def get_latest_sequence(self, lookback_days: int = 60) -> np.ndarray:
        """Get the latest sequence for prediction."""
        if self.data is None:
            raise ValueError("Must fetch data first")

        prices = self.data['Close'].values
        if len(prices) < lookback_days:
            raise ValueError(f"Not enough data. Need at least {lookback_days} days")

        return prices[-lookback_days:].reshape(1, lookback_days, 1)

    def get_current_price(self) -> float:
        """Get the current (latest) price."""
        if self.data is None:
            raise ValueError("Must fetch data first")
        return self.data['Close'].iloc[-1]
