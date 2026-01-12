"""
Module for fetching cryptocurrency price data using CoinGecko API (free, no key needed).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .indicators import prepare_indicators

try:
    from pycoingecko import CoinGeckoAPI
except ImportError:
    CoinGeckoAPI = None


class CoinGeckoFetcher:
    """Fetches cryptocurrency data from CoinGecko (free, no API key needed)."""

    def __init__(self, ticker: str, period: str = "5y", currency: str = "eur"):
        """
        Initialize CoinGecko fetcher.
        
        Args:
            ticker: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
            period: Time period ('5y', '1y', '6mo', '3mo', '1mo')
            currency: Currency for prices (e.g., 'eur', 'usd')
        """
        self.ticker = ticker.replace('-USD', '').replace('-EUR', '').upper()
        self.period = period
        self.currency = currency.lower()
        self.data = None
        self.news_api_key = None
        self.cg = CoinGeckoAPI() if CoinGeckoAPI else None

    def _get_coingecko_id(self):
        """Map ticker to CoinGecko ID."""
        ticker_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOT': 'polkadot',
            'XRP': 'ripple',
            'LTC': 'litecoin',
            'DOGE': 'dogecoin',
        }
        return ticker_map.get(self.ticker, self.ticker.lower())

    def _days_from_period(self):
        """Convert period string to number of days.
        Note: CoinGecko free tier limits to ~365 days max.
        """
        periods = {
            '5y': 365,      # CoinGecko free tier max is ~365 days, so cap here
            '1y': 365,
            '6mo': 180,
            '3mo': 90,
            '1mo': 30
        }
        # Always cap at 365 days for free tier compatibility
        days = periods.get(self.period, 365)
        return min(days, 365)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data from CoinGecko.
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.cg:
            raise ImportError("pycoingecko not installed. Install with: pip install pycoingecko")

        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching {self.period} of {self.ticker} data from CoinGecko... (Attempt {attempt + 1}/{max_retries})")
                
                coin_id = self._get_coingecko_id()
                days = self._days_from_period()
                
                # Fetch market data
                market_data = self.cg.get_coin_market_chart_by_id(
                    id=coin_id,
                    vs_currency=self.currency,
                    days=days,
                    interval='daily'
                )
                
                # Parse data
                prices = market_data['prices']  # [timestamp, price]
                volumes = market_data['market_caps']  # [timestamp, market_cap]
                
                dates = [datetime.fromtimestamp(p[0]/1000) for p in prices]
                closes = [p[1] for p in prices]
                
                # Create OHLCV DataFrame (C for close, V for volume approximation)
                df = pd.DataFrame({
                    'Open': closes,
                    'High': closes,
                    'Low': closes,
                    'Close': closes,
                    'Volume': [0] * len(closes)  # CoinGecko doesn't provide volume in this endpoint
                }, index=pd.DatetimeIndex(dates))
                
                # Add some OHLC variation (±1% from close)
                np.random.seed(42)
                noise = np.random.uniform(0.99, 1.01, len(df))
                df['Open'] = df['Close'] * noise
                df['High'] = df['Close'] * np.random.uniform(1.00, 1.01, len(df))
                df['Low'] = df['Close'] * np.random.uniform(0.99, 1.00, len(df))
                
                self.data = df
                print(f"Successfully fetched {len(self.data)} days of {self.ticker} data from CoinGecko")
                return self.data
                
            except Exception as e:
                error_str = str(e)
                print(f"Attempt {attempt + 1} failed: {error_str[:100]}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Provide helpful error message based on error type
                    if '10012' in error_str or 'time range' in error_str.lower():
                        raise ValueError(
                            f"CoinGecko Free Tier Limit: {self.ticker} query exceeded time range.\n"
                            f"→ Free tier limited to ~365 days of history\n"
                            f"→ For longer history, try Yahoo Finance data source\n"
                            f"→ Or select a shorter period (e.g., 1y instead of 5y)"
                        )
                    elif 'error' in error_str.lower():
                        raise ValueError(
                            f"CoinGecko API Error for {self.ticker}:\n{error_str[:250]}\n"
                            f"→ Try again in a few moments\n"
                            f"→ Or use Yahoo Finance data source instead"
                        )
                    else:
                        raise ValueError(
                            f"Failed to fetch {self.ticker} from CoinGecko after {max_retries} attempts:\n"
                            f"{error_str[:300]}\n"
                            f"→ Try using Yahoo Finance or Alpha Vantage instead"
                        )

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
