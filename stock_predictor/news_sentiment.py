"""
News fetching and sentiment analysis utilities.

Supports NewsAPI (requires API key) and local/text fallback.
Performs sentiment scoring using VADER and aggregates daily scores
to be merged into feature pipelines.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Optional


def fetch_news_newsapi(api_key: str, query: str, from_date: datetime, to_date: datetime, page_size: int = 100) -> List[dict]:
    """
    Fetch news articles from NewsAPI.org for a query between from_date and to_date.

    Returns list of article dicts. Requires a valid NewsAPI key.
    """
    url = "https://newsapi.org/v2/everything"
    headers = {"Authorization": api_key}
    params = {
        'q': query,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'pageSize': page_size,
        'sortBy': 'relevancy'
    }
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get('articles', [])


def analyze_sentiment_texts(texts: List[str]) -> List[float]:
    """Return compound sentiment scores for a list of texts using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for t in texts:
        if not t:
            scores.append(0.0)
            continue
        s = analyzer.polarity_scores(t)
        scores.append(s['compound'])
    return scores


def aggregate_daily_sentiment(articles: List[dict]) -> pd.Series:
    """
    Aggregate article sentiments by publication date and return a pandas Series
    indexed by date with average compound score for each day.
    """
    if not articles:
        return pd.Series(dtype=float)

    rows = []
    for a in articles:
        pub = a.get('publishedAt')
        # publishedAt is ISO8601 like '2023-01-01T12:34:56Z'
        try:
            dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
            text = ' '.join(filter(None, [a.get('title'), a.get('description'), a.get('content')]))
            rows.append({'date': dt.date(), 'text': text})
        except Exception:
            continue

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)
    df['sentiment'] = analyze_sentiment_texts(df['text'].tolist())
    daily = df.groupby('date')['sentiment'].mean()
    # return as Series indexed by datetime.date
    return daily


def get_sentiment_series_for_range(api_key: Optional[str], query: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Fetch articles in the date range and return a daily sentiment series.
    If api_key is None, returns an empty Series.
    """
    if not api_key:
        print("⚠️ No API key provided for sentiment analysis")
        return pd.Series(dtype=float)

    try:
        print(f"Fetching news for '{query}' from NewsAPI...")
        articles = fetch_news_newsapi(api_key, query, start_date, end_date)
        print(f"Got {len(articles)} articles from NewsAPI")
        
        daily = aggregate_daily_sentiment(articles)
        print(f"Aggregated to {len(daily)} daily sentiment records")
        
        if daily.empty:
            print("⚠️ No sentiment data after aggregation")
        
        return daily
    except requests.exceptions.HTTPError as e:
        print(f"❌ NewsAPI HTTP Error: {e}")
        if hasattr(e.response, 'json'):
            try:
                error_data = e.response.json()
                print(f"   API Response: {error_data}")
            except:
                pass
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"❌ Error fetching sentiment: {str(e)}")
        return pd.Series(dtype=float)
