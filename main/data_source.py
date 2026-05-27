import os

import yfinance as yf
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_stock_data(stock_name: str, duration: str, period: str) -> pd.DataFrame:
    """Download historical stock data and save it to CSV.
    Returns:
        DataFrame of the downloaded OHLCV data.
    """
    fetched_data = yf.download(stock_name, period=duration, interval=period)
    fetched_data = fetched_data.droplevel("Ticker", axis=1)
    csv_path = os.path.join(BASE_DIR, "processed_data/historical_data.csv")
    fetched_data.to_csv(csv_path)
    return fetched_data

def add_technical_indicators(stock_name: str) -> pd.DataFrame:
    from constant import INPUT_CSV
    csv_path = os.path.join(BASE_DIR, INPUT_CSV)
    df = pd.read_csv(csv_path)
    df['stock_name'] = stock_name
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['close_lag1'] = df['Close'].shift(1)
    df['close_lag2'] = df['Close'].shift(2)
    df['close_lag3'] = df['Close'].shift(3)
    df['volume_lag1'] = df['Volume'].shift(1)

    df['ema_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['returns'] = df['Close'].pct_change()
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['vol_ma_10'] = df['Volume'].rolling(10).mean()
    df['vol_change'] = df['Volume'].pct_change()
    df['prev_high'] = df['High'].shift(2)
    df['prev_low'] = df['Low'].shift(2)

    df['bullish_fvg'] = (df['Low'] > df['prev_high']).astype(int)
    df['bearish_fvg'] = (df['High'] < df['prev_low']).astype(int)
    df['target_range'] = (df['High'].shift(-1) - df['Low'].shift(-1))
    df['ema_diff'] = df['ema_9'] - df['ema_21']
    df['rolling_high_10'] = df['High'].shift(1).rolling(10).max()
    df['rolling_low_10'] = df['Low'].shift(1).rolling(10).min()
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['atr_14'] = df['range'].rolling(14).mean()
    df['target'] = df['High'].shift(-1) - df['Low'].shift(-1)
    df["vol_ma_10"] = df["Volume"].rolling(10).mean()
    df["volume_spike"] = (df["Volume"] > 1.5 * df["vol_ma_10"]).astype(int)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    processed_path = os.path.join(BASE_DIR, "processed_data/processed_data.csv")
    file_exists = os.path.isfile(processed_path)
    df.to_csv(processed_path,
              mode='a',
              header=not file_exists,
              index=False)
    print(f"CSV updated with technical indicators: {csv_path}")
    return df
