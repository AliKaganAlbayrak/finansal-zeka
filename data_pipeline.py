import yfinance as yf
import pandas as pd
import sqlite3

SYMBOLS = ["NVDA", "AAPL", "MSFT", "THYAO.IS",]
DB_NAME = "finansal_zeka.db"


def calculate_indicators(df):
    # Hareketli Ortalamalar
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD & Signal Line
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Günlük Getiri
    df['Daily_Return'] = df['Close'].pct_change()

    # Target (Hedef Değişken)
    next_close = df['Close'].shift(-1)
    df['Target'] = (next_close > df['Close'] * 1.005).astype('Int64').where(next_close.notna())

    # Son satır tahmin için korunur; henüz bilinmeyen hedefi NULL olarak saklanır.
    feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Daily_Return']
    return df.dropna(subset=['Close'] + feature_cols).copy()


def fetch_and_save_data():
    conn = sqlite3.connect(DB_NAME)

    for symbol in SYMBOLS:
        print(f"Veri çekiliyor: {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")

        if df.empty:
            continue

        df.reset_index(inplace=True)
        # Date sütununu string formata çevirelim (SQLite uyumu için)
        df['Date'] = df['Date'].astype(str)
        df['Symbol'] = symbol
        df = calculate_indicators(df)

        if symbol == SYMBOLS[0]:
            df.to_sql("stock_prices", conn, if_exists="replace", index=False)
        else:
            df.to_sql("stock_prices", conn, if_exists="append", index=False)

    conn.close()
    print("Tüm veriler yenilendi ve veritabanı başarıyla güncellendi!")


if __name__ == "__main__":
    fetch_and_save_data()
