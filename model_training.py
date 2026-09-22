import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_model():
    conn = sqlite3.connect('finansal_zeka.db')
    df = pd.read_sql_query("SELECT * FROM stock_prices", conn)
    conn.close()

    # Yeni Özellik Kümemiz (Features)
    feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Daily_Return']
    # Yerel işlem gününü koru; farklı piyasalar için ortak tarih sınırı kullan.
    df['Trade_Date'] = pd.to_datetime(df['Date'].str[:10], format='%Y-%m-%d')
    df = df.sort_values(['Symbol', 'Trade_Date'])
    df['Target_Date'] = df.groupby('Symbol')['Trade_Date'].shift(-1)
    # Eski veritabanındaki son satırlar 0 olsa bile eğitim/test dışında kalır.
    df = df.dropna(subset=feature_cols + ['Target', 'Target_Date'])
    dates = df['Trade_Date'].sort_values().unique()
    if len(dates) < 2:
        raise ValueError('Eğitim/test ayrımı için yeterli etiketli işlem günü yok.')
    test_start = dates[int(len(dates) * 0.8)]

    # Test dönemindeki fiyatla etiketlenen sınır satırlarını eğitimden çıkar.
    train_df = df[(df['Trade_Date'] < test_start) & (df['Target_Date'] < test_start)]
    test_df = df[df['Trade_Date'] >= test_start]
    if train_df.empty or test_df.empty:
        raise ValueError('Tarih ayrımından sonra eğitim veya test kümesi boş kaldı.')
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train, y_test = train_df['Target'].astype(int), test_df['Target'].astype(int)

    # Model Tanımlama ve Eğitme
    # class_weight='balanced' ile sınıf dengesizliğini önüne geçiyoruz
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Test
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 40)
    print(f"YENİ MODEL DOĞRULUK ORANI (Accuracy): %{acc * 100:.2f}")
    print("=" * 40)
    print("\nDetaylı Performans Raporu:\n")
    print(classification_report(y_test, y_pred))

    # Modeli Kaydet
    joblib.dump(model, 'stock_model.pkl')
    print("Yeni model başarıyla 'stock_model.pkl' olarak kaydedildi!")


if __name__ == "__main__":
    train_model()
