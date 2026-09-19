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
    X = df[feature_cols]
    y = df['Target']

    # Zaman serisi yapısına uygun olarak veriyi bölme (Son %20 test verisi)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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