import streamlit as st
import pandas as pd
import sqlite3
import joblib

# Sayfa Yapılandırması
st.set_page_config(page_title="Finansal Zeka Dashboard", page_icon="📈", layout="wide")

st.title("📈 Finansal Zeka: Hisse Yönü ve Grafik Analiz Paneli")
st.markdown("Makine öğrenmesi modeli ve teknik göstergelerle profesyonel hisse senedi analiz paneli.")


# Modeli Yükle
@st.cache_resource
def load_model():
    return joblib.load('stock_model.pkl')


try:
    model = load_model()
except Exception as e:
    st.sidebar.error(f"Model yüklenirken hata oluştu: {e}")

# Yan Panel - Kullanıcı Girdileri
st.sidebar.header("Hisse Seçimi")
symbol = st.sidebar.text_input("Hisse Sembolü Girin (Örn: NVDA, AAPL, THYAO.IS):", "NVDA").upper()

if st.sidebar.button("Analiz Et ve Görselleştir"):
    conn = sqlite3.connect('finansal_zeka.db')

    # Tüm tarihsel veriyi çek (Grafik ve trend analizi için)
    query_history = f"SELECT * FROM stock_prices WHERE Symbol = '{symbol}' ORDER BY Date ASC"
    df_history = pd.read_sql_query(query_history, conn)
    conn.close()

    if not df_history.empty:
        # Tarih sütununu index yapalım
        df_history['Date'] = pd.to_datetime(df_history['Date'], utc=True)
        df_history.set_index('Date', inplace=True)

        # En son gün verisi (Model tahmini için)
        df_last = df_history.iloc[-1:]

        # Özellikleri Hazırla ve Tahmin Yap
        features = df_last[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Daily_Return']]
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        # Geçmiş getiri özeti; modelin gelecek fiyat değişimi tahmini değildir.
        recent_avg_return = df_history['Close'].pct_change().tail(10).mean() * 100

        # Üst Metrikler (4 Sütun)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Son Kapanış Fiyatı", f"${df_last['Close'].values[0]:.2f}")
        col2.metric("RSI (14)", f"{df_last['RSI'].values[0]:.2f}")

        if prediction == 1:
            col3.metric("Model Yön Tahmini", "🚀 YÜKSELİŞ (1)", delta=f"%{prob[1] * 100:.1f} Olasılık")
        else:
            col3.metric("Model Yön Tahmini", "📉 DÜŞÜŞ / NÖTR (0)", delta=f"-%{prob[0] * 100:.1f} Olasılık",
                        delta_color="inverse")

        col4.metric("Son 10 İşlem Dönemi Ort. Günlük Getiri", f"%{recent_avg_return:.2f}",
                    help="Geçmiş getirilerin aritmetik ortalamasıdır; gelecek getiri tahmini değildir.")

        st.divider()

        # Fiyat ve İndikatör Grafiği
        st.subheader(f"📊 {symbol} Fiyat ve Hareketli Ortalamalar (Son 90 Gün)")
        chart_data = df_history[['Close', 'SMA_20', 'SMA_50']].tail(90)
        st.line_chart(chart_data)

        # Detaylı JSON / Tablo Özeti
        st.subheader("📋 Model Analiz ve Metrik Özeti")
        st.json({
            "Hisse": symbol,
            "Son Tarih": str(df_last.index[0].date()),
            "SMA_20": round(df_last['SMA_20'].values[0], 2),
            "SMA_50": round(df_last['SMA_50'].values[0], 2),
            "RSI": round(df_last['RSI'].values[0], 2),
            "Yükseliş Olasılığı": f"%{prob[1] * 100:.2f}",
            "Son 10 İşlem Dönemi Ortalama Günlük Getiri": f"%{recent_avg_return:.2f}"
        })
    else:
        st.warning(f"'{symbol}' için veritabanında veri bulunamadı. Lütfen önce `data_pipeline.py` çalıştırın.")
