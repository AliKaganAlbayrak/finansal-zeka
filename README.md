# Finansal Zeka

Teknik göstergeler ve basit bir makine öğrenmesi modeli kullanarak seçili hisselerin kısa vadeli yönünü inceleyen bir Streamlit öğrenci projesi.

## Amaç

Proje, Yahoo Finance üzerinden alınan tarihsel fiyat verilerini işleyerek `NVDA`, `AAPL`, `MSFT` ve `THYAO.IS` sembolleri için bir sonraki işlem döneminde kapanış fiyatının mevcut kapanıştan %0,5'ten fazla yükselme olasılığını sınıflandırmayı dener. Streamlit paneli, modelin sınıf tahminini ve temel fiyat/gösterge grafiklerini gösterir.

Bu uygulama finansal yatırım tavsiyesi vermez. Çıktılar yalnızca eğitim ve portföy gösterimi amacıyla üretilir.

## Kullanılan teknolojiler

- Python 3.11 veya üzeri
- Streamlit: etkileşimli panel
- pandas: veri işleme ve özellik hesaplama
- yfinance: Yahoo Finance tarihsel fiyat verisi
- scikit-learn: Random Forest sınıflandırma modeli
- joblib: eğitilmiş modeli kaydetme ve yükleme
- SQLite: yerel veri saklama (`sqlite3`, Python standart kütüphanesi)

## Nasıl çalışır?

Proje üç adımdan oluşur:

1. `data_pipeline.py`, seçili sembollerin yaklaşık iki yıllık geçmişini indirir ve teknik göstergeleri hesaplayarak `finansal_zeka.db` içindeki `stock_prices` tablosuna yazar.
2. `model_training.py`, veritabanındaki etiketli kayıtları tarih sırasına göre eğitim ve test dönemlerine ayırır, bir `RandomForestClassifier` eğitir ve modeli `stock_model.pkl` olarak kaydeder.
3. `app.py`, kaydedilmiş modeli ve veritabanını yükler. Kullanıcının seçtiği sembol için son kaydın özellikleriyle yön tahmini üretir; kapanış, RSI, hareketli ortalamalar ve geçmiş ortalama getiriyi gösterir.

## Veri pipeline'ı

Genel akış şöyledir:

```text
Yahoo Finance
    -> tarihsel OHLCV verisi
    -> SMA_20, SMA_50, RSI, MACD, MACD_Signal, Daily_Return
    -> Target: sonraki kapanış mevcut kapanıştan %0,5'ten fazla yüksek mi?
    -> SQLite / stock_prices
    -> model_training.py
    -> stock_model.pkl
    -> Streamlit dashboard
```

`Target=1`, sonraki işlem döneminde kapanışın %0,5 eşiğini aştığı anlamına gelir. Sonraki kapanışı henüz bilinmeyen son kayıt tahmin için tutulur, eğitim ve testte kullanılmaz.

## Machine learning yaklaşımı

Model olarak `RandomForestClassifier` kullanılır. Model altı teknik özellikle iki sınıftan birini tahmin eder:

- `1`: sonraki kapanışta %0,5'ten fazla yükseliş
- `0`: bu eşiği aşmayan değişim

Model 200 ağaç, `max_depth=8`, `random_state=42` ve `class_weight='balanced'` ayarlarıyla eğitilir. Eğitim/test ayrımı ortak tarih sınırı kullanır; değerlendirme accuracy ve classification report ile yapılır.

Panelde gösterilen son 10 işlem döneminin ortalama günlük getirisi modelin gelecek fiyat değişimi tahmini değildir; yalnızca geçmiş veriye ait özet bir istatistiktir.

## Temel feature'lar

| Feature | Açıklama |
| --- | --- |
| `SMA_20` | 20 dönemlik basit hareketli ortalama |
| `SMA_50` | 50 dönemlik basit hareketli ortalama |
| `RSI` | 14 dönemlik kazanç/kayıp oranından hesaplanan RSI |
| `MACD` | 12 ve 26 dönemlik üstel hareketli ortalamaların farkı |
| `MACD_Signal` | MACD'nin 9 dönemlik üstel hareketli ortalaması |
| `Daily_Return` | Önceki kapanışa göre günlük oransal değişim |

## Kurulum

Python 3.11 veya üzeri bir sanal ortam oluşturun ve bağımlılıkları yükleyin:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Çalıştırma

Verileri yeniden almak ve göstergeleri oluşturmak için:

```bash
python data_pipeline.py
```

Modeli yeniden eğitmek için:

```bash
python model_training.py
```

Paneli başlatmak için:

```bash
streamlit run app.py
```

Veri indirme ve model eğitimi ağ erişimi gerektirir. Repository'de bulunan mevcut `.db` ve `.pkl` dosyalarıyla panel doğrudan açılabilir; güncel veya düzeltilmiş çıktılar için ilk iki komutun sırasıyla çalıştırılması gerekir.

## Proje dosya yapısı

```text
finansal_zeka/
├── app.py                # Streamlit arayüzü ve tahmin ekranı
├── data_pipeline.py     # Veri indirme ve teknik gösterge hesaplama
├── model_training.py    # Model eğitimi ve değerlendirme
├── finansal_zeka.db     # Yerel SQLite fiyat/gösterge verisi
├── stock_model.pkl      # Kaydedilmiş Random Forest modeli
├── requirements.txt      # Python bağımlılıkları
└── .gitignore
```

## Sınırlamalar

- Veri kaynağı Yahoo Finance'tır; veri güncelliği ve veri sağlayıcısının koşulları garanti edilmez.
- Model yalnızca dört sembol ve tarihsel veriler üzerinde çalışır.
- Hedef, gelecekteki fiyatı veya getiriyi sayısal olarak tahmin etmez; yalnızca %0,5 eşiğine dayalı ikili sınıflandırma yapar.
- Accuracy tek başına yatırım performansını veya kârlılığı göstermez; işlem maliyeti, kayma ve portföy simülasyonu yoktur.
- Teknik göstergeler ve Random Forest seçimi eğitim amaçlıdır; kapsamlı model seçimi veya hiperparametre optimizasyonu yapılmamıştır.
- Uygulama yerel bir öğrenci projesidir ve güvenilmeyen kullanıcı girdileri için production seviyesinde sertleştirilmemiştir.
- Tarihsel veriler ve kaydedilmiş model, yeniden üretildiklerinde farklı sonuçlar verebilir.

Bu repository eğitim ve portföy amaçlı hazırlanmıştır. Finansal karar vermek için kullanılmamalıdır.
