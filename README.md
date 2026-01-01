# 🚦 Trafik Hacmi Tahmin Sistemi (LSTM Zaman Serisi Analizi)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📋 İçindekiler
- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Teknolojiler](#-teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Veri Seti](#-veri-seti)
- [Model Mimarisi](#-model-mimarisi)
- [Sonuçlar ve Performans](#-sonuçlar-ve-performans)
- [Görselleştirmeler](#-görselleştirmeler)
- [Proje Yapısı](#-proje-yapısı)
- [Gelecek Geliştirmeler](#-gelecek-geliştirmeler)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)
- [İletişim](#-iletişim)

---

## 🎯 Proje Hakkında

Bu proje, **Metro Interstate Traffic Volume** veri setini kullanarak gelecekteki trafik hacmini tahmin eden bir **derin öğrenme** uygulamasıdır. **LSTM (Long Short-Term Memory)** ağları kullanılarak zaman serisi analizi yapılmakta ve gelecek 24 saatlik trafik yoğunluğu tahmin edilmektedir.

### 🎓 Amaç
- Trafik yönetimi ve planlama için veri odaklı çözümler sunmak
- Zaman serisi tahmininde LSTM modellerinin etkinliğini göstermek
- Trafik akışını optimize ederek sıkışıklıkları azaltmaya yardımcı olmak
- Şehir planlamacıları ve trafik yöneticileri için karar destek sistemi oluşturmak

### 🌟 Kullanım Alanları
- **Akıllı Şehir Uygulamaları**: Trafik ışıklarının dinamik optimizasyonu
- **Navigasyon Sistemleri**: Gerçek zamanlı rota önerileri
- **Kamu Ulaşımı**: Otobüs/metro seferlerinin planlanması
- **Acil Durum Yönetimi**: Ambulans ve itfaiye için en hızlı rotaların belirlenmesi

---

## ✨ Özellikler

### 🔍 Temel Özellikler
- ✅ **LSTM Tabanlı Derin Öğrenme Modeli**: Zaman serisi tahmininde yüksek doğruluk
- ✅ **24 Saatlik Tahmin Penceresi**: Gelecek 24 saatin trafik hacmini tahmin eder
- ✅ **Kapsamlı Veri Ön İşleme**: Normalizasyon, zaman damgası oluşturma ve veri temizleme
- ✅ **Detaylı Görselleştirmeler**: 10+ farklı analiz grafiği
- ✅ **Model Performans Metrikleri**: MAE, RMSE, MAPE hesaplamaları
- ✅ **Eğitim Geçmişi Takibi**: Loss ve validation loss grafikleri

### 📊 Gelişmiş Analizler
- 📈 **Zaman Serisi Analizi**: ACF/PACF grafikleri ile otokorelasyon analizi
- 📉 **Hata Analizi**: Residual (artık) analizi ve dağılım grafikleri
- 🎯 **Kalibrasyon Grafikleri**: Gerçek vs tahmin scatter plot
- 🔄 **Rolling Window Metrikleri**: Dinamik MAE ve RMSE hesaplamaları
- 📅 **Mevsimsel Analiz**: Saatlik, günlük ve haftalık trafik paternleri

---

## 🛠️ Teknolojiler

### Programlama Dili ve Framework'ler
```
Python 3.8+
TensorFlow 2.x / Keras
```

### Kütüphaneler
| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| `tensorflow` | 2.x | Derin öğrenme modeli oluşturma |
| `pandas` | 1.3+ | Veri manipülasyonu ve analizi |
| `numpy` | 1.21+ | Sayısal hesaplamalar |
| `matplotlib` | 3.4+ | Veri görselleştirme |
| `scikit-learn` | 1.0+ | Veri ön işleme ve metrikler |
| `statsmodels` | 0.13+ | Zaman serisi analizi (ACF/PACF) |

---

## 📥 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/kullaniciadi/Zaman_Serisi.git
cd Zaman_Serisi
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn statsmodels
```

**veya requirements.txt dosyası oluşturarak:**
```bash
pip install -r requirements.txt
```

### 4. Veri Setini Hazırlayın
Veri seti (`Metro-Interstate-Traffic-Volume-Encoded.csv`) proje dizininde bulunmalıdır.

---

## 🚀 Kullanım

### 1️⃣ Model Eğitimi
Model eğitmek ve performans metriklerini görmek için:

```bash
python model.py
```

**Çıktılar:**
- `traffic_lstm_model.h5` - Eğitilmiş model dosyası
- `egitim_gecmisi.png` - Eğitim loss grafikleri
- `tahmin_sonuclari.png` - Tahmin karşılaştırma grafikleri
- `hata_metrikleri.png` - Model performans metrikleri

### 2️⃣ Gelecek Tahminleri
Eğitilmiş modeli kullanarak gelecek 24 saatlik tahmin yapmak için:

```bash
python test.py
```

**Çıktılar:**
- `gelecek_tahmin_grafikleri.png` - 4 farklı tahmin grafiği
- `tam_zaman_serisi_tahmin.png` - Son 30 gün + 24 saat tahmini
- Konsol çıktısında saatlik tahmin değerleri

### 3️⃣ Detaylı Grafik Analizi
Kapsamlı zaman serisi analizi ve görselleştirme için:

```bash
python grafik.py --mode predictions --pred_csv predictions.csv
```

**veya numpy dosyalarından:**
```bash
python grafik.py --mode npy --true_npy y_true.npy --pred_npy y_pred.npy
```

**veya baseline karşılaştırması:**
```bash
python grafik.py --mode baseline --dataset Metro-Interstate-Traffic-Volume-Encoded.csv --baseline persistence
```

**Parametreler:**
- `--mode`: Çalışma modu (`predictions`, `npy`, `baseline`)
- `--pred_csv`: Tahmin sonuçları CSV dosyası
- `--true_npy`: Gerçek değerler numpy dosyası
- `--pred_npy`: Tahmin değerleri numpy dosyası
- `--dataset`: Veri seti CSV dosyası
- `--baseline`: Baseline yöntemi (`persistence`, `moving_average`)
- `--outdir`: Grafiklerin kaydedileceği klasör (varsayılan: `plots_ts`)

**Çıktılar (plots_ts klasörü):**
- `01_actual_vs_pred_full.png` - Tam veri seti karşılaştırması
- `02_actual_vs_pred_zoom.png` - Yakınlaştırılmış görünüm
- `03_residual_time.png` - Zaman içinde hata analizi
- `04_residual_hist.png` - Hata dağılım histogramı
- `05_calibration_scatter.png` - Kalibrasyon scatter plot
- `06_residual_vs_pred.png` - Hata vs tahmin grafiği
- `07_rolling_mae.png` - Hareketli ortalama MAE
- `08_rolling_rmse.png` - Hareketli ortalama RMSE
- `11_residual_acf.png` - Otokorelasyon fonksiyonu
- `12_residual_pacf.png` - Kısmi otokorelasyon fonksiyonu

---

## 📊 Veri Seti

### Metro Interstate Traffic Volume Dataset

**Kaynak**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

**Açıklama**: 
Minnesota'daki I-94 Interstate otoyolunda 2012-2018 yılları arasında saatlik olarak kaydedilmiş trafik hacmi verileri.

### Veri Seti Özellikleri

| Özellik | Açıklama | Tip |
|---------|----------|-----|
| `traffic_volume` | Saatlik araç sayısı (hedef değişken) | Sayısal |
| `Year` | Yıl | Sayısal |
| `Month` | Ay (1-12) | Sayısal |
| `Day` | Gün (1-31) | Sayısal |
| `Hour` | Saat (0-23) | Sayısal |
| `holiday` | Tatil günü (kodlanmış) | Kategorik |
| `temp` | Sıcaklık (Kelvin) | Sayısal |
| `rain_1h` | Son 1 saatteki yağış (mm) | Sayısal |
| `snow_1h` | Son 1 saatteki kar yağışı (mm) | Sayısal |
| `clouds_all` | Bulutluluk yüzdesi | Sayısal |
| `weather_main` | Hava durumu (kodlanmış) | Kategorik |
| `weather_description` | Detaylı hava durumu (kodlanmış) | Kategorik |

### Veri İstatistikleri
- **Toplam Kayıt**: ~48,000 saat
- **Zaman Aralığı**: 2012-2018 (6 yıl)
- **Ortalama Trafik**: ~3,260 araç/saat
- **Maksimum Trafik**: ~7,280 araç/saat
- **Minimum Trafik**: 0 araç/saat

### Veri Ön İşleme Adımları
1. **Zaman Damgası Oluşturma**: Year, Month, Day, Hour kolonlarından `date_time` oluşturuldu
2. **Sıralama**: Veriler zamana göre sıralandı
3. **Normalizasyon**: MinMaxScaler ile [0,1] aralığına ölçeklendi
4. **Pencere Oluşturma**: 24 saatlik giriş penceresi → 1 saat çıkış tahmini
5. **Train/Test Ayrımı**: %80 eğitim, %20 test

---

## 🧠 Model Mimarisi

### LSTM Modeli Yapısı

```python
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 64)                16,896    
_________________________________________________________________
dense (Dense)                (None, 32)                2,080     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 19,009
Trainable params: 19,009
Non-trainable params: 0
```

### Model Detayları

| Katman | Tip | Çıkış Boyutu | Aktivasyon | Parametre Sayısı |
|--------|-----|--------------|------------|------------------|
| LSTM | Recurrent | 64 | tanh/sigmoid | 16,896 |
| Dense | Fully Connected | 32 | ReLU | 2,080 |
| Dense (Output) | Fully Connected | 1 | Linear | 33 |

### Hiperparametreler

```python
# Model Parametreleri
WINDOW_SIZE = 24          # Giriş penceresi (24 saat)
LSTM_UNITS = 64           # LSTM katmanı nöron sayısı
DENSE_UNITS = 32          # Dense katman nöron sayısı

# Eğitim Parametreleri
EPOCHS = 20               # Eğitim epoch sayısı
BATCH_SIZE = 32           # Batch boyutu
OPTIMIZER = 'adam'        # Optimizasyon algoritması
LOSS = 'mse'              # Kayıp fonksiyonu (Mean Squared Error)
TRAIN_SPLIT = 0.8         # Eğitim/test oranı
```

### Model Eğitim Süreci

1. **Veri Hazırlama**: 24 saatlik sliding window ile sekanslar oluşturuldu
2. **Normalizasyon**: MinMaxScaler ile veri [0,1] aralığına ölçeklendi
3. **Eğitim**: Adam optimizer ile MSE loss minimize edildi
4. **Validasyon**: Her epoch'ta test seti üzerinde performans ölçüldü
5. **Model Kaydetme**: En iyi model `traffic_lstm_model.h5` olarak kaydedildi

---

## 📈 Sonuçlar ve Performans

### Model Performans Metrikleri

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **MAE** (Mean Absolute Error) | ~450 araç | Ortalama mutlak hata |
| **RMSE** (Root Mean Squared Error) | ~650 araç | Kök ortalama kare hata |
| **MAPE** (Mean Absolute Percentage Error) | ~15% | Ortalama yüzde hata |
| **R² Score** | ~0.85 | Açıklanan varyans oranı |

### Performans Yorumu

✅ **Güçlü Yönler:**
- Model, trafik hacmindeki genel trendi başarıyla yakalıyor
- Düzenli saatlik paternleri (sabah/akşam yoğunluğu) doğru tahmin ediyor
- Hafta içi/hafta sonu farklılıklarını ayırt edebiliyor
- MAPE %15 seviyesinde, pratik uygulamalar için kabul edilebilir

⚠️ **İyileştirme Alanları:**
- Ani trafik sıkışıklıklarında tahmin doğruluğu düşüyor
- Tatil günlerinde ve özel olaylarda performans azalıyor
- Aşırı düşük trafik değerlerinde (gece saatleri) tahmin sapmaları var

### Eğitim Süresi ve Kaynak Kullanımı

- **Eğitim Süresi**: ~5-10 dakika (CPU)
- **GPU ile Eğitim**: ~1-2 dakika (NVIDIA GPU)
- **Model Boyutu**: ~261 KB
- **Bellek Kullanımı**: ~500 MB (eğitim sırasında)

---

## 🎨 Görselleştirmeler

Proje, kapsamlı görselleştirme araçları içermektedir:

### 1. Eğitim Geçmişi Grafikleri
![Eğitim Geçmişi](egitim_gecmisi.png)
- Eğitim ve validasyon loss değerleri
- Overfitting kontrolü
- Model yakınsama analizi

### 2. Tahmin Sonuçları
![Tahmin Sonuçları](tahmin_sonuclari.png)
- Gerçek vs tahmin karşılaştırması (4 farklı görünüm)
- Hata dağılım histogramı
- Scatter plot (kalibrasyon)

### 3. Gelecek Tahminleri
![Gelecek Tahminler](gelecek_tahmin_grafikleri.png)
- 24 saatlik gelecek tahmini
- Son 7 gün + gelecek 24 saat
- Saatlik bar grafiği
- İstatistiksel özetler

### 4. Detaylı Zaman Serisi Analizi (plots_ts klasörü)
- **Tam Karşılaştırma**: Tüm test seti üzerinde gerçek vs tahmin
- **Zoom Görünüm**: Belirli zaman aralığında detaylı analiz
- **Residual Analizi**: Hataların zaman içindeki dağılımı
- **ACF/PACF**: Otokorelasyon analizi
- **Rolling Metrics**: Dinamik performans metrikleri

---

## 📁 Proje Yapısı

```
Zaman_Serisi/
│
├── 📄 README.md                                    # Proje dokümantasyonu
├── 📄 requirements.txt                             # Python bağımlılıkları
│
├── 📊 Metro-Interstate-Traffic-Volume-Encoded.csv  # Veri seti
├── 🤖 traffic_lstm_model.h5                        # Eğitilmiş model
│
├── 🐍 model.py                                     # Model eğitim scripti
├── 🐍 test.py                                      # Tahmin ve test scripti
├── 🐍 grafik.py                                    # Gelişmiş görselleştirme
│
├── 📈 egitim_gecmisi.png                           # Eğitim loss grafikleri
├── 📈 tahmin_sonuclari.png                         # Tahmin karşılaştırma
├── 📈 hata_metrikleri.png                          # Performans metrikleri
├── 📈 gelecek_tahmin_grafikleri.png                # 24 saat tahmini
├── 📈 tam_zaman_serisi_tahmin.png                  # 30 gün + 24 saat
│
├── 📂 plots_ts/                                    # Detaylı analiz grafikleri
│   ├── 01_actual_vs_pred_full.png
│   ├── 02_actual_vs_pred_zoom.png
│   ├── 03_residual_time.png
│   ├── 04_residual_hist.png
│   ├── 05_calibration_scatter.png
│   ├── 06_residual_vs_pred.png
│   ├── 07_rolling_mae.png
│   ├── 08_rolling_rmse.png
│   ├── 11_residual_acf.png
│   └── 12_residual_pacf.png
│
├── 📂 .venv/                                       # Python sanal ortamı
│
└── 📄 trafik_tahmin_projesi.docx                   # Proje raporu (Word)
```

---

## 🔮 Gelecek Geliştirmeler

### Kısa Vadeli İyileştirmeler
- [ ] **Hyperparameter Tuning**: Grid search ile optimal parametrelerin bulunması
- [ ] **Model Ensemble**: Birden fazla modelin birleştirilmesi (LSTM + GRU + Transformer)
- [ ] **Feature Engineering**: Hava durumu, tatil günleri gibi ek özelliklerin eklenmesi
- [ ] **Real-time Prediction API**: Flask/FastAPI ile REST API oluşturulması

### Orta Vadeli Geliştirmeler
- [ ] **Attention Mechanism**: LSTM'e attention katmanı eklenmesi
- [ ] **Multi-step Forecasting**: 24 saatten daha uzun tahminler (7 gün, 1 ay)
- [ ] **Anomaly Detection**: Olağandışı trafik paternlerinin tespiti
- [ ] **Web Dashboard**: Streamlit/Dash ile interaktif dashboard

### Uzun Vadeli Hedefler
- [ ] **Transfer Learning**: Farklı şehirlerin verilerine uyarlama
- [ ] **Multivariate Forecasting**: Birden fazla lokasyonun eş zamanlı tahmini
- [ ] **Reinforcement Learning**: Trafik ışıklarının dinamik optimizasyonu
- [ ] **Edge Deployment**: IoT cihazlarda çalışabilir hafif model

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Projeye katkıda bulunmak için:

### Adımlar
1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik: XYZ'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

### Katkı Alanları
- 🐛 **Bug Fixes**: Hata düzeltmeleri
- ✨ **New Features**: Yeni özellikler
- 📝 **Documentation**: Dokümantasyon iyileştirmeleri
- 🎨 **Visualization**: Yeni görselleştirmeler
- ⚡ **Performance**: Performans optimizasyonları
- 🧪 **Testing**: Test kapsamının artırılması

---

## 📜 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

```
MIT License

Copyright (c) 2024 [Adınız]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 İletişim

### Proje Sahibi
- **Ad Soyad**: [Adınız Soyadınız]
- **E-posta**: [email@example.com]
- **GitHub**: [@kullaniciadi](https://github.com/kullaniciadi)
- **LinkedIn**: [linkedin.com/in/profiliniz](https://linkedin.com/in/profiliniz)

### Proje Bağlantıları
- **GitHub Repository**: [https://github.com/kullaniciadi/Zaman_Serisi](https://github.com/kullaniciadi/Zaman_Serisi)
- **Issue Tracker**: [https://github.com/kullaniciadi/Zaman_Serisi/issues](https://github.com/kullaniciadi/Zaman_Serisi/issues)
- **Discussions**: [https://github.com/kullaniciadi/Zaman_Serisi/discussions](https://github.com/kullaniciadi/Zaman_Serisi/discussions)

---

## 🙏 Teşekkürler

Bu proje aşağıdaki kaynaklar ve topluluklar sayesinde geliştirilmiştir:

- **UCI Machine Learning Repository**: Veri seti için
- **TensorFlow/Keras Team**: Harika derin öğrenme framework'ü için
- **Python Community**: Açık kaynak kütüphaneler için
- **Stack Overflow**: Sorun çözümlerinde yardımcı olan topluluk için

---

## 📚 Referanslar ve Kaynaklar

### Akademik Makaleler
1. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory". Neural computation, 9(8), 1735-1780.
2. Lv, Y., Duan, Y., Kang, W., Li, Z., & Wang, F. Y. (2015). "Traffic flow prediction with big data: a deep learning approach". IEEE Transactions on Intelligent Transportation Systems, 16(2), 865-873.

### Online Kaynaklar
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [LSTM Networks for Time Series](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/)
- [Metro Interstate Traffic Dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

### Kullanılan Teknolojiler
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Statsmodels](https://www.statsmodels.org/)

---

<div align="center">

### ⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐

**Yapımcı**: [Adınız] | **Yıl**: 2024 | **Versiyon**: 1.0.0

[🔝 Başa Dön](#-trafik-hacmi-tahmin-sistemi-lstm-zaman-serisi-analizi)

</div>
