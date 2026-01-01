# =============================
# 🚦 TRAFİK HACMİ TAHMİNİ (LSTM)
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------------
# 1️⃣ VERİYİ YÜKLE ve ÖN İŞLEME YAP
# -----------------------------------
df = pd.read_csv("Metro-Interstate-Traffic-Volume-Encoded.csv")

# Tarihi oluştur
df["date_time"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]])
df = df.sort_values("date_time")
df = df.drop(columns=["Year", "Month", "Day", "Hour"])
df.set_index("date_time", inplace=True)

print("✅ Veri başarıyla yüklendi.")
print(df.head())

# -----------------------------------
# 2️⃣ VERİYİ GÖRSELLEŞTİR (isteğe bağlı)
# -----------------------------------
plt.figure(figsize=(14,5))
plt.plot(df["traffic_volume"])
plt.title("Zaman Serisi: Trafik Hacmi (2012 - 2018)")
plt.xlabel("Zaman")
plt.ylabel("Araç Sayısı")
plt.show()

# -----------------------------------
# 3️⃣ LSTM MODELİ İÇİN VERİ HAZIRLA
# -----------------------------------
data = df[["traffic_volume"]].values

# Normalizasyon
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 24 saatlik pencere → 1 saat sonrası tahmin
window_size = 24
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("🔹 X şekli:", X.shape)
print("🔹 y şekli:", y.shape)

# -----------------------------------
# 4️⃣ EĞİTİM ve TEST AYRIMI
# -----------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------------
# 5️⃣ LSTM MODELİNİ OLUŞTUR
# -----------------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# -----------------------------------
# 6️⃣ MODELİ EĞİT
# -----------------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# -----------------------------------
# 7️⃣ MODEL TAHMİNİ
# -----------------------------------
predictions = model.predict(X_test)

# Ölçeklemeyi geri çevir
predictions = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------------
# 8️⃣ EĞİTİM GEÇMİŞİ GRAFİKLERİ
# -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss grafiği
axes[0].plot(history.history['loss'], label='Eğitim Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validasyon Loss', color='red')
axes[0].set_title('Model Loss (Eğitim vs Validasyon)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss farkı
axes[1].plot(history.history['loss'], label='Eğitim Loss', color='blue', linestyle='-')
axes[1].plot(history.history['val_loss'], label='Validasyon Loss', color='red', linestyle='-')
axes[1].fill_between(range(len(history.history['loss'])), 
                     history.history['loss'], 
                     history.history['val_loss'], 
                     alpha=0.3, color='gray')
axes[1].set_title('Loss Karşılaştırması')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('egitim_gecmisi.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Eğitim geçmişi grafiği kaydedildi: egitim_gecmisi.png")

# -----------------------------------
# 9️⃣ SONUÇLARI GÖRSELLEŞTİR
# -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Grafik 1: Genel karşılaştırma
axes[0, 0].plot(y_test_real[:500], label='Gerçek Trafik', color='blue', alpha=0.7)
axes[0, 0].plot(predictions[:500], label='Tahmin (LSTM)', color='red', linestyle='--', alpha=0.8)
axes[0, 0].set_title("Gerçek vs Tahmin Trafik Hacmi (İlk 500 Örnek)")
axes[0, 0].set_xlabel("Zaman Adımı")
axes[0, 0].set_ylabel("Araç Sayısı")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Grafik 2: Son 200 örnek detaylı
axes[0, 1].plot(y_test_real[-200:], label='Gerçek Trafik', color='blue', marker='o', markersize=3, alpha=0.7)
axes[0, 1].plot(predictions[-200:], label='Tahmin (LSTM)', color='red', marker='s', markersize=3, linestyle='--', alpha=0.8)
axes[0, 1].set_title("Gerçek vs Tahmin (Son 200 Örnek - Detaylı)")
axes[0, 1].set_xlabel("Zaman Adımı")
axes[0, 1].set_ylabel("Araç Sayısı")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Grafik 3: Hata dağılımı
errors = y_test_real.flatten() - predictions.flatten()
axes[1, 0].hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sıfır Hatası')
axes[1, 0].set_title("Hata Dağılımı (Gerçek - Tahmin)")
axes[1, 0].set_xlabel("Hata (Araç Sayısı)")
axes[1, 0].set_ylabel("Frekans")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Grafik 4: Scatter plot (Gerçek vs Tahmin)
axes[1, 1].scatter(y_test_real, predictions, alpha=0.5, s=10, color='green')
min_val = min(y_test_real.min(), predictions.min())
max_val = max(y_test_real.max(), predictions.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin')
axes[1, 1].set_title("Gerçek vs Tahmin Scatter Plot")
axes[1, 1].set_xlabel("Gerçek Trafik Hacmi")
axes[1, 1].set_ylabel("Tahmin Edilen Trafik Hacmi")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tahmin_sonuclari.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Tahmin sonuçları grafiği kaydedildi: tahmin_sonuclari.png")

# -----------------------------------
# 🔟 MODELİN HATA ANALİZİ
# -----------------------------------
mae = mean_absolute_error(y_test_real, predictions)
rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
mape = np.mean(np.abs((y_test_real - predictions) / y_test_real)) * 100

print(f"\n📊 MODEL PERFORMANS METRİKLERİ:")
print(f"   • MAE (Ortalama Mutlak Hata): {mae:.2f} araç")
print(f"   • RMSE (Kök Ortalama Kare Hata): {rmse:.2f} araç")
print(f"   • MAPE (Ortalama Mutlak Yüzde Hata): {mape:.2f}%")

# Hata metrikleri grafiği
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['MAE', 'RMSE']
values = [mae, rmse]
colors = ['skyblue', 'lightcoral']
bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
ax.set_title('Model Hata Metrikleri', fontsize=14, fontweight='bold')
ax.set_ylabel('Hata (Araç Sayısı)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Değerleri çubukların üzerine yaz
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('hata_metrikleri.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Hata metrikleri grafiği kaydedildi: hata_metrikleri.png")

print("\n✅ Model başarıyla eğitildi ve test edildi.")
model.save("traffic_lstm_model.h5")
print("✅ Model kaydedildi!")
