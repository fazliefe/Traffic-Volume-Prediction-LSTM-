import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1️⃣ Modeli ve veriyi yükle
# -----------------------------
from tensorflow.keras.models import load_model
model = load_model("traffic_lstm_model.h5", compile=False)

df = pd.read_csv("Metro-Interstate-Traffic-Volume-Encoded.csv")

# -----------------------------
# 2️⃣ Zaman serisi hazırlığı
# -----------------------------
df["date_time"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]])
df = df.sort_values("date_time")
df = df.drop(columns=["Year", "Month", "Day", "Hour"])
df.set_index("date_time", inplace=True)

# -----------------------------
# 3️⃣ Normalizasyon (aynı scaler mantığı)
# -----------------------------
data = df[["traffic_volume"]].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# 4️⃣ Son 24 saati al ve 24 saat ileri tahmin et
# -----------------------------
last_window = scaled_data[-24:]
forecast = []

for i in range(24):
    pred = model.predict(last_window.reshape(1, 24, 1))
    forecast.append(pred[0, 0])
    last_window = np.append(last_window[1:], pred)

# -----------------------------
# 5️⃣ Tahminleri orijinal ölçeğe döndür
# -----------------------------
forecast = np.array(forecast).reshape(-1, 1)
forecast_real = scaler.inverse_transform(forecast)

# -----------------------------
# 6️⃣ Son 7 günlük veriyi al (karşılaştırma için)
# -----------------------------
last_7_days = df["traffic_volume"].tail(168).values  # 7 gün * 24 saat = 168 saat
last_7_days_time = df.index[-168:]

# -----------------------------
# 7️⃣ Gelecek 24 saat için zaman damgası oluştur
# -----------------------------
forecast_time = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

# -----------------------------
# 8️⃣ Görselleştir - Kapsamlı Grafikler
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Grafik 1: Gelecek 24 saatlik tahmin
axes[0, 0].plot(range(1, 25), forecast_real, marker='o', linestyle='-', 
                color='orange', linewidth=2, markersize=8, label='Tahmin')
axes[0, 0].fill_between(range(1, 25), forecast_real.flatten(), alpha=0.3, color='orange')
axes[0, 0].set_title("Gelecek 24 Saatlik Trafik Hacmi Tahmini", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Saat Sonrası", fontsize=12)
axes[0, 0].set_ylabel("Araç Sayısı", fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].set_xticks(range(1, 25, 2))

# Grafik 2: Son 7 gün + Gelecek 24 saat
axes[0, 1].plot(range(-167, 1), last_7_days, label='Son 7 Gün (Gerçek)', 
                color='blue', linewidth=1.5, alpha=0.7)
axes[0, 1].plot(range(1, 25), forecast_real, marker='o', linestyle='--', 
                color='orange', linewidth=2, markersize=6, label='Gelecek 24 Saat (Tahmin)')
axes[0, 1].axvline(x=0, color='red', linestyle=':', linewidth=2, label='Şimdi')
axes[0, 1].set_title("Son 7 Gün + Gelecek 24 Saat Tahmini", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Saat (Son 7 günden itibaren)", fontsize=12)
axes[0, 1].set_ylabel("Araç Sayısı", fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_xticks(range(-168, 25, 24))

# Grafik 3: Günlük ortalama tahmin (24 saatlik döngü)
hourly_avg = forecast_real.flatten()
axes[1, 0].bar(range(1, 25), hourly_avg, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_title("Gelecek 24 Saatlik Trafik Hacmi (Bar Grafik)", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Saat (0-23)", fontsize=12)
axes[1, 0].set_ylabel("Araç Sayısı", fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_xticks(range(1, 25, 2))

# En yüksek ve en düşük değerleri işaretle
max_idx = np.argmax(hourly_avg) + 1
min_idx = np.argmin(hourly_avg) + 1
axes[1, 0].bar(max_idx, hourly_avg[max_idx-1], color='red', edgecolor='black', alpha=0.9)
axes[1, 0].bar(min_idx, hourly_avg[min_idx-1], color='green', edgecolor='black', alpha=0.9)
axes[1, 0].text(max_idx, hourly_avg[max_idx-1], f'Max\n{hourly_avg[max_idx-1]:.0f}', 
                ha='center', va='bottom', fontweight='bold')
axes[1, 0].text(min_idx, hourly_avg[min_idx-1], f'Min\n{hourly_avg[min_idx-1]:.0f}', 
                ha='center', va='top', fontweight='bold')

# Grafik 4: Tahmin istatistikleri
stats_data = {
    'Ortalama': np.mean(forecast_real),
    'Maksimum': np.max(forecast_real),
    'Minimum': np.min(forecast_real),
    'Standart Sapma': np.std(forecast_real)
}
stats_names = list(stats_data.keys())
stats_values = list(stats_data.values())
colors_stats = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
bars = axes[1, 1].bar(stats_names, stats_values, color=colors_stats, edgecolor='black', linewidth=2)
axes[1, 1].set_title("Tahmin İstatistikleri", fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel("Araç Sayısı", fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Değerleri çubukların üzerine yaz
for bar, value in zip(bars, stats_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('gelecek_tahmin_grafikleri.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gelecek tahmin grafikleri kaydedildi: gelecek_tahmin_grafikleri.png")

# -----------------------------
# 9️⃣ Ekstra: Zaman serisi grafiği (tüm veri + tahmin)
# -----------------------------
plt.figure(figsize=(16, 6))
# Son 30 günü göster
last_30_days = df["traffic_volume"].tail(720).values  # 30 gün * 24 saat
last_30_days_time = df.index[-720:]

plt.plot(range(len(last_30_days)), last_30_days, label='Son 30 Gün (Gerçek)', 
         color='blue', linewidth=1, alpha=0.7)
plt.plot(range(len(last_30_days), len(last_30_days) + 24), forecast_real, 
         marker='o', linestyle='--', color='orange', linewidth=2, 
         markersize=5, label='Gelecek 24 Saat (Tahmin)')
plt.axvline(x=len(last_30_days), color='red', linestyle=':', linewidth=2, label='Şimdi')
plt.title("Trafik Hacmi: Son 30 Gün + Gelecek 24 Saat Tahmini", fontsize=14, fontweight='bold')
plt.xlabel("Saat", fontsize=12)
plt.ylabel("Araç Sayısı", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tam_zaman_serisi_tahmin.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Tam zaman serisi grafiği kaydedildi: tam_zaman_serisi_tahmin.png")

# -----------------------------
# 🔟 Konsol çıktısı
# -----------------------------
print("\n" + "="*60)
print("🚗 GELECEK 24 SAATLİK TRAFİK HACMİ TAHMİNİ")
print("="*60)
for i, value in enumerate(forecast_real.flatten(), 1):
    print(f"   {i:02d}. saat → {value:.0f} araç")
print("="*60)
print(f"\n📊 TAHMİN İSTATİSTİKLERİ:")
print(f"   • Ortalama: {np.mean(forecast_real):.0f} araç")
print(f"   • Maksimum: {np.max(forecast_real):.0f} araç (Saat {np.argmax(forecast_real)+1})")
print(f"   • Minimum: {np.min(forecast_real):.0f} araç (Saat {np.argmin(forecast_real)+1})")
print(f"   • Standart Sapma: {np.std(forecast_real):.0f} araç")
print("="*60)
