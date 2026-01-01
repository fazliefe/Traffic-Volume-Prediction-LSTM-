import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Opsiyonel ileri TS analizleri
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

# -------------------------
# Varsayılanlar (Sizin yol)
# -------------------------
DEFAULT_DATASET_PATH = r"C:\Users\USER\Desktop\projeler1\gite yüklenecekler\Zaman_Serisi\Metro-Interstate-Traffic-Volume-Encoded.csv"
DEFAULT_TIME_COL = "date_time"
DEFAULT_TARGET_COL = "traffic_volume"

# -------------------------
# Yardımcılar
# -------------------------
def ensure_dir(d="plots_ts"):
    os.makedirs(d, exist_ok=True)
    return d

def to_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)

def metrics(y_true, y_pred):
    y_true = to_1d(y_true)
    y_pred = to_1d(y_pred)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100.0)
    return mae, rmse, mape

def safe_datetime_index(x):
    if x is None:
        return None
    try:
        return pd.to_datetime(x)
    except Exception:
        return None

def load_from_predictions_csv(path, true_col="y_true", pred_col="y_pred", time_col=None):
    df = pd.read_csv(path)
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(
            f"CSV kolonları bulunamadı. Beklenen: {true_col}, {pred_col}. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    dt_index = None
    if time_col and time_col in df.columns:
        dt_index = safe_datetime_index(df[time_col].values)
    return y_true, y_pred, dt_index, df

def load_from_npy(true_path, pred_path, time_path=None):
    y_true = np.load(true_path, allow_pickle=True)
    y_pred = np.load(pred_path, allow_pickle=True)
    dt_index = None
    if time_path:
        dt_raw = np.load(time_path, allow_pickle=True)
        dt_index = safe_datetime_index(dt_raw)
    return to_1d(y_true), to_1d(y_pred), dt_index

def load_dataset_make_baseline(
    dataset_csv: str,
    target_col: str = DEFAULT_TARGET_COL,
    time_col: str = DEFAULT_TIME_COL,
    baseline: str = "persistence",
    ma_window: int = 24,
    sort_by_time: bool = True,
    dropna: bool = True,
):
    df = pd.read_csv(dataset_csv)

    if target_col not in df.columns:
        raise ValueError(
            f"Target kolonu bulunamadı: {target_col}. Mevcut kolonlar: {list(df.columns)}"
        )

    dt_index = None
    if time_col and time_col in df.columns:
        dt_index = safe_datetime_index(df[time_col].values)
        if dt_index is not None:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            if sort_by_time:
                df = df.sort_values(time_col)

    y_true = df[target_col].astype(float).values

    # Baseline tahmin üretimi
    baseline = (baseline or "").lower().strip()
    if baseline == "persistence":
        # y_hat[t] = y[t-1]
        y_pred = pd.Series(y_true).shift(1).values
    elif baseline in ("moving_avg", "moving-average", "ma"):
        # y_hat[t] = rolling_mean(y)[t-1] -> ileri sızıntıyı önlemek için shift
        y_pred = (
            pd.Series(y_true)
            .rolling(ma_window, min_periods=1)
            .mean()
            .shift(1)
            .values
        )
    else:
        raise ValueError("baseline sadece 'persistence' veya 'moving_avg' olabilir.")

    # İlk satır(lar) NaN olacağından hizalama
    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    if dt_index is not None and time_col in df.columns:
        out[time_col] = df[time_col].values
        dt_index_out = safe_datetime_index(out[time_col].values)
    else:
        dt_index_out = None

    if dropna:
        out = out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
        if dt_index_out is not None:
            dt_index_out = pd.to_datetime(out[time_col].values)

    return out["y_true"].values, out["y_pred"].values, dt_index_out, out

# -------------------------
# Grafik Üretimi
# -------------------------
def plot_all(
    y_true,
    y_pred,
    dt_index=None,
    outdir="plots_ts",
    zoom_window=24*7,
    rolling_window=200,
    seasonal_period=24,
):
    outdir = ensure_dir(outdir)

    y_true = to_1d(y_true)
    y_pred = to_1d(y_pred)

    n = min(len(y_true), len(y_pred))
    if n < 10:
        raise ValueError("y_true / y_pred çok kısa. En az 10 örnek gerekli.")

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    if dt_index is not None:
        dt_index = pd.to_datetime(dt_index)[:n]
        x = dt_index
        x_label = "Zaman"
        use_datetime = True
    else:
        x = np.arange(n)
        x_label = "Zaman Adımı"
        use_datetime = False

    err = y_true - y_pred
    abs_err = np.abs(err)

    mae, rmse, mape = metrics(y_true, y_pred)
    print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")
    print(f"Çıktı klasörü: {os.path.abspath(outdir)}")

    # 01) Full actual vs pred
    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true, label="Gerçek", linewidth=1)
    plt.plot(x, y_pred, label="Tahmin", linewidth=1)
    plt.title("Gerçek vs Tahmin (Tam Seri)")
    plt.xlabel(x_label)
    plt.ylabel("Değer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_actual_vs_pred_full.png"), dpi=200)
    plt.close()

    # 02) Zoom window
    if n > zoom_window:
        start = max(0, n // 3)
        end = min(n, start + zoom_window)
        plt.figure(figsize=(12, 4))
        plt.plot(x[start:end], y_true[start:end], label="Gerçek", linewidth=1.5)
        plt.plot(x[start:end], y_pred[start:end], label="Tahmin", linewidth=1.5)
        plt.title(f"Gerçek vs Tahmin (Yakınlaştırma: {zoom_window} örnek)")
        plt.xlabel(x_label)
        plt.ylabel("Değer")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "02_actual_vs_pred_zoom.png"), dpi=200)
        plt.close()

    # 03) Residual over time
    plt.figure(figsize=(12, 4))
    plt.plot(x, err, linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.title("Residual (Hata) Zaman Serisi: y - y_hat")
    plt.xlabel(x_label)
    plt.ylabel("Hata")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_residual_time.png"), dpi=200)
    plt.close()

    # 04) Residual histogram
    plt.figure(figsize=(8, 4))
    plt.hist(err, bins=60)
    plt.title("Residual Histogramı")
    plt.xlabel("Hata")
    plt.ylabel("Frekans")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_residual_hist.png"), dpi=200)
    plt.close()

    # 05) Calibration scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.35)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linewidth=1.5)
    plt.title("Kalibrasyon: Gerçek vs Tahmin (y=x referansı)")
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "05_calibration_scatter.png"), dpi=200)
    plt.close()

    # 06) Residual vs Pred
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, err, s=6, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.title("Tahmin vs Residual")
    plt.xlabel("Tahmin (y_hat)")
    plt.ylabel("Residual (y - y_hat)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_residual_vs_pred.png"), dpi=200)
    plt.close()

    # 07) Rolling MAE
    roll_mae = pd.Series(abs_err).rolling(rolling_window, min_periods=1).mean().values
    plt.figure(figsize=(12, 4))
    plt.plot(x, roll_mae, linewidth=1.5)
    plt.title(f"Rolling MAE (pencere={rolling_window})")
    plt.xlabel(x_label)
    plt.ylabel("MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "07_rolling_mae.png"), dpi=200)
    plt.close()

    # 08) Rolling RMSE
    roll_rmse = (
        pd.Series(err**2).rolling(rolling_window, min_periods=1).mean().pow(0.5).values
    )
    plt.figure(figsize=(12, 4))
    plt.plot(x, roll_rmse, linewidth=1.5)
    plt.title(f"Rolling RMSE (pencere={rolling_window})")
    plt.xlabel(x_label)
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "08_rolling_rmse.png"), dpi=200)
    plt.close()

    # 09-10) Saat/Gün profili (timestamp varsa)
    if use_datetime:
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=x)
        df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
        df["hour"] = df.index.hour
        df["dow"] = df.index.dayofweek  # 0=Pzt

        hour_mae = df.groupby("hour")["abs_err"].mean()
        plt.figure(figsize=(10, 4))
        plt.plot(hour_mae.index, hour_mae.values, marker="o", linewidth=1.5)
        plt.title("Saat Bazında Ortalama Mutlak Hata (MAE)")
        plt.xlabel("Saat")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "09_mae_by_hour.png"), dpi=200)
        plt.close()

        dow_mae = df.groupby("dow")["abs_err"].mean()
        plt.figure(figsize=(8, 4))
        plt.bar(dow_mae.index.astype(int), dow_mae.values)
        plt.title("Haftanın Günü Bazında MAE (0=Pzt ... 6=Paz)")
        plt.xlabel("Gün")
        plt.ylabel("MAE")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "10_mae_by_dayofweek.png"), dpi=200)
        plt.close()

    # 11-13) ACF/PACF + Seasonal decomposition (statsmodels varsa)
    if STATSMODELS_OK:
        # ACF
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(err, lags=min(60, n // 2), zero=False, ax=ax)
        ax.set_title("Residual ACF")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "11_residual_acf.png"), dpi=200)
        plt.close(fig)

        # PACF
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(err, lags=min(60, n // 2), zero=False, method="ywm", ax=ax)
        ax.set_title("Residual PACF")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "12_residual_pacf.png"), dpi=200)
        plt.close(fig)

        # Decomposition (timestamp varsa ve veri yeterliyse)
        if use_datetime and n >= seasonal_period * 4:
            try:
                series = pd.Series(y_true, index=x)
                decomp = seasonal_decompose(series, model="additive", period=seasonal_period)
                fig = decomp.plot()
                fig.set_size_inches(12, 8)
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, "13_seasonal_decompose.png"), dpi=200)
                plt.close(fig)
            except Exception:
                pass

# -------------------------
# CLI / Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Zaman serisi tahmin grafikleri: residual, zoom, rolling MAE/RMSE, calibration, ACF/PACF, decomposition."
    )

    # A) Model çıktı CSV (y_true/y_pred)
    p.add_argument("--pred-csv", type=str, default=None, help="Tahmin CSV (y_true/y_pred kolonları).")
    p.add_argument("--true-col", type=str, default="y_true", help="Tahmin CSV true kolon adı.")
    p.add_argument("--pred-col", type=str, default="y_pred", help="Tahmin CSV pred kolon adı.")
    p.add_argument("--time-col", type=str, default=None, help="Tahmin CSV timestamp kolon adı (opsiyonel).")

    # B) Ham dataset CSV -> baseline üret
    p.add_argument("--data-csv", type=str, default=None, help="Ham dataset CSV (örn. Metro-Interstate...).")
    p.add_argument("--dataset-time-col", type=str, default=DEFAULT_TIME_COL, help="Dataset timestamp kolonu (vars: date_time).")
    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL, help="Dataset hedef kolonu (vars: traffic_volume).")
    p.add_argument("--baseline", type=str, default="persistence", choices=["persistence", "moving_avg"],
                   help="Ham datasetten üretilecek baseline tahmin tipi.")
    p.add_argument("--ma-window", type=int, default=24, help="moving_avg için rolling pencere (örn saatlikte 24).")

    # C) NPY opsiyonu (eski)
    p.add_argument("--true-npy", type=str, default=None, help="y_true .npy yolu (CSV yoksa).")
    p.add_argument("--pred-npy", type=str, default=None, help="y_pred .npy yolu (CSV yoksa).")
    p.add_argument("--time-npy", type=str, default=None, help="timestamp .npy yolu (opsiyonel).")

    # Genel
    p.add_argument("--outdir", type=str, default="plots_ts", help="Çıktı klasörü.")
    p.add_argument("--zoom-window", type=int, default=24*7, help="Zoom pencere boyutu.")
    p.add_argument("--rolling-window", type=int, default=200, help="Rolling metrik pencere boyutu.")
    p.add_argument("--seasonal-period", type=int, default=24, help="Decomposition period (ör. saatlik veri için 24).")
    p.add_argument("--save-preds", action="store_true", help="Kullanılan y_true/y_pred'i outdir altına predictions_used.csv olarak kaydet.")
    return p.parse_args()

def main():
    args = parse_args()

    # 0) Hiçbir şey verilmediyse: sizin dataset yolunu otomatik dene
    if args.pred_csv is None and args.data_csv is None and args.true_npy is None and args.pred_npy is None:
        if Path(DEFAULT_DATASET_PATH).exists():
            args.data_csv = DEFAULT_DATASET_PATH
        elif Path("predictions.csv").exists():
            args.pred_csv = "predictions.csv"
        elif Path("preds.csv").exists():
            args.pred_csv = "preds.csv"

    # 1) Veri yükle (öncelik: pred-csv > data-csv > npy)
    used_df = None

    if args.pred_csv:
        y_true, y_pred, dt_index, used_df = load_from_predictions_csv(
            args.pred_csv,
            true_col=args.true_col,
            pred_col=args.pred_col,
            time_col=args.time_col,
        )
    elif args.data_csv:
        y_true, y_pred, dt_index, used_df = load_dataset_make_baseline(
            args.data_csv,
            target_col=args.target_col,
            time_col=args.dataset_time_col,
            baseline=args.baseline,
            ma_window=args.ma_window,
            sort_by_time=True,
            dropna=True,
        )
    else:
        if not args.true_npy or not args.pred_npy:
            raise SystemExit(
                "Veri bulunamadı.\n"
                "Çözüm A: Tahmin CSV ile: python grafik_ts.py --pred-csv predictions.csv --time-col date_time\n"
                "Çözüm B: Ham dataset ile baseline: python grafik_ts.py --data-csv \"...Encoded.csv\" --target-col traffic_volume --dataset-time-col date_time\n"
                "Çözüm C: NPY ile: python grafik_ts.py --true-npy y_true.npy --pred-npy y_pred.npy"
            )
        y_true, y_pred, dt_index = load_from_npy(args.true_npy, args.pred_npy, args.time_npy)

    # 2) İstenirse kullanılan y_true/y_pred kaydet
    if args.save_preds and used_df is not None:
        outdir = ensure_dir(args.outdir)
        out_path = os.path.join(outdir, "predictions_used.csv")
        used_df.to_csv(out_path, index=False)
        print(f"Kullanılan seri kaydedildi: {out_path}")

    # 3) Grafik üret
    plot_all(
        y_true,
        y_pred,
        dt_index=dt_index,
        outdir=args.outdir,
        zoom_window=args.zoom_window,
        rolling_window=args.rolling_window,
        seasonal_period=args.seasonal_period,
    )
    print("Bitti. plots_ts klasörünü kontrol et.")

if __name__ == "__main__":
    main()
