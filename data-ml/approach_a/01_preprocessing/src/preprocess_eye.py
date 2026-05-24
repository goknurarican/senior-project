#!/usr/bin/env python3
"""
BİTİRMEEG - Eye Tracking Preprocessing Pipeline
=================================================
Kullanım:
    python preprocess_eye.py                  # Tüm göz verisini işle
    python preprocess_eye.py --figures-only   # Sadece görseller

Bu script:
1. gaze_data_log.xlsx verisini yükler
2. Göz kırpma anlarını tespit eder (pupil = 0 veya çok küçük)
3. Kayıp verileri interpolasyonla doldurur
4. Outlier'ları temizler
5. Sunum için karşılaştırmalı grafikler üretir
6. Temizlenmiş veriyi kaydeder

Gereksinimler:
    pip install pandas numpy matplotlib scipy openpyxl
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from config import (
    BASE_DIR, FIGURES_DIR, PROCESSED_DIR, RAW_DIR,
    PUPIL_BLINK_THRESHOLD, PUPIL_SMOOTHING_WINDOW,
)

warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: YÜKLEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_gaze_data() -> pd.DataFrame:
    """Göz izleyici verisini yükle."""
    # Olası dosya isimleri
    possible_names = [
        "gaze_data_log.xlsx",
        "gaze_data_log (1).xlsx",
        "gaze_data_log.csv",
    ]

    gaze_path = None
    for name in possible_names:
        path = RAW_DIR / name
        if path.exists():
            gaze_path = path
            break

    if gaze_path is None:
        print("ERROR: gaze_data_log file not found!")
        print(f"Looked in: {RAW_DIR}")
        print(f"Tried: {possible_names}")
        sys.exit(1)

    print(f"Loading: {gaze_path}")

    if gaze_path.suffix == ".xlsx":
        df = pd.read_excel(gaze_path, engine="openpyxl")
    else:
        df = pd.read_csv(gaze_path)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  First few rows:")
    print(df.head(3).to_string())

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: BLINK TESPİTİ VE İNTERPOLASYON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_blinks(pupil_series: pd.Series, threshold: float = None) -> np.ndarray:
    """
    Göz kırpma anlarını tespit et.
    Pupil çapı 0 veya threshold altındaysa → blink.
    """
    if threshold is None:
        threshold = PUPIL_BLINK_THRESHOLD

    # NaN, 0, veya çok küçük değerler = blink
    blink_mask = (
        pupil_series.isna() |
        (pupil_series == 0) |
        (pupil_series < threshold)
    )

    n_blinks = blink_mask.sum()
    pct = n_blinks / len(pupil_series) * 100
    print(f"    Blinks detected: {n_blinks} ({pct:.1f}%)")

    return blink_mask.values


def interpolate_blinks(pupil_series: pd.Series, blink_mask: np.ndarray) -> pd.Series:
    """
    Göz kırpma anlarındaki kayıp verileri doğrusal interpolasyonla doldur.
    """
    result = pupil_series.copy()
    result[blink_mask] = np.nan

    # Doğrusal interpolasyon
    valid = ~result.isna()
    if valid.sum() < 2:
        print("    WARNING: Not enough valid data for interpolation")
        return result.fillna(0)

    x_valid = np.where(valid)[0]
    y_valid = result[valid].values

    # Tüm indeksler için interpolasyon
    f_interp = interp1d(
        x_valid, y_valid,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    result_interpolated = pd.Series(
        f_interp(np.arange(len(result))),
        index=result.index,
    )

    return result_interpolated


def smooth_pupil(pupil_series: pd.Series, window: int = None) -> pd.Series:
    """Median filtre ile gürültü temizleme."""
    if window is None:
        window = PUPIL_SMOOTHING_WINDOW
    if window % 2 == 0:
        window += 1  # scipy median_filter tek sayı ister

    smoothed = scipy_signal.medfilt(pupil_series.values, kernel_size=window)
    return pd.Series(smoothed, index=pupil_series.index)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: TÜM PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_gaze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Göz izleyici verisini temizle.
    Sütun isimleri cihaza göre değişebilir - otomatik tespit eder.
    """
    df_clean = df.copy()

    # Pupil çapı sütunlarını bul
    pupil_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["lpd", "rpd", "pupil", "diameter"]):
            pupil_cols.append(col)

    if not pupil_cols:
        print("  WARNING: No pupil diameter columns found!")
        print(f"  Available columns: {list(df.columns)}")
        # En azından sayısal sütunları dene
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  Numeric columns: {numeric_cols}")
        return df_clean

    print(f"  Pupil columns found: {pupil_cols}")

    for col in pupil_cols:
        print(f"\n  Processing: {col}")
        print(f"    Original stats: mean={df[col].mean():.4f}, "
              f"std={df[col].std():.4f}, zeros={( df[col] == 0).sum()}")

        # 1. Blink tespiti
        blink_mask = detect_blinks(df[col])

        # 2. İnterpolasyon
        interpolated = interpolate_blinks(df[col], blink_mask)

        # 3. Smoothing
        smoothed = smooth_pupil(interpolated)

        df_clean[f"{col}_raw"] = df[col]
        df_clean[f"{col}_blink_mask"] = blink_mask
        df_clean[col] = smoothed

        print(f"    Cleaned stats: mean={smoothed.mean():.4f}, "
              f"std={smoothed.std():.4f}, zeros={(smoothed == 0).sum()}")

    # Gaze koordinat sütunlarını bul
    gaze_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["bpogx", "bpogy", "gaze", "fixation"]):
            gaze_cols.append(col)

    if gaze_cols:
        print(f"\n  Gaze position columns: {gaze_cols}")

    return df_clean


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUNUM GÖRSELLERİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_eye_figures(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """Göz izleyici preprocessing görselleri."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Pupil sütununu bul
    pupil_col = None
    for col in df_raw.columns:
        if any(k in col.lower() for k in ["lpd", "rpd", "pupil"]):
            pupil_col = col
            break

    if pupil_col is None:
        print("  No pupil column found for figures")
        return

    raw_col_name = f"{pupil_col}_raw" if f"{pupil_col}_raw" in df_clean.columns else pupil_col

    # ─── KANIT 3: Ham vs İnterpolasyon Göz Bebeği ────────
    print("\n  [Eye Figure 1] Raw vs Interpolated Pupil...")

    # 500 sample'lık bir pencere göster (blink'lerin olduğu bölge)
    raw_pupil = df_raw[pupil_col].values if pupil_col in df_raw.columns else df_clean[raw_col_name].values
    clean_pupil = df_clean[pupil_col].values

    # Blink olan bölgeyi bul (göstermek için)
    if f"{pupil_col}_blink_mask" in df_clean.columns:
        blink_mask = df_clean[f"{pupil_col}_blink_mask"].values
        # İlk blink cluster'ını bul
        blink_indices = np.where(blink_mask)[0]
        if len(blink_indices) > 0:
            center = blink_indices[len(blink_indices) // 4]  # İlk çeyrekteki bir blink
            start = max(0, center - 250)
            end = min(len(raw_pupil), center + 250)
        else:
            start, end = 0, min(500, len(raw_pupil))
    else:
        start, end = 0, min(500, len(raw_pupil))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(start, end)

    axes[0].plot(x, raw_pupil[start:end], color="#E24B4A", linewidth=0.8, alpha=0.9)
    if f"{pupil_col}_blink_mask" in df_clean.columns:
        blink_region = blink_mask[start:end]
        axes[0].fill_between(x, raw_pupil[start:end].min(), raw_pupil[start:end].max(),
                            where=blink_region, alpha=0.2, color="orange",
                            label="Göz kırpma anları")
    axes[0].set_title(f"Ham Göz Bebeği Çapı ({pupil_col}) - Blink'ler ile Kesintili",
                     fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Çap (mm)")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, clean_pupil[start:end], color="#1D9E75", linewidth=0.8, alpha=0.9)
    axes[1].set_title("Temizlenmiş Göz Bebeği Çapı - İnterpolasyon + Smoothing",
                     fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Çap (mm)")
    axes[1].set_xlabel("Sample")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eye_01_raw_vs_interpolated.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR}/eye_01_raw_vs_interpolated.png")

    # ─── Tam Kayıt Görünümü ──────────────────────────────
    print("  [Eye Figure 2] Full recording overview...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(raw_pupil, color="#E24B4A", linewidth=0.3, alpha=0.6)
    axes[0].set_title("Ham Göz Bebeği - Tam Kayıt", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Çap (mm)")

    axes[1].plot(clean_pupil, color="#1D9E75", linewidth=0.3, alpha=0.6)
    axes[1].set_title("Temizlenmiş Göz Bebeği - Tam Kayıt", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Çap (mm)")
    axes[1].set_xlabel("Sample")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eye_02_full_recording.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR}/eye_02_full_recording.png")

    # ─── Gaze Heatmap (varsa) ────────────────────────────
    gaze_x_col = None
    gaze_y_col = None
    for col in df_clean.columns:
        cl = col.lower()
        if "bpogx" in cl or "gaze_x" in cl:
            gaze_x_col = col
        elif "bpogy" in cl or "gaze_y" in cl:
            gaze_y_col = col

    if gaze_x_col and gaze_y_col:
        print("  [Eye Figure 3] Gaze heatmap...")
        fig, ax = plt.subplots(figsize=(10, 8))

        x_data = df_clean[gaze_x_col].dropna()
        y_data = df_clean[gaze_y_col].dropna()

        # Aynı uzunluğa getir
        min_len = min(len(x_data), len(y_data))
        ax.hist2d(
            x_data.values[:min_len], y_data.values[:min_len],
            bins=50, cmap="YlOrRd", alpha=0.9,
        )
        ax.set_title("Bakış Noktası Isı Haritası", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (ekran koordinatı)")
        ax.set_ylabel("Y (ekran koordinatı)")
        ax.invert_yaxis()  # Ekran koordinatları Y aşağı doğru artar

        plt.colorbar(ax.collections[0], ax=ax, label="Bakış süresi (sample)")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eye_03_gaze_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {FIGURES_DIR}/eye_03_gaze_heatmap.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BİTİRMEEG Eye Tracking Preprocessing")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    # Yükle
    df = load_gaze_data()

    # İşle
    df_clean = preprocess_gaze(df)

    # Görseller
    generate_eye_figures(df, df_clean)

    if not args.figures_only:
        # Kaydet
        out_path = PROCESSED_DIR / "gaze_data_cleaned.csv"
        df_clean.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
