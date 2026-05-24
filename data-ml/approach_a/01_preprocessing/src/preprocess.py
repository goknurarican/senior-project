#!/usr/bin/env python3
"""
BİTİRMEEG - EEG Preprocessing Pipeline
========================================
Kullanım:
    python preprocess.py                    # Tüm subject'leri işle
    python preprocess.py --subject varyant_a  # Tek subject
    python preprocess.py --subject varyant_a --figures-only  # Sadece görseller

Bu script:
1. BrainVision EEG verisini yükler (vhdr referanslarını otomatik düzeltir)
2. Akselerometre kanallarını ayırır
3. 50Hz Notch + 1-45Hz Bandpass filtre uygular
4. ICA ile göz kırpma artefaktlarını temizler
5. Platform event'lerine göre epoklama yapar
6. Sunum için karşılaştırmalı grafikler üretir
7. Temizlenmiş veriyi kaydeder

Gereksinimler:
    pip install mne numpy pandas matplotlib scipy
"""

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalış
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from config import (
    ACCEL_CHANNELS, BANDPASS_HIGH, BANDPASS_LOW, BASE_DIR, EEG_CHANNELS,
    EPOCH_BASELINE, EPOCH_TMAX, EPOCH_TMIN, FEATURES_DIR, FIGURES_DIR,
    FREQ_BANDS, FRONTAL_CHANNELS, ICA_MAX_ITER, ICA_N_COMPONENTS,
    ICA_RANDOM_STATE, KEY_CHANNELS, N_EEG_CHANNELS, NOTCH_FREQ,
    PROCESSED_DIR, RAW_DIR, SFREQ, SUBJECTS, get_subject_eeg_start,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0: VHDR DOSYASINI DÜZELT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fix_vhdr_references(subject_name: str) -> Path:
    """
    vhdr dosyaları içinde DataFile=dhl_000092.eeg gibi orijinal isimler var.
    Bizim dosyalarımız yeniden adlandırıldı (control.eeg, varyant_a.eeg vs.)
    Bu fonksiyon geçici bir vhdr kopyası oluşturup referansları düzeltir.
    """
    info = SUBJECTS[subject_name]
    vhdr_path = RAW_DIR / info["vhdr"]
    eeg_file = info["eeg"]
    vmrk_file = info["vmrk"]

    # Geçici düzeltilmiş vhdr
    fixed_dir = PROCESSED_DIR / "temp"
    fixed_dir.mkdir(exist_ok=True)

    # eeg ve vmrk dosyalarını temp'e kopyala
    for src_name in [eeg_file, vmrk_file]:
        src = RAW_DIR / src_name
        if src.exists():
            dst = fixed_dir / src_name
            if not dst.exists():
                shutil.copy2(src, dst)

    # vhdr'yı oku ve DataFile/MarkerFile referanslarını düzelt
    with open(vhdr_path, "r", encoding="utf-8") as f:
        content = f.read()

    # DataFile ve MarkerFile satırlarını değiştir
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        clean = line.strip().replace("\r", "")
        if clean.startswith("DataFile="):
            new_lines.append(f"DataFile={eeg_file}")
        elif clean.startswith("MarkerFile="):
            new_lines.append(f"MarkerFile={vmrk_file}")
        else:
            new_lines.append(line.rstrip("\r"))

    fixed_vhdr = fixed_dir / info["vhdr"]
    with open(fixed_vhdr, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    return fixed_vhdr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: EEG VERİSİNİ YÜKLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_eeg(subject_name: str) -> mne.io.Raw:
    """BrainVision EEG verisini yükle, akselerometre kanallarını ayır."""
    print(f"\n{'='*60}")
    print(f"  Loading: {subject_name}")
    print(f"{'='*60}")

    fixed_vhdr = fix_vhdr_references(subject_name)
    raw = mne.io.read_raw_brainvision(str(fixed_vhdr), preload=True)

    print(f"  Channels: {raw.info['nchan']}")
    print(f"  Sfreq: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} s ({raw.times[-1]/60:.1f} min)")
    print(f"  Channel names: {raw.ch_names[:5]}... {raw.ch_names[-3:]}")

    # Akselerometre kanallarını misc olarak işaretle
    accel_picks = [ch for ch in raw.ch_names if ch in ACCEL_CHANNELS]
    if accel_picks:
        raw.set_channel_types({ch: "misc" for ch in accel_picks})
        print(f"  Accelerometer channels marked as misc: {accel_picks}")

    # EEG kanallarının tipini ayarla (bazen eksik olabiliyor)
    eeg_picks = [ch for ch in raw.ch_names if ch in EEG_CHANNELS]
    raw.set_channel_types({ch: "eeg" for ch in eeg_picks})

    # Montaj ayarla (10-20 pozisyonları)
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")
        print("  Montage: standard_1020 applied")
    except Exception as e:
        print(f"  Montage warning: {e}")

    return raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: FİLTRELEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_filters(raw: mne.io.Raw) -> mne.io.Raw:
    """50Hz Notch + 1-45Hz Bandpass filtre uygula."""
    print("\n  [Filter] Applying 50Hz Notch filter...")
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(
        freqs=NOTCH_FREQ,
        picks="eeg",
        method="spectrum_fit",
        filter_length="auto",
    )

    print(f"  [Filter] Applying {BANDPASS_LOW}-{BANDPASS_HIGH}Hz Bandpass filter...")
    raw_filtered.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        picks="eeg",
        method="fir",
        fir_design="firwin",
    )

    return raw_filtered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: ICA ARTEFAKTTEMİZLİĞİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_ica(raw_filtered: mne.io.Raw, subject_name: str) -> tuple:
    """ICA uygula, göz kırpma bileşenlerini tespit et ve çıkar."""
    print("\n  [ICA] Fitting ICA...")

    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        method="fastica",
    )

    # ICA için 1Hz highpass gerekli (daha iyi ayrıştırma)
    raw_for_ica = raw_filtered.copy().filter(l_freq=1.0, h_freq=None, picks="eeg")
    ica.fit(raw_for_ica, picks="eeg")
    print(f"  [ICA] Fitted {ica.n_components_} components")

    # Göz kırpma bileşenlerini otomatik tespit
    # Fp1 ve Fp2 frontal kanallarla korelasyon
    eog_indices = []
    for eog_ch in ["Fp1", "Fp2"]:
        if eog_ch in raw_filtered.ch_names:
            try:
                indices, scores = ica.find_bads_eog(
                    raw_filtered,
                    ch_name=eog_ch,
                    threshold=2.5,
                )
                eog_indices.extend(indices)
            except Exception:
                pass

    eog_indices = list(set(eog_indices))
    if not eog_indices:
        # Fallback: İlk 2 bileşen genellikle göz artefaktı
        eog_indices = [0, 1]
        print(f"  [ICA] Auto-detection failed, using default: {eog_indices}")
    else:
        print(f"  [ICA] Detected EOG components: {eog_indices}")

    ica.exclude = eog_indices

    # Temizlenmiş veriyi al
    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean)
    print(f"  [ICA] Removed {len(eog_indices)} components")

    return raw_clean, ica, eog_indices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: PLATFORM EVENT'LERİNDEN MARKER OLUŞTUR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_platform_events(subject_name: str) -> pd.DataFrame:
    """Platform events.csv'den ilgili session'ın olaylarını yükle."""
    events_path = RAW_DIR / "events.csv"
    if not events_path.exists():
        print("  [Events] events.csv not found!")
        return pd.DataFrame()

    df = pd.read_csv(events_path)
    session_id = SUBJECTS[subject_name].get("session_id")

    if session_id is None:
        print(f"  [Events] No session_id mapped for {subject_name}")
        return pd.DataFrame()

    # Session'a ait olayları filtrele
    session_events = df[df["session_id"] == session_id].copy()
    print(f"  [Events] Found {len(session_events)} events for session {session_id[:8]}...")

    # SCENARIO_TRIGGERED olaylarını parse et
    scenarios = session_events[session_events["event_type"] == "SCENARIO_TRIGGERED"].copy()
    if len(scenarios) > 0:
        parsed = []
        for _, row in scenarios.iterrows():
            try:
                data = json.loads(row["event_data"].replace('""', '"'))
                parsed.append({
                    "event_id": row["id"],
                    "timestamp_ms": data["details"]["timestamp"],
                    "scenario_name": data["details"]["name"],
                    "scenario_type": data["details"]["type"],
                    "page_url": row["page_url"],
                    "created_at": row["created_at"],
                })
            except (json.JSONDecodeError, KeyError):
                continue
        scenarios_df = pd.DataFrame(parsed)
        print(f"  [Events] Parsed {len(scenarios_df)} scenario triggers")
        return scenarios_df

    return pd.DataFrame()


def create_annotations_from_events(
    raw: mne.io.Raw,
    subject_name: str,
    scenarios_df: pd.DataFrame,
    timezone_offset_hours: float = 0.0,
) -> mne.io.Raw:
    """
    Platform event timestamp'lerini EEG zaman eksenine çevirerek
    MNE Annotations olarak ekle.

    Senkronizasyon mantığı:
    - EEG başlangıcı: vmrk'daki New Segment timestamp (ör: 2025-12-22 04:28:52)
    - Platform event: created_at (ör: 2025-12-22 12:43:37)
    - Fark = event_time - eeg_start → EEG'deki saniye pozisyonu
    - timezone_offset_hours: EEG UTC ise ve platform local ise fark (ör: 3.0 for UTC+3)
    """
    if scenarios_df.empty:
        print("  [Annotations] No scenarios to annotate")
        return raw

    eeg_start = get_subject_eeg_start(subject_name)
    eeg_duration = raw.times[-1]

    onsets = []
    descriptions = []
    durations = []

    for _, row in scenarios_df.iterrows():
        try:
            event_time = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue

        # Saat dilimi düzeltmesi
        event_time_adjusted = event_time - timedelta(hours=timezone_offset_hours)

        # EEG başlangıcından itibaren saniye
        offset_seconds = (event_time_adjusted - eeg_start).total_seconds()

        # EEG kaydı içinde mi kontrol et
        if 0 <= offset_seconds <= eeg_duration:
            onsets.append(offset_seconds)
            descriptions.append(f"scenario/{row['scenario_type']}")
            durations.append(3.0)  # Her senaryo ~3 saniye sürer
        else:
            print(f"    Skipped: {row['scenario_type']} at {offset_seconds:.1f}s "
                  f"(outside EEG range 0-{eeg_duration:.1f}s)")

    if onsets:
        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
        )
        raw.set_annotations(annotations)
        print(f"  [Annotations] Added {len(onsets)} scenario markers to EEG")
    else:
        print("  [Annotations] WARNING: No events fell within EEG recording window!")
        print(f"    EEG start: {eeg_start}")
        print(f"    EEG duration: {eeg_duration:.1f}s ({eeg_duration/60:.1f} min)")
        if len(scenarios_df) > 0:
            first_event = scenarios_df.iloc[0]["created_at"]
            print(f"    First event: {first_event}")
            print(f"    Try adjusting timezone_offset_hours (current: {timezone_offset_hours})")

    return raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: FEATURE EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_band_power(epoch_data: np.ndarray, sfreq: float, ch_names: list) -> dict:
    """
    Tek bir epoch'tan frekans bandı güçlerini (power) hesapla.
    Welch yöntemi ile PSD → her bant için ortalama güç.
    """
    features = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        for i, ch_name in enumerate(ch_names):
            if ch_name not in EEG_CHANNELS:
                continue
            freqs, psd = scipy_signal.welch(
                epoch_data[i], fs=sfreq, nperseg=min(256, epoch_data.shape[1])
            )
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psd[band_mask]) if band_mask.any() else 0.0
            features[f"{ch_name}_{band_name}_power"] = band_power

    # Frontal theta/beta ratio (bilişsel yük göstergesi)
    theta_powers = [features.get(f"{ch}_theta_power", 0) for ch in FRONTAL_CHANNELS
                    if f"{ch}_theta_power" in features]
    beta_powers = [features.get(f"{ch}_beta_power", 0) for ch in FRONTAL_CHANNELS
                   if f"{ch}_beta_power" in features]

    if theta_powers and beta_powers:
        avg_theta = np.mean(theta_powers)
        avg_beta = np.mean(beta_powers)
        features["frontal_theta_beta_ratio"] = avg_theta / avg_beta if avg_beta > 0 else 0

    return features


def extract_features_from_epochs(epochs: mne.Epochs) -> pd.DataFrame:
    """Tüm epoch'lardan feature matrix oluştur."""
    print("\n  [Features] Extracting features from epochs...")
    all_features = []

    for i, epoch_data in enumerate(epochs.get_data()):
        features = extract_band_power(epoch_data, epochs.info["sfreq"], epochs.ch_names)
        features["epoch_idx"] = i
        features["event_description"] = epochs.events[i, 2]  # Event ID
        if hasattr(epochs, "event_id") and epochs.event_id:
            # Event açıklamasını bul
            reverse_map = {v: k for k, v in epochs.event_id.items()}
            features["scenario_type"] = reverse_map.get(epochs.events[i, 2], "unknown")
        all_features.append(features)

    df = pd.DataFrame(all_features)
    print(f"  [Features] Extracted {len(df)} epochs × {len(df.columns)} features")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUNUM GÖRSELLERİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_figures(raw, raw_filtered, raw_clean, ica, eog_indices,
                     subject_name, scenarios_df=None):
    """Sunum için tüm kanıt grafiklerini üret."""

    fig_prefix = FIGURES_DIR / subject_name
    plt.style.use("seaborn-v0_8-whitegrid")

    # ─── KANIT 1: Ham vs Filtrelenmiş EEG ─────────────────
    print("\n  [Figure 1] Raw vs Filtered EEG...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 10 saniyelik segment göster (ortadan)
    t_start = raw.times[-1] / 2  # Kaydın ortası
    t_end = t_start + 10.0
    show_channels = ["Fz", "Cz", "Pz"]

    for ch in show_channels:
        if ch in raw.ch_names:
            idx = raw.ch_names.index(ch)
            times_mask = (raw.times >= t_start) & (raw.times <= t_end)

            # Ham sinyal
            raw_data = raw.get_data(picks=[idx])[0]
            axes[0].plot(raw.times[times_mask], raw_data[times_mask] * 1e6,
                        label=ch, alpha=0.8, linewidth=0.7)

            # Filtrelenmiş sinyal
            filt_data = raw_filtered.get_data(picks=[idx])[0]
            axes[1].plot(raw_filtered.times[times_mask], filt_data[times_mask] * 1e6,
                        label=ch, alpha=0.8, linewidth=0.7)

    axes[0].set_title("Ham EEG Sinyali (Filtrelenmemiş)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Genlik (µV)")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim([-150, 150])

    axes[1].set_title(f"Filtrelenmiş EEG ({NOTCH_FREQ}Hz Notch + {BANDPASS_LOW}-{BANDPASS_HIGH}Hz Bandpass)",
                      fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Genlik (µV)")
    axes[1].set_xlabel("Zaman (s)")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim([-150, 150])

    plt.tight_layout()
    fig.savefig(f"{fig_prefix}_01_raw_vs_filtered.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_prefix}_01_raw_vs_filtered.png")

    # ─── KANIT 2: ICA Topografik Haritalar ────────────────
    print("  [Figure 2] ICA Topographic maps...")
    try:
        fig_ica = ica.plot_components(
            picks=range(min(10, ica.n_components_)),
            show=False,
        )
        # plot_components bir liste döndürebilir
        if isinstance(fig_ica, list):
            for i, f in enumerate(fig_ica):
                f.savefig(f"{fig_prefix}_02_ica_components_{i}.png", dpi=200, bbox_inches="tight")
                plt.close(f)
        else:
            fig_ica.savefig(f"{fig_prefix}_02_ica_components.png", dpi=200, bbox_inches="tight")
            plt.close(fig_ica)
        print(f"    Saved: {fig_prefix}_02_ica_components*.png")
    except Exception as e:
        print(f"    ICA plot failed: {e}")

    # ICA bileşen zaman serileri (artefakt olanları işaretle)
    try:
        fig_sources = ica.plot_sources(raw_filtered, show=False, start=t_start, stop=t_end)
        fig_sources.savefig(f"{fig_prefix}_02b_ica_sources.png", dpi=200, bbox_inches="tight")
        plt.close(fig_sources)
        print(f"    Saved: {fig_prefix}_02b_ica_sources.png")
    except Exception as e:
        print(f"    ICA sources plot failed: {e}")

    # ─── KANIT 3: ICA Öncesi vs Sonrası ──────────────────
    print("  [Figure 3] Before vs After ICA...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ch in ["Fp1", "Fp2"]:  # Frontal - göz artefaktı en çok burada
        if ch in raw_filtered.ch_names:
            idx = raw_filtered.ch_names.index(ch)
            times_mask = (raw_filtered.times >= t_start) & (raw_filtered.times <= t_end)

            before_data = raw_filtered.get_data(picks=[idx])[0]
            axes[0].plot(raw_filtered.times[times_mask], before_data[times_mask] * 1e6,
                        label=ch, alpha=0.8, linewidth=0.7)

            after_data = raw_clean.get_data(picks=[idx])[0]
            axes[1].plot(raw_clean.times[times_mask], after_data[times_mask] * 1e6,
                        label=ch, alpha=0.8, linewidth=0.7)

    axes[0].set_title("ICA Öncesi (Fp1, Fp2 - Göz Artefaktı Baskın)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Genlik (µV)")
    axes[0].legend(loc="upper right")

    axes[1].set_title("ICA Sonrası (Göz Kırpma Artefaktları Temizlendi)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Genlik (µV)")
    axes[1].set_xlabel("Zaman (s)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(f"{fig_prefix}_03_before_after_ica.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_prefix}_03_before_after_ica.png")

    # ─── KANIT 4: PSD (Güç Spektral Yoğunluğu) ──────────
    print("  [Figure 4] Power Spectral Density...")
    fig, ax = plt.subplots(figsize=(12, 6))
    psd_raw = raw.compute_psd(picks="eeg", fmax=60)
    psd_filt = raw_clean.compute_psd(picks="eeg", fmax=60)

    psd_raw.plot(axes=ax, show=False, color="red", alpha=0.4, spatial_colors=False)
    psd_filt.plot(axes=ax, show=False, color="blue", alpha=0.4, spatial_colors=False)

    # Frekans bantlarını arka plana çiz
    band_colors = {"delta": "#E8E8E8", "theta": "#FFE0B2", "alpha": "#C8E6C9",
                   "beta": "#BBDEFB", "gamma": "#E1BEE7"}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        ax.axvspan(fmin, fmax, alpha=0.15, color=band_colors.get(band_name, "#ccc"),
                  label=f"{band_name} ({fmin}-{fmax} Hz)")

    ax.set_title("Güç Spektral Yoğunluğu: Ham (Kırmızı) vs Temizlenmiş (Mavi)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Frekans (Hz)")
    ax.set_ylabel("Güç Yoğunluğu (µV²/Hz)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 55])

    plt.tight_layout()
    fig.savefig(f"{fig_prefix}_04_psd_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_prefix}_04_psd_comparison.png")

    # ─── KANIT 5: Senkronizasyon (varsa) ─────────────────
    if scenarios_df is not None and not scenarios_df.empty and raw_clean.annotations:
        print("  [Figure 5] Synchronization proof...")
        annotations = raw_clean.annotations
        if len(annotations) > 0:
            # İlk senaryo olayını göster
            first_onset = annotations[0]["onset"]
            scenario_desc = annotations[0]["description"]

            fig, ax = plt.subplots(figsize=(14, 6))

            window_start = max(0, first_onset - 3)
            window_end = min(raw_clean.times[-1], first_onset + 6)

            for ch in ["Fz", "Cz", "Pz"]:
                if ch in raw_clean.ch_names:
                    idx = raw_clean.ch_names.index(ch)
                    times_mask = (raw_clean.times >= window_start) & (raw_clean.times <= window_end)
                    data = raw_clean.get_data(picks=[idx])[0]
                    ax.plot(raw_clean.times[times_mask], data[times_mask] * 1e6,
                           label=ch, linewidth=1.0)

            # Trigger çizgisi
            ax.axvline(x=first_onset, color="red", linestyle="--", linewidth=2,
                      label=f"Trigger: {scenario_desc}")
            ax.axvspan(first_onset, first_onset + 3, alpha=0.1, color="red",
                      label="Senaryo aktif (3s)")

            ax.set_title(f"EEG-Platform Senkronizasyonu: {scenario_desc}",
                        fontsize=14, fontweight="bold")
            ax.set_xlabel("Zaman (s)")
            ax.set_ylabel("Genlik (µV)")
            ax.legend(loc="upper right")

            plt.tight_layout()
            fig.savefig(f"{fig_prefix}_05_sync_proof.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {fig_prefix}_05_sync_proof.png")

    # ─── KANIT 6: Topografik Harita (Zaman Dilimleri) ────
    print("  [Figure 6] Topographic map...")
    try:
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        times_to_plot = np.linspace(t_start, t_start + 8, 5)
        for i, t in enumerate(times_to_plot):
            # t anındaki veriyi al
            sample_idx = int(t * raw_clean.info["sfreq"])
            if sample_idx < raw_clean.n_times:
                data = raw_clean.get_data(picks="eeg")[:, sample_idx]
                mne.viz.plot_topomap(
                    data, raw_clean.info, axes=axes[i], show=False,
                    names=EEG_CHANNELS[:N_EEG_CHANNELS],
                )
                axes[i].set_title(f"t = {t:.1f}s", fontsize=10)

        fig.suptitle("Topografik Beyin Haritası - Zaman Dilimleri", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(f"{fig_prefix}_06_topomap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


        print(f"    Saved: {fig_prefix}_06_topomap.png")
    except Exception as e:
        print(f"    Topomap failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_subject(subject_name: str, timezone_offset: float = 0.0,
                    figures_only: bool = False):
    """Tek bir subject için tam preprocessing pipeline."""
    info = SUBJECTS[subject_name]

    # Dosya kontrolü
    eeg_path = RAW_DIR / info["eeg"]
    if not eeg_path.exists():
        print(f"  ERROR: EEG file not found: {eeg_path}")
        print(f"  Available files in {RAW_DIR}:")
        if RAW_DIR.exists():
            for f in sorted(RAW_DIR.glob("*.eeg")):
                print(f"    {f.name}")
        return None

    # Step 1: Yükle
    raw = load_eeg(subject_name)

    # Step 2: Filtrele
    raw_filtered = apply_filters(raw)

    # Step 3: ICA
    raw_clean, ica, eog_indices = apply_ica(raw_filtered, subject_name)

    # Step 4: Platform olaylarını yükle
    scenarios_df = load_platform_events(subject_name)

    # Senkronizasyon - event'leri EEG'ye ekle
    raw_clean = create_annotations_from_events(
        raw_clean, subject_name, scenarios_df,
        timezone_offset_hours=timezone_offset,
    )

    # Görseller
    generate_figures(raw, raw_filtered, raw_clean, ica, eog_indices,
                     subject_name, scenarios_df)

    if figures_only:
        print(f"\n  [Done] Figures saved to {FIGURES_DIR}/")
        return None

    # Step 5: Epoklama (eğer annotation varsa)
    features_df = None
    if raw_clean.annotations and len(raw_clean.annotations) > 0:
        print("\n  [Epoching] Creating epochs from annotations...")
        try:
            events, event_id = mne.events_from_annotations(raw_clean)
            epochs = mne.Epochs(
                raw_clean, events, event_id,
                tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                baseline=EPOCH_BASELINE,
                preload=True,
                reject=dict(eeg=200e-6),  # ±200µV üstü reject
                picks="eeg",
            )
            print(f"  [Epoching] {len(epochs)} epochs created (rejected: "
                  f"{len(events) - len(epochs)})")

            # Feature extraction
            features_df = extract_features_from_epochs(epochs)
            features_df["subject"] = subject_name
            features_df["group"] = info["group"]

            # Kaydet
            features_path = FEATURES_DIR / f"{subject_name}_features.csv"
            features_df.to_csv(features_path, index=False)
            print(f"  [Features] Saved: {features_path}")

            # Epoch'ları kaydet
            epochs_path = PROCESSED_DIR / f"{subject_name}_epochs-epo.fif"
            epochs.save(epochs_path, overwrite=True)
            print(f"  [Epochs] Saved: {epochs_path}")

        except Exception as e:
            print(f"  [Epoching] Failed: {e}")
    else:
        print("\n  [Epoching] Skipped - no annotations found")
        print("    (Senkronizasyon için timezone_offset ayarını kontrol et)")

    # Temizlenmiş veriyi kaydet
    clean_path = PROCESSED_DIR / f"{subject_name}_clean-raw.fif"
    raw_clean.save(clean_path, overwrite=True)
    print(f"\n  [Saved] Clean EEG: {clean_path}")

    return features_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIMEZONE AUTO-DETECT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_timezone_offset() -> float:
    """
    EEG başlangıç zamanı ile platform ilk event zamanını karşılaştırarak
    saat farkını otomatik tespit et.
    """
    events_path = RAW_DIR / "events.csv"
    if not events_path.exists():
        return 0.0

    df = pd.read_csv(events_path)

    # varyant_a ile deneyelim - en net eşleşme
    va_info = SUBJECTS["varyant_a"]
    va_session = va_info["session_id"]
    va_eeg_start = get_subject_eeg_start("varyant_a")

    session_events = df[df["session_id"] == va_session]
    if session_events.empty:
        return 0.0

    first_event_str = session_events.iloc[0]["created_at"]
    try:
        first_event = datetime.strptime(first_event_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return 0.0

    # EEG ~8 dakika → event EEG süresi içinde düşmeli
    raw_diff_hours = (first_event - va_eeg_start).total_seconds() / 3600

    # Olası offset'ler: 0, 1, 2, 3, -1, -2, -3
    best_offset = 0
    best_fit = abs(raw_diff_hours)

    for offset in [0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, 7, 8, 9]:
        adjusted = raw_diff_hours - offset
        # Event, EEG kaydının 0-10 dakika aralığında düşmeli
        if 0 <= adjusted * 60 <= 10:
            if abs(adjusted) < abs(best_fit - offset) or best_fit > 0.2:
                best_offset = offset
                best_fit = adjusted
                break

    print(f"\n  [Timezone] EEG start: {va_eeg_start}")
    print(f"  [Timezone] First event: {first_event}")
    print(f"  [Timezone] Raw diff: {raw_diff_hours:.2f} hours")
    print(f"  [Timezone] Detected offset: {best_offset} hours")
    print(f"  [Timezone] Adjusted event position in EEG: {best_fit*60:.1f} min")

    return float(best_offset)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="BİTİRMEEG EEG Preprocessing Pipeline")
    parser.add_argument("--subject", type=str, default=None,
                       help="Subject name (e.g. varyant_a). Default: all subjects")
    parser.add_argument("--figures-only", action="store_true",
                       help="Only generate figures, skip saving processed data")
    parser.add_argument("--tz-offset", type=float, default=None,
                       help="Timezone offset hours (auto-detect if not set)")
    parser.add_argument("--list", action="store_true",
                       help="List available subjects")
    args = parser.parse_args()

    if args.list:
        print("Available subjects:")
        for name, info in SUBJECTS.items():
            eeg_exists = (RAW_DIR / info["eeg"]).exists()
            status = "✓" if eeg_exists else "✗ (eeg missing)"
            print(f"  {name:15s} | {info['group']:12s} | {status}")
        return

    # Timezone offset
    if args.tz_offset is not None:
        tz_offset = args.tz_offset
    else:
        tz_offset = detect_timezone_offset()

    subjects_to_process = (
        [args.subject] if args.subject
        else list(SUBJECTS.keys())
    )

    all_features = []
    for subject in subjects_to_process:
        if subject not in SUBJECTS:
            print(f"ERROR: Unknown subject '{subject}'. Use --list to see available.")
            continue

        features_df = process_subject(
            subject,
            timezone_offset=tz_offset,
            figures_only=args.figures_only,
        )
        if features_df is not None:
            all_features.append(features_df)

    # Tüm feature'ları birleştir
    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        combined_path = FEATURES_DIR / "all_features.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"  Combined features: {combined_path}")
        print(f"  Shape: {combined.shape}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
