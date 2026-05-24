"""
BİTİRMEEG - Preprocessing Pipeline Configuration
==================================================
Tüm parametreler burada. Yeni deney geldiğinde sadece SUBJECTS'e ekle.
"""
from pathlib import Path
from datetime import datetime

# ─── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR.parent / "data"
PROCESSED_DIR = BASE_DIR.parent / "processed"
FIGURES_DIR = BASE_DIR.parent / "figures"
FEATURES_DIR = BASE_DIR.parent / "features"

# Klasörleri oluştur
for d in [PROCESSED_DIR, FIGURES_DIR, FEATURES_DIR]:
    d.mkdir(exist_ok=True)

# ─── Subject / Recording Mapping ───────────────────────
# Her kayıt: dosya adı prefix → experiment group + EEG/platform eşleştirme
#
# ÖNEMLİ: vhdr dosyaları içindeki DataFile referansları orijinal isimlerle
# (dhl_000092.eeg vs.) kaydedilmiş. MNE yüklerken vhdr'yı düzelteceğiz.
#
# vmrk timestamp formatı: YYYYMMDDHHMMSSffffff (mikrosaniye dahil)
# Örnek: 20251222042852115701 → 2025-12-22 04:28:52.115701

SUBJECTS = {
    "control": {
        "vhdr": "control.vhdr",
        "eeg": "controll.eeg",       # NOT: dosya adı "controll" (çift L)
        "vmrk": "control.vmrk",
        "group": "control",
        "session_id": "8888c7b1-8f92-43f6-830e-007748322043",
        "eeg_start": "20251222042852115701",  # vmrk'dan
    },
    "varyant_a": {
        "vhdr": "varyant_a.vhdr",
        "eeg": "varyant_a.eeg",
        "vmrk": "varyaant_a.vmrk",   # NOT: typo "varyaant"
        "group": "variant_a",
        "session_id": "d63f258f-d166-466a-9c28-4314e59d1f24",
        "eeg_start": "20251222044246877280",
    },
    "control1": {
        "vhdr": "control1.vhdr",
        "eeg": "control1.eeg",
        "vmrk": "control1.vmrk",
        "group": "control",
        "session_id": "ff43e5bb-0fe6-4374-afab-abd28f59ee74",  # Tahmin
        "eeg_start": "20251226002112692348",
    },
    "varyantb": {
        "vhdr": "varyantb.vhdr",
        "eeg": "varyantb.eeg",
        "vmrk": "varyantb.vmrk",
        "group": "variant_b",
        "session_id": "47460be4-ab0c-453a-8eaf-8588fde55007",
        "eeg_start": "20251226003018879027",
    },
    "control2": {
        "vhdr": "control2.vhdr",
        "eeg": "control2.eeg",
        "vmrk": "control2.vmrk",
        "group": "control",
        "session_id": None,  # Birden fazla control session var, eşleşme belirsiz
        "eeg_start": "20251226014956670541",
    },
    "varyantc": {
        "vhdr": "varyantc.vhdr",
        "eeg": "varyantc.eeg",
        "vmrk": "varyantc.vmrk",
        "group": "variant_c",
        "session_id": "3e735bad-5746-4afa-a7ea-81f929178658",
        "eeg_start": "20251226020017767148",
    },
}

# ─── EEG Parameters ───────────────────────────────────
SFREQ = 500                    # Hz (SamplingInterval=2000µs → 500Hz)
N_EEG_CHANNELS = 32            # İlk 32 kanal EEG
ACCEL_CHANNELS = ["x_dir", "y_dir", "z_dir"]  # Son 3 kanal akselerometre

# EEG kanal isimleri (10-20 sistemi)
EEG_CHANNELS = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
    "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1",
    "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz",
    "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2",
]

# Sunum ve analiz için önemli kanallar
KEY_CHANNELS = ["Fz", "Cz", "Pz", "Oz", "Fp1", "Fp2"]

# Frontal kanallar (dikkat ve bilişsel yük göstergeleri)
FRONTAL_CHANNELS = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8"]

# ─── Filtering ─────────────────────────────────────────
NOTCH_FREQ = 50.0              # Hz - Türkiye şebeke frekansı
BANDPASS_LOW = 1.0             # Hz
BANDPASS_HIGH = 45.0           # Hz

# ─── ICA ───────────────────────────────────────────────
ICA_N_COMPONENTS = 20          # 32 kanaldan 20 bileşen yeterli
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = 800

# ─── Epoching ──────────────────────────────────────────
EPOCH_TMIN = -2.0              # Senaryo öncesi (saniye)
EPOCH_TMAX = 5.0               # Senaryo sonrası (saniye)
EPOCH_BASELINE = (-2.0, 0.0)   # Baseline düzeltme penceresi

# ─── Feature Extraction ───────────────────────────────
# Frekans bantları (Hz)
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),      # Bilişsel yük ↑ → theta ↑
    "alpha": (8, 13),     # Dikkat ↓ → alpha ↑ (gözler kapalı/rahat)
    "beta": (13, 30),     # Aktif düşünme, stres → beta ↑
    "gamma": (30, 45),    # Yüksek bilişsel işlem → gamma ↑
}

# ─── Eye Tracking ──────────────────────────────────────
PUPIL_BLINK_THRESHOLD = 0.1   # Bu değerin altındaki pupil ölçümleri = blink
PUPIL_SMOOTHING_WINDOW = 5    # Median filtre penceresi (sample sayısı)

# ─── Helpers ───────────────────────────────────────────
def parse_vmrk_timestamp(ts_str: str) -> datetime:
    """vmrk timestamp'ini datetime'a çevir.
    Format: YYYYMMDDHHMMSSffffff
    Örnek: 20251222042852115701
    """
    ts_str = ts_str.strip().replace("\r", "")
    return datetime.strptime(ts_str[:14], "%Y%m%d%H%M%S")


def get_subject_eeg_start(subject_name: str) -> datetime:
    """Bir subject'in EEG kayıt başlangıç zamanını döndür."""
    return parse_vmrk_timestamp(SUBJECTS[subject_name]["eeg_start"])
