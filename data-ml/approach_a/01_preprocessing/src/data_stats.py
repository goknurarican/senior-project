"""
Per-subject statistics + cross-modal alignment check.

Modaliteler: EEG (BrainVision) | Eye (eye_data_db.csv) | Platform (all_events.csv)
Zaman birimi: wall_time_ms (Unix epoch ms) - üç modalitede ortak referans.
"""

import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW = Path(__file__).parent.parent / "data" / "raw"
SEP = "─" * 68


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def fmt_ms(ms: float) -> str:
    s = ms / 1000
    return f"{int(s // 60)}m {s % 60:.1f}s"


def overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    if hi <= lo:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return (hi - lo) / union


# ──────────────────────────────────────────────────────────────
# Per-modality loaders
# ──────────────────────────────────────────────────────────────

def load_eeg(subject_dir: Path) -> dict:
    vhdr_files = list(subject_dir.glob("*.vhdr"))
    if not vhdr_files:
        return {"ok": False, "reason": "no .vhdr file"}

    vhdr = vhdr_files[0]
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)

    # Marker dosyasından wall_time_ms aralığı
    marker_csv = subject_dir / "eeg" / "eeg_markers.csv"
    marker_wall = None
    if marker_csv.exists():
        mdf = pd.read_csv(marker_csv)
        if "wall_time_ms" in mdf.columns:
            valid = mdf["wall_time_ms"].dropna()
            if len(valid):
                marker_wall = (valid.min(), valid.max())

    return {
        "ok": True,
        "file": vhdr.name,
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "duration_s": raw.times[-1],
        "n_samples": raw.n_times,
        "marker_wall_ms": marker_wall,
        "n_marker_rows": len(pd.read_csv(marker_csv)) if marker_csv.exists() else None,
    }


def load_eye(subject_dir: Path) -> dict:
    csv = subject_dir / "eye" / "eye_data_db.csv"
    if not csv.exists():
        return {"ok": False, "reason": "eye_data_db.csv not found"}

    df = pd.read_csv(csv)
    if df.empty:
        return {"ok": False, "reason": "empty file"}

    wt = df["wall_time_ms"]
    duration_ms = wt.max() - wt.min()
    n = len(df)
    eff_sfreq = n / (duration_ms / 1000) if duration_ms > 0 else 0

    # Geçerlilik oranları
    bpogv = df["bpogv"].mean() if "bpogv" in df.columns else None
    fpogv = df["fpogv"].mean() if "fpogv" in df.columns else None

    # Pupil varlığı
    pl_ok = df["pupil_left"].gt(0).mean() if "pupil_left" in df.columns else None
    pr_ok = df["pupil_right"].gt(0).mean() if "pupil_right" in df.columns else None

    # Zaman boşlukları
    diffs = wt.sort_values().diff().dropna()
    gaps = diffs[diffs > 1000]  # >1 s gap

    return {
        "ok": True,
        "n_rows": n,
        "wall_start_ms": wt.min(),
        "wall_end_ms": wt.max(),
        "duration_ms": duration_ms,
        "eff_sfreq": eff_sfreq,
        "bpogv_mean": bpogv,
        "fpogv_mean": fpogv,
        "pupil_left_valid": pl_ok,
        "pupil_right_valid": pr_ok,
        "n_gaps_1s": len(gaps),
        "max_gap_s": gaps.max() / 1000 if len(gaps) else 0,
    }


def load_platform(subject_dir: Path) -> dict:
    csv = subject_dir / "platform" / "all_events.csv"
    if not csv.exists():
        return {"ok": False, "reason": "all_events.csv not found"}

    df = pd.read_csv(csv)
    if df.empty:
        return {"ok": False, "reason": "empty file"}

    ts = df["timestamp"]
    duration_ms = ts.max() - ts.min()

    event_counts = df["event_type"].value_counts().to_dict()

    # Scenario triggers
    sc_csv = subject_dir / "platform" / "scenario_triggers.csv"
    n_scenarios = None
    if sc_csv.exists():
        sc = pd.read_csv(sc_csv)
        n_scenarios = len(sc)

    # Mouse trajectory summary
    mt_csv = subject_dir / "platform" / "mouse_trajectory_summary.csv"
    n_traj = None
    if mt_csv.exists():
        mt = pd.read_csv(mt_csv)
        n_traj = len(mt)

    return {
        "ok": True,
        "n_events": len(df),
        "wall_start_ms": ts.min(),
        "wall_end_ms": ts.max(),
        "duration_ms": duration_ms,
        "event_counts": event_counts,
        "n_scenarios_triggered": n_scenarios,
        "n_mouse_traj": n_traj,
    }


# ──────────────────────────────────────────────────────────────
# Alignment check
# ──────────────────────────────────────────────────────────────

def check_alignment(eeg: dict, eye: dict, plat: dict) -> dict:
    results = {}

    # Hangi modaliteler wall_time_ms taşıyor?
    windows = {}
    if eeg.get("ok") and eeg.get("marker_wall_ms"):
        windows["eeg"] = eeg["marker_wall_ms"]
    if eye.get("ok"):
        windows["eye"] = (eye["wall_start_ms"], eye["wall_end_ms"])
    if plat.get("ok"):
        windows["platform"] = (plat["wall_start_ms"], plat["wall_end_ms"])

    # Pairwise overlap
    pairs = [("eeg", "eye"), ("eeg", "platform"), ("eye", "platform")]
    for a, b in pairs:
        if a in windows and b in windows:
            r = overlap_ratio(*windows[a], *windows[b])
            offset_ms = windows[b][0] - windows[a][0]
            results[f"{a}_vs_{b}"] = {
                "overlap_ratio": r,
                "offset_ms": offset_ms,
                "ok": r > 0.5,
            }

    # Genel ortak pencere
    if len(windows) >= 2:
        global_start = max(v[0] for v in windows.values())
        global_end   = min(v[1] for v in windows.values())
        results["common_window_ms"] = max(0, global_end - global_start)

    return results


# ──────────────────────────────────────────────────────────────
# Printer
# ──────────────────────────────────────────────────────────────

def print_subject(meta: dict, eeg: dict, eye: dict, plat: dict, align: dict):
    uid   = meta.get("user_id", "?")
    name  = meta.get("name", "?")
    group = meta.get("group", "?")
    print(f"\n{SEP}")
    print(f"  USER {uid:>3} | {name:<28} | {group}")
    print(SEP)

    # ── EEG ──
    print("  [EEG]")
    if eeg["ok"]:
        dur = fmt_ms(eeg["duration_s"] * 1000)
        mw  = eeg["marker_wall_ms"]
        mw_str = f"{mw[0]}  →  {mw[1]}" if mw else "N/A"
        print(f"    Dosya       : {eeg['file']}")
        print(f"    Süre        : {dur}  ({eeg['duration_s']:.1f} s)")
        print(f"    Kanallar    : {eeg['n_channels']}  |  {eeg['sfreq']} Hz")
        print(f"    Örnek sayısı: {eeg['n_samples']:,}")
        print(f"    Marker satır: {eeg['n_marker_rows']:,}")
        print(f"    Marker wall : {mw_str}")
    else:
        print(f"    HATA: {eeg['reason']}")

    # ── Eye ──
    print("  [EYE / GAZE]")
    if eye["ok"]:
        bpog = f"{eye['bpogv_mean']*100:.1f}%" if eye["bpogv_mean"] is not None else "N/A"
        fpog = f"{eye['fpogv_mean']*100:.1f}%" if eye["fpogv_mean"] is not None else "N/A"
        pl   = f"{eye['pupil_left_valid']*100:.1f}%" if eye["pupil_left_valid"] is not None else "N/A"
        pr   = f"{eye['pupil_right_valid']*100:.1f}%" if eye["pupil_right_valid"] is not None else "N/A"
        print(f"    Satır sayısı: {eye['n_rows']:,}")
        print(f"    Süre        : {fmt_ms(eye['duration_ms'])}")
        print(f"    Eff. sfreq  : {eye['eff_sfreq']:.1f} Hz")
        print(f"    BPOG geçerl : {bpog}  |  FPOG geçerl: {fpog}")
        print(f"    Pupil L/R   : {pl} / {pr}")
        gap_str = f"{eye['n_gaps_1s']} gap (max {eye['max_gap_s']:.1f}s)" if eye["n_gaps_1s"] else "gap yok"
        print(f"    Zaman boşluk: {gap_str}")
    else:
        print(f"    HATA: {eye['reason']}")

    # ── Platform ──
    print("  [PLATFORM / MOUSE]")
    if plat["ok"]:
        ec = plat["event_counts"]
        print(f"    Toplam event: {plat['n_events']:,}")
        print(f"    Süre        : {fmt_ms(plat['duration_ms'])}")
        print(f"    Scenario trig: {plat['n_scenarios_triggered']}  |  Traj: {plat['n_mouse_traj']}")
        type_line = "  ".join(f"{k}:{v}" for k, v in sorted(ec.items(), key=lambda x: -x[1])[:6])
        print(f"    Event tipleri: {type_line}")
    else:
        print(f"    HATA: {plat['reason']}")

    # ── Alignment ──
    print("  [ALIGNMENT]")
    if not align:
        print("    Yeterli modalite yok.")
    else:
        cw = align.get("common_window_ms", 0)
        print(f"    Ortak pencere: {fmt_ms(cw)}" if cw else "    Ortak pencere: N/A")
        for pair, info in align.items():
            if pair == "common_window_ms":
                continue
            status = "OK" if info["ok"] else "UYARIM"
            sign   = "+" if info["offset_ms"] >= 0 else ""
            print(
                f"    {pair:<22}  overlap={info['overlap_ratio']*100:.1f}%"
                f"  offset={sign}{info['offset_ms']/1000:.2f}s  [{status}]"
            )


# ──────────────────────────────────────────────────────────────
# Global summary table
# ──────────────────────────────────────────────────────────────

def print_global_summary(rows: list):
    print(f"\n\n{'═'*68}")
    print("  GENEL ÖZET")
    print(f"{'═'*68}")
    hdr = f"{'Denek':<32} {'EEG':>5} {'Eye':>5} {'Plat':>5} {'Align':>6} {'EEG_dur':>8} {'Eye_dur':>8}"
    print(hdr)
    print("─" * 68)
    for r in rows:
        print(
            f"  {r['name']:<30} "
            f"{'OK' if r['eeg_ok'] else 'ERR':>5} "
            f"{'OK' if r['eye_ok'] else 'ERR':>5} "
            f"{'OK' if r['plat_ok'] else 'ERR':>5} "
            f"{r['align']:>6} "
            f"{r['eeg_dur']:>8} "
            f"{r['eye_dur']:>8}"
        )
    print("─" * 68)
    print(f"  {len(rows)} denek  |  tam veri: {sum(1 for r in rows if r['eeg_ok'] and r['eye_ok'] and r['plat_ok'])}/{len(rows)}")
    print(f"{'═'*68}\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    subject_dirs = sorted(d for d in RAW.iterdir() if d.is_dir())
    if not subject_dirs:
        print("data/raw/ altında denek klasörü bulunamadı.")
        return

    summary_rows = []

    for sdir in subject_dirs:
        meta_path = sdir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        eeg  = load_eeg(sdir)
        eye  = load_eye(sdir)
        plat = load_platform(sdir)
        align = check_alignment(eeg, eye, plat)

        print_subject(meta, eeg, eye, plat, align)

        # Tüm pairwise align OK mu?
        pair_oks = [v["ok"] for k, v in align.items() if k != "common_window_ms"]
        align_str = "OK" if pair_oks and all(pair_oks) else ("UYARI" if pair_oks else "N/A")

        summary_rows.append({
            "name": meta.get("name", sdir.name),
            "eeg_ok": eeg["ok"],
            "eye_ok": eye["ok"],
            "plat_ok": plat["ok"],
            "align": align_str,
            "eeg_dur": fmt_ms(eeg["duration_s"] * 1000) if eeg["ok"] else "N/A",
            "eye_dur": fmt_ms(eye["duration_ms"]) if eye["ok"] else "N/A",
        })

    print_global_summary(summary_rows)


if __name__ == "__main__":
    main()
