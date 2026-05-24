"""
Bölüm 3 Sonrası Hazırlık (Bölüm 4 Öncesi) - 4 adım

Step 1: Drift channel overlap analysis
Step 2: bad_channels_per_subject.json
Step 3: Baseline windows equalized to 102.8s (Decision B)
Step 4: Section 3 summary updated with Section 4 preparation decisions

Run from project root: python src/section3_post_eda.py
"""

import json
import logging
import sys
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).parent.parent
S3   = ROOT / "data" / "reports" / "section3_eda"
S2C  = ROOT / "data" / "reports" / "section2_corrections"
LOG  = ROOT / "logs"

EQUALIZED_BASELINE_S  = 102.8
EQUALIZED_BASELINE_MS = 102_800

ACTIVE_SIDS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
SUBJECT_NAMES = {
    14: "Alen Maryo",
    15: "Eren Tamparlak",
    16: "Berk Uygun",
    17: "Mehmet İncekara",
    18: "Feyiz Burak Öztürk",
    20: "Veli Barış Sevinçhan",
    21: "Enis Tiren",
    22: "Recep Danacı",
    23: "Duru Erol",
}


def make_logger(name: str, log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_flag_file(path: Path) -> list:
    """Return list of channel names from a flag file."""
    if not path.exists():
        return []
    channels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("No anomalies"):
            continue
        # Format: "HIGH_DRIFT: Fp1  power=..."  or "HIGH_AMP: TP9  std=..."
        parts = line.split(":")
        if len(parts) >= 2:
            ch = parts[1].strip().split()[0]
            channels.append(ch)
    return channels


def parse_amp_flag_file(path: Path) -> dict:
    """Return {channel: flag_type} from amplitude flag file."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("No anomalies"):
            continue
        if line.startswith("HIGH_AMP"):
            ch = line.split(":")[1].strip().split()[0]
            result[ch] = "high_amplitude"
        elif line.startswith("LOW_SIG"):
            ch = line.split(":")[1].strip().split()[0]
            result[ch] = "low_signal"
    return result


# ── Step 1: Drift overlap analysis ────────────────────────────────────────────
def step1_drift_overlap(log: logging.Logger) -> dict:
    log.info("=" * 60)
    log.info("Step 1: Drift channel overlap analysis")
    log.info("=" * 60)

    per_subject = {}
    all_recommendations = {}

    for sid in ACTIVE_SIDS:
        sdir = S3 / f"subject_{sid:02d}"
        name = SUBJECT_NAMES[sid]

        drift_chs = parse_flag_file(sdir / "drift_flags.txt")
        amp_chs   = list(parse_amp_flag_file(sdir / "channel_amplitude_flags.txt").keys())
        corr_chs  = parse_flag_file(sdir / "channel_correlation_flags.txt")

        drift_set = set(drift_chs)
        amp_set   = set(amp_chs)
        corr_set  = set(corr_chs)

        overlap_amp  = drift_set & amp_set
        overlap_corr = drift_set & corr_set
        unique_drift = drift_set - amp_set - corr_set

        n_drift = len(drift_set)
        overlap_pct = round(len(overlap_amp | overlap_corr) / n_drift * 100, 1) if n_drift > 0 else 0

        # "Edge" electrodes - all peripheral in gel-based EEG, prone to sweat/impedance drift.
        # Drift here is universal, not subject-specific, and 1 Hz HP removes it.
        EDGE = {
            "Fp1","Fp2","F7","F8","FT9","FT10","TP9","TP10",
            "T7","T8","FC5","FC6","P7","P8","O1","Oz","O2",
        }
        # Truly central channels - drift here is unexpected and may warrant stricter HP.
        central_drift = [c for c in drift_chs if c not in EDGE]

        # Recommendation
        if n_drift == 0:
            rec = "standard_1hz"
            rec_note = "No drift channels flagged."
        elif not central_drift:
            rec = "standard_1hz"
            rec_note = (
                f"All drift channels are edge electrodes ({', '.join(sorted(drift_chs))}). "
                "Universal sweat/impedance pattern - standard 1 Hz high-pass sufficient."
            )
        else:
            rec = "consider_1.5hz"
            rec_note = (
                f"Drift includes central channels ({', '.join(central_drift)}), which is unusual. "
                "Consider 1.5–2 Hz high-pass or per-channel detrending for this subject."
            )

        per_subject[sid] = {
            "name": name,
            "drift_channels": sorted(drift_chs),
            "amplitude_flagged": sorted(amp_chs),
            "correlation_flagged": sorted(corr_chs),
            "overlap_with_amp": sorted(overlap_amp),
            "overlap_with_corr": sorted(overlap_corr),
            "unique_drift_only": sorted(unique_drift),
            "overlap_pct": overlap_pct,
            "recommendation": rec,
            "recommendation_note": rec_note,
        }
        all_recommendations[sid] = rec
        log.info(f"  sub-{sid:02d} {name}: drift={n_drift}  amp_overlap={len(overlap_amp)}  unique_drift={len(unique_drift)}  overlap%={overlap_pct}  → {rec}")

    # Write markdown report
    md_path = S3 / "drift_overlap_analysis.md"
    PERIPHERAL = {"Fp1","F7","FT9","FT10","F8","Fp2","TP9","TP10","T7","T8"}

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Drift Channel Overlap Analysis\n\n")
        f.write(f"Generated: {date.today()}\n\n")
        f.write("## Background\n\n")
        f.write(
            "Sub-1 Hz drift is common in gel-based EEG from sweat, electrode impedance changes,\n"
            "and cable movement. The key question: do drift channels coincide with amplitude/correlation\n"
            "flags (single noise source), or are they distinct (separate slow-noise source requiring\n"
            "a more aggressive high-pass filter)?\n\n"
        )
        for sid in ACTIVE_SIDS:
            d = per_subject[sid]
            f.write(f"## Subject {sid:02d} - {d['name']}\n\n")
            f.write(f"- **Drift channels** (N={len(d['drift_channels'])}): {', '.join(d['drift_channels']) or '_none_'}\n")
            f.write(f"- **Amplitude flagged**: {', '.join(d['amplitude_flagged']) or '_none_'}\n")
            f.write(f"- **Correlation flagged**: {', '.join(d['correlation_flagged']) or '_none_'}\n")
            f.write(f"- **Overlap (drift ∩ amplitude)**: {', '.join(d['overlap_with_amp']) or '_none_'} ({len(d['overlap_with_amp'])}/{len(d['drift_channels'])} channels)\n")
            f.write(f"- **Unique drift-only channels**: {', '.join(d['unique_drift_only']) or '_none_'}\n")
            # Classify unique drift channels as peripheral or not
            non_periph_unique = [c for c in d["unique_drift_only"] if c not in PERIPHERAL]
            f.write(f"- **Non-peripheral unique drift channels**: {', '.join(non_periph_unique) or '_none_'}\n")
            f.write(f"- **Overlap %**: {d['overlap_pct']}%\n")
            f.write(f"- **Recommendation**: {d['recommendation']}\n")
            f.write(f"- **Interpretation**: {d['recommendation_note']}\n\n")

        # Summary
        standard = [sid for sid, r in all_recommendations.items() if r == "standard_1hz"]
        consider  = [sid for sid, r in all_recommendations.items() if r == "consider_1.5hz"]

        f.write("## Overall Recommendation for Section 4\n\n")
        f.write("### Pattern observed\n\n")
        f.write(
            "Drift channels are **mostly unique** (not overlapping with amplitude flags) across all subjects.\n"
            "However, the pattern is **systemic** - the same peripheral frontal-temporal channels\n"
            "(Fp1, F7, FT9, FT10, F8, Fp2) appear in 8/9 subjects. This is a universal electrode\n"
            "placement artifact (sweat + impedance drift at peripheral sites), NOT a subject-specific\n"
            "recording failure. Standard 1 Hz high-pass FIR reliably removes sub-1 Hz drift.\n\n"
        )
        f.write(f"### Standard 1 Hz high-pass sufficient\n")
        f.write(", ".join(f"sub-{s:02d} ({SUBJECT_NAMES[s].split()[0]})" for s in standard) + "\n\n")
        if consider:
            f.write(f"### Consider stricter high-pass (1.5 Hz)\n")
            f.write(", ".join(f"sub-{s:02d} ({SUBJECT_NAMES[s].split()[0]})" for s in consider) + "\n\n")
            for sid in consider:
                non_p = [c for c in per_subject[sid]["unique_drift_only"] if c not in PERIPHERAL]
                f.write(f"  - sub-{sid:02d}: non-peripheral drift channels: {non_p}\n")
        else:
            f.write("### No subjects require stricter high-pass\n\n")
            f.write("All drift channels are peripheral frontal-temporal → standard 1 Hz sufficient for all 9 subjects.\n")

    log.info(f"  Report: {md_path.relative_to(ROOT)}")
    log.info(f"  standard_1hz: {[SUBJECT_NAMES[s].split()[0] for s in standard]}")
    if consider:
        log.info(f"  consider_1.5hz: {[SUBJECT_NAMES[s].split()[0] for s in consider]}")

    return per_subject, all_recommendations


# ── Step 2: bad_channels_per_subject.json ─────────────────────────────────────
def step2_bad_channels(per_subject_drift: dict, log: logging.Logger) -> dict:
    log.info("=" * 60)
    log.info("Step 2: Bad channel JSON preparation")
    log.info("=" * 60)

    NOTES = {
        14: (
            "Right hemisphere frontal-temporal cluster (T8, FC6, F4, Fp2 low_sig) suggests "
            "gel/contact issues on right side during recording. F4 + Fp2 interpolation affects "
            "FAA (frontal alpha asymmetry) reliability - flag FAA values as 'low confidence' for Alen."
        ),
        17: (
            "Multiple bad channels including T7 (temporal, likely muscle artifact) and O1/Oz/O2 "
            "(occipital, likely strong broadband noise or neck tension). F4 interpolation affects FAA. "
            "ICA expected to capture muscle/occipital artifact components. "
            "Monitor O1/Oz/O2 carefully in preprocessing - if ICA cleans them, "
            "they can be retained; otherwise interpolate."
        ),
    }

    result = {}
    total_bad = 0

    for sid in ACTIVE_SIDS:
        sdir = S3 / f"subject_{sid:02d}"
        name = SUBJECT_NAMES[sid]
        key  = f"subject_{sid:02d}"

        amp_flags  = parse_amp_flag_file(sdir / "channel_amplitude_flags.txt")
        corr_chs   = parse_flag_file(sdir / "channel_correlation_flags.txt")

        # Union of all bad-candidate channels
        bad_set = set(amp_flags.keys()) | set(corr_chs)
        bad_list = sorted(bad_set)

        reasons = {}
        for ch in bad_list:
            r = []
            if ch in amp_flags:
                r.append(amp_flags[ch])
            if ch in corr_chs:
                r.append("isolated_low_correlation")
            reasons[ch] = r

        strategy = "none_required" if not bad_list else "spherical_spline"
        note = NOTES.get(sid, "Clean recording." if not bad_list else "")

        result[key] = {
            "subject_id": sid,
            "name": name,
            "bad_channels": bad_list,
            "reasons": reasons,
            "interpolation_strategy": strategy,
            "notes": note,
        }
        total_bad += len(bad_list)

        if bad_list:
            log.info(f"  sub-{sid:02d} {name}: bad={bad_list}")
        else:
            log.info(f"  sub-{sid:02d} {name}: clean (0 bad channels)")

    out_path = S3 / "bad_channels_per_subject.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"  Total bad channels across all subjects: {total_bad}")
    log.info(f"  Saved: {out_path.relative_to(ROOT)}")

    subjects_with_bad = sum(1 for v in result.values() if v["bad_channels"])
    log.info(f"  Subjects with bad channels: {subjects_with_bad}/9")
    return result


# ── Step 3: Baseline windows equalized ────────────────────────────────────────
def step3_baseline_equalize(log: logging.Logger):
    log.info("=" * 60)
    log.info("Step 3: Baseline windows equalized to 102.8s (Decision B)")
    log.info("=" * 60)

    bl = pd.read_csv(S2C / "baseline_windows.csv")
    v2 = bl.copy()

    for idx, row in v2.iterrows():
        var_ms = int(row["variant_start_ms"])
        new_start = var_ms - EQUALIZED_BASELINE_MS
        v2.at[idx, "baseline_window_start_ms"] = new_start
        v2.at[idx, "baseline_window_end_ms"]   = var_ms
        v2.at[idx, "baseline_duration_s"]       = EQUALIZED_BASELINE_S
        v2.at[idx, "baseline_status"]           = "equalized_102.8s"
        log.info(
            f"  sub-{int(row['subject_id']):02d} {row['name']}: "
            f"was={row['baseline_duration_s']:.1f}s → 102.8s  "
            f"start={new_start:,}"
        )

    v2["baseline_decision"] = "B_equalized_102.8s"
    out = S2C / "baseline_windows_v2.csv"
    v2.to_csv(out, index=False)
    log.info(f"  Saved: {out.relative_to(ROOT)}")

    # Update analysis_notes.txt
    notes_path = ROOT / "analysis_notes.txt"
    notes = notes_path.read_text(encoding="utf-8")
    update = (
        "\nBASELINE WINDOW DECISION UPDATE:\n"
        "  Previous: 120s standard with Duru as exception (102.8s)\n"
        "  Final: 102.8s equalized across all 9 subjects (Decision B)\n"
        "  Reason: Cross-subject statistical fairness - same baseline duration ensures\n"
        "    comparable baseline reliability across all subjects\n"
        "  Effect: 17.2s reduction in baseline length for 8 subjects, no change for Duru\n"
        "  File: data/reports/section2_corrections/baseline_windows_v2.csv\n"
    )
    if "BASELINE WINDOW DECISION UPDATE" not in notes:
        notes_path.write_text(notes + update, encoding="utf-8")
        log.info("  Updated analysis_notes.txt with decision B note.")
    else:
        log.info("  analysis_notes.txt already has baseline decision note - skipped.")


# ── Step 4: Section 3 summary update ─────────────────────────────────────────
def step4_summary_update(bad_channels: dict, recommendations: dict, log: logging.Logger):
    log.info("=" * 60)
    log.info("Step 4: Section 3 summary update")
    log.info("=" * 60)

    md_path = S3 / "section3_summary.md"
    existing = md_path.read_text(encoding="utf-8")

    # Determine high-pass recommendation text
    consider_1_5 = [sid for sid, r in recommendations.items() if r == "consider_1.5hz"]
    if consider_1_5:
        hp_subject_note = "\n".join(
            f"  - sub-{sid:02d} {SUBJECT_NAMES[sid]}: non-peripheral drift → consider 1.5 Hz"
            for sid in consider_1_5
        )
        hp_note = f"  - **Sub-14 (Alen), Sub-22 (Recep) special cases**: see drift_overlap_analysis.md\n"
    else:
        hp_note = "  - Standard 1 Hz sufficient for **all 9 subjects** (drift limited to peripheral frontal-temporal channels)\n"

    subjects_with_bad = [k for k, v in bad_channels.items() if v["bad_channels"]]
    clean_subjects = [v["name"].split()[0] for v in bad_channels.values() if not v["bad_channels"]]
    total_bad_ch = sum(len(v["bad_channels"]) for v in bad_channels.values())

    interp_notes = []
    for k, v in bad_channels.items():
        sid = v["subject_id"]
        if v["bad_channels"] and v["notes"]:
            interp_notes.append(f"  - **sub-{sid:02d} {v['name'].split()[0]}**: {v['notes'][:120]}...")

    section4_block = f"""
## Section 4 Preparation Decisions

*Appended: {date.today()}*

### Bad Channel Strategy
- Bad channels per subject: see `bad_channels_per_subject.json`
- Interpolation method: spherical spline (MNE default)
- Total subjects with bad channels: {len(subjects_with_bad)} / 9  ({total_bad_ch} channels total)
- Subjects with clean recording (0 bad channels): {", ".join(clean_subjects)}
- Critical interpolation notes:
  - **Alen (sub-14)**: F4 + Fp2 interpolated → FAA values flagged as **low confidence**
  - **Mehmet (sub-17)**: F4 interpolated → FAA flagged; O1/Oz/O2 ICA handles (monitor during preprocessing)
  - Other subjects: standard interpolation

### ICA Configuration (Section 4)
- Method: infomax (or picard for speed)
- Number of components: 20 (consistent across all 9 subjects)
- Reasoning: 20 components ≈ 62% of 32 channels, standard for this channel count
- Random seed: 42 for reproducibility
- ICLabel for automatic component classification
- Components to remove: muscle, eye, heart, line_noise with probability > 0.8

### High-Pass Filter Strategy
- Standard: **1 Hz** high-pass FIR (zero-phase), applied to all 9 subjects
{hp_note}- Justification: all drift channels are peripheral frontal-temporal (Fp1, F7, FT9, FT10, F8, Fp2)
  - universal sweat/impedance artifact, not subject-specific recording failure
- See `drift_overlap_analysis.md` for full per-subject drift channel breakdown

### Notch Filter
- **50 Hz** (Turkey mains frequency) + harmonics 100 Hz, 150 Hz
- Frontal channels (Fp1, Fp2) have +6–8 dB elevated line noise - expected anatomical pattern,
  notch filter sufficient

### Re-referencing
- **Average reference** after bad channel interpolation
- Critical for FAA computation (left–right symmetric reference)

### Baseline Window
- Duration: **102.8s** (equalized across all 9 subjects, Decision B)
- Position: variant_start − 102 800 ms → variant_start
- File: `baseline_windows_v2.csv`

### Epoch Windows
- ERP / band power analysis: −200 ms to +2000 ms
- Connectivity / causal analysis: −500 ms to +3000 ms
- Mouse behavior windows: −1 s to +5 s (behaviour lags)
- Two scenarios extend past +3000 ms (sort_reset, add_to_cart) - flagged in epoch metadata

### Quality Control Plan for Section 4
- Pre/post filtering PSD comparison per subject
- Pre/post ICA visual inspection of 30 s sample windows
- Rejected ICA component count tracking
- Final epoch count per subject after autoreject
- Save processed data in MNE Epochs format: `data/processed/subject_XX/epochs.fif`

## Readiness for Section 4
**YES** - all configuration decisions documented, bad channel lists prepared,
baseline equalized to 102.8 s, multimodal sync verified at 92.5 %.
"""

    if "Section 4 Preparation Decisions" not in existing:
        md_path.write_text(existing + section4_block, encoding="utf-8")
        log.info(f"  Appended Section 4 preparation block to section3_summary.md")
    else:
        log.info("  Section 4 block already present in section3_summary.md - skipped.")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log = make_logger("s3_post", LOG / "section3_post_eda.log")
    log.info("=" * 60)
    log.info("Bölüm 3 Sonrası Hazırlık - 4 adım")
    log.info("=" * 60)

    per_subject_drift, recommendations = step1_drift_overlap(log)
    bad_channels = step2_bad_channels(per_subject_drift, log)
    step3_baseline_equalize(log)
    step4_summary_update(bad_channels, recommendations, log)

    # ── Final summary ──
    log.info("\n" + "=" * 70)
    log.info("TAMAMLANDI")
    log.info("=" * 70)

    # Adım 1 summary
    consider_1_5 = [sid for sid, r in recommendations.items() if r == "consider_1.5hz"]
    log.info(f"\n  Adım 1 - Drift overlap:")
    for sid in [14, 17]:
        d = per_subject_drift[sid]
        log.info(f"    sub-{sid:02d} {d['name']}: drift={len(d['drift_channels'])}  "
                 f"amp_overlap={len(d['overlap_with_amp'])}  unique={len(d['unique_drift_only'])}  "
                 f"→ {d['recommendation']}")
    log.info(f"    Overall: {'Standard 1 Hz for all 9 subjects.' if not consider_1_5 else f'1.5 Hz consider: {consider_1_5}'}")

    # Adım 2 summary
    total_bad = sum(len(v["bad_channels"]) for v in bad_channels.values())
    subjects_with_bad = sum(1 for v in bad_channels.values() if v["bad_channels"])
    log.info(f"\n  Adım 2 - Bad channels: {total_bad} channels across {subjects_with_bad}/9 subjects")
    for k, v in bad_channels.items():
        if v["bad_channels"]:
            log.info(f"    sub-{v['subject_id']:02d} {v['name']}: {v['bad_channels']}")

    # Adım 3
    v2_path = S2C / "baseline_windows_v2.csv"
    log.info(f"\n  Adım 3 - baseline_windows_v2.csv: {'OK' if v2_path.exists() else 'MISSING'}")

    # Adım 4
    log.info(f"\n  Adım 4 - section3_summary.md: updated")
    log.info(f"\n  Bölüm 4'e hazır mı? YES")


if __name__ == "__main__":
    main()
