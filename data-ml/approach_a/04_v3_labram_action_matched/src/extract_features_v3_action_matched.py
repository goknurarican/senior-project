"""
extract_features_v3_action_matched.py
======================================
Extracts action-matched control epochs from the control phase and builds
a balanced v3 dataset for frustration detection.

Action markers are derived from real user actions (product views, add-to-cart)
during the free-browsing control phase, eliminating the "task vs rest" confound
present in v2 pseudo-markers.

Outputs per subject (data/processed/subject_XX/):
  control_action_matched_markers.csv
  epochs_erp_action_matched-epo.fif
  eye_epoch_features_action_matched_erp.csv
  eye_timeseries_action_matched_erp.npy
  mouse_epoch_features_action_matched_erp.csv
  mouse_timeseries_action_matched_erp.npy

Outputs global (approach_a/features/subject_XX/):
  eeg_embeddings_action_matched.npy

Outputs global (approach_a/features/):
  all_eeg_embeddings_v3.npy
  all_eye_timeseries_v3.npy
  all_mouse_timeseries_v3.npy
  labels_v3.csv
  all_eeg_embeddings_v3_metadata.csv
"""

import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "approach_a" / "src"))

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from labram_wrapper import load_labram_base
from sklearn.utils import resample

# Import all helpers from the existing extraction script
from extract_features_8a_with_control import (
    ERP_TMIN, ERP_TMAX,
    EYE_CHANNELS, EYE_STEPS,
    MOUSE_CHANNELS, MOUSE_STEPS,
    get_raw_start_wall_ms,
    _survived_markers,
    step3_eye_features,
    step4_mouse_features,
    step5_labram,
    subject_info,
    load_config,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]

SUBJECT_FOLDERS = {
    14: "user_014_alen_maryo_variant_b",
    15: "user_015_eren_tamparlak_variant_c",
    16: "user_016_berk_uygun_variant_b",
    17: "user_017_mehmet_i̇ncekara_variant_b",
    18: "user_018_feyiz_burak_öztürk_variant_b",
    20: "user_020_veli_barış_sevinçhan_variant_b",
    21: "user_021_enis_tiren_variant_a",
    22: "user_022_recep_danacı_variant_c",
    23: "user_023_duru_erol_variant_c",
}

FEAT_DIR = ROOT / "approach_a" / "features"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR  = ROOT / "data" / "raw"

MIN_GAP_MS   = 3000   # minimum gap between action markers
PRE_BUFFER_MS  = 200  # epoch starts 200ms before marker
POST_BUFFER_MS = 2000 # epoch ends 2000ms after marker


# ── Step 1: Action-matched markers ────────────────────────────────────────────
def step1_action_markers(sid: int, raw_start_wall_ms: float) -> pd.DataFrame:
    """
    Extract action-matched markers from the control phase using real user clicks
    on product pages or add-to-cart buttons.
    """
    proc_dir = PROC_DIR / f"subject_{sid}"
    folder   = SUBJECT_FOLDERS[sid]

    # Load control window boundaries from pseudo-markers
    ctrl_pm_path = proc_dir / "control_pseudo_markers.csv"
    if not ctrl_pm_path.exists():
        log.warning(f"  sub-{sid}: control_pseudo_markers.csv not found")
        return pd.DataFrame()

    ctrl_pm = pd.read_csv(ctrl_pm_path)
    ctrl_start_ms = float(ctrl_pm["wall_time_ms"].min())
    ctrl_end_ms   = float(ctrl_pm["wall_time_ms"].max())
    log.info(f"  sub-{sid}: ctrl window [{ctrl_start_ms:.0f}, {ctrl_end_ms:.0f}]  "
             f"({(ctrl_end_ms - ctrl_start_ms)/1000:.1f}s)")

    # Load mouse_clicks_flat.csv
    clicks_path = RAW_DIR / folder / "platform" / "mouse_clicks_flat.csv"
    if not clicks_path.exists():
        log.warning(f"  sub-{sid}: mouse_clicks_flat.csv not found at {clicks_path}")
        return pd.DataFrame()

    clicks = pd.read_csv(clicks_path)

    # Filter to control window
    clicks = clicks[
        (clicks["wall_time_ms"] >= ctrl_start_ms) &
        (clicks["wall_time_ms"] <= ctrl_end_ms)
    ].copy()

    if clicks.empty:
        log.warning(f"  sub-{sid}: no clicks in control window")
        return pd.DataFrame()

    # Classify actions
    # S32 (product_viewed): '/product' in page_url
    # S30 (add_to_cart): class_name contains add-to-cart / add_to_cart / addtocart
    is_product = clicks["page_url"].str.contains("/product", na=False, case=False)
    is_cart    = clicks["class_name"].str.contains(
        "add.to.cart|addtocart|add_to_cart", na=False, case=False, regex=True
    )
    action_mask = is_product | is_cart
    action_df = clicks[action_mask].copy()

    if action_df.empty:
        log.warning(f"  sub-{sid}: no product/cart actions in control window")
        return pd.DataFrame()

    # Assign original_marker_type
    def _marker_type(row):
        cn = str(row.get("class_name", ""))
        if pd.notna(row.get("class_name")) and any(
            k in cn.lower() for k in ["add-to-cart", "add_to_cart", "addtocart"]
        ):
            return "S30"
        return "S32"

    action_df["original_marker_type"] = action_df.apply(_marker_type, axis=1)
    action_df = action_df.sort_values("wall_time_ms").reset_index(drop=True)

    # Apply pre/post buffer filters
    pre_ok  = action_df["wall_time_ms"] - PRE_BUFFER_MS  >= ctrl_start_ms
    post_ok = action_df["wall_time_ms"] + POST_BUFFER_MS <= ctrl_end_ms
    action_df = action_df[pre_ok & post_ok].copy()

    if action_df.empty:
        log.warning(f"  sub-{sid}: all actions filtered by buffer constraints")
        return pd.DataFrame()

    # Apply 3000ms minimum gap (keep first of overlapping pair)
    kept_indices = []
    last_kept_ms = -np.inf
    for idx, row in action_df.iterrows():
        t = float(row["wall_time_ms"])
        if t - last_kept_ms >= MIN_GAP_MS:
            kept_indices.append(idx)
            last_kept_ms = t

    action_df = action_df.loc[kept_indices].copy().reset_index(drop=True)

    log.info(f"  sub-{sid}: {len(action_df)} action markers "
             f"(S30={int((action_df['original_marker_type']=='S30').sum())}, "
             f"S32={int((action_df['original_marker_type']=='S32').sum())})")

    # Get sfreq from existing epochs
    epo_path = proc_dir / "epochs_erp-epo.fif"
    epochs   = mne.read_epochs(str(epo_path), preload=False, verbose=False)
    sfreq    = epochs.info["sfreq"]

    # Build output DataFrame with all columns needed by helpers
    action_df["action_marker_id"] = np.arange(len(action_df))
    action_df["epoch_id"]         = action_df["action_marker_id"]
    action_df["event_code"]       = 88
    action_df["scenario_name"]    = "control_action_matched"
    action_df["phase"]            = "control"
    action_df["raw_sample"]       = ((action_df["wall_time_ms"] - raw_start_wall_ms) / 1000.0 * sfreq).astype(int)
    action_df["raw_time_s"]       = (action_df["wall_time_ms"] - raw_start_wall_ms) / 1000.0

    out_cols = [
        "action_marker_id", "wall_time_ms", "original_marker_type",
        "scenario_name", "phase", "epoch_id", "event_code",
        "raw_sample", "raw_time_s",
    ]
    result = action_df[out_cols].copy()
    result.to_csv(str(proc_dir / "control_action_matched_markers.csv"), index=False)
    return result


# ── Step 2: EEG epochs with AR ────────────────────────────────────────────────
def step2_eeg_action_matched(sid: int, markers: pd.DataFrame):
    """
    Cut ERP epochs around action markers and apply AutoReject.
    Returns cleaned MNE Epochs object.
    """
    if markers.empty:
        return None

    proc_dir = PROC_DIR / f"subject_{sid}"
    raw = mne.io.read_raw_fif(str(proc_dir / "raw_clean-raw.fif"),
                               preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    samples = markers["raw_sample"].values.astype(int)

    # Guard: drop any samples outside the raw data range
    n_times = raw.n_times
    valid_mask = (samples >= 0) & (samples + int(ERP_TMAX * sfreq) < n_times)
    if valid_mask.sum() < len(samples):
        log.warning(f"  sub-{sid}: {(~valid_mask).sum()} samples out of raw range - dropped")
        samples = samples[valid_mask]
        markers = markers[valid_mask].reset_index(drop=True)

    if len(samples) == 0:
        log.warning(f"  sub-{sid}: no valid samples after range check")
        return None

    events = np.column_stack([
        samples,
        np.zeros(len(samples), int),
        np.full(len(samples), 88, int),
    ])

    epo = mne.Epochs(
        raw, events, event_id={"action_matched": 88},
        tmin=ERP_TMIN, tmax=ERP_TMAX,
        baseline=(ERP_TMIN, 0), preload=True, verbose=False,
    )
    n_before = len(epo)

    if len(epo) > 1:
        ar = AutoReject(n_interpolate=[1, 2, 4], random_state=42, verbose=False)
        epo, _ = ar.fit_transform(epo, return_log=True)

    log.info(f"  sub-{sid} EEG: {n_before} → {len(epo)} epochs after AR")
    epo.save(str(proc_dir / "epochs_erp_action_matched-epo.fif"),
             overwrite=True, verbose=False)
    return epo


# ── Survived markers after AR ─────────────────────────────────────────────────
def survived_action_markers(epo_erp, original_markers: pd.DataFrame,
                             raw_start_wall_ms: float, sid: int) -> pd.DataFrame:
    """
    Build a DataFrame of survived markers based on samples remaining after AR,
    matching back to original action marker metadata.
    """
    if epo_erp is None or len(epo_erp) == 0:
        return pd.DataFrame()

    sfreq         = epo_erp.info["sfreq"]
    surv_samples  = epo_erp.events[:, 0]
    surv_wall_ms  = raw_start_wall_ms + surv_samples / sfreq * 1000

    # Match survived samples back to original markers
    orig_samples = original_markers["raw_sample"].values
    rows = []
    for i, (samp, wms) in enumerate(zip(surv_samples, surv_wall_ms)):
        # Find closest original marker (within 5-sample tolerance)
        dists = np.abs(orig_samples - samp)
        best = int(np.argmin(dists))
        if dists[best] <= 5:
            orig_row = original_markers.iloc[best].to_dict()
        else:
            # Fallback: just keep the sample information
            orig_row = {
                "original_marker_type": "S32",
                "scenario_name": "control_action_matched",
                "phase": "control",
                "event_code": 88,
            }
        rows.append({
            "epoch_id":             i,
            "action_marker_id":     i,
            "event_code":           88,
            "scenario_name":        "control_action_matched",
            "phase":                "control",
            "raw_sample":           int(samp),
            "raw_time_s":           float(samp / sfreq),
            "wall_time_ms":         float(wms),
            "original_marker_type": orig_row.get("original_marker_type", "S32"),
        })

    return pd.DataFrame(rows)


# ── Step 6: Build balanced v3 dataset ─────────────────────────────────────────
def step6_build_v3(ctrl_v3: dict) -> dict:
    """
    Build a balanced v3 dataset by pairing action-matched control epochs with
    stratified-sampled variant epochs from v2.

    ctrl_v3: {sid -> {eeg, eye, mouse, markers}}
    """
    # Load v2 arrays and metadata
    v2_eeg    = np.load(str(FEAT_DIR / "all_eeg_embeddings_v2.npy"))
    v2_eye    = np.load(str(FEAT_DIR / "all_eye_timeseries_v2.npy"))
    v2_mouse  = np.load(str(FEAT_DIR / "all_mouse_timeseries_v2.npy"))
    v2_meta   = pd.read_csv(str(FEAT_DIR / "all_eeg_embeddings_v2_metadata.csv"))
    v2_labels = pd.read_csv(str(FEAT_DIR / "labels_v2.csv"))

    all_eeg_parts   = []
    all_eye_parts   = []
    all_mouse_parts = []
    label_rows      = []
    meta_rows       = []
    global_idx      = 0

    summary_per_sub = {}

    for sid in SUBJECTS:
        if sid not in ctrl_v3:
            log.warning(f"  sub-{sid}: no control data - skipped in v3")
            continue

        d          = ctrl_v3[sid]
        ctrl_eeg   = d["eeg"]
        ctrl_eye   = d["eye"]
        ctrl_mouse = d["mouse"]
        ctrl_marks = d["markers"]
        n_ctrl     = len(ctrl_eeg)

        if n_ctrl < 3:
            log.warning(f"  sub-{sid}: WARNING only {n_ctrl} control epochs (< 3) - including anyway")

        # ── Variant epochs for this subject from v2 ──
        v2_sub_mask = (
            (v2_meta["subject_id"] == sid) &
            (v2_meta["phase"].isin(["variant_a", "variant_b", "variant_c"]))
        )
        v2_sub_meta = v2_meta[v2_sub_mask].copy()
        n_variant   = len(v2_sub_meta)

        if n_variant == 0:
            log.warning(f"  sub-{sid}: no variant epochs in v2 metadata - skip variant part")
            # Still add control epochs with 0 variant
            for i, (_, mr) in enumerate(ctrl_marks.iterrows()):
                all_eeg_parts.append(ctrl_eeg[i:i+1])
                all_eye_parts.append(ctrl_eye[i:i+1])
                all_mouse_parts.append(ctrl_mouse[i:i+1])
                label_rows.append({
                    "global_idx": global_idx, "subject_id": sid, "label": 0,
                    "phase": "control", "wall_time_ms": float(mr["wall_time_ms"]),
                    "source": "action_matched_control",
                })
                meta_rows.append({
                    "global_idx": global_idx, "subject_id": sid,
                    "epoch_id": int(mr["epoch_id"]), "event_id": 88,
                    "scenario_name": "control_action_matched", "phase": "control",
                    "wall_time_ms": float(mr["wall_time_ms"]),
                    "label_class": 0,
                })
                global_idx += 1
            summary_per_sub[sid] = {"n_ctrl": n_ctrl, "n_variant_sampled": 0}
            continue

        # ── Stratified sampling of variant epochs ──
        n_sample = min(n_ctrl, n_variant)
        if n_ctrl >= n_variant:
            # No sampling needed - take all variant epochs
            sampled_meta = v2_sub_meta.copy()
        else:
            # Stratified sample by scenario_name
            scenario_col = "scenario_name"
            sampled_rows = []
            strat_groups = v2_sub_meta.groupby(scenario_col)
            total_weight = len(v2_sub_meta)
            remaining    = n_sample
            # Proportional allocation
            for grp_name, grp_df in strat_groups:
                proportion = len(grp_df) / total_weight
                n_grp = max(1, round(proportion * n_sample))
                n_grp = min(n_grp, len(grp_df))
                sampled_rows.append(
                    grp_df.sample(n=n_grp, random_state=42)
                )
            sampled_meta = pd.concat(sampled_rows, ignore_index=True)
            # Trim or pad to exact n_sample
            if len(sampled_meta) > n_sample:
                sampled_meta = sampled_meta.sample(n=n_sample, random_state=42)
            elif len(sampled_meta) < n_sample:
                # Fill from remaining rows
                remaining_pool = v2_sub_meta[~v2_sub_meta.index.isin(sampled_meta.index)]
                n_fill = min(n_sample - len(sampled_meta), len(remaining_pool))
                if n_fill > 0:
                    extra = remaining_pool.sample(n=n_fill, random_state=42)
                    sampled_meta = pd.concat([sampled_meta, extra], ignore_index=True)

        n_variant_sampled = len(sampled_meta)
        log.info(f"  sub-{sid}: {n_ctrl} ctrl, {n_variant} variant total, "
                 f"{n_variant_sampled} variant sampled")

        # ── Add control epochs ──
        for i in range(n_ctrl):
            mr = ctrl_marks.iloc[i]
            all_eeg_parts.append(ctrl_eeg[i:i+1])
            all_eye_parts.append(ctrl_eye[i:i+1])
            all_mouse_parts.append(ctrl_mouse[i:i+1])
            label_rows.append({
                "global_idx": global_idx, "subject_id": sid, "label": 0,
                "phase": "control", "wall_time_ms": float(mr["wall_time_ms"]),
                "source": "action_matched_control",
            })
            meta_rows.append({
                "global_idx": global_idx, "subject_id": sid,
                "epoch_id": int(mr["epoch_id"]), "event_id": 88,
                "scenario_name": "control_action_matched", "phase": "control",
                "wall_time_ms": float(mr["wall_time_ms"]),
                "label_class": 0,
            })
            global_idx += 1

        # ── Add variant epochs (from v2 arrays) ──
        for _, vr in sampled_meta.iterrows():
            v_idx = int(vr["global_idx"])  # index into v2 arrays
            all_eeg_parts.append(v2_eeg[v_idx:v_idx+1])
            all_eye_parts.append(v2_eye[v_idx:v_idx+1])
            all_mouse_parts.append(v2_mouse[v_idx:v_idx+1])
            label_rows.append({
                "global_idx": global_idx, "subject_id": sid, "label": 1,
                "phase": str(vr.get("phase", "variant")),
                "wall_time_ms": float(vr.get("wall_time_ms", 0.0)),
                "source": "variant_v2",
            })
            meta_rows.append({
                "global_idx": global_idx, "subject_id": sid,
                "epoch_id": int(vr.get("epoch_id", 0)),
                "event_id": str(vr.get("event_id", "1")),
                "scenario_name": str(vr.get("scenario_name", "variant")),
                "phase": str(vr.get("phase", "variant")),
                "wall_time_ms": float(vr.get("wall_time_ms", 0.0)),
                "label_class": 1,
            })
            global_idx += 1

        summary_per_sub[sid] = {
            "n_ctrl": n_ctrl,
            "n_variant_sampled": n_variant_sampled,
        }

    if not all_eeg_parts:
        log.error("No data to build v3 dataset!")
        return {}

    # ── Concatenate ──
    v3_eeg   = np.concatenate(all_eeg_parts, axis=0)
    v3_eye   = np.concatenate(all_eye_parts, axis=0)
    v3_mouse = np.concatenate(all_mouse_parts, axis=0)
    v3_labels = pd.DataFrame(label_rows)
    v3_meta   = pd.DataFrame(meta_rows)

    n_ctrl    = int((v3_labels["label"] == 0).sum())
    n_variant = int((v3_labels["label"] == 1).sum())

    log.info(f"  v3: control={n_ctrl}, variant={n_variant}, total={n_ctrl+n_variant}")
    log.info(f"  v3 shapes: EEG={v3_eeg.shape}, Eye={v3_eye.shape}, Mouse={v3_mouse.shape}")

    # ── Save ──
    np.save(str(FEAT_DIR / "all_eeg_embeddings_v3.npy"),   v3_eeg)
    np.save(str(FEAT_DIR / "all_eye_timeseries_v3.npy"),   v3_eye)
    np.save(str(FEAT_DIR / "all_mouse_timeseries_v3.npy"), v3_mouse)
    v3_labels.to_csv(str(FEAT_DIR / "labels_v3.csv"), index=False)
    v3_meta.to_csv(str(FEAT_DIR / "all_eeg_embeddings_v3_metadata.csv"), index=False)

    log.info(f"  Saved v3 arrays → {FEAT_DIR}")
    return {
        "n_control": n_ctrl,
        "n_variant": n_variant,
        "n_total": n_ctrl + n_variant,
        "shapes": {
            "eeg":   v3_eeg.shape,
            "eye":   v3_eye.shape,
            "mouse": v3_mouse.shape,
        },
        "per_subject": summary_per_sub,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg     = load_config()
    log.info("Loading LaBraM encoder...")
    encoder = load_labram_base()

    ctrl_v3 = {}   # sid -> {eeg, eye, mouse, markers}
    marker_counts = {}  # sid -> (before_ar, after_ar)

    for sid in SUBJECTS:
        log.info(f"\n{'='*60}")
        log.info(f"Subject {sid}")
        log.info(f"{'='*60}")

        raw_start_wall_ms = get_raw_start_wall_ms(sid)

        # Step 1: action-matched markers
        markers = step1_action_markers(sid, raw_start_wall_ms)
        n_before_ar = len(markers)
        if markers.empty or n_before_ar == 0:
            log.warning(f"  sub-{sid}: WARNING no action markers found")
            marker_counts[sid] = (0, 0)
            continue

        # Step 2: EEG epochs + AR
        epo_erp = step2_eeg_action_matched(sid, markers)
        if epo_erp is None or len(epo_erp) == 0:
            log.warning(f"  sub-{sid}: WARNING all epochs rejected by AR")
            marker_counts[sid] = (n_before_ar, 0)
            continue

        n_after_ar = len(epo_erp)
        marker_counts[sid] = (n_before_ar, n_after_ar)

        if n_after_ar < 3:
            log.warning(f"  sub-{sid}: WARNING only {n_after_ar} epochs survived AR (< 3)")

        # Build survived markers DataFrame
        survived = survived_action_markers(epo_erp, markers, raw_start_wall_ms, sid)
        if survived.empty:
            log.warning(f"  sub-{sid}: survived markers empty")
            marker_counts[sid] = (n_before_ar, 0)
            continue

        # Steps 3-5: eye, mouse, LaBraM
        info = subject_info(cfg, sid)

        log.info(f"  sub-{sid}: [3] Eye features")
        _, _, eye_ts = step3_eye_features(sid, survived, info, raw_start_wall_ms)

        # Rename eye files to action_matched naming
        proc_dir = PROC_DIR / f"subject_{sid}"
        for old_name, new_name in [
            ("eye_epoch_features_control_erp.csv",
             "eye_epoch_features_action_matched_erp.csv"),
            ("eye_epoch_features_control_causal.csv",
             "eye_epoch_features_action_matched_causal.csv"),
            ("eye_timeseries_control_erp.npy",
             "eye_timeseries_action_matched_erp.npy"),
        ]:
            old_path = proc_dir / old_name
            new_path = proc_dir / new_name
            if old_path.exists():
                import shutil
                shutil.copy2(str(old_path), str(new_path))

        log.info(f"  sub-{sid}: [4] Mouse features")
        _, _, mouse_ts = step4_mouse_features(sid, survived)

        # Rename mouse files to action_matched naming
        for old_name, new_name in [
            ("mouse_epoch_features_control_erp.csv",
             "mouse_epoch_features_action_matched_erp.csv"),
            ("mouse_epoch_features_control_causal.csv",
             "mouse_epoch_features_action_matched_causal.csv"),
            ("mouse_timeseries_control_erp.npy",
             "mouse_timeseries_action_matched_erp.npy"),
        ]:
            old_path = proc_dir / old_name
            new_path = proc_dir / new_name
            if old_path.exists():
                import shutil
                shutil.copy2(str(old_path), str(new_path))

        log.info(f"  sub-{sid}: [5] LaBraM embeddings")
        eeg_embs = step5_labram(sid, epo_erp, encoder)

        # Save to action_matched-specific path
        feat_sub = FEAT_DIR / f"subject_{sid}"
        feat_sub.mkdir(parents=True, exist_ok=True)
        am_emb_path = feat_sub / "eeg_embeddings_action_matched.npy"
        np.save(str(am_emb_path), eeg_embs)
        log.info(f"  sub-{sid}: EEG embeddings → {am_emb_path}")

        # Align lengths across modalities
        n = min(len(eeg_embs), len(eye_ts), len(mouse_ts), len(survived))
        ctrl_v3[sid] = {
            "eeg":     eeg_embs[:n],
            "eye":     eye_ts[:n],
            "mouse":   mouse_ts[:n],
            "markers": survived.head(n),
        }
        log.info(f"  sub-{sid}: {n} action-matched control epochs ready")

    # Step 6: balanced v3 dataset
    log.info("\n[6] Building balanced v3 dataset")
    summary = step6_build_v3(ctrl_v3)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("v3 EXTRACTION REPORT")
    print("="*70)
    print("\nPer-subject marker counts:")
    print(f"{'Subject':<12} {'Before AR':<12} {'After AR':<12} {'Status'}")
    print("-" * 50)
    for sid in SUBJECTS:
        before, after = marker_counts.get(sid, (0, 0))
        status = ""
        if before == 0:
            status = "WARNING: no markers"
        elif after == 0:
            status = "WARNING: all rejected"
        elif after < 3:
            status = f"WARNING: only {after} survived"
        print(f"sub-{sid:<8} {before:<12} {after:<12} {status}")

    if summary:
        print(f"\nv3 Dataset Summary:")
        print(f"  Control (action-matched): {summary['n_control']}")
        print(f"  Variant (task):           {summary['n_variant']}")
        print(f"  Total:                    {summary['n_total']}")
        print(f"\nv3 Array Shapes:")
        print(f"  EEG:   {summary['shapes']['eeg']}")
        print(f"  Eye:   {summary['shapes']['eye']}")
        print(f"  Mouse: {summary['shapes']['mouse']}")

        print("\nlabels_v3.csv (first 5 rows):")
        ldf = pd.read_csv(str(FEAT_DIR / "labels_v3.csv"))
        print(ldf.head(5).to_string(index=False))
        print("\nlabels_v3.csv value_counts(label):")
        print(ldf["label"].value_counts().to_string())
        print("\nlabels_v3.csv value_counts(source):")
        print(ldf["source"].value_counts().to_string())

    print("\n" + "="*70)
    print("v3 extraction complete.")
    print("="*70)


if __name__ == "__main__":
    main()
