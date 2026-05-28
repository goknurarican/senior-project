from pathlib import Path
import yaml
import numpy as np


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    data_root = Path(cfg["data_root"])
    eeg_dir = data_root / cfg["input_dirs"]["eeg"]
    eye_dir = data_root / cfg["input_dirs"]["eye"]
    mouse_dir = data_root / cfg["input_dirs"]["mouse"]
    align_dir = data_root / cfg["input_dirs"]["alignment"]

    print("=" * 70)
    print("RF BASELINE INPUT CHECK")
    print("=" * 70)
    print(f"data_root: {data_root}")

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    rows = []

    for sid in cfg["subjects"]:
        eeg_path = eeg_dir / f"subject_{sid}" / "epochs_causal-epo.fif"
        eye_path = eye_dir / f"subject_{sid}" / "eye_timeseries_causal.npy"
        mouse_path = mouse_dir / f"subject_{sid}" / "mouse_timeseries_causal.npy"
        align_path = align_dir / f"alignment_master_subject_{sid}.csv"

        eeg_ok = eeg_path.exists()
        eye_ok = eye_path.exists()
        mouse_ok = mouse_path.exists()
        align_ok = align_path.exists()

        eye_shape = None
        mouse_shape = None

        if eye_ok:
            try:
                eye_shape = np.load(eye_path, mmap_mode="r").shape
            except Exception as e:
                eye_shape = f"ERROR: {e}"

        if mouse_ok:
            try:
                mouse_shape = np.load(mouse_path, mmap_mode="r").shape
            except Exception as e:
                mouse_shape = f"ERROR: {e}"

        rows.append({
            "subject": sid,
            "eeg": eeg_ok,
            "eye": eye_ok,
            "mouse": mouse_ok,
            "alignment": align_ok,
            "eye_shape": eye_shape,
            "mouse_shape": mouse_shape,
        })

    print()
    print(f"{'Subject':<10} {'EEG':<6} {'Eye':<6} {'Mouse':<7} {'Align':<7} {'Eye shape':<20} {'Mouse shape'}")
    print("-" * 100)

    all_ok = True

    for r in rows:
        ok = r["eeg"] and r["eye"] and r["mouse"] and r["alignment"]
        all_ok = all_ok and ok

        print(
            f"{r['subject']:<10} "
            f"{'OK' if r['eeg'] else 'MISS':<6} "
            f"{'OK' if r['eye'] else 'MISS':<6} "
            f"{'OK' if r['mouse'] else 'MISS':<7} "
            f"{'OK' if r['alignment'] else 'MISS':<7} "
            f"{str(r['eye_shape']):<20} "
            f"{str(r['mouse_shape'])}"
        )

    print("-" * 100)

    if all_ok:
        print("All required input files were found.")
    else:
        print("Some files are missing. Fix paths before continuing.")


if __name__ == "__main__":
    main()