"""
data/raw/ içindeki ham verilerin analitik görselleştirmesi.
Her denek için ayrı bir dashboard + ortak bir özet figür üretir.
Çıktı: data/raw/figures/
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW     = Path(__file__).parent.parent / "data" / "raw"
FIGURES = RAW / "figures"
FIGURES.mkdir(exist_ok=True)

PHASE_COLORS  = {"control": "#4C9BE8", "variant_b": "#E8834C", "variant_c": "#6CC08B"}
MODAL_COLORS  = {"EEG": "#2E86AB", "Eye": "#A23B72", "Platform": "#F18F01"}
SCENARIO_PALETTE = plt.cm.tab20.colors

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4A",
    "axes.labelcolor":  "#D0D3E0",
    "xtick.color":      "#9A9DB0",
    "ytick.color":      "#9A9DB0",
    "text.color":       "#D0D3E0",
    "grid.color":       "#2A2D3A",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "axes.titlesize":   10,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
})


# ──────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────

def load_subject(sdir: Path) -> dict:
    meta  = json.loads((sdir / "metadata.json").read_text())
    eeg_s = meta.get("eeg_start_wall_ms") if "eeg_start_wall_ms" in meta else None

    eye   = pd.read_csv(sdir / "eye" / "eye_data_db.csv")
    em    = pd.read_csv(sdir / "eeg" / "eeg_markers.csv")
    ae    = pd.read_csv(sdir / "platform" / "all_events.csv")
    mt    = pd.read_csv(sdir / "platform" / "mouse_trajectory_summary.csv")

    # EEG penceresi → platform + eye'ı kırp
    eeg_t0 = em["wall_time_ms"].min()
    eeg_t1 = em["wall_time_ms"].max()
    ae_clipped = ae[(ae["timestamp"] >= eeg_t0) & (ae["timestamp"] <= eeg_t1)].copy()
    mt_clipped = mt[(mt["wall_time_ms_start"] >= eeg_t0) & (mt["wall_time_ms_end"] <= eeg_t1)].copy()

    # Klikler (event_data'dan x,y çek)
    clicks_raw = ae_clipped[ae_clipped["event_type"] == "mouse_click"].copy()
    def parse_click(row):
        try:
            d = json.loads(row["event_data"])
            return pd.Series({"cx": d["x"], "cy": d["y"], "sw": d.get("screen_w", 1920), "sh": d.get("screen_h", 1080)})
        except Exception:
            return pd.Series({"cx": np.nan, "cy": np.nan, "sw": 1920, "sh": 1080})
    if len(clicks_raw):
        clicks_xy = clicks_raw.apply(parse_click, axis=1)
        clicks_raw = pd.concat([clicks_raw.reset_index(drop=True), clicks_xy], axis=1)

    return dict(
        meta=meta, eye=eye, em=em,
        ae=ae_clipped, mt=mt_clipped, clicks=clicks_raw,
        eeg_t0=eeg_t0, eeg_t1=eeg_t1,
    )


def rel_sec(ms_series, t0):
    return (ms_series - t0) / 1000


# ──────────────────────────────────────────────────────────────
# Figure 1 - Per-subject dashboard
# ──────────────────────────────────────────────────────────────

def plot_subject_dashboard(d: dict, out_path: Path):
    meta = d["meta"]
    eye  = d["eye"]
    em   = d["em"]
    ae   = d["ae"]
    mt   = d["mt"]
    t0   = d["eeg_t0"]

    uid   = meta["user_id"]
    name  = meta["name"]
    group = meta["group"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle(
        f"USER {uid:02d} - {name}  |  {group.upper()}",
        fontsize=13, color="#E8EAF6", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.52, wspace=0.32,
        top=0.93, bottom=0.06, left=0.07, right=0.97
    )

    # ── Row 0: Modality alignment timeline (spans all 3 cols) ──
    ax_align = fig.add_subplot(gs[0, :])
    _plot_alignment(ax_align, d, t0)

    # ── Row 1 col 0: EEG marker density ──
    ax_eeg = fig.add_subplot(gs[1, 0])
    _plot_eeg_markers(ax_eeg, em, t0)

    # ── Row 1 col 1: Eye validity over time ──
    ax_eye_v = fig.add_subplot(gs[1, 1])
    _plot_eye_validity(ax_eye_v, eye, t0)

    # ── Row 1 col 2: Pupil size over time ──
    ax_pupil = fig.add_subplot(gs[1, 2])
    _plot_pupil(ax_pupil, eye, t0)

    # ── Row 2 col 0: Gaze heatmap ──
    ax_gaze = fig.add_subplot(gs[2, 0])
    _plot_gaze_heatmap(ax_gaze, eye)

    # ── Row 2 col 1: Mouse trajectory stats ──
    ax_mt = fig.add_subplot(gs[2, 1])
    _plot_mouse_traj(ax_mt, mt, t0)

    # ── Row 2 col 2: Platform event timeline ──
    ax_plat = fig.add_subplot(gs[2, 2])
    _plot_platform_events(ax_plat, ae, t0)

    # ── Row 3 col 0: Scenario type distribution ──
    ax_sc = fig.add_subplot(gs[3, 0])
    _plot_scenario_dist(ax_sc, em)

    # ── Row 3 col 1: Click heatmap ──
    ax_cl = fig.add_subplot(gs[3, 1])
    _plot_click_heatmap(ax_cl, d["clicks"])

    # ── Row 3 col 2: Data quality card ──
    ax_q = fig.add_subplot(gs[3, 2])
    _plot_quality_card(ax_q, d)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


# ── subplot functions ──────────────────────────────────────────

def _plot_alignment(ax, d, t0):
    eye, em, ae = d["eye"], d["em"], d["ae"]
    eeg_t0, eeg_t1 = d["eeg_t0"], d["eeg_t1"]

    rows = [
        ("EEG markers", em["wall_time_ms"].min(), em["wall_time_ms"].max(), MODAL_COLORS["EEG"]),
        ("Eye / Gaze",  eye["wall_time_ms"].min(), eye["wall_time_ms"].max(), MODAL_COLORS["Eye"]),
        ("Platform",    ae["timestamp"].min(),      ae["timestamp"].max(),     MODAL_COLORS["Platform"]),
    ]

    for i, (label, start, end, color) in enumerate(rows):
        s = (start - t0) / 1000
        e = (end   - t0) / 1000
        ax.barh(i, e - s, left=s, height=0.5, color=color, alpha=0.85)
        ax.text(s + (e - s) / 2, i, f"{(e-s)/60:.1f} min", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

    # Senaryo trigger anları
    sc = ae[ae["event_type"] == "SCENARIO_TRIGGERED"]
    for ts in sc["timestamp"]:
        ax.axvline((ts - t0) / 1000, color="#FFD700", alpha=0.25, linewidth=0.6)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("EEG kaydı başından itibaren (s)")
    ax.set_title("Modalite Zaman Pencereleri  (sarı çizgiler = senaryo tetikleri)")
    ax.grid(axis="x")
    ax.set_xlim(left=-30)


def _plot_eeg_markers(ax, em, t0):
    t = rel_sec(em["wall_time_ms"], t0)
    # Blink vs non-blink
    blink = em["scenario_type"] == "blink"
    ax.scatter(t[blink],  [1]*blink.sum(),  s=1.5, alpha=0.3, color="#9A9DB0", label="blink")
    ax.scatter(t[~blink], [1]*(~blink).sum(), s=8,  alpha=0.9, color="#FFD700", label="senaryo", zorder=3)

    # Kernel density estimate - non-blink marker yoğunluğu
    from scipy.stats import gaussian_kde
    nb = t[~blink]
    if len(nb) > 2:
        kde = gaussian_kde(nb, bw_method=0.15)
        xs  = np.linspace(t.min(), t.max(), 400)
        ax2 = ax.twinx()
        ax2.fill_between(xs, kde(xs), alpha=0.25, color="#FFD700")
        ax2.set_yticks([])
        ax2.set_ylabel("Yoğunluk", fontsize=6)

    ax.set_yticks([])
    ax.set_xlabel("Süre (s)")
    ax.set_title(f"EEG Marker Zaman Dağılımı  (n={len(em):,})")
    ax.legend(loc="upper right", fontsize=6, markerscale=3)
    ax.grid(axis="x")


def _plot_eye_validity(ax, eye, t0):
    t = rel_sec(eye["wall_time_ms"], t0)
    # 5 saniyelik pencerede ortalama al
    bin_s = 5
    n_bins = int((t.max() - t.min()) / bin_s) + 1
    bins   = np.arange(t.min(), t.max() + bin_s, bin_s)
    bpog_m = []
    fpog_m = []
    bin_c  = []
    for i in range(len(bins) - 1):
        mask = (t >= bins[i]) & (t < bins[i+1])
        if mask.sum() > 0:
            bpog_m.append(eye.loc[mask, "bpogv"].mean())
            fpog_m.append(eye.loc[mask, "fpogv"].mean())
            bin_c.append((bins[i] + bins[i+1]) / 2)

    ax.plot(bin_c, bpog_m, lw=1.2, color="#4CAF50", label="BPOG")
    ax.plot(bin_c, fpog_m, lw=1.2, color="#FF9800", label="FPOG", alpha=0.8)
    ax.fill_between(bin_c, bpog_m, alpha=0.15, color="#4CAF50")
    ax.axhline(0.8, color="#FF5252", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Süre (s)")
    ax.set_ylabel("Geçerlilik oranı")
    ax.set_title("Göz İzleme Geçerliliği (5s pencere)")
    ax.legend(fontsize=7)
    ax.grid()


def _plot_pupil(ax, eye, t0):
    t  = rel_sec(eye["wall_time_ms"], t0)
    # Downsample: her 10. nokta
    step = max(1, len(t) // 2000)
    ts   = t.iloc[::step]
    pl   = eye["pupil_left"].iloc[::step]
    pr   = eye["pupil_right"].iloc[::step]

    # Bölgesel renklendirme: phase
    phase_colors_local = PHASE_COLORS
    for phase, grp in eye.groupby("phase"):
        tg = rel_sec(grp["wall_time_ms"], t0)
        color = phase_colors_local.get(phase, "#AAAAAA")
        ax.axvspan(tg.min(), tg.max(), alpha=0.08, color=color)

    ax.plot(ts, pl, lw=0.6, alpha=0.7, color="#A23B72", label="Sol")
    ax.plot(ts, pr, lw=0.6, alpha=0.7, color="#E8834C", label="Sağ")
    ax.set_xlabel("Süre (s)")
    ax.set_ylabel("Pupil büyüklüğü")
    ax.set_title("Pupil Boyutu Zaman Serisi")
    ax.legend(fontsize=7)
    ax.grid()


def _plot_gaze_heatmap(ax, eye):
    # Normalize gaze_x/y [0,1] - 0=sol/üst, 1=sağ/alt
    gx = eye["gaze_x"].clip(0, 1)
    gy = 1 - eye["gaze_y"].clip(0, 1)  # y'yi çevir (0=üst)

    h, xedges, yedges = np.histogram2d(gx, gy, bins=60, range=[[0,1],[0,1]])
    h = np.log1p(h.T)
    ax.imshow(h, origin="upper", aspect="auto",
              extent=[0, 1, 0, 1], cmap="plasma", interpolation="bilinear")
    ax.set_xlabel("Gaze X (normalize)")
    ax.set_ylabel("Gaze Y (normalize)")
    ax.set_title("Gaze Isı Haritası")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def _plot_mouse_traj(ax, mt, t0):
    if mt.empty:
        ax.text(0.5, 0.5, "Veri yok", ha="center", va="center")
        return
    t = rel_sec(mt["wall_time_ms_start"], t0)
    sc = ax.scatter(t, mt["mean_velocity"], c=mt["path_length_px"],
                    cmap="viridis", s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Path length (px)", pad=0.02)
    ax.set_xlabel("Süre (s)")
    ax.set_ylabel("Ortalama hız (px/s)")
    ax.set_title("Mouse Trajectory Hız & Uzunluk")
    ax.grid()


def _plot_platform_events(ax, ae, t0):
    event_types = [e for e in ae["event_type"].unique() if e != "mouse_trajectory"]
    colors_map  = {e: SCENARIO_PALETTE[i % 20] for i, e in enumerate(event_types)}

    for i, etype in enumerate(event_types):
        sub = ae[ae["event_type"] == etype]
        ts  = rel_sec(sub["timestamp"], t0)
        ax.scatter(ts, [i] * len(ts), s=12, color=colors_map[etype], alpha=0.7, edgecolors="none")
        ax.text(-5, i, etype, ha="right", va="center", fontsize=5.5, color=colors_map[etype])

    ax.set_yticks([])
    ax.set_xlabel("Süre (s)")
    ax.set_title("Platform Event Zaman Çizelgesi")
    ax.grid(axis="x")


def _plot_scenario_dist(ax, em):
    sc = em[em["scenario_type"] != "blink"]["scenario_type"].value_counts()
    colors = [SCENARIO_PALETTE[i % 20] for i in range(len(sc))]
    bars = ax.barh(sc.index[::-1], sc.values[::-1], color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, sc.values[::-1]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=6)
    ax.set_xlabel("Marker sayısı")
    ax.set_title("EEG Senaryo Marker Dağılımı")
    ax.grid(axis="x")


def _plot_click_heatmap(ax, clicks):
    if clicks.empty or "cx" not in clicks.columns:
        ax.text(0.5, 0.5, "Klik verisi yok", ha="center", va="center")
        return
    valid = clicks.dropna(subset=["cx", "cy"])
    sw = valid["sw"].mode().iloc[0] if len(valid) else 1920
    sh = valid["sh"].mode().iloc[0] if len(valid) else 1080
    # Normalize
    nx = (valid["cx"] / sw).clip(0, 1)
    ny = 1 - (valid["cy"] / sh).clip(0, 1)

    h, xe, ye = np.histogram2d(nx, ny, bins=40, range=[[0,1],[0,1]])
    h = np.log1p(h.T)
    ax.imshow(h, origin="upper", aspect="auto",
              extent=[0, 1, 0, 1], cmap="hot", interpolation="bilinear")
    # Ham noktaları üste ekle
    ax.scatter(nx, ny, s=6, alpha=0.4, color="white", edgecolors="none")
    ax.set_xlabel("X (normalize)")
    ax.set_ylabel("Y (normalize)")
    ax.set_title(f"Klik Isı Haritası  (n={len(valid)})")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def _plot_quality_card(ax, d):
    eye, em, ae, mt = d["eye"], d["em"], d["ae"], d["mt"]
    eeg_dur = (em["wall_time_ms"].max() - em["wall_time_ms"].min()) / 1000
    eye_dur = (eye["wall_time_ms"].max() - eye["wall_time_ms"].min()) / 1000
    plat_dur = (ae["timestamp"].max() - ae["timestamp"].min()) / 1000 if len(ae) > 1 else 0

    bpog = eye["bpogv"].mean() * 100
    fpog = eye["fpogv"].mean() * 100
    pupil_l = eye["pupil_left"].gt(0).mean() * 100
    pupil_r = eye["pupil_right"].gt(0).mean() * 100
    n_sc = len(em[em["scenario_type"] != "blink"])

    lines = [
        ("EEG süresi",    f"{eeg_dur/60:.1f} dk"),
        ("EEG kanallar",  f"{d['meta'].get('counts', {}).get('all_events', '?')} ev  |  35ch @ 500Hz"),
        ("EEG marker",    f"{len(em):,} satır  ({n_sc} senaryo)"),
        ("─────────────", "──────────────────"),
        ("Eye süresi",    f"{eye_dur/60:.1f} dk"),
        ("Eye satır",     f"{len(eye):,}  @ ~62 Hz"),
        ("BPOG geçerl",   f"{bpog:.1f}%"),
        ("FPOG geçerl",   f"{fpog:.1f}%"),
        ("Pupil L/R",     f"{pupil_l:.1f}% / {pupil_r:.1f}%"),
        ("─────────────", "──────────────────"),
        ("Platform ev",   f"{len(ae)} (EEG penceresinde)"),
        ("Scenario trig", f"{ae['event_type'].eq('SCENARIO_TRIGGERED').sum()}"),
        ("Mouse traj",    f"{len(mt)} hareket"),
    ]

    ax.axis("off")
    ax.set_title("Veri Kalite Özeti", pad=6)
    for i, (k, v) in enumerate(lines):
        y = 1 - (i + 0.5) / len(lines)
        color = "#6CC08B" if "─" not in k else "#3A3D4A"
        ax.text(0.02, y, k, transform=ax.transAxes, fontsize=7,
                color="#9A9DB0", va="center")
        ax.text(0.52, y, v, transform=ax.transAxes, fontsize=7,
                color=color, va="center", fontweight="bold")


# ──────────────────────────────────────────────────────────────
# Figure 2 - Cross-subject comparison
# ──────────────────────────────────────────────────────────────

def plot_cross_subject(subjects: list, out_path: Path):
    n = len(subjects)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Denekler Arası Karşılaştırma", fontsize=13,
                 color="#E8EAF6", fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#0F1117")

    names   = [d["meta"]["name"] for d in subjects]
    groups  = [d["meta"]["group"] for d in subjects]
    colors  = [PHASE_COLORS.get(g, "#AAAAAA") for g in groups]

    # 0,0 - EEG süre
    ax = axes[0, 0]
    eeg_durs = [(d["em"]["wall_time_ms"].max() - d["em"]["wall_time_ms"].min()) / 60000 for d in subjects]
    ax.bar(names, eeg_durs, color=colors, alpha=0.85)
    ax.set_title("EEG Kayıt Süresi (dk)")
    ax.set_ylabel("Dakika")
    ax.grid(axis="y")
    for i, v in enumerate(eeg_durs):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)

    # 0,1 - Eye geçerlilik
    ax = axes[0, 1]
    bpogs = [d["eye"]["bpogv"].mean() * 100 for d in subjects]
    fpogs = [d["eye"]["fpogv"].mean() * 100 for d in subjects]
    x = np.arange(n)
    ax.bar(x - 0.18, bpogs, width=0.35, label="BPOG", color="#4CAF50", alpha=0.85)
    ax.bar(x + 0.18, fpogs, width=0.35, label="FPOG", color="#FF9800", alpha=0.85)
    ax.axhline(80, color="#FF5252", ls="--", lw=0.8, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_title("Eye Tracking Geçerliliği (%)")
    ax.legend(fontsize=7); ax.grid(axis="y")

    # 0,2 - Pupil boyutu (sol)
    ax = axes[0, 2]
    for d, nm, c in zip(subjects, names, colors):
        eye = d["eye"]
        t   = rel_sec(eye["wall_time_ms"], d["eeg_t0"])
        step = max(1, len(t) // 800)
        ax.plot(t.iloc[::step], eye["pupil_left"].iloc[::step], lw=0.8, alpha=0.75, label=nm, color=c)
    ax.set_xlabel("Süre (s)"); ax.set_ylabel("Pupil L")
    ax.set_title("Pupil Boyutu Karşılaştırması")
    ax.legend(fontsize=7); ax.grid()

    # 1,0 - Senaryo sayısı
    ax = axes[1, 0]
    sc_counts = [
        d["em"][d["em"]["scenario_type"] != "blink"]["scenario_type"].value_counts()
        for d in subjects
    ]
    all_types = sorted(set().union(*[s.index for s in sc_counts]))
    x = np.arange(len(all_types))
    w = 0.25
    for i, (d, nm, c) in enumerate(zip(subjects, names, colors)):
        vals = [sc_counts[i].get(t, 0) for t in all_types]
        ax.bar(x + i * w, vals, width=w, label=nm, color=c, alpha=0.85)
    ax.set_xticks(x + w); ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=6)
    ax.set_title("Senaryo Marker Dağılımı")
    ax.legend(fontsize=7); ax.grid(axis="y")

    # 1,1 - Mouse ortalama hız dağılımı (violin)
    ax = axes[1, 1]
    data_v = [d["mt"]["mean_velocity"].dropna().values for d in subjects]
    vp = ax.violinplot(data_v, positions=range(n), showmedians=True, showextrema=True)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.7)
    vp["cmedians"].set_color("white")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=7)
    ax.set_title("Mouse Hız Dağılımı (px/s)")
    ax.set_ylabel("Ortalama hız"); ax.grid(axis="y")

    # 1,2 - Path length dağılımı
    ax = axes[1, 2]
    data_p = [d["mt"]["path_length_px"].dropna().values for d in subjects]
    vp2 = ax.violinplot(data_p, positions=range(n), showmedians=True)
    for body, c in zip(vp2["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.7)
    vp2["cmedians"].set_color("white")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=7)
    ax.set_title("Mouse Path Uzunluğu (px)")
    ax.set_ylabel("Path length"); ax.grid(axis="y")

    # 2,0 - Gaze heatmap yan yana (küçük)
    for col, d in enumerate(subjects[:3]):
        ax = axes[2, col]
        eye = d["eye"]
        gx  = eye["gaze_x"].clip(0, 1)
        gy  = 1 - eye["gaze_y"].clip(0, 1)
        h, _, _ = np.histogram2d(gx, gy, bins=50, range=[[0,1],[0,1]])
        h = np.log1p(h.T)
        ax.imshow(h, origin="upper", aspect="auto",
                  extent=[0,1,0,1], cmap="plasma", interpolation="bilinear")
        ax.set_title(f"Gaze - {subjects[col]['meta']['name']}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


# ──────────────────────────────────────────────────────────────
# Figure 3 - EEG marker timeline (all subjects stacked)
# ──────────────────────────────────────────────────────────────

def plot_eeg_timeline(subjects: list, out_path: Path):
    fig, axes = plt.subplots(len(subjects), 1, figsize=(16, 5 * len(subjects)),
                             sharex=False)
    fig.suptitle("EEG Marker Zaman Serisi - Tüm Denekler",
                 fontsize=13, color="#E8EAF6", fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#0F1117")

    if len(subjects) == 1:
        axes = [axes]

    scenario_types = sorted(set().union(*[
        set(d["em"][d["em"]["scenario_type"] != "blink"]["scenario_type"].unique())
        for d in subjects
    ]))
    color_map = {st: SCENARIO_PALETTE[i % 20] for i, st in enumerate(scenario_types)}

    for ax, d in zip(axes, subjects):
        em   = d["em"]
        t0   = d["eeg_t0"]
        name = d["meta"]["name"]
        group = d["meta"]["group"]

        t_all = rel_sec(em["wall_time_ms"], t0)
        blink = em["scenario_type"] == "blink"

        # Blink markers (çok sayıda - binned density)
        ax.hist(t_all[blink], bins=120, color="#3A3D4A", alpha=0.6, label="blink yoğunluk")

        # Senaryo markerleri
        for st in scenario_types:
            mask = (em["scenario_type"] == st) & (~blink)
            t_sc = t_all[mask]
            if len(t_sc):
                ax.scatter(t_sc, [-30] * len(t_sc), s=60,
                           color=color_map[st], label=st, zorder=5,
                           marker="|", linewidths=1.5)

        # Phase bölgeleri
        for phase, pc in PHASE_COLORS.items():
            mask_p = em["phase"] == phase
            if mask_p.sum():
                tp = t_all[mask_p]
                ax.axvspan(tp.min(), tp.max(), alpha=0.06, color=pc)

        ax.set_title(f"{name}  ({group})", fontsize=9)
        ax.set_xlabel("EEG kaydı başından (s)")
        ax.set_ylabel("Blink sayısı")
        ax.grid(axis="x")

        # Legend sağ tarafta
        handles = [mpatches.Patch(color=color_map[st], label=st) for st in scenario_types]
        ax.legend(handles=handles, fontsize=6, loc="upper right",
                  ncol=3, framealpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    subject_dirs = sorted(d for d in RAW.iterdir() if d.is_dir() and (d / "metadata.json").exists())
    print(f"\n{len(subject_dirs)} denek bulundu. Görselleştirme başlıyor...\n")

    all_data = []
    for sdir in subject_dirs:
        print(f"Yükleniyor: {sdir.name}")
        d = load_subject(sdir)
        all_data.append(d)

        out = FIGURES / f"dashboard_{d['meta']['user_id']:02d}_{d['meta']['name'].replace(' ','_')}.png"
        plot_subject_dashboard(d, out)

    print("\nKarşılaştırma figürleri üretiliyor...")
    plot_cross_subject(all_data, FIGURES / "cross_subject_comparison.png")
    plot_eeg_timeline(all_data,  FIGURES / "eeg_marker_timeline.png")

    print(f"\nTüm figürler kaydedildi → {FIGURES}")
    for f in sorted(FIGURES.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
