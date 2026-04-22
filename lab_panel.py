#!/usr/bin/env python3
"""
BİTİRMEEG — Lab Panel
=======================
Deney teknisyeni için tek-tuş veri paketleme paneli.

Kullanım:
  SONRAKI_DENEK.bat dosyasına çift tıkla — bu script arka planda çalışır.

Teknisyen adımları (panel açıkken):
  1. Gazepoint Control'de kaydı durdur
  2. BrainVision Recorder'da kaydı durdur
  3. Paneldeki yeşil butona bas
  4. Panel "Hazır" diyene kadar bekle
  5. Sonraki denek için Gazepoint kalibrasyon + yeni kayıt başlat
  6. BrainVision'da yeni kayıt başlat
"""

import contextlib
import io
import json
import os
import sqlite3
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).parent
DB_PATH           = SCRIPT_DIR / "experiment.db"
PACKAGED_LOG_PATH = SCRIPT_DIR / "packaged_log.json"

sys.path.insert(0, str(SCRIPT_DIR))

# ── Optional Google Drive backup ──────────────────────────────────────────────
try:
    from backup_drive import backup_subject as _drive_backup
    HAS_BACKUP = True
except ImportError:
    HAS_BACKUP = False   # google packages not installed — backup silently skipped

# ── Packaged-user log ─────────────────────────────────────────────────────────

def _load_log() -> dict:
    if PACKAGED_LOG_PATH.exists():
        try:
            return json.loads(PACKAGED_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _mark_packaged(user_id: int, folder: Path):
    log = _load_log()
    log[str(user_id)] = {
        "packaged_at": datetime.now().isoformat(),
        "folder": str(folder),
    }
    PACKAGED_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def _clean_stale_packaged_log():
    """packaged_log.json'dan artık DB'de bulunmayan user ID'leri temizle.

    DB silinip yeniden oluşturulduğunda eski log kayıtları aynı ID'li yeni
    katılımcıları 'zaten paketlendi' olarak işaretleyebilir. Bu fonksiyon
    DB'deki gerçek kullanıcı listesiyle karşılaştırarak bu sorunu önler.
    """
    if not DB_PATH.exists():
        return
    log = _load_log()
    if not log:
        return
    try:
        conn = _db()
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE role != 'admin'")
        valid_ids = {str(row["id"]) for row in c.fetchall()}
        conn.close()
        stale = [uid for uid in log if uid not in valid_ids]
        if stale:
            for uid in stale:
                del log[uid]
            PACKAGED_LOG_PATH.write_text(
                json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  [OK] {len(stale)} eski paket kaydı temizlendi (DB sıfırlanmış).")
    except Exception as exc:
        print(f"  [WARN] Paket log temizleme hatası: {exc}")

# ── DB helpers ────────────────────────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _get_next_unpackaged():
    # type: () -> Optional[dict]
    """Return info dict for the most-recent participant not yet packaged."""
    if not DB_PATH.exists():
        return None
    packaged = _load_log()
    conn = _db()
    c = conn.cursor()

    # ── 1. sessions JOIN users (user_id set — new code) ───────────────────
    try:
        c.execute("""
            SELECT s.user_id, u.name, s.experiment_group AS grp,
                   s.id AS session_id, s.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE u.role != 'admin'
            ORDER BY s.created_at DESC
        """)
        for row in c.fetchall():
            if str(row["user_id"]) not in packaged:
                conn.close()
                return dict(row)
    except Exception:
        pass

    # ── 2. events JOIN users (works when cookie sent correctly) ───────────
    try:
        c.execute("""
            SELECT e.user_id, u.name,
                   MAX(e.experiment_group)  AS grp,
                   MAX(e.session_id)        AS session_id,
                   MAX(e.timestamp)         AS created_at
            FROM events e
            JOIN users u ON u.id = e.user_id
            WHERE u.role != 'admin'
            GROUP BY e.user_id
            ORDER BY MAX(e.timestamp) DESC
        """)
        for row in c.fetchall():
            if str(row["user_id"]) not in packaged:
                conn.close()
                return dict(row)
    except Exception:
        pass

    # ── 3. Last resort: iterate all non-admin users newest-first ─────────
    try:
        c.execute("""
            SELECT u.id AS user_id, u.name,
                   s.experiment_group AS grp,
                   s.id AS session_id, s.created_at
            FROM users u
            LEFT JOIN sessions s ON s.user_id = u.id
            WHERE u.role != 'admin'
            ORDER BY u.id DESC
        """)
        for row in c.fetchall():
            if str(row["user_id"]) not in packaged:
                conn.close()
                return dict(row)
    except Exception:
        pass

    # ── 4. sessions without user_id (old code — match by recency) ─────────
    try:
        c.execute("""
            SELECT s.id AS session_id, s.experiment_group AS grp,
                   s.created_at, s.user_id
            FROM sessions s
            WHERE s.user_id IS NULL
            ORDER BY s.created_at DESC
            LIMIT 1
        """)
        row = c.fetchone()
        if row:
            # Try to find the matching user from events for this session
            c.execute("""
                SELECT e.user_id, u.name
                FROM events e
                JOIN users u ON u.id = e.user_id
                WHERE e.session_id = ? AND u.role != 'admin'
                LIMIT 1
            """, (row["session_id"],))
            user_row = c.fetchone()
            if user_row and str(user_row["user_id"]) not in packaged:
                conn.close()
                return {
                    "user_id":    user_row["user_id"],
                    "name":       user_row["name"],
                    "grp":        row["grp"],
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                }
    except Exception:
        pass

    conn.close()
    return None

def _count_all_users() -> int:
    if not DB_PATH.exists():
        return 0
    try:
        conn = _db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role != 'admin'")
        n = c.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0

# ── Colour / font constants ───────────────────────────────────────────────────
BG          = "#F5F6FA"
PANEL_BG    = "#FFFFFF"
HEADER_BG   = "#2D3561"
HEADER_FG   = "#FFFFFF"
GREEN       = "#27AE60"
GREEN_HOV   = "#219150"
ORANGE      = "#E67E22"
RED         = "#E74C3C"
GREY        = "#95A5A6"
TEXT_DARK   = "#2C3E50"
TEXT_LIGHT  = "#7F8C8D"
MONO        = "Consolas" if sys.platform == "win32" else "Courier New"

# ── Main application window ───────────────────────────────────────────────────

class LabPanel(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("BİTİRMEEG — Lab Panel")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("720x680")
        self.minsize(600, 560)

        self._user       = None   # current unpackaged user dict
        self._busy       = False  # packaging in progress?
        self._packaged   = 0      # count packaged this session

        self._build_ui()
        _clean_stale_packaged_log()
        self._refresh()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=HEADER_BG, pady=18)
        hdr.pack(fill="x")
        tk.Label(hdr, text="BİTİRMEEG", bg=HEADER_BG, fg=HEADER_FG,
                 font=("Segoe UI", 22, "bold")).pack()
        tk.Label(hdr, text="Deney Veri Paketleme Paneli",
                 bg=HEADER_BG, fg="#A9B4D3",
                 font=("Segoe UI", 11)).pack()

        # ── Status card ───────────────────────────────────────────────────
        card = tk.Frame(self, bg=PANEL_BG, bd=0, relief="flat",
                        padx=24, pady=18)
        card.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(card, text="SON DENEK", bg=PANEL_BG, fg=TEXT_LIGHT,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        self._lbl_user = tk.Label(card, text="—", bg=PANEL_BG, fg=TEXT_DARK,
                                  font=("Segoe UI", 16, "bold"))
        self._lbl_user.grid(row=1, column=0, sticky="w")

        tk.Label(card, text="GRUP", bg=PANEL_BG, fg=TEXT_LIGHT,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(32, 0))
        self._lbl_group = tk.Label(card, text="—", bg=PANEL_BG, fg=TEXT_DARK,
                                   font=("Segoe UI", 16, "bold"))
        self._lbl_group.grid(row=1, column=1, sticky="w", padx=(32, 0))

        tk.Label(card, text="BU OTURUMDA PAKETLENDİ", bg=PANEL_BG,
                 fg=TEXT_LIGHT, font=("Segoe UI", 9, "bold")).grid(
                     row=0, column=2, sticky="e", padx=(32, 0))
        self._lbl_count = tk.Label(card, text="0", bg=PANEL_BG, fg=GREEN,
                                   font=("Segoe UI", 16, "bold"))
        self._lbl_count.grid(row=1, column=2, sticky="e", padx=(32, 0))

        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=1)

        # ── Main button ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=20, pady=16)

        self._btn = tk.Button(
            btn_frame,
            text="✔  VERİYİ KAYDET  →  SONRAKİ DENEĞE GEÇ",
            font=("Segoe UI", 14, "bold"),
            bg=GREEN, fg="white", activebackground=GREEN_HOV, activeforeground="white",
            bd=0, relief="flat", padx=16, pady=14, cursor="hand2",
            command=self._on_package
        )
        self._btn.pack(fill="x")
        self._btn.bind("<Enter>", lambda e: self._btn.configure(bg=GREEN_HOV)
                       if not self._busy else None)
        self._btn.bind("<Leave>", lambda e: self._btn.configure(bg=GREEN)
                       if not self._busy else None)

        self._lbl_status = tk.Label(
            self, text="Veri tabanı kontrol ediliyor…",
            bg=BG, fg=TEXT_LIGHT, font=("Segoe UI", 10)
        )
        self._lbl_status.pack()

        # ── Log area ──────────────────────────────────────────────────────
        tk.Label(self, text="İşlem Günlüğü", bg=BG, fg=TEXT_LIGHT,
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(
                     fill="x", padx=20, pady=(10, 2))
        self._log = scrolledtext.ScrolledText(
            self, height=12, font=(MONO, 9), bg="#1E1E2E", fg="#D4D4D4",
            insertbackground="white", relief="flat", bd=0, wrap="word",
            state="disabled"
        )
        self._log.pack(fill="both", expand=True, padx=20, pady=(0, 4))

        # ── Next-steps checklist (hidden until success) ───────────────────
        self._checklist_frame = tk.Frame(self, bg="#EBF5EB", bd=0, pady=12, padx=18)
        # Not packed yet — shown after success

        steps = [
            "1.  Gazepoint Control → yeni kalibrasyon yap",
            "2.  Gazepoint Control → ▶ Record (yeni kayıt başlat)",
            "3.  BrainVision Recorder → ▶ Start Recording",
            "4.  Yeni deneği oturuma yönlendir (platform URL'i aç)",
        ]
        tk.Label(self._checklist_frame, text="✔ Paketleme tamamlandı — sonraki denek için:",
                 bg="#EBF5EB", fg=GREEN, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        for s in steps:
            tk.Label(self._checklist_frame, text=s, bg="#EBF5EB", fg=TEXT_DARK,
                     font=("Segoe UI", 10)).pack(anchor="w", pady=1)

    # ── State updates (always called from main thread via after()) ────────────

    def _refresh(self):
        """Re-query the DB and update the status card."""
        self._user = _get_next_unpackaged()
        if self._user:
            name = self._user.get("name") or f"User {self._user['user_id']}"
            uid  = self._user["user_id"]
            grp  = self._user.get("grp") or "?"
            self._lbl_user.configure(text=f"#{uid} — {name}", fg=TEXT_DARK)
            self._lbl_group.configure(text=grp, fg=TEXT_DARK)
            self._lbl_status.configure(
                text=f"Hazır: User {uid} paketlenebilir.", fg=GREEN
            )
            self._btn.configure(state="normal", bg=GREEN)
        else:
            all_n = _count_all_users()
            if not DB_PATH.exists():
                user_text   = "experiment.db bulunamadı"
                status_text = f"DB yolu: {DB_PATH}\nnpm start çalışıyor mu? Denek giriş yaptı mı?"
                status_fg   = RED
            elif all_n == 0:
                user_text   = "Veritabanında denek yok"
                status_text = f"DB bulundu ama kullanıcı yok. Denek kayıt oldu mu?"
                status_fg   = ORANGE
            else:
                user_text   = "Paketlenecek denek yok"
                status_text = f"Tüm {all_n} denek zaten paketlendi."
                status_fg   = GREY
            self._lbl_user.configure(text=user_text, fg=GREY)
            self._lbl_group.configure(text="—", fg=GREY)
            self._lbl_status.configure(text=status_text, fg=status_fg)
            self._btn.configure(state="disabled", bg=GREY)

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy:
            self._btn.configure(
                state="disabled", bg=ORANGE,
                text="⏳  Paketleniyor… lütfen bekleyin"
            )
            self._lbl_status.configure(
                text="Veriler kaydediliyor, lütfen bekleyin…", fg=ORANGE
            )
            self._checklist_frame.pack_forget()
        else:
            self._btn.configure(
                text="✔  VERİYİ KAYDET  →  SONRAKİ DENEĞE GEÇ"
            )

    def _show_success(self, folder: Path):
        self._lbl_count.configure(text=str(self._packaged))
        self._checklist_frame.pack(fill="x", padx=20, pady=(0, 12))
        self._lbl_status.configure(
            text=f"✔ Kaydedildi: {folder.name}", fg=GREEN
        )

    def _show_error(self):
        self._lbl_status.configure(
            text="✘ Hata oluştu — günlüğe bakın veya bir araştırmacıyı arayın.",
            fg=RED
        )

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        """Thread-safe log append via after()."""
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", text)
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    # ── Button handler ────────────────────────────────────────────────────────

    def _on_package(self):
        if self._busy or not self._user:
            return

        uid = self._user["user_id"]

        # Double-confirm (safeguard against accidental clicks)
        name = self._user.get("name") or f"User {uid}"
        if not messagebox.askyesno(
            "Onay",
            f"User {uid} ({name}) verisi paketlensin mi?\n\n"
            "Gazepoint ve BrainVision kayıtları durduruldu mu?",
            icon="question",
        ):
            return

        self._set_busy(True)
        self._append_log(
            f"\n{'─'*60}\n"
            f"  Paketleniyor: User {uid} — {name}\n"
            f"  Zaman: {datetime.now().strftime('%H:%M:%S')}\n"
            f"{'─'*60}\n\n"
        )

        threading.Thread(
            target=self._package_thread, args=(uid,), daemon=True
        ).start()

    def _package_thread(self, user_id: int):
        """Runs in background thread. Posts results back via after()."""
        buf = io.StringIO()
        result_folder = None
        success = False

        try:
            # Capture all print() output from package_subject and data_logger
            with contextlib.redirect_stdout(buf):
                from package_subject import package_subject
                result_folder = package_subject(user_id)
            success = result_folder is not None
        except Exception as exc:
            buf.write(f"\n[HATA] {exc}\n")

        captured = buf.getvalue()
        self._append_log(captured + "\n")

        if success:
            _mark_packaged(user_id, result_folder)
            self._packaged += 1

            # ── Google Drive backup (runs right after packaging) ──────────
            if HAS_BACKUP:
                self._append_log(f"\n{'─'*60}\n  GOOGLE DRIVE YEDEK\n{'─'*60}\n")
                _drive_backup(Path(result_folder), log=self._append_log)
            else:
                self._append_log(
                    "\n[BACKUP] Google Drive paketi kurulu değil.\n"
                    "  Kurmak için: pip install google-api-python-client "
                    "google-auth-httplib2 google-auth-oauthlib\n"
                )

        def _finish():
            self._set_busy(False)
            if success:
                self._show_success(Path(result_folder))
            else:
                self._show_error()
            self._refresh()   # find next unpackaged user

        self.after(0, _finish)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = LabPanel()
    app.mainloop()
