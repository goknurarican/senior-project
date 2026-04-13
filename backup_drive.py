"""
Google Drive backup — OAuth 2.0 Desktop App.

First run on a new machine: a browser window opens once for Google login.
Every run after that: fully automatic, no user interaction needed.

Install once on the lab PC:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import json
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
def _find_creds_file() -> Path:
    """Return credentials file path — accepts credentials.json or client_secret_*.json."""
    standard = SCRIPT_DIR / "credentials.json"
    if standard.exists():
        return standard
    candidates = sorted(SCRIPT_DIR.glob("client_secret_*.json"))
    if candidates:
        return candidates[0]
    return standard  # will fail gracefully later with a clear message

CREDS_FILE  = _find_creds_file()
TOKEN_FILE  = SCRIPT_DIR / "token.json"
CONFIG_FILE = SCRIPT_DIR / "lab_config.json"

SCOPES = ["https://www.googleapis.com/auth/drive"]


def _get_service():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Silently refresh with the saved token — no browser needed
            creds.refresh(Request())
        else:
            # First time only: opens browser for Google login
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def _get_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return the Drive folder ID, creating it if it does not exist."""
    q = (f"name='{name}' "
         f"and mimeType='application/vnd.google-apps.folder' "
         f"and '{parent_id}' in parents "
         f"and trashed=false")
    results = service.files().list(q=q, fields="files(id)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(body=meta, fields="id").execute()["id"]


def _upload_file(service, path: Path, parent_id: str, log=None):
    from googleapiclient.http import MediaFileUpload

    # Update if already exists, create if not
    q = f"name='{path.name}' and '{parent_id}' in parents and trashed=false"
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])

    # Use resumable upload for files larger than 5 MB (e.g. EEG .eeg files)
    resumable = path.stat().st_size > 5 * 1024 * 1024
    media = MediaFileUpload(str(path), resumable=resumable)

    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        service.files().create(
            body={"name": path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        ).execute()

    if log:
        size_kb = path.stat().st_size // 1024
        log(f"  ↑ {path.name}  ({size_kb} KB)\n")


def _upload_folder(service, local: Path, parent_id: str, log=None):
    """Recursively mirror a local folder to Drive under parent_id."""
    folder_id = _get_or_create_folder(service, local.name, parent_id)
    for item in sorted(local.iterdir()):
        if item.is_file():
            _upload_file(service, item, folder_id, log)
        elif item.is_dir():
            _upload_folder(service, item, folder_id, log)


def backup_subject(subject_folder: Path, log=None) -> bool:
    """
    Upload a packaged subject folder to Google Drive.

    subject_folder : Path  e.g. subjects/user_005_ahmet_variant_a
    log            : callable(str) | None  — receives progress lines

    Returns True on success, False on any failure (non-fatal).
    """
    # ── Pre-flight checks ─────────────────────────────────────────────────
    if not CREDS_FILE.exists():
        if log:
            log("[BACKUP] credentials.json bulunamadı — yedekleme atlandı.\n")
        return False

    config = {}
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    backup_cfg = config.get("backup", {})

    if not backup_cfg.get("enabled", False):
        if log:
            log("[BACKUP] lab_config.json'da backup kapalı — atlandı.\n")
        return False

    folder_id = backup_cfg.get("drive_folder_id", "").strip()
    if not folder_id:
        if log:
            log("[BACKUP] lab_config.json'da drive_folder_id yok — atlandı.\n")
        return False

    # ── Upload ────────────────────────────────────────────────────────────
    try:
        if log:
            log("[BACKUP] Google Drive'a bağlanıyor...\n")
        service = _get_service()

        if log:
            log(f"[BACKUP] Yükleniyor: {subject_folder.name}\n")
        _upload_folder(service, subject_folder, folder_id, log)

        # Optionally also back up the full database
        if backup_cfg.get("copy_db", True):
            db = SCRIPT_DIR / "experiment.db"
            if db.exists():
                if log:
                    log("[BACKUP] experiment.db yükleniyor...\n")
                _upload_file(service, db, folder_id, log)

        if log:
            log("[BACKUP] ✓ Google Drive yüklemesi tamamlandı.\n")
        return True

    except Exception as exc:
        if log:
            log(f"[BACKUP] HATA: {exc}\n")
        return False
