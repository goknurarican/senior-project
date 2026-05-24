import io
from pathlib import Path
from collections import defaultdict

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES      = ["https://www.googleapis.com/auth/drive"]
CREDS_FILE  = Path("/Users/goknurarican/Downloads/senior-project-main-3/credentials.json")
TOKEN_FILE  = Path("/Users/goknurarican/Downloads/senior-project-main-3/token.json")
ROOT_FOLDER = "1ocVHY1s70b4OsBxktsJPo6Vk1QIisYFq"
DEST        = Path(__file__).parent / "data" / "raw"

# denek_adı → {"mouse": bool, "eye": bool, "eeg": bool}
summary = defaultdict(lambda: {"mouse": False, "eye": False, "eeg": False})


def get_service():
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    return build("drive", "v3", credentials=creds)


def list_folder(service, folder_id):
    items, token = [], None
    while True:
        r = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken,files(id,name,mimeType,size)",
            pageToken=token,
        ).execute()
        items += r.get("files", [])
        token = r.get("nextPageToken")
        if not token:
            break
    return items


def classify_file(name: str, local_path: Path):
    """Dosya uzantısına/adına göre veri tipini belirle."""
    name_lower = name.lower()
    ext = Path(name).suffix.lower()
    # Denek klasörünün adını bul (data/raw/<denek>/ altında)
    parts = local_path.parts
    try:
        raw_idx = parts.index("raw")
        if raw_idx + 1 < len(parts):
            return parts[raw_idx + 1]  # denek klasörü adı
    except ValueError:
        pass
    return None


def update_summary(item_name: str, local_path: Path):
    """İndirilen dosyaya göre özet tablosunu güncelle."""
    parts = local_path.parts
    try:
        raw_idx = list(parts).index("raw")
    except ValueError:
        return

    if raw_idx + 1 >= len(parts):
        return  # raw/ kökündeki dosya (örn. experiment.db)

    denek = parts[raw_idx + 1]
    name_lower = item_name.lower()
    ext = Path(item_name).suffix.lower()

    if ext in (".eeg", ".vhdr", ".vmrk"):
        summary[denek]["eeg"] = True
    elif ext == ".csv" and ("gaze" in name_lower or "eye" in name_lower):
        summary[denek]["eye"] = True
    elif ext == ".csv" and ("mouse" in name_lower or "platform" in name_lower or "click" in name_lower):
        summary[denek]["mouse"] = True
    elif ext == ".csv":
        # Klasör adına bak
        if "eye" in str(local_path).lower():
            summary[denek]["eye"] = True
        elif "platform" in str(local_path).lower() or "mouse" in str(local_path).lower():
            summary[denek]["mouse"] = True


def download_file(service, item, local_path: Path):
    dest_file = local_path / item["name"]
    if dest_file.exists():
        print(f"  [SKIP] {item['name']} (zaten mevcut)")
        return

    request = service.files().get_media(fileId=item["id"])
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    dest_file.write_bytes(buf.getvalue())
    size_kb = len(buf.getvalue()) // 1024
    rel = dest_file.relative_to(DEST)
    print(f"  [OK]   {rel}  ({size_kb} KB)")
    update_summary(item["name"], local_path)


def download_folder(service, folder_id, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    for item in list_folder(service, folder_id):
        if item["mimeType"] == "application/vnd.google-apps.folder":
            download_folder(service, item["id"], local_path / item["name"])
        else:
            download_file(service, item, local_path)


def print_summary():
    print("\n" + "=" * 55)
    print("ÖZET")
    print("=" * 55)

    # raw/ kökündeki dosyaları say
    root_files = [f for f in DEST.iterdir() if f.is_file()] if DEST.exists() else []
    denek_dirs = [d for d in DEST.iterdir() if d.is_dir()] if DEST.exists() else []

    print(f"\nBulunan denek klasörü sayısı : {len(denek_dirs)}")
    if root_files:
        print(f"Kök dosyalar               : {', '.join(f.name for f in root_files)}")

    if not summary:
        # Klasörleri disk'ten tara
        for d in sorted(denek_dirs):
            files = list(d.rglob("*"))
            has_eeg   = any(f.suffix.lower() in (".eeg", ".vhdr", ".vmrk") for f in files)
            has_eye   = any("gaze" in f.name.lower() or "eye" in str(f).lower() for f in files if f.suffix == ".csv")
            has_mouse = any("mouse" in f.name.lower() or "platform" in str(f).lower() for f in files if f.suffix == ".csv")
            summary[d.name]["eeg"]   = has_eeg
            summary[d.name]["eye"]   = has_eye
            summary[d.name]["mouse"] = has_mouse

    print(f"\n{'Denek':<30} {'Mouse':^7} {'Eye':^5} {'EEG':^5}")
    print("-" * 50)
    for denek in sorted(summary):
        d = summary[denek]
        mouse = "VAR" if d["mouse"] else "yok"
        eye   = "VAR" if d["eye"]   else "yok"
        eeg   = "VAR" if d["eeg"]   else "yok"
        print(f"{denek:<30} {mouse:^7} {eye:^5} {eeg:^5}")
    print("=" * 55)


if __name__ == "__main__":
    service = get_service()
    print("Drive'a bağlandı. İndirme başlıyor...\n")
    download_folder(service, ROOT_FOLDER, DEST)
    print("\nTamamlandı.")
    print_summary()
