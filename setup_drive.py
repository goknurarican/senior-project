#!/usr/bin/env python3
"""
BİTİRMEEG — Google Drive Kurulum Scripti (tek seferlik)
========================================================
Bu scripti lab bilgisayarında SADECE BİR KEZ çalıştır.
Bundan sonra backup otomatik çalışır, tekrar çalıştırmana gerek yok.

Önce credentials.json dosyasını bu klasöre kopyala, sonra:
    python setup_drive.py
"""

import json
import sys
import webbrowser
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
TOKEN_FILE  = SCRIPT_DIR / "token.json"
CONFIG_FILE = SCRIPT_DIR / "lab_config.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

SEP = "=" * 60

def find_creds():
    for name in ["credentials.json"] + sorted(
        str(p) for p in SCRIPT_DIR.glob("client_secret_*.json")
    ):
        p = SCRIPT_DIR / name if not Path(name).is_absolute() else Path(name)
        if p.exists():
            return p
    return None

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── 1. Credentials dosyası ────────────────────────────────────────────────────
section("1. CREDENTIALS DOSYASI")

creds_file = find_creds()
if not creds_file:
    print("""
  [HATA] credentials.json bulunamadı!

  Yapman gerekenler:
    1. Araştırmacının Mac/PC'sinden credentials.json dosyasını al
       (proje klasöründe: senior-project/credentials.json)
    2. USB veya e-posta ile lab bilgisayarına kopyala
    3. Şu klasöre yapıştır:
""")
    print(f"       {SCRIPT_DIR}")
    print("""
    4. Bu scripti tekrar çalıştır: python setup_drive.py

  NEDEN GİTHUB'DA YOK?
    Güvenlik nedeniyle credentials.json git'e yüklenmez.
    Sadece el ile kopyalanır.
""")
    sys.exit(1)

print(f"  [OK]  Credentials dosyası bulundu: {creds_file.name}")

# ── 2. Google Drive paketleri ─────────────────────────────────────────────────
section("2. GOOGLE DRIVE PAKETLERİ")
missing = []
for pkg in ["google.oauth2", "google_auth_oauthlib", "googleapiclient"]:
    try:
        __import__(pkg.replace(".", "_") if "_" not in pkg else pkg.split(".")[0])
        print(f"  [OK]  {pkg}")
    except ImportError:
        try:
            __import__(pkg)
            print(f"  [OK]  {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [HATA]  {pkg} kurulu değil")

if missing:
    print(f"""
  Eksik paketler kuruluyor...
  Çalıştır:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
""")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib"
    ])
    print("  [OK]  Paketler kuruldu.")

# ── 3. Google Drive klasör ID kontrolü ───────────────────────────────────────
section("3. DRIVE KLASÖR AYARI")
config = {}
folder_id = ""
if CONFIG_FILE.exists():
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        folder_id = config.get("backup", {}).get("drive_folder_id", "").strip()
    except Exception:
        pass

if not folder_id:
    print("""
  [HATA] lab_config.json'da drive_folder_id bulunamadı!
  lab_config.json şöyle görünmeli:

  {
    "backup": {
      "enabled": true,
      "drive_folder_id": "KLASOR_ID_BURAYA",
      "copy_db": true
    }
  }

  Drive klasör ID'sini bulmak için:
    1. drive.google.com aç
    2. Yedek klasörüne gir
    3. URL'deki son parça: drive.google.com/drive/folders/BURASI
""")
    sys.exit(1)

print(f"  [OK]  Klasör ID: {folder_id}")
print(f"  [OK]  Backup aktif: {config.get('backup', {}).get('enabled', False)}")

# ── 4. Google hesabı girişi ───────────────────────────────────────────────────
section("4. GOOGLE HESABI GİRİŞİ")

if TOKEN_FILE.exists():
    print("  token.json zaten mevcut — yeniden giriş gerekmeyebilir.")
    yenile = input("  Token'ı yenilemek istiyor musun? (e/h): ").strip().lower()
    if yenile == "e":
        TOKEN_FILE.unlink()

print("""
  Tarayıcıda Google hesabına giriş yapman gerekecek.
  Şimdi tarayıcı açılıyor...
  (Açılmazsa birkaç saniye bekle)
""")

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        if creds and creds.valid:
            print("  [OK]  Mevcut token geçerli, yeniden giriş gerekmez.")
        elif creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
            print("  [OK]  Token yenilendi.")
        else:
            TOKEN_FILE.unlink(missing_ok=True)
            creds = None
    else:
        creds = None

    if creds is None:
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        print("  [OK]  Giriş başarılı, token.json kaydedildi.")

    # ── 5. Bağlantı testi ────────────────────────────────────────────────────
    section("5. DRIVE BAĞLANTI TESTİ")
    service = build("drive", "v3", credentials=creds)

    result = service.files().get(fileId=folder_id, fields="id,name").execute()
    print(f"  [OK]  Drive klasörü bulundu: '{result.get('name', '?')}'")
    print(f"        ID: {folder_id}")

    # Test: küçük bir dosya yükle
    import io
    from googleapiclient.http import MediaIoBaseUpload
    test_content = b"BITIRMEEG drive test OK"
    media = MediaIoBaseUpload(io.BytesIO(test_content), mimetype="text/plain")
    test_file = service.files().create(
        body={"name": "_test_baglanti.txt", "parents": [folder_id]},
        media_body=media,
        fields="id"
    ).execute()
    service.files().delete(fileId=test_file["id"]).execute()
    print("  [OK]  Test dosyasi yuklendi ve silindi.")

except Exception as e:
    print(f"  [HATA] {e}")
    print("""
  Olası nedenler:
    - Klasör ID yanlış
    - Google hesabının bu klasöre erişim izni yok
    - İnternet bağlantısı yok
""")
    sys.exit(1)

# ── Özet ─────────────────────────────────────────────────────────────────────
section("KURULUM TAMAMLANDI")
print(f"""
  [OK]  credentials: {creds_file.name}
  [OK]  token.json oluşturuldu (bir daha giriş gerekmez)
  [OK]  Drive klasörü erişilebilir

  Bundan sonra lab_panel.py yeşil butona her basıldığında
  veriler otomatik olarak Drive'a yüklenir.

  Dosya konumu: {SCRIPT_DIR}
""")
