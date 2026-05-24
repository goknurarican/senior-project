"""Tek seferlik OAuth akışı — token.json üretir."""
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES     = ["https://www.googleapis.com/auth/drive"]
CREDS_FILE = Path("/Users/goknurarican/Downloads/senior-project-main-3/credentials.json")
TOKEN_FILE = Path("/Users/goknurarican/Downloads/senior-project-main-3/token.json")

flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
creds = flow.run_local_server(port=0)
TOKEN_FILE.write_text(creds.to_json())
print(f"token.json kaydedildi → {TOKEN_FILE}")
