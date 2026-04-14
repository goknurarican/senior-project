"""
Gazepoint'ten RAW XML satırlarını yazdırır.
Hangi attribute'ların geldiğini ve değerlerini gösterir.

Kullanım:
    python debug_gaze_raw.py
"""
import socket, re, time

HOST, PORT = "127.0.0.1", 4242

cmds = [
    '<SET ID="ENABLE_SEND_POG_BEST" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_LEFT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n',
]

TRACK_ATTRS = [
    "BPOGX", "BPOGY", "BPOGV",
    "FPOGX", "FPOGY", "FPOGV",
    "LPOGX", "LPOGY", "LPOGV",
    "RPOGX", "RPOGY", "RPOGV",
    "LPMMV", "RPMMV", "LPV", "RPV",
    "TIME",
]

print(f"Gazepoint {HOST}:{PORT} bağlanıyor...")
s = socket.socket()
s.settimeout(5)
s.connect((HOST, PORT))
print("Bağlandı. Komutlar gönderiliyor...")

for cmd in cmds:
    s.send(cmd.encode())

print("\n10 saniye ham veri bekleniyor...\n")
s.settimeout(10)

buffer = ""
data_lines = 0
deadline = time.time() + 10

print(f"{'Satır':>5}  {' | '.join(f'{a:>8}' for a in TRACK_ATTRS)}")
print("-" * (7 + 11 * len(TRACK_ATTRS)))

while time.time() < deadline and data_lines < 20:
    try:
        chunk = s.recv(4096).decode("utf-8", errors="ignore")
        if not chunk:
            break
        buffer += chunk
        while "\r\n" in buffer:
            line, buffer = buffer.split("\r\n", 1)
            line = line.strip()
            if not line or "<ACK" in line:
                continue

            # Sadece REC satırlarını işle
            if "<REC" not in line:
                continue

            # Her attribute değerini çıkar
            vals = {}
            for attr in TRACK_ATTRS:
                m = re.search(rf'{attr}="([^"]+)"', line)
                vals[attr] = m.group(1) if m else "—"

            data_lines += 1
            row = " | ".join(f"{vals[a]:>8}" for a in TRACK_ATTRS)
            print(f"{data_lines:>5}  {row}")

            # POG sıfır mı değil mi özet
            bpogv = vals.get("BPOGV", "—")
            fpogv = vals.get("FPOGV", "—")
            lpogv = vals.get("LPOGV", "—")
            bpogx = vals.get("BPOGX", "—")

            if data_lines == 1:
                print()
                print("  → İlk satır tam XML:")
                print(f"     {line[:200]}")
                print()

    except socket.timeout:
        break

s.close()

print("\n" + "=" * 60)
print("ÖZET:")
print(f"  Toplam REC satırı: {data_lines}")
if data_lines > 0:
    print(f"  BPOGX sütunu var mı: {'Evet' if bpogx != '—' else 'HAYIR — BPOG gönderilmiyor'}")
    print(f"  BPOGV (son): {bpogv}")
    print(f"  FPOGV (son): {fpogv}")
    print(f"  LPOGV (son): {lpogv}")
    if bpogx not in ("—", "0.0000", "0"):
        print(f"\n  ✓ BPOGX non-zero: {bpogx} — gaze verisi geliyor!")
    else:
        print(f"\n  ✗ Tüm gaze koordinatları sıfır.")
        print(f"    Olası nedenler:")
        print(f"    1. Kalibrasyon kaydedilmemiş (Gazepoint restart sonrası sıfırlanır)")
        print(f"    2. Denek kameraya düzgün bakmıyor")
        print(f"    3. BPOG Gazepoint bu versiyonda 'LBESTX'/'RBESTX' adıyla geliyor olabilir")
