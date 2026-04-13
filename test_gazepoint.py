"""Gazepoint'ten ham veri okur — python test_gazepoint.py"""
import socket, time

HOST, PORT = "127.0.0.1", 4242

cmds = [
    '<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_LEFT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />\r\n',
]

print(f"Gazepoint {HOST}:{PORT} baglaniliyor...")
s = socket.socket()
s.settimeout(5)
s.connect((HOST, PORT))
print("Baglandi. Komutlar gonderiliyor...")

for cmd in cmds:
    s.send(cmd.encode())

print("10 saniye veri bekleniyor...\n")
s.settimeout(10)

lines_seen = 0
has_fpog = False
has_lpog = False
buffer = ""

deadline = time.time() + 10
while time.time() < deadline:
    try:
        chunk = s.recv(4096).decode("utf-8", errors="ignore")
        if not chunk:
            break
        buffer += chunk
        while "\r\n" in buffer:
            line, buffer = buffer.split("\r\n", 1)
            line = line.strip()
            if not line:
                continue
            lines_seen += 1
            if "<ACK" in line:
                print(f"  ACK: {line[:80]}")
            elif "FPOGX" in line:
                has_fpog = True
                print(f"  GOZ VERISI (FPOG): {line[:120]}")
            elif "LPOGX" in line or "RPOGX" in line:
                has_lpog = True
                print(f"  GOZ VERISI (LPOG/RPOG): {line[:120]}")
            else:
                print(f"  DIGER: {line[:80]}")
            if lines_seen >= 20:
                break
    except socket.timeout:
        break

s.close()
print()
print(f"Toplam satir: {lines_seen}")
print(f"FPOG verisi var mi: {has_fpog}")
print(f"LPOG/RPOG verisi var mi: {has_lpog}")

if not has_fpog and not has_lpog:
    print()
    print("SORUN: Goz verisi gelmiyor.")
    print("Kontrol et:")
    print("  1. Gazepoint Control'de kalibrasyon yapildi mi?")
    print("  2. Kamera aktif mi, yesil isik var mi?")
    print("  3. Birisi kameraya bakiyor mu?")
else:
    print()
    print("OK: Goz verisi akiyor, trigger_server kaydedecek.")
