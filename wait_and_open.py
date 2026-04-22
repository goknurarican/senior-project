"""
Localhost:3000 (Next.js) ve 5001 (Trigger Server) hazir olana kadar
bekler, sonra tarayiciyi acar. DENEYI_BASLAT.bat tarafindan cagrilir.
"""
import socket, time, webbrowser, sys

URL     = "http://localhost:3000/login"
TIMEOUT = 120  # maks 2 dakika bekle

def wait_for_port(host, port, label, timeout=TIMEOUT):
    print(f"{label} bekleniyor", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            print(f" hazir!")
            return True
        except OSError:
            print(".", end="", flush=True)
            time.sleep(2)
    print(f"\nZaman asimi — {label} baslamadi.")
    return False

# Trigger Server (port 5001) — goreceli olarak cok hizli baslar
if not wait_for_port("127.0.0.1", 5001, "Trigger Server", timeout=30):
    print("UYARI: Trigger server baslamadi. EEG/goz takibi calismayabilir.")
    print("       trigger_server.py penceresini kontrol edin.")

# Next.js (port 3000) — derlenme gerektirirse uzun surebilir
if not wait_for_port("127.0.0.1", 3000, "Next.js"):
    print("Hata: Next.js baslamadi. npm start penceresini kontrol et.")
    sys.exit(1)

time.sleep(1)
webbrowser.open(URL)
print(f"Tarayici acildi: {URL}")
