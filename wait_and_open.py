"""
Localhost:3000 hazir olana kadar bekler, sonra tarayiciyi acar.
DENEYI_BASLAT.bat tarafindan cagrilir.
"""
import socket, time, webbrowser, sys

URL = "http://localhost:3000/login"
HOST, PORT = "127.0.0.1", 3000
TIMEOUT = 120  # maks 2 dakika bekle

print("Next.js baslamasi bekleniyor", end="", flush=True)

start = time.time()
while time.time() - start < TIMEOUT:
    try:
        s = socket.create_connection((HOST, PORT), timeout=1)
        s.close()
        print("\nSite hazir!")
        break
    except OSError:
        print(".", end="", flush=True)
        time.sleep(2)
else:
    print("\nZaman asimi — Next.js baslamadi. npm start penceresini kontrol et.")
    sys.exit(1)

time.sleep(1)
webbrowser.open(URL)
print(f"Tarayici acildi: {URL}")
