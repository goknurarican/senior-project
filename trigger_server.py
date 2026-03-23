import ctypes
import time
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PORT_ADDRESS = 0xEEFC

try:
    parallel_port = ctypes.WinDLL("inpoutx64.dll")
    print("Paralel port sürücüsü başarıyla yüklendi.")
except Exception as e:
    print("HATA: inpoutx64.dll bulunamadı! Lütfen dosyayı bu dizine ekleyin.")

def send_marker(value=1, duration_ms=10):
    try:
        parallel_port.Out32(PORT_ADDRESS, value)
        time.sleep(duration_ms / 1000.0)
        parallel_port.Out32(PORT_ADDRESS, 0)
        print(f"Marker {value} gönderildi.")
    except Exception as e:
        print("Marker gönderilirken hata oluştu (Sürücü yok veya port yanlış olabilir).")

@app.route('/send_negative_trigger', methods=['POST'])
def trigger_negative():
    send_marker(value=1)
    return {"status": "success", "message": "EEG'ye marker iletildi."}

if __name__ == '__main__':
    app.run(port=5000)