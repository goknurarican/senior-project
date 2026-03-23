import ctypes
import time
import socket
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) #runs accross different os envs
# ==========================================
# 1. EEG PARALLEL PORT SETUP
# ==========================================
PORT_ADDRESS = 0xEEFC

try:
    parallel_port = ctypes.WinDLL("inpoutx64.dll")
    print("[INFO] EEG: Parallel port driver loaded successfully.")
except Exception as e:
    print(f"[ERROR] EEG: inpoutx64.dll not found. Hardware markers will NOT be sent. Details: {e}")
    parallel_port = None


# ==========================================
# 2. GAZEPOINT (EYE TRACKER) SOCKET SETUP
# ==========================================
GAZEPOINT_IP = '127.0.0.1'
GAZEPOINT_PORT = 4242
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.connect((GAZEPOINT_IP, GAZEPOINT_PORT))
    print("[INFO] EYE TRACKER: Successfully connected to Gazepoint API.")
    # Command Gazepoint to accept external USER_DATA markers
    s.send(str.encode('<SET ID="ENABLE_SEND_USER_DATA" STATE="1" />\r\n'))
except Exception as e:
    print(f"[ERROR] EYE TRACKER: Could not connect to Gazepoint. Is Open Gaze API enabled? Details: {e}")
    s = None


# ==========================================
# MARKER TRANSMISSION FUNCTIONS
# ==========================================
def send_eeg_marker(value=1, duration_ms=10):
    if parallel_port:
        try:
            parallel_port.Out32(PORT_ADDRESS, value)
            time.sleep(duration_ms / 1000.0)
            parallel_port.Out32(PORT_ADDRESS, 0)
            print(f"[EVENT] EEG: Marker {value} dispatched.")
        except Exception as e:
            print(f"[ERROR] EEG: Failed to send marker. Details: {e}")

def send_eye_marker(scenario_name):
    if s:
        try:
            # XML format required by Gazepoint
            msg = f'<SET ID="USER_DATA" VALUE="{scenario_name}" />\r\n'
            s.send(str.encode(msg))
            print(f"[EVENT] EYE TRACKER: Marker '{scenario_name}' dispatched.")
        except Exception as e:
            print(f"[ERROR] EYE TRACKER: Network error while sending marker. Details: {e}")


# ==========================================
# WEB API (Endpoint for the frontend SDK)
# ==========================================
@app.route('/send_negative_trigger', methods=['POST'])
def trigger_negative():
    data = request.get_json() or {}
    scenario_name = data.get('scenario', 'UNKNOWN_SCENARIO')
    
    # 1. Send hardware TTL trigger to EEG
    send_eeg_marker(value=1)
    
    # 2. Send software string trigger to Eye Tracker
    send_eye_marker(scenario_name)
    
    return {"status": "success", "message": f"Markers dispatched for: {scenario_name}"}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001) #5000 olması lazım windowsta