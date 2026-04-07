import ctypes
import time
import socket
import json
import re
import threading
from flask import Flask, request
from flask_cors import CORS
from pylsl import StreamInfo, StreamOutlet, local_clock

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
s.settimeout(1.0)

try:
    s.connect((GAZEPOINT_IP, GAZEPOINT_PORT))
    print("[INFO] EYE TRACKER: Successfully connected to Gazepoint API.")
    # Command Gazepoint to accept external USER_DATA markers
    s.send(str.encode('<SET ID="ENABLE_SEND_USER_DATA" STATE="1" />\r\n'))
    
     # gaze data stream alanlarını aç
    s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
    s.send(str.encode('<SET ID="ENABLE_SEND_POG_LEFT" STATE="1" />\r\n'))
    s.send(str.encode('<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1" />\r\n'))
    s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />\r\n'))
    s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />\r\n'))
except Exception as e:
    print(f"[ERROR] EYE TRACKER: Could not connect to Gazepoint. Is Open Gaze API enabled? Details: {e}")
    s = None

# ==========================================
# 3. LSL MARKER STREAM SETUP
# ==========================================
marker_info = StreamInfo(
    name='WebMarkers',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='string',
    source_id='senior_project_web_markers'
)

eye_info = StreamInfo(
    name='EyeGaze',
    type='EyeTracking',
    channel_count=9,
    nominal_srate=0,
    channel_format='float32',
    source_id='senior_project_eye_gaze'
)

eye_channels = eye_info.desc().append_child("channels")
for label in ["time", "fpogx", "fpogy", "lpogx", "lpogy", "rpogx", "rpogy", "lpv", "rpv"]:
    ch = eye_channels.append_child("channel")
    ch.append_child_value("label", label)

eye_outlet = StreamOutlet(eye_info)
print("[INFO] LSL: EyeGaze stream initialized.")
marker_outlet = StreamOutlet(marker_info)
print("[INFO] LSL: WebMarkers stream initialized.")

SCENARIO_MARKER_MAP = {
    "slow_image": 11,
    "broken_image": 12,
    "skeleton_prolong": 13,
    "search_irrelevant": 14,
    "button_delay": 15,
    "first_click_miss": 16,
    "feedback_late": 17,
    "network_jitter": 18,
    "overlay_blocking": 19,
    "price_change": 20,
    "coupon_min_spend": 21,
    "coupon_expired": 22,
    "facet_reset_once": 23,
    "sort_reset": 24
}
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


def extract_attr(xml_line, key, default=0.0):
    match = re.search(rf'{key}="([^"]+)"', xml_line)
    if match:
        try:
            return float(match.group(1))
        except:
            return default
    return default

def eye_reader_loop():
    if not s:
        return

    buffer = ""

    while True:
        try:
            chunk = s.recv(4096).decode("utf-8", errors="ignore")
            if not chunk:
                continue

            buffer += chunk

            while "\r\n" in buffer:
                line, buffer = buffer.split("\r\n", 1)
                line = line.strip()

                if not line:
                    continue

                if "<ACK" in line:
                    continue

                if 'FPOGX="' in line or 'LPOGX="' in line or 'RPOGX="' in line:

                    row = [
                        extract_attr(line, "TIME"),
                        extract_attr(line, "FPOGX"),
                        extract_attr(line, "FPOGY"),
                        extract_attr(line, "LPOGX"),
                        extract_attr(line, "LPOGY"),
                        extract_attr(line, "RPOGX"),
                        extract_attr(line, "RPOGY"),
                        extract_attr(line, "LPV"),
                        extract_attr(line, "RPV"),
                    ]
                    eye_outlet.push_sample(row, local_clock())
        except socket.timeout:
            continue
        
        except Exception as e:
            print(f"[ERROR] EYE READER: {e}")
            break
        
def send_lsl_marker(payload):
    try:
        marker_outlet.push_sample([json.dumps(payload, ensure_ascii=False)], local_clock())
        print(f"[EVENT] LSL: Marker dispatched -> {payload}")
    except Exception as e:
        print(f"[ERROR] LSL: Failed to send marker. Details: {e}")
# ==========================================
# WEB API (Endpoint for the frontend SDK)
# ==========================================
@app.route('/send_negative_trigger', methods=['POST'])
def trigger_negative():
    data = request.get_json() or {}

    scenario_name = data.get('scenario_name', 'UNKNOWN_SCENARIO')
    scenario_type = data.get('scenario_type', 'unknown')
    session_id = data.get('session_id')
    experiment_group = data.get('experiment_group')
    phase = data.get('phase')
    page_url = data.get('page_url')
    timestamp = data.get('timestamp', int(time.time() * 1000))

    eeg_marker = SCENARIO_MARKER_MAP.get(scenario_type, 1)

    # 1. EEG TTL
    send_eeg_marker(value=eeg_marker)

    # 2. Eye tracker
    send_eye_marker(scenario_name)

    # 3. LSL
    lsl_payload = {
        "session_id": session_id,
        "experiment_group": experiment_group,
        "phase": phase,
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "event_type": "scenario_start",
        "page_url": page_url,
        "timestamp": timestamp,
        "eeg_marker": eeg_marker
    }
    send_lsl_marker(lsl_payload)

    return {
        "status": "success",
        "message": f"Markers dispatched for: {scenario_name}",
        "eeg_marker": eeg_marker
    }

if __name__ == '__main__':
<<<<<<< HEAD
    app.run(host='127.0.0.1', port=5001) #5000 olması lazım windowsta
=======
    if s:
        threading.Thread(target=eye_reader_loop, daemon=True).start()
        print("[INFO] EYE TRACKER: Reader thread started.")

    app.run(host='127.0.0.1', port=5001)
>>>>>>> lab
