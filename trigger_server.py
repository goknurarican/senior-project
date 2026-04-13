import ctypes
import time
import socket
import json
import re
import threading
from pathlib import Path
from flask import Flask, request
from flask_cors import CORS
from pylsl import StreamInfo, StreamOutlet, local_clock
from data_logger import save_event, save_eye_data

_SCRIPT_DIR = Path(__file__).resolve().parent

# ==========================================
# SESSION TRACKING (thread-safe)
# ==========================================
_session_lock = threading.Lock()
current_session_id = None

def get_current_session_id():
    with _session_lock:
        return current_session_id

def set_current_session_id(sid):
    global current_session_id
    with _session_lock:
        current_session_id = sid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# 1. EEG PARALLEL PORT SETUP
# ==========================================
PORT_ADDRESS = 0xEEFC

# Search for the DLL in the script folder first, then fall back to PATH/CWD.
# Place inpoutx64.dll in the same folder as trigger_server.py.
_dll_candidates = [
    str(_SCRIPT_DIR / "inpoutx64.dll"),  # preferred: next to this script
    "inpoutx64.dll",                      # fallback: Windows PATH or CWD
]
parallel_port = None
for _dll in _dll_candidates:
    try:
        parallel_port = ctypes.WinDLL(_dll)
        print(f"[INFO] EEG: Parallel port driver loaded ({_dll}).")
        break
    except Exception:
        continue
if parallel_port is None:
    print("[ERROR] EEG: inpoutx64.dll not found. Place it next to trigger_server.py.")
    print("        Hardware markers will NOT be sent.")


# ==========================================
# 2. GAZEPOINT (EYE TRACKER) SOCKET SETUP
# ==========================================
GAZEPOINT_IP = '127.0.0.1'
GAZEPOINT_PORT = 4242

_GAZE_ENABLE_CMDS = [
    '<SET ID="ENABLE_SEND_USER_DATA" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_LEFT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />\r\n',
    '<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />\r\n',
]

# Global Gazepoint socket (updated on reconnect)
gaze_socket = None
_gaze_socket_lock = threading.Lock()


def _make_gazepoint_socket():
    """Create and configure a fresh Gazepoint socket. Raises on failure."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    sock.connect((GAZEPOINT_IP, GAZEPOINT_PORT))
    for cmd in _GAZE_ENABLE_CMDS:
        sock.send(str.encode(cmd))
    return sock


def reconnect_gazepoint(retry_interval=5):
    """Block until Gazepoint reconnects, then update the global socket."""
    global gaze_socket
    while True:
        try:
            sock = _make_gazepoint_socket()
            with _gaze_socket_lock:
                gaze_socket = sock
            print("[INFO] EYE TRACKER: Reconnected to Gazepoint API.")
            return
        except Exception as e:
            print(f"[WARN] EYE TRACKER: Reconnect failed ({e}), retrying in {retry_interval}s...")
            time.sleep(retry_interval)


# Initial connection attempt
try:
    gaze_socket = _make_gazepoint_socket()
    print("[INFO] EYE TRACKER: Successfully connected to Gazepoint API.")
except Exception as e:
    print(f"[ERROR] EYE TRACKER: Could not connect to Gazepoint. Is Open Gaze API enabled? Details: {e}")
    gaze_socket = None

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

# Keys are scenario_name values — must use scenario_name for lookup (not scenario_type)
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
    with _gaze_socket_lock:
        sock = gaze_socket
    if sock:
        try:
            msg = f'<SET ID="USER_DATA" VALUE="{scenario_name}" />\r\n'
            sock.send(str.encode(msg))
            print(f"[EVENT] EYE TRACKER: Marker '{scenario_name}' dispatched.")
        except Exception as e:
            print(f"[ERROR] EYE TRACKER: Network error while sending marker. Details: {e}")


def extract_attr(xml_line, key, default=0.0):
    match = re.search(rf'{key}="([^"]+)"', xml_line)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return default
    return default


def eye_reader_loop():
    """
    Continuously reads gaze data from Gazepoint.
    Reconnects automatically on any socket error instead of stopping.
    """
    global gaze_socket
    buffer = ""

    while True:
        with _gaze_socket_lock:
            sock = gaze_socket

        if sock is None:
            time.sleep(1)
            continue

        try:
            chunk = sock.recv(4096).decode("utf-8", errors="ignore")
            if not chunk:
                # Gazepoint closed the connection
                print("[WARN] EYE READER: Connection dropped, reconnecting...")
                with _gaze_socket_lock:
                    gaze_socket = None
                reconnect_gazepoint()
                buffer = ""
                continue

            buffer += chunk

            while "\r\n" in buffer:
                line, buffer = buffer.split("\r\n", 1)
                line = line.strip()

                if not line or "<ACK" in line:
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

                    # Wall-clock ms at the moment this sample arrives — used for
                    # cross-stream alignment (Gazepoint TIME is relative seconds,
                    # not Unix epoch).
                    wall_time_ms = int(time.time() * 1000)

                    sid = get_current_session_id()
                    save_eye_data(
                        session_id=sid,
                        gazepoint_time=row[0],
                        wall_time_ms=wall_time_ms,
                        gaze_x=row[1],
                        gaze_y=row[2],
                        pupil_left=row[7],
                        pupil_right=row[8]
                    )

                    eye_outlet.push_sample(row, local_clock())

        except socket.timeout:
            continue

        except Exception as e:
            print(f"[ERROR] EYE READER: {e}, attempting reconnect...")
            with _gaze_socket_lock:
                gaze_socket = None
            reconnect_gazepoint()
            buffer = ""


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

    # Update session id in a thread-safe way
    set_current_session_id(session_id)

    experiment_group = data.get('experiment_group')
    phase = data.get('phase')
    page_url = data.get('page_url')
    timestamp = data.get('timestamp', int(time.time() * 1000))

    # FIX: look up by scenario_name (map keys are names, not types)
    eeg_marker = SCENARIO_MARKER_MAP.get(scenario_name, 1)

    # Wall-clock reference recorded at the same moment for all three streams
    wall_time_ms = int(time.time() * 1000)

    # 1. EEG TTL
    send_eeg_marker(value=eeg_marker)

    # 2. Eye tracker marker
    send_eye_marker(scenario_name)

    # 3. LSL marker — include wall_time_ms so all streams can be aligned offline
    lsl_payload = {
        "session_id": session_id,
        "experiment_group": experiment_group,
        "phase": phase,
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "event_type": "scenario_start",
        "page_url": page_url,
        "timestamp": timestamp,
        "wall_time_ms": wall_time_ms,
        "eeg_marker": eeg_marker
    }
    send_lsl_marker(lsl_payload)
    save_event(lsl_payload)

    return {
        "status": "success",
        "message": f"Markers dispatched for: {scenario_name}",
        "eeg_marker": eeg_marker
    }


if __name__ == '__main__':
    if gaze_socket:
        threading.Thread(target=eye_reader_loop, daemon=True).start()
        print("[INFO] EYE TRACKER: Reader thread started.")
    else:
        # Start reader anyway — it will reconnect when Gazepoint becomes available
        threading.Thread(target=eye_reader_loop, daemon=True).start()
        print("[INFO] EYE TRACKER: Reader thread started (will connect when Gazepoint is available).")

    app.run(host='127.0.0.1', port=5001)
