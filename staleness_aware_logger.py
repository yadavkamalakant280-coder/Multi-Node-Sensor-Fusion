"""
Staleness-Aware Asynchronous Multi-Node Sensor Fusion
=====================================================

Dataset logging server for four independent PIR + HC-SR04 sensor nodes.

Pipeline:
    4 asynchronous nodes
        -> /log ingestion
        -> thread-safe per-node state
        -> per-node staleness estimation
        -> adaptive freshness threshold
        -> staleness-aware confidence weighting
        -> 200-ms fused snapshot
        -> per-subject / per-pattern CSV

Expected POST fields from each ESP32:
    node   : "1", "2", "3", or "4"
    dist   : distance in cm
    status : "MOTION_DETECTED" or another status

Optional POST fields:
    node_ts: node timestamp in milliseconds (if available)

The logger records both raw node-level values and derived
staleness-aware fusion values. It does NOT claim that a single
weighted distance is a physical position estimate; the fused
values are confidence-weighted representations for downstream
feature extraction/classification.

Requires:
    pip install flask
"""

from flask import Flask, request
from datetime import datetime
from collections import deque
from threading import Thread, Lock
from pathlib import Path
import csv
import math
import os
import statistics
import time
import logging

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("Gait_Dataset")

NODE_IDS = ["1", "2", "3", "4"]

# Sensor records are approximately 150-250 ms apart.
SNAPSHOT_INTERVAL_SEC = 0.200

# Adaptive staleness settings.
MIN_STALE_SEC = 0.30
MAX_STALE_SEC = 0.80
JITTER_K = 3.0

# A reading older than this is excluded completely from fusion.
# It is still retained in the CSV with its staleness value.
HARD_MAX_AGE_SEC = 1.50

# Weight decay:
# weight = exp(-LAMBDA * staleness)
STALENESS_LAMBDA = 2.0

# Number of recent inter-arrival intervals used to estimate jitter.
INTERVAL_HISTORY_SIZE = 30

# Default threshold before enough timing history exists.
DEFAULT_STALE_SEC = 0.50

# Minimum distance accepted from HC-SR04.
MIN_DISTANCE_CM = 2.0

# Maximum distance accepted from HC-SR04.
MAX_DISTANCE_CM = 400.0

# ============================================================
# GLOBAL STATE
# ============================================================

current_sub_id = "000"
current_pattern = "None"
is_recording = False

data_count = 0
snapshot_count = 0

state_lock = Lock()

node_state = {}
for nid in NODE_IDS:
    node_state[nid] = {
        "motion": False,
        "distance": None,
        "last_update": 0.0,
        "last_node_timestamp": None,
        "sequence": 0,
        "interval_history": deque(maxlen=INTERVAL_HISTORY_SIZE),
        "last_interval": None,
        "packets_received": 0,
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def now_timestamp():
    """Return local timestamp with millisecond precision."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def setup_folders(subject_id):
    path = BASE_DIR / f"Subject_{subject_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_filepath():
    subject_folder = setup_folders(current_sub_id)
    return subject_folder / f"S{current_sub_id}_{current_pattern}.csv"


def calculate_adaptive_threshold(state):
    """
    Adaptive staleness threshold based on recent inter-arrival timing.

    tau_i = mean_interval + K * std_interval

    The result is bounded to avoid an unrealistically small or
    excessively large freshness window.
    """
    history = list(state["interval_history"])

    if len(history) < 3:
        return DEFAULT_STALE_SEC

    mean_interval = statistics.mean(history)
    std_interval = statistics.pstdev(history) if len(history) > 1 else 0.0

    adaptive_tau = mean_interval + JITTER_K * std_interval

    return clamp(adaptive_tau, MIN_STALE_SEC, MAX_STALE_SEC)


def calculate_staleness(now, state):
    """
    Staleness is the age of the latest measurement at fusion time.

        s_i(t) = t_f - t_i,last
    """
    if state["last_update"] <= 0:
        return float("inf")

    return max(0.0, now - state["last_update"])


def calculate_weight(staleness):
    """
    Exponential freshness/confidence weighting.

        w_i = exp(-lambda * s_i)

    Very old readings are excluded before this function is used.
    """
    if not math.isfinite(staleness):
        return 0.0

    return math.exp(-STALENESS_LAMBDA * staleness)


def reset_node_state():
    """Clear all per-node state at the beginning of a recording."""
    with state_lock:
        for nid in NODE_IDS:
            node_state[nid] = {
                "motion": False,
                "distance": None,
                "last_update": 0.0,
                "last_node_timestamp": None,
                "sequence": 0,
                "interval_history": deque(maxlen=INTERVAL_HISTORY_SIZE),
                "last_interval": None,
                "packets_received": 0,
            }


# ============================================================
# ASYNCHRONOUS INGESTION
# ============================================================

@app.route("/log", methods=["POST"])
def log_data():
    """
    Receives one asynchronous measurement from one sensor node.

    The request updates only that node's latest state.
    No node waits for another node.
    """
    global data_count

    if not is_recording:
        return "Idle", 200

    node_id = request.form.get("node")
    distance_raw = request.form.get("dist")
    status = request.form.get("status", "")
    node_timestamp = request.form.get("node_ts")

    if node_id not in NODE_IDS:
        return "Unknown node", 400

    distance = safe_float(distance_raw)

    if distance is not None:
        if not (MIN_DISTANCE_CM <= distance <= MAX_DISTANCE_CM):
            distance = None

    receipt_time = time.time()

    with state_lock:
        state = node_state[node_id]

        # Estimate inter-arrival time from server receipt timestamps.
        if state["last_update"] > 0:
            interval = receipt_time - state["last_update"]

            # Ignore obviously invalid timing gaps from the jitter model.
            if 0.001 <= interval <= 5.0:
                state["interval_history"].append(interval)
                state["last_interval"] = interval

        state["motion"] = (status == "MOTION_DETECTED")
        state["distance"] = distance
        state["last_update"] = receipt_time
        state["last_node_timestamp"] = node_timestamp
        state["sequence"] += 1
        state["packets_received"] += 1

        data_count += 1

    return "Logged", 200


# ============================================================
# STALENESS-AWARE FUSION
# ============================================================

def build_fused_snapshot(now):
    """
    Build a four-node snapshot.

    For each node:
        staleness = current time - last update
        adaptive threshold = timing mean + 3*std, bounded
        freshness = whether staleness <= adaptive threshold
        weight = exp(-lambda * staleness)

    A node older than HARD_MAX_AGE_SEC is excluded from fusion.

    The returned fused representation contains:
        weighted motion
        weighted distance
        total fusion weight
        number of contributing nodes
        mean staleness
        maximum staleness

    The raw node measurements remain in the CSV.
    """
    node_records = []

    with state_lock:
        for nid in NODE_IDS:
            state = node_state[nid]

            staleness = calculate_staleness(now, state)
            adaptive_tau = calculate_adaptive_threshold(state)

            if math.isfinite(staleness):
                freshness = staleness <= adaptive_tau
            else:
                freshness = False

            if (
                state["last_update"] > 0
                and staleness <= HARD_MAX_AGE_SEC
            ):
                weight = calculate_weight(staleness)
            else:
                weight = 0.0

            distance = state["distance"]
            motion = 1 if state["motion"] else 0

            node_records.append({
                "node": nid,
                "motion": motion,
                "distance": distance,
                "staleness": staleness,
                "adaptive_tau": adaptive_tau,
                "fresh": int(freshness),
                "weight": weight,
                "sequence": state["sequence"],
                "packets_received": state["packets_received"],
            })

    # --------------------------------------------------------
    # Confidence-weighted multi-node representation
    # --------------------------------------------------------

    valid = [r for r in node_records if r["weight"] > 0]

    total_weight = sum(r["weight"] for r in valid)

    if total_weight > 0:
        weighted_motion = (
            sum(r["weight"] * r["motion"] for r in valid)
            / total_weight
        )

        distance_records = [
            r for r in valid
            if r["distance"] is not None
        ]

        distance_weight = sum(r["weight"] for r in distance_records)

        if distance_weight > 0:
            weighted_distance = (
                sum(
                    r["weight"] * r["distance"]
                    for r in distance_records
                )
                / distance_weight
            )
        else:
            weighted_distance = None

        mean_staleness = (
            sum(r["weight"] * r["staleness"] for r in valid)
            / total_weight
        )

        max_staleness = max(r["staleness"] for r in valid)

    else:
        weighted_motion = None
        weighted_distance = None
        mean_staleness = None
        max_staleness = None

    return {
        "nodes": node_records,
        "fused_motion": weighted_motion,
        "fused_distance": weighted_distance,
        "total_weight": total_weight,
        "contributing_nodes": len(valid),
        "mean_staleness": mean_staleness,
        "max_staleness": max_staleness,
    }


# ============================================================
# CSV DATASET LOGGING
# ============================================================

CSV_HEADER = [
    "Timestamp",
    "Subject_ID",
    "Pattern",

    "N1_Status",
    "N1_Distance_CM",
    "N1_Staleness_MS",
    "N1_Adaptive_Threshold_MS",
    "N1_Fresh",
    "N1_Weight",
    "N1_Sequence",

    "N2_Status",
    "N2_Distance_CM",
    "N2_Staleness_MS",
    "N2_Adaptive_Threshold_MS",
    "N2_Fresh",
    "N2_Weight",
    "N2_Sequence",

    "N3_Status",
    "N3_Distance_CM",
    "N3_Staleness_MS",
    "N3_Adaptive_Threshold_MS",
    "N3_Fresh",
    "N3_Weight",
    "N3_Sequence",

    "N4_Status",
    "N4_Distance_CM",
    "N4_Staleness_MS",
    "N4_Adaptive_Threshold_MS",
    "N4_Fresh",
    "N4_Weight",
    "N4_Sequence",

    "Fused_Motion",
    "Fused_Distance_CM",
    "Total_Fusion_Weight",
    "Contributing_Nodes",
    "Mean_Staleness_MS",
    "Max_Staleness_MS",
]


def value_or_empty(value):
    if value is None:
        return ""
    return value


def build_csv_row(timestamp, fusion):
    row = [
        timestamp,
        current_sub_id,
        current_pattern,
    ]

    for record in fusion["nodes"]:
        status = (
            "MOTION_DETECTED"
            if record["motion"] == 1
            else "NO_MOTION"
        )

        row.extend([
            status,
            value_or_empty(record["distance"]),
            round(record["staleness"] * 1000, 3)
            if math.isfinite(record["staleness"])
            else "",
            round(record["adaptive_tau"] * 1000, 3),
            record["fresh"],
            round(record["weight"], 6),
            record["sequence"],
        ])

    row.extend([
        round(fusion["fused_motion"], 6)
        if fusion["fused_motion"] is not None
        else "",

        round(fusion["fused_distance"], 3)
        if fusion["fused_distance"] is not None
        else "",

        round(fusion["total_weight"], 6),

        fusion["contributing_nodes"],

        round(fusion["mean_staleness"] * 1000, 3)
        if fusion["mean_staleness"] is not None
        else "",

        round(fusion["max_staleness"] * 1000, 3)
        if fusion["max_staleness"] is not None
        else "",
    ])

    return row


def snapshot_writer():
    """
    Periodic fusion/snapshot thread.

    Every 200 ms:
        1. captures the current time
        2. computes per-node staleness
        3. computes adaptive freshness thresholds
        4. computes staleness weights
        5. constructs the multi-node fused representation
        6. writes one wide-format row
    """
    global snapshot_count

    while True:
        if not is_recording:
            time.sleep(0.05)
            continue

        loop_start = time.time()

        filepath = current_filepath()
        timestamp = now_timestamp()

        fusion = build_fused_snapshot(loop_start)
        row = build_csv_row(timestamp, fusion)

        file_exists = filepath.is_file()

        with open(
            filepath,
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(CSV_HEADER)

            writer.writerow(row)

        snapshot_count += 1

        # Maintain approximately fixed 200-ms snapshot spacing.
        elapsed = time.time() - loop_start
        sleep_for = max(
            0.0,
            SNAPSHOT_INTERVAL_SEC - elapsed
        )
        time.sleep(sleep_for)


# ============================================================
# LIVE DISPLAY
# ============================================================

def timer_display(start_time):
    global is_recording

    while is_recording:
        elapsed = int(time.time() - start_time)

        with state_lock:
            live_nodes = []

            for nid in NODE_IDS:
                state = node_state[nid]

                if state["last_update"] <= 0:
                    continue

                staleness = time.time() - state["last_update"]
                tau = calculate_adaptive_threshold(state)

                if staleness <= tau:
                    live_nodes.append(nid)

        print(
            f"\r[TIMER: {elapsed:03d}s] "
            f"Raw events: {data_count} | "
            f"Snapshots: {snapshot_count} | "
            f"Fresh nodes: {live_nodes or 'None'} | "
            f"PRESS ENTER TO STOP -> ",
            end="",
            flush=True,
        )

        time.sleep(0.1)


# ============================================================
# RECORDING CONTROL PANEL
# ============================================================

def control_panel():
    global current_sub_id
    global current_pattern
    global is_recording
    global data_count
    global snapshot_count

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    patterns = {
        "1": "Direct",
        "2": "Pacing",
        "3": "Lapping",
        "4": "Random",
    }

    while True:
        print("\n" + "=" * 72)
        print(" STALENESS-AWARE ASYNCHRONOUS MULTI-NODE SENSOR FUSION")
        print("             DATASET COLLECTION SERVER")
        print("=" * 72)

        current_sub_id = input(
            "Enter Subject ID (e.g., 001): "
        ).strip().zfill(3)

        setup_folders(current_sub_id)

        print(
            "\nSelect Pattern:"
            "\n[1] Direct"
            "\n[2] Pacing"
            "\n[3] Lapping"
            "\n[4] Random"
        )

        choice = input("Enter Choice (1-4): ").strip()
        current_pattern = patterns.get(choice, "Unknown")

        if current_pattern == "Unknown":
            print("Invalid pattern selection.")
            continue

        print("\n" + "-" * 72)
        print(f"Subject : {current_sub_id}")
        print(f"Pattern : {current_pattern}")
        print(f"Output  : {current_filepath()}")
        print(
            f"Snapshot interval : {SNAPSHOT_INTERVAL_SEC * 1000:.0f} ms"
        )
        print(
            f"Initial staleness threshold : "
            f"{DEFAULT_STALE_SEC * 1000:.0f} ms"
        )
        print(
            f"Adaptive threshold range : "
            f"{MIN_STALE_SEC * 1000:.0f}-"
            f"{MAX_STALE_SEC * 1000:.0f} ms"
        )
        print(f"Hard maximum age : {HARD_MAX_AGE_SEC * 1000:.0f} ms")
        print("-" * 72)

        input(">>> Press ENTER to START recording...")

        data_count = 0
        snapshot_count = 0

        reset_node_state()

        is_recording = True
        start_time = time.time()

        display_thread = Thread(
            target=timer_display,
            args=(start_time,),
            daemon=True,
        )
        display_thread.start()

        input("")

        is_recording = False

        print("\n\nRecording stopped.")
        print(f"Raw events received : {data_count}")
        print(f"Snapshots generated : {snapshot_count}")
        print(f"File saved          : {current_filepath()}")

        continue_recording = input(
            "\nContinue with another trial? (y/n): "
        ).strip().lower()

        if continue_recording != "y":
            print("Exiting logger.")
            os._exit(0)


# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    panel_thread = Thread(
        target=control_panel,
        daemon=True,
    )
    panel_thread.start()

    writer_thread = Thread(
        target=snapshot_writer,
        daemon=True,
    )
    writer_thread.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
