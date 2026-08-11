from __future__ import annotations

import csv
import logging
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


@dataclass
class AttentionSnapshot:
    brightness: float = 50.0
    eye_openness: float = 70.0
    movement: float = 20.0
    energy_score: float = 63.5
    gaze_score: float = 0.58
    total_score: float = 0.62
    attention_score: float = 0.0
    state: str = "no_face"
    face_detected: bool = False
    source: str = "fallback"
    status: str = "idle"
    timestamp: float = 0.0
    eye_aspect_ratio: float = 0.25
    eye_movement: float = 0.0
    face_distance: float = 0.5
    distance_ok: bool = True
    head_movement: float = 0.0
    normalized_attention: float = 0.5


LATEST = AttentionSnapshot(timestamp=time.time())
LATEST_FRAME: bytes = b""
_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()
_SNAPSHOT_LOCK = threading.Lock()   # guards LATEST + LATEST_FRAME

CSV_FILE = Path("webcam_metrics.csv")
CSV_HEADERS = [
    "timestamp",
    "brightness",
    "eye_openness",
    "movement",
    "energy_score",
    "gaze_score",
    "total_score",
    "attention_score",
    "state",
]

# Eye landmark indices for MediaPipe FaceMesh
_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# Iris center indices (refine_landmarks=True, 478 total)
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473


def compute_brightness(face_crop: np.ndarray) -> float:
    if face_crop.size == 0:
        return 50.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    median_val = np.median(gray)
    return float(np.clip(median_val / 2.55, 0, 100))


def _ear(pts, a, b, c, d, e, f):
    vertical = np.linalg.norm(pts[b] - pts[f]) + np.linalg.norm(pts[c] - pts[e])
    horizontal = np.linalg.norm(pts[a] - pts[d]) + 1e-6
    return vertical / horizontal


def compute_eye_openness(frame: np.ndarray, face_landmarks) -> tuple[float, float]:
    """Return (eye_openness_0_100, raw_ear_value)."""
    if not face_landmarks:
        return 70.0, 0.25
    h, w = frame.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in face_landmarks.landmark])
    left = _ear(pts, 33, 160, 158, 133, 153, 144)
    right = _ear(pts, 362, 385, 387, 263, 373, 380)
    ear_avg = (left + right) / 2
    openness = float(np.clip(np.interp(ear_avg, [0.15, 0.35], [0, 100]), 0, 100))
    return openness, float(ear_avg)


def compute_movement(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    if prev_gray is None or prev_gray.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.clip(np.mean(mag) * 12, 0, 100))


def compute_scores(brightness: float, eye_openness: float, movement: float) -> tuple[float, float, float]:
    energy = 0.35 * brightness + 0.30 * eye_openness + 0.35 * (100 - movement)
    b, e, m = brightness / 100, eye_openness / 100, movement / 100
    gaze = 0.5 * e + 0.3 * (1 - m) + 0.2 * b
    total = 0.7 * (energy / 100) + 0.3 * gaze
    return energy, gaze, total


def compute_energy_score(brightness: float, eye_openness: float, movement: float) -> float:
    return float(0.35 * brightness + 0.30 * eye_openness + 0.35 * (100 - movement))


def compute_gaze_score(brightness: float, eye_openness: float, movement: float, eye_movement: float = 0.0) -> float:
    b, e, m = brightness / 100, eye_openness / 100, movement / 100
    return float(0.5 * e + 0.3 * (1 - m) + 0.2 * b)


def compute_attention_score(energy_score: float, gaze_score: float) -> float:
    return float(0.7 * (energy_score / 100.0) + 0.3 * gaze_score)


def compute_vfas(brightness: float, eye_openness: float, movement: float) -> float:
    return 0.4 * (brightness / 100) + 0.3 * (eye_openness / 100) + 0.2 * (1 - movement / 100)


def classify_state(gaze: float, face_detected: bool) -> str:
    if not face_detected:
        return "no_face"
    if gaze < 0.2:
        return "distracted"
    elif gaze <= 0.6:
        return "neutral"
    return "immersive"


def get_latest_attention_snapshot() -> dict:
    with _SNAPSHOT_LOCK:
        return asdict(LATEST)


def get_latest_frame() -> bytes:
    with _SNAPSHOT_LOCK:
        return LATEST_FRAME


def init_csv():
    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)


def append_to_csv(data: dict):
    with open(CSV_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(data)


def _open_camera(index: int) -> cv2.VideoCapture:
    if _IS_WINDOWS:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap

def _probe_camera(camera_index: int = 0) -> tuple[bool, str]:
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        cap.release()
        return False, f"Unable to open camera index {camera_index}"
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return False, f"Camera opened but failed to read from index {camera_index}"
    return True, "camera available"


def _find_camera() -> tuple[int, str]:
    """Try indices 0, 1, 2 and return first working one."""
    for idx in range(3):
        ok, msg = _probe_camera(idx)
        if ok:
            return idx, msg
    return -1, "No working camera found at indices 0, 1, 2"


def start_background_attention(camera_index: int = -1, show_window: bool = False) -> dict[str, str]:
    global _THREAD, LATEST
    if _THREAD and _THREAD.is_alive():
        return {"status": "already_running", "message": "Attention monitor already running"}

    if camera_index < 0:
        camera_index, message = _find_camera()
    else:
        _, message = _probe_camera(camera_index)

    if camera_index < 0:
        LATEST.status = "camera_unavailable"
        LATEST.source = "webcam"
        LATEST.timestamp = time.time()
        return {"status": "camera_unavailable", "message": message}

    # Windows needs a moment after probe releases the device
    time.sleep(0.5)

    _STOP_EVENT.clear()
    _THREAD = threading.Thread(target=_attention_loop, args=(camera_index, show_window), daemon=True)
    _THREAD.start()
    LATEST.status = "starting"
    LATEST.source = "webcam"
    LATEST.timestamp = time.time()
    return {"status": "started", "message": f"Attention monitor started (camera {camera_index})"}


def stop_background_attention() -> dict[str, str]:
    global _THREAD
    _STOP_EVENT.set()
    if _THREAD:
        _THREAD.join(timeout=2.0)
        _THREAD = None
    return {"status": "stopped", "message": "Attention monitor stopped"}


def _draw_overlays(frame: np.ndarray, face_landmarks, x_min, y_min, x_max, y_max, state: str, snapshot: AttentionSnapshot):
    """Draw face bounding box, eye landmarks, iris markers, and metrics onto the frame."""
    h, w = frame.shape[:2]

    # Face bounding box
    state_color = {
        "immersive": (0, 220, 0),
        "neutral": (0, 180, 255),
        "distracted": (0, 60, 255),
        "no_face": (160, 160, 160),
    }.get(state, (0, 220, 0))
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), state_color, 2)

    # State label above bounding box
    label = state.upper()
    cv2.putText(frame, label, (int(x_min), max(int(y_min) - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)

    # Eye corner landmarks (yellow dots)
    for idx in _LEFT_EYE_IDX + _RIGHT_EYE_IDX:
        px = int(face_landmarks.landmark[idx].x * w)
        py = int(face_landmarks.landmark[idx].y * h)
        cv2.circle(frame, (px, py), 2, (0, 220, 220), -1)

    # Iris centers (blue circle with crosshair)
    total_landmarks = len(face_landmarks.landmark)
    for iris_idx in (_LEFT_IRIS_CENTER, _RIGHT_IRIS_CENTER):
        if iris_idx < total_landmarks:
            ix = int(face_landmarks.landmark[iris_idx].x * w)
            iy = int(face_landmarks.landmark[iris_idx].y * h)
            cv2.circle(frame, (ix, iy), 5, (255, 80, 0), 2)
            cv2.line(frame, (ix - 7, iy), (ix + 7, iy), (255, 80, 0), 1)
            cv2.line(frame, (ix, iy - 7), (ix, iy + 7), (255, 80, 0), 1)

    # Metrics overlay (bottom-left)
    lines = [
        f"EAR:{snapshot.eye_aspect_ratio:.2f}  Eye:{snapshot.eye_openness:.0f}",
        f"Bright:{snapshot.brightness:.0f}  Move:{snapshot.movement:.0f}",
        f"Energy:{snapshot.energy_score:.0f}  Gaze:{snapshot.gaze_score:.2f}",
        f"Attn:{snapshot.normalized_attention:.2f}  Dist:{snapshot.face_distance:.2f}",
    ]
    y_off = h - 10 - (len(lines) - 1) * 17
    for line in lines:
        cv2.putText(frame, line, (6, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y_off += 17


def _attention_loop(camera_index: int, show_window: bool):
    global LATEST, LATEST_FRAME
    logger.info("Camera thread started")
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )

    # Retry opening up to 3 times (Windows DirectShow release can be slow)
    cap = None
    for attempt in range(3):
        cap = _open_camera(camera_index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                logger.info("Camera opened successfully")
                break
            cap.release()
        time.sleep(0.4)
    if cap is None or not cap.isOpened():
        with _SNAPSHOT_LOCK:
            globals()["LATEST"] = AttentionSnapshot(status="camera_unavailable", timestamp=time.time())
        return

    # Request MJPEG from camera for higher FPS on Windows
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if show_window:
        cv2.namedWindow("Attention Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Attention Monitor", 640, 480)

    init_csv()

    prev_gray = None
    prev_iris_pos: tuple[float, float] | None = None
    prev_face_center: tuple[float, float] | None = None
    face_detected_start: float | None = None
    focus_time = 0.0
    frame_count = 0

    while not _STOP_EVENT.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_count += 1
        # Mirror for natural feel
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (480, 360))

        # Process every 2nd frame for performance
        if frame_count % 2 != 0:
            # Still encode and store the raw frame so stream doesn't stutter
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                with _SNAPSHOT_LOCK:
                    globals()["LATEST_FRAME"] = buf.tobytes()
                continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        results = face_mesh.process(rgb)
        face_detected = bool(results.multi_face_landmarks)

        if face_detected:
            face_landmarks = results.multi_face_landmarks[0]
            lm = face_landmarks.landmark

            # Bounding box from landmarks
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            x_min, x_max = max(min(xs) - 5, 0), min(max(xs) + 5, w)
            y_min, y_max = max(min(ys) - 5, 0), min(max(ys) + 5, h)
            face_crop = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            brightness = compute_brightness(face_crop)
            eye_openness, ear_val = compute_eye_openness(frame, face_landmarks)
            movement = compute_movement(prev_gray, gray)
            energy, gaze, total_score = compute_scores(brightness, eye_openness, movement)
            vfas = compute_vfas(brightness, eye_openness, movement)

            # Focus time
            if face_detected_start is None:
                face_detected_start = time.time()
            focus_time = time.time() - face_detected_start
            # 🔥 FIXED ATTENTION (REAL-TIME, NOT TIME-BIASED)
            attention_score = compute_attention_score(energy, gaze)
            normalized_attention = float(np.clip(attention_score, 0.0, 1.0))

            # Iris position → eye movement
            total_lm = len(lm)
            iris_pos: tuple[float, float] | None = None
            if _LEFT_IRIS_CENTER < total_lm and _RIGHT_IRIS_CENTER < total_lm:
                li = lm[_LEFT_IRIS_CENTER]
                ri = lm[_RIGHT_IRIS_CENTER]
                iris_pos = ((li.x + ri.x) / 2 * w, (li.y + ri.y) / 2 * h)
            eye_movement = 0.0
            if prev_iris_pos is not None and iris_pos is not None:
                eye_movement = float(np.linalg.norm(
                    np.array(iris_pos) - np.array(prev_iris_pos)
                ))
            prev_iris_pos = iris_pos

            # Face center → head movement
            face_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            head_movement = 0.0
            if prev_face_center is not None:
                head_movement = float(np.linalg.norm(
                    np.array(face_center) - np.array(prev_face_center)
                ))
            prev_face_center = face_center

            # Face distance (normalized face width)
            face_distance = float(np.clip(1.0 - (x_max - x_min) / w, 0.05, 1.0))
            distance_ok = 0.15 < face_distance < 0.75

            state = classify_state(gaze, True)

            new_snap = AttentionSnapshot(
                brightness=brightness,
                eye_openness=eye_openness,
                movement=movement,
                energy_score=energy,
                gaze_score=gaze,
                total_score=total_score,
                attention_score=attention_score,
                state=state,
                face_detected=True,
                source="webcam",
                status="running",
                timestamp=time.time(),
                eye_aspect_ratio=ear_val,
                eye_movement=float(np.clip(eye_movement, 0, 50)),
                face_distance=face_distance,
                distance_ok=distance_ok,
                head_movement=float(np.clip(head_movement, 0, 100)),
                normalized_attention=normalized_attention,
            )
            with _SNAPSHOT_LOCK:
                globals()["LATEST"] = new_snap

            _draw_overlays(frame, face_landmarks, x_min, y_min, x_max, y_max, state, new_snap)

            append_to_csv({
                "timestamp": time.time(),
                "brightness": brightness,
                "eye_openness": eye_openness,
                "movement": movement,
                "energy_score": energy,
                "gaze_score": gaze,
                "total_score": total_score,
                "attention_score": attention_score,
                "state": state,
            })
        else:
            face_detected_start = None
            focus_time = 0.0
            prev_iris_pos = None
            prev_face_center = None
            with _SNAPSHOT_LOCK:
                globals()["LATEST"] = AttentionSnapshot(
                    face_detected=False,
                    state="no_face",
                    source="webcam",
                    status="running",
                    timestamp=time.time(),
                    normalized_attention=0.0,
                    attention_score=0.0,
                )

            # "No face" overlay
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 200), 2, cv2.LINE_AA)

        # Encode frame (with overlays)
        success, buf = cv2.imencode(".jpg", frame)
        if success:
            with _SNAPSHOT_LOCK:
                globals()["LATEST_FRAME"] = buf.tobytes()
        else:
            logger.warning("Frame encoding failed")

        if show_window:
            cv2.imshow("Attention Monitor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        prev_gray = gray.copy()

    cap.release()
    face_mesh.close()
    if show_window:
        cv2.destroyAllWindows()
