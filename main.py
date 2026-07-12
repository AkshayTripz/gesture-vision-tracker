"""
Gesture Vision Tracker
----------------------
Real-time hand and face tracking with MediaPipe, adaptive low-light
enhancement, per-hand finger counting, keyboard number-key triggers,
and a custom dark-themed control panel with 10 hand filters and
10 face filters.

Run: python main.py
Controls window: click tabs, drag sliders, click filter buttons.
Keyboard: N = cycle night-vision mode, H = cycle hand filter,
          F = cycle face filter, K = toggle keyboard trigger, Esc = quit.
"""

import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

WHEAT = (130, 216, 245)
DOT_RADIUS = 4

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

SMOOTHING_WINDOW = 5
TRAIL_LENGTH = 15

PANEL_W, PANEL_H = 480, 680
TAB_H = 44
TAB_NAMES = ["NIGHT VISION", "HAND FILTERS", "FACE FILTERS", "SYSTEM"]

BG_DARK = (18, 18, 18)
PANEL_BG = (25, 25, 25)
WIDGET_BG = (45, 45, 45)
WIDGET_BORDER = (80, 80, 80)
TEXT_DIM = (160, 160, 160)
TEXT_BRIGHT = (220, 220, 220)

HAND_FILTERS = [
    "Dots Only", "Skeleton Lines", "Neon Glow", "Wireframe Mesh",
    "Fire Trail", "Rainbow Gradient", "Laser Fingertip", "Minimal Joints",
    "Bounding Box", "Hidden (Off)",
]

FACE_FILTERS = [
    "Dots Mesh", "Wireframe Tesselation", "Contour Outline", "Face Oval Only",
    "Eye Highlight", "Glow Mask", "Privacy Blur", "Pixelate",
    "Edge Highlight", "Hidden (Off)",
]

NIGHT_SLIDER_DEFS = [
    ("gamma", "Gamma", 1.0, 3.0, 1.7, True),
    ("clip", "CLAHE Clip", 1.0, 8.0, 3.5, True),
    ("sharpen", "Sharpen Strength", 0.0, 3.0, 1.0, True),
    ("gain", "Brightness Gain", 0, 100, 20, False),
    ("denoise", "Denoise Kernel", 1, 15, 3, False),
    ("temporal", "Temporal Avg Frames", 1, 8, 1, False),
    ("det_conf", "Detection Confidence", 0.3, 0.95, 0.6, True),
]

SYSTEM_SLIDER_DEFS = [
    ("cooldown", "Trigger Cooldown (s)", 0.1, 2.0, 0.6, True),
]


# ---------------------------------------------------------------------------
# Low-level widget primitives (custom-drawn, no native OS trackbar dialogs -
# that's the only way to get a real dark theme across platforms)
# ---------------------------------------------------------------------------

def clamp(v, a, b):
    return max(a, min(b, v))


def slider_hit(s, mx, my):
    return s["x"] <= mx <= s["x"] + s["w"] and s["y"] <= my <= s["y"] + s["h"]


def set_slider_from_x(s, mx):
    frac = clamp((mx - s["x"]) / s["w"], 0, 1)
    val = s["min"] + frac * (s["max"] - s["min"])
    if not s["float"]:
        val = round(val)
    s["value"] = val


def draw_slider(canvas, s):
    x, y, w, h = s["x"], s["y"], s["w"], s["h"]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), WIDGET_BG, -1)
    frac = (s["value"] - s["min"]) / (s["max"] - s["min"])
    fill_w = int(w * frac)
    cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), WHEAT, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), WIDGET_BORDER, 1)
    label = f"{s['label']}: {s['value']:.2f}" if s["float"] else f"{s['label']}: {int(s['value'])}"
    cv2.putText(canvas, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_BRIGHT, 1, cv2.LINE_AA)


def button_hit(b, mx, my):
    return b["x"] <= mx <= b["x"] + b["w"] and b["y"] <= my <= b["y"] + b["h"]


def draw_button(canvas, b, active):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    bg = WHEAT if active else WIDGET_BG
    fg = (20, 20, 20) if active else TEXT_BRIGHT
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), WIDGET_BORDER, 1)
    (tw, th), _ = cv2.getTextSize(b["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(canvas, b["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def draw_tabs(canvas, active_tab):
    tab_w = PANEL_W // len(TAB_NAMES)
    for i, name in enumerate(TAB_NAMES):
        x0 = i * tab_w
        active = i == active_tab
        bg = (35, 35, 35) if active else (20, 20, 20)
        cv2.rectangle(canvas, (x0, 0), (x0 + tab_w, TAB_H), bg, -1)
        if active:
            cv2.rectangle(canvas, (x0, TAB_H - 3), (x0 + tab_w, TAB_H), WHEAT, -1)
        color = WHEAT if active else TEXT_DIM
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = x0 + (tab_w - tw) // 2
        ty = (TAB_H + th) // 2
        cv2.putText(canvas, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# State construction - all sliders/buttons built once, positions fixed
# ---------------------------------------------------------------------------

def build_state():
    state = {
        "active_tab": 0,
        "dragging": None,
        "mode_idx": 0,          # 0=AUTO 1=DAY 2=NIGHT
        "hand_filter_idx": 0,
        "face_filter_idx": 0,
        "keyboard_trigger": True,
        "fps": 0.0,
        "brightness": 0.0,
        "resolution": "",
        "sliders": {},
        "buttons": [],
    }

    y = TAB_H + 40
    for key, label, mn, mx, default, is_float in NIGHT_SLIDER_DEFS:
        state["sliders"][key] = {
            "key": key, "label": label, "min": mn, "max": mx,
            "value": default, "default": default, "float": is_float,
            "x": 24, "y": y, "w": PANEL_W - 48, "h": 20, "tab": 0,
        }
        y += 50

    mode_labels = ["AUTO", "DAY", "NIGHT"]
    btn_w = (PANEL_W - 48 - 20) // 3
    for i, lbl in enumerate(mode_labels):
        state["buttons"].append({
            "tab": 0, "label": lbl, "x": 24 + i * (btn_w + 10), "y": y + 10,
            "w": btn_w, "h": 36,
            "action": (lambda st, idx=i: st.__setitem__("mode_idx", idx)),
            "is_active": (lambda st, idx=i: st["mode_idx"] == idx),
        })

    y0 = TAB_H + 20
    for i, name in enumerate(HAND_FILTERS):
        state["buttons"].append({
            "tab": 1, "label": f"{i+1}. {name}", "x": 24, "y": y0 + i * 38,
            "w": PANEL_W - 48, "h": 34,
            "action": (lambda st, idx=i: st.__setitem__("hand_filter_idx", idx)),
            "is_active": (lambda st, idx=i: st["hand_filter_idx"] == idx),
        })

    for i, name in enumerate(FACE_FILTERS):
        state["buttons"].append({
            "tab": 2, "label": f"{i+1}. {name}", "x": 24, "y": y0 + i * 38,
            "w": PANEL_W - 48, "h": 34,
            "action": (lambda st, idx=i: st.__setitem__("face_filter_idx", idx)),
            "is_active": (lambda st, idx=i: st["face_filter_idx"] == idx),
        })

    sy = TAB_H + 40
    for key, label, mn, mx, default, is_float in SYSTEM_SLIDER_DEFS:
        state["sliders"][key] = {
            "key": key, "label": label, "min": mn, "max": mx,
            "value": default, "default": default, "float": is_float,
            "x": 24, "y": sy, "w": PANEL_W - 48, "h": 20, "tab": 3,
        }
        sy += 50

    state["buttons"].append({
        "tab": 3, "label": "Keyboard Trigger: ON", "x": 24, "y": sy + 10,
        "w": PANEL_W - 48, "h": 36,
        "action": (lambda st: st.__setitem__("keyboard_trigger", not st["keyboard_trigger"])),
        "is_active": (lambda st: st["keyboard_trigger"]),
        "dynamic_label": True,
    })

    state["buttons"].append({
        "tab": 3, "label": "Reset All To Defaults", "x": 24, "y": sy + 56,
        "w": PANEL_W - 48, "h": 36,
        "action": (lambda st: reset_defaults(st)),
        "is_active": (lambda st: False),
    })

    state["system_info_y"] = sy + 110
    return state


def reset_defaults(state):
    for s in state["sliders"].values():
        s["value"] = s["default"]
    state["mode_idx"] = 0
    state["hand_filter_idx"] = 0
    state["face_filter_idx"] = 0
    state["keyboard_trigger"] = True


def render_panel(state):
    canvas = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    draw_tabs(canvas, state["active_tab"])

    for s in state["sliders"].values():
        if s["tab"] == state["active_tab"]:
            draw_slider(canvas, s)

    for b in state["buttons"]:
        if b["tab"] == state["active_tab"]:
            label = b["label"]
            if b.get("dynamic_label"):
                label = f"Keyboard Trigger: {'ON' if state['keyboard_trigger'] else 'OFF'}"
                b["label"] = label
            draw_button(canvas, b, b["is_active"](state))

    if state["active_tab"] == 3:
        lines = [
            f"Resolution : {state['resolution']}",
            f"FPS        : {state['fps']:.1f}",
            f"Brightness : {state['brightness']:.0f} / 255",
            f"Mode       : {['AUTO','DAY','NIGHT'][state['mode_idx']]}",
            f"Hand filter: {HAND_FILTERS[state['hand_filter_idx']]}",
            f"Face filter: {FACE_FILTERS[state['face_filter_idx']]}",
            "",
            "Keys: N=cycle mode  H=cycle hand filter",
            "      F=cycle face filter  K=toggle trigger  Esc=quit",
        ]
        yy = state["system_info_y"]
        for line in lines:
            cv2.putText(canvas, line, (24, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_DIM, 1, cv2.LINE_AA)
            yy += 22

    return canvas


def on_mouse(event, x, y, flags, state):
    tab_w = PANEL_W // len(TAB_NAMES)

    if event == cv2.EVENT_LBUTTONDOWN:
        if y < TAB_H:
            state["active_tab"] = min(x // tab_w, len(TAB_NAMES) - 1)
            return
        for s in state["sliders"].values():
            if s["tab"] == state["active_tab"] and slider_hit(s, x, y):
                state["dragging"] = s["key"]
                set_slider_from_x(s, x)
                return
        for b in state["buttons"]:
            if b["tab"] == state["active_tab"] and button_hit(b, x, y):
                b["action"](state)
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["dragging"] and (flags & cv2.EVENT_FLAG_LBUTTON):
            s = state["sliders"].get(state["dragging"])
            if s:
                set_slider_from_x(s, x)

    elif event == cv2.EVENT_LBUTTONUP:
        state["dragging"] = None


# ---------------------------------------------------------------------------
# Night vision pipeline
# ---------------------------------------------------------------------------

def get_brightness(frame):
    small = cv2.resize(frame, (80, 60))
    return float(np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))


def gamma_lut(gamma):
    inv_g = 1.0 / gamma
    return np.array([((i / 255.0) ** inv_g) * 255 for i in range(256)]).astype("uint8")


def sharpen_kernel(strength):
    return np.array([
        [0, -strength, 0],
        [-strength, 1 + 4 * strength, -strength],
        [0, -strength, 0],
    ])


def enhance_low_light(frame, gamma, clip, sharpen_strength, gain, denoise_k):
    if gain > 0:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=gain)

    if denoise_k > 1:
        frame = cv2.medianBlur(frame, denoise_k)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    result = cv2.LUT(enhanced, gamma_lut(gamma))

    if sharpen_strength > 0:
        result = cv2.filter2D(result, -1, sharpen_kernel(sharpen_strength))

    return result


# ---------------------------------------------------------------------------
# Finger counting
# ---------------------------------------------------------------------------

def count_fingers(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers_up = []

    if handedness_label == "Right":
        fingers_up.append(1 if lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x else 0)
    else:
        fingers_up.append(1 if lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x else 0)

    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers_up.append(1 if lm[tip].y < lm[pip].y else 0)

    return sum(fingers_up)


# ---------------------------------------------------------------------------
# Hand filters (10)
# ---------------------------------------------------------------------------

def lm_px(hand_landmarks, idx, w, h):
    lm = hand_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def draw_hand_dots(frame, hl):
    h, w, _ = frame.shape
    for lm in hl.landmark:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), DOT_RADIUS, WHEAT, -1, cv2.LINE_AA)


def draw_hand_skeleton(frame, hl):
    h, w, _ = frame.shape
    for a, b in mp_hands.HAND_CONNECTIONS:
        cv2.line(frame, lm_px(hl, a, w, h), lm_px(hl, b, w, h), WHEAT, 1, cv2.LINE_AA)
    draw_hand_dots(frame, hl)


def draw_hand_neon(frame, hl):
    h, w, _ = frame.shape
    glow = np.zeros_like(frame)
    for lm in hl.landmark:
        cv2.circle(glow, (int(lm.x * w), int(lm.y * h)), 8, WHEAT, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (15, 15), 0)
    frame[:] = cv2.add(frame, glow)
    draw_hand_dots(frame, hl)


def draw_hand_wireframe(frame, hl):
    h, w, _ = frame.shape
    for a, b in mp_hands.HAND_CONNECTIONS:
        cv2.line(frame, lm_px(hl, a, w, h), lm_px(hl, b, w, h), WHEAT, 1, cv2.LINE_AA)


def draw_hand_fire_trail(frame, hl, trail):
    h, w, _ = frame.shape
    tip = lm_px(hl, 8, w, h)
    trail.append(tip)
    n = len(trail)
    for i, (px, py) in enumerate(trail):
        alpha = (i + 1) / n
        color = (int(30 * alpha), int(120 * alpha), int(255 * alpha))
        radius = max(int(10 * alpha), 2)
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
    draw_hand_dots(frame, hl)


def draw_hand_rainbow(frame, hl):
    h, w, _ = frame.shape
    connections = list(mp_hands.HAND_CONNECTIONS)
    n = len(connections)
    for i, (a, b) in enumerate(connections):
        hue = int(180 * i / n)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.line(frame, lm_px(hl, a, w, h), lm_px(hl, b, w, h), tuple(int(c) for c in color), 2, cv2.LINE_AA)


def draw_hand_laser(frame, hl):
    h, w, _ = frame.shape
    x, y = lm_px(hl, 8, w, h)
    for r, alpha in [(18, 0.15), (12, 0.3), (6, 0.6), (3, 1.0)]:
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), r, WHEAT, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hand_minimal(frame, hl):
    h, w, _ = frame.shape
    for idx in [0, 4, 8, 12, 16, 20]:
        cv2.circle(frame, lm_px(hl, idx, w, h), 5, WHEAT, -1, cv2.LINE_AA)


def draw_hand_bbox(frame, hl):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hl.landmark]
    ys = [lm.y * h for lm in hl.landmark]
    x0, x1 = int(min(xs)) - 10, int(max(xs)) + 10
    y0, y1 = int(min(ys)) - 10, int(max(ys)) + 10
    cv2.rectangle(frame, (x0, y0), (x1, y1), WHEAT, 2, cv2.LINE_AA)
    draw_hand_dots(frame, hl)


def apply_hand_filter(frame, hl, idx, trail):
    if idx == 0:
        draw_hand_dots(frame, hl)
    elif idx == 1:
        draw_hand_skeleton(frame, hl)
    elif idx == 2:
        draw_hand_neon(frame, hl)
    elif idx == 3:
        draw_hand_wireframe(frame, hl)
    elif idx == 4:
        draw_hand_fire_trail(frame, hl, trail)
    elif idx == 5:
        draw_hand_rainbow(frame, hl)
    elif idx == 6:
        draw_hand_laser(frame, hl)
    elif idx == 7:
        draw_hand_minimal(frame, hl)
    elif idx == 8:
        draw_hand_bbox(frame, hl)
    # idx == 9 -> hidden, draw nothing


# ---------------------------------------------------------------------------
# Face filters (10)
# ---------------------------------------------------------------------------

def draw_face_dots(frame, fl):
    h, w, _ = frame.shape
    for lm in fl.landmark:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, WHEAT, -1, cv2.LINE_AA)


def draw_face_connections(frame, fl, connections, thickness=1):
    h, w, _ = frame.shape
    for a, b in connections:
        ax, ay = int(fl.landmark[a].x * w), int(fl.landmark[a].y * h)
        bx, by = int(fl.landmark[b].x * w), int(fl.landmark[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), WHEAT, thickness, cv2.LINE_AA)


def face_bbox(fl, w, h, pad=10):
    xs = [lm.x * w for lm in fl.landmark]
    ys = [lm.y * h for lm in fl.landmark]
    x0 = max(int(min(xs)) - pad, 0)
    x1 = min(int(max(xs)) + pad, w)
    y0 = max(int(min(ys)) - pad, 0)
    y1 = min(int(max(ys)) + pad, h)
    return x0, y0, x1, y1


def draw_face_glow(frame, fl):
    h, w, _ = frame.shape
    glow = np.zeros_like(frame)
    for lm in fl.landmark[::3]:
        cv2.circle(glow, (int(lm.x * w), int(lm.y * h)), 3, WHEAT, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (21, 21), 0)
    frame[:] = cv2.add(frame, glow)


def apply_privacy_blur(frame, fl):
    h, w, _ = frame.shape
    x0, y0, x1, y1 = face_bbox(fl, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (35, 35), 0)


def apply_pixelate(frame, fl):
    h, w, _ = frame.shape
    x0, y0, x1, y1 = face_bbox(fl, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def apply_edge_highlight(frame, fl):
    h, w, _ = frame.shape
    x0, y0, x1, y1 = face_bbox(fl, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    colored = np.zeros_like(roi)
    colored[edges > 0] = WHEAT
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.6, colored, 1.0, 0)


def apply_face_filter(frame, fl, idx):
    if idx == 0:
        draw_face_dots(frame, fl)
    elif idx == 1:
        draw_face_connections(frame, fl, mp_face_mesh.FACEMESH_TESSELATION, 1)
    elif idx == 2:
        draw_face_connections(frame, fl, mp_face_mesh.FACEMESH_CONTOURS, 1)
    elif idx == 3:
        draw_face_connections(frame, fl, mp_face_mesh.FACEMESH_FACE_OVAL, 2)
    elif idx == 4:
        draw_face_connections(frame, fl, mp_face_mesh.FACEMESH_LEFT_EYE, 1)
        draw_face_connections(frame, fl, mp_face_mesh.FACEMESH_RIGHT_EYE, 1)
    elif idx == 5:
        draw_face_glow(frame, fl)
    elif idx == 6:
        apply_privacy_blur(frame, fl)
    elif idx == 7:
        apply_pixelate(frame, fl)
    elif idx == 8:
        apply_edge_highlight(frame, fl)
    # idx == 9 -> hidden


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    state = build_state()
    state["resolution"] = f"{actual_w}x{actual_h}"

    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", PANEL_W, PANEL_H)
    cv2.setMouseCallback("Control Panel", on_mouse, state)

    last_trigger_time = {}
    prev_time = time.time()
    count_history = [deque(maxlen=SMOOTHING_WINDOW), deque(maxlen=SMOOTHING_WINDOW)]
    trail_buffers = [deque(maxlen=TRAIL_LENGTH), deque(maxlen=TRAIL_LENGTH)]
    frame_buffer = deque(maxlen=8)

    hands = None
    face_mesh = None
    current_det_conf = None

    def rebuild_models(det_conf):
        nonlocal hands, face_mesh
        if hands is not None:
            hands.close()
        if face_mesh is not None:
            face_mesh.close()
        hands = mp_hands.Hands(
            max_num_hands=2, model_complexity=1,
            min_detection_confidence=det_conf, min_tracking_confidence=det_conf,
        )
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=max(det_conf - 0.15, 0.3),
            min_tracking_confidence=max(det_conf - 0.15, 0.3),
        )

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)

            night = state["sliders"]
            gamma = night["gamma"]["value"]
            clip = night["clip"]["value"]
            sharpen_strength = night["sharpen"]["value"]
            gain = night["gain"]["value"]
            denoise_k = int(night["denoise"]["value"])
            denoise_k = denoise_k if denoise_k % 2 == 1 else denoise_k + 1
            temporal_frames = int(night["temporal"]["value"])
            det_conf = round(night["det_conf"]["value"], 2)
            cooldown = state["sliders"]["cooldown"]["value"]

            if current_det_conf != det_conf:
                rebuild_models(det_conf)
                current_det_conf = det_conf

            mode = state["mode_idx"]  # 0 AUTO 1 DAY 2 NIGHT
            brightness = get_brightness(frame)
            state["brightness"] = brightness

            if mode == 1:  # DAY
                frame_buffer.clear()
            else:
                apply_enhance = (mode == 2) or (mode == 0 and brightness < 100)
                if apply_enhance:
                    if temporal_frames > 1:
                        frame_buffer.append(frame.astype(np.float32))
                        stacked = np.mean(list(frame_buffer)[-temporal_frames:], axis=0)
                        frame = stacked.astype(np.uint8)
                    else:
                        frame_buffer.clear()
                    frame = enhance_low_light(frame, gamma, clip, sharpen_strength, gain, denoise_k)
                else:
                    frame_buffer.clear()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if face_results.multi_face_landmarks:
                for fl in face_results.multi_face_landmarks:
                    apply_face_filter(frame, fl, state["face_filter_idx"])

            active_counts = set()

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for idx, (hl, handedness) in enumerate(zip(
                    hand_results.multi_hand_landmarks, hand_results.multi_handedness
                )):
                    label = handedness.classification[0].label
                    slot = idx if idx < 2 else 1

                    apply_hand_filter(frame, hl, state["hand_filter_idx"], trail_buffers[slot])

                    raw_count = count_fingers(hl, label)
                    count_history[slot].append(raw_count)
                    stable_count = Counter(count_history[slot]).most_common(1)[0][0]

                    if 1 <= stable_count <= 5:
                        active_counts.add(stable_count)

                    h, w, _ = frame.shape
                    wx, wy = lm_px(hl, 0, w, h)
                    cv2.putText(frame, str(stable_count), (wx - 10, wy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHEAT, 2, cv2.LINE_AA)
            else:
                for hist in count_history:
                    hist.clear()
                for tr in trail_buffers:
                    tr.clear()

            now = time.time()
            if state["keyboard_trigger"]:
                for count in active_counts:
                    last = last_trigger_time.get(count, 0)
                    if now - last >= cooldown:
                        pyautogui.press(str(count))
                        last_trigger_time[count] = now

            fps = 1 / (now - prev_time) if now != prev_time else 0
            prev_time = now
            state["fps"] = fps

            mode_label = ["AUTO", "DAY", "NIGHT"][mode]
            cv2.putText(frame, f"FPS: {fps:.1f}  MODE: {mode_label}  BR: {brightness:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHEAT, 2, cv2.LINE_AA)

            cv2.imshow("Tracker", frame)
            cv2.imshow("Control Panel", render_panel(state))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('n'), ord('N')):
                state["mode_idx"] = (state["mode_idx"] + 1) % 3
            elif key in (ord('h'), ord('H')):
                state["hand_filter_idx"] = (state["hand_filter_idx"] + 1) % len(HAND_FILTERS)
            elif key in (ord('f'), ord('F')):
                state["face_filter_idx"] = (state["face_filter_idx"] + 1) % len(FACE_FILTERS)
            elif key in (ord('k'), ord('K')):
                state["keyboard_trigger"] = not state["keyboard_trigger"]

    finally:
        if hands is not None:
            hands.close()
        if face_mesh is not None:
            face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
