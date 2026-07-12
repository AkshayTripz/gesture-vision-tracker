# Gesture Vision Tracker

Real-time hand and face tracking built on MediaPipe and OpenCV, with:

- Per-hand finger counting (1–5), tracked independently for left and right hands
- Keyboard number-key triggers when a finger count is held up (with debounce/cooldown)
- Adaptive low-light enhancement pipeline (AUTO / DAY / NIGHT modes) with manual fader control over every stage
- 10 selectable hand-tracking visual filters
- 10 selectable face-tracking visual filters (including privacy blur and pixelate)
- A custom dark-themed control panel window — no native OS dialogs

![mode](https://img.shields.io/badge/python-3.10-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## Requirements

- **Python 3.10 or 3.11** (MediaPipe does not yet support 3.13/3.14 reliably — this project pins `mediapipe==0.10.9`, which is the last version with a fully stable `mp.solutions` API)
- A webcam
- Windows, macOS, or Linux (tested primarily on Windows)
## Setup

### Windows (automated)

Double-click `run.bat`, or run it from a terminal:

```bat
run.bat
```

This will:
1. Verify Python 3.10 is available via the `py` launcher
2. Create a virtual environment (`venv/`) if one doesn't exist
3. Install all dependencies from `requirements.txt`
4. Verify `mediapipe.solutions` imports correctly before running anything
5. Launch `main.py`

### Manual setup (any OS)

```bash
python3.10 -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

## Usage

Running `main.py` opens two windows:

- **Tracker** — the live camera feed with hand/face overlays
- **Control Panel** — a dark-themed tabbed settings window

### Control Panel tabs

| Tab | Contents |
|---|---|
| **NIGHT VISION** | Gamma, CLAHE clip limit, sharpen strength, brightness gain, denoise kernel size, temporal frame averaging, detection confidence — plus AUTO/DAY/NIGHT mode buttons |
| **HAND FILTERS** | 10 selectable visualization styles for hand landmarks |
| **FACE FILTERS** | 10 selectable visualization styles for face mesh landmarks |
| **SYSTEM** | Live resolution/FPS/brightness readout, keyboard trigger on/off toggle, trigger cooldown slider, reset-to-defaults button |

Click a tab to switch. Click and drag sliders. Click filter buttons to select.

### Keyboard shortcuts (work while either window has focus)

| Key | Action |
|---|---|
| `N` | Cycle night-vision mode: AUTO → DAY → NIGHT |
| `H` | Cycle to the next hand filter |
| `F` | Cycle to the next face filter |
| `K` | Toggle keyboard number-key triggering on/off |
| `Esc` | Quit |

### Hand filters

1. Dots Only
2. Skeleton Lines
3. Neon Glow
4. Wireframe Mesh
5. Fire Trail (fingertip motion trail)
6. Rainbow Gradient
7. Laser Fingertip
8. Minimal Joints
9. Bounding Box
10. Hidden (tracking still runs, nothing drawn)

### Face filters

1. Dots Mesh
2. Wireframe Tesselation
3. Contour Outline
4. Face Oval Only
5. Eye Highlight
6. Glow Mask
7. Privacy Blur
8. Pixelate
9. Edge Highlight
10. Hidden (tracking still runs, nothing drawn)

## How the night-vision pipeline works

Order matters — each stage depends on the previous one being roughly correct:

1. **Brightness Gain** — linear additive lift, recovers shadow detail that gamma curves alone compress unevenly
2. **Denoise** — median blur, applied before contrast enhancement so noise isn't amplified along with signal
3. **CLAHE** (adaptive histogram equalization on the L channel in LAB color space) — boosts local contrast without blowing out already-lit areas
4. **Gamma correction** — via precomputed lookup table, tunes midtone brightness
5. **Sharpen** — an unsharp-style kernel that restores edge definition lost to noise and blurring; this is what actually lets MediaPipe's landmark model lock onto joints and facial features in low light, since it keys off edge gradients rather than raw brightness
6. **Temporal averaging** (optional) — stacks 2–8 recent frames to reduce random sensor noise beyond what a single-frame filter can achieve, at the cost of motion smear on moving subjects

**Hardware ceiling note:** software enhancement can't recover detail a camera sensor never captured. If your hand looks like a featureless gray blob to your own eyes at maximum gain/gamma, that's the sensor's noise floor — the fix at that point is better lighting or an IR illuminator, not further filter tuning.

## Known limitation

Keyboard triggering (`pyautogui.press`) sends real keystrokes to whichever window currently has OS focus — it is **not** scoped to any particular target application. If the Tracker window isn't focused, number keys will type into whatever is (a text editor, browser, terminal, etc). Scoping triggers to a specific target window requires OS-level window-focus detection and is a natural next step, not something OpenCV/MediaPipe provide.

## Project structure

```
gesture-vision-tracker/
├── main.py            # Application entry point
├── requirements.txt   # Pinned dependencies
├── run.bat            # Windows one-click setup + run
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
