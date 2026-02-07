"""
Zoom Mood — Real-time facial emotion classifier for Zoom meetings.

Captures your Zoom window, detects faces via MediaPipe, classifies
emotions with HSEmotion (ONNX), and displays a live dashboard in
your browser.

Usage:
    python zoom_mood.py                  # auto-detect Zoom window
    python zoom_mood.py --region         # manually select screen region
    python zoom_mood.py --monitor 1      # capture a specific monitor
"""

import argparse
import base64
import io
import json
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from PIL import Image  # noqa: F401 — used by hsemotion internally

# ---------------------------------------------------------------------------
# Emotion Engine — MediaPipe face detection + HSEmotion classification
# ---------------------------------------------------------------------------

EMOTIONS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]

# Map emotions to simpler "mood" buckets for the dashboard
MOOD_MAP = {
    "Happiness": ("Happy", "#4ade80"),
    "Surprise":  ("Surprised", "#facc15"),
    "Neutral":   ("Neutral", "#94a3b8"),
    "Sadness":   ("Sad", "#60a5fa"),
    "Anger":     ("Frustrated", "#f87171"),
    "Contempt":  ("Skeptical", "#fb923c"),
    "Disgust":   ("Displeased", "#c084fc"),
    "Fear":      ("Anxious", "#f472b6"),
}


@dataclass
class FaceResult:
    """Result for a single detected face."""
    face_id: int
    thumbnail_b64: str          # JPEG thumbnail as base64
    dominant_emotion: str
    mood_label: str
    mood_color: str
    confidence: float
    scores: dict                # all 8 emotion scores
    x: int = 0
    y: int = 0


@dataclass
class FrameResult:
    """Result for a full frame analysis."""
    timestamp: float
    faces: list = field(default_factory=list)


class EmotionEngine:
    """Detects faces and classifies emotions locally."""

    def __init__(self):
        self._init_face_detector()
        self._init_emotion_model()

    # -- Face detection (MediaPipe) ----------------------------------------

    def _init_face_detector(self):
        import mediapipe as mp
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,      # full-range model (better for screen caps)
            min_detection_confidence=0.5,
        )

    # -- Emotion classification (HSEmotion ONNX) ---------------------------

    def _init_emotion_model(self):
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.recognizer = HSEmotionRecognizer(
            model_name="enet_b0_8_best_vgaf",
            device="cpu",
        )

    # -- Pipeline ----------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """Run full pipeline on a BGR frame. Returns FrameResult."""
        result = FrameResult(timestamp=time.time())
        h, w, _ = frame_bgr.shape

        # MediaPipe wants RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.detector.process(frame_rgb)

        if not detections.detections:
            return result

        for i, det in enumerate(detections.detections):
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)

            if bw < 30 or bh < 30:
                continue  # skip tiny faces

            # Add margin around the face for better emotion classification
            margin_x = int(bw * 0.15)
            margin_y = int(bh * 0.15)
            cx1 = max(0, x1 - margin_x)
            cy1 = max(0, y1 - margin_y)
            cx2 = min(w, x2 + margin_x)
            cy2 = min(h, y2 + margin_y)

            face_crop = frame_bgr[cy1:cy2, cx1:cx2]
            if face_crop.size == 0:
                continue

            # Classify emotion
            try:
                emotion, scores = self.recognizer.predict_emotions(
                    face_crop, logits=False
                )
                score_dict = {
                    EMOTIONS[j]: float(scores[j]) for j in range(len(EMOTIONS))
                }
                dominant = max(score_dict, key=score_dict.get)
                confidence = score_dict[dominant]
            except Exception as e:
                print(f"⚠️  Emotion classification failed: {e}")
                dominant = "Neutral"
                confidence = 0.0
                score_dict = {e: 0.0 for e in EMOTIONS}

            mood_label, mood_color = MOOD_MAP.get(
                dominant, ("Unknown", "#94a3b8")
            )

            # Create thumbnail
            thumb = cv2.resize(face_crop, (80, 80))
            _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            thumb_b64 = base64.b64encode(buf).decode("ascii")

            result.faces.append(FaceResult(
                face_id=i,
                thumbnail_b64=thumb_b64,
                dominant_emotion=dominant,
                mood_label=mood_label,
                mood_color=mood_color,
                confidence=confidence,
                scores=score_dict,
                x=x1,
                y=y1,
            ))

        return result


# ---------------------------------------------------------------------------
# Screen Capture
# ---------------------------------------------------------------------------

class ScreenCapture:
    """Captures a region of the screen using mss."""

    def __init__(self, region: Optional[dict] = None, monitor: Optional[int] = None):
        self.sct = mss.mss()
        if region:
            self.region = region
        elif monitor is not None:
            self.region = self.sct.monitors[monitor]
        else:
            # Default: primary monitor
            self.region = self.sct.monitors[1]

    def grab(self) -> np.ndarray:
        """Grab a frame as a BGR numpy array."""
        shot = self.sct.grab(self.region)
        frame = np.array(shot)
        # mss returns BGRA; convert to BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def select_region_interactive() -> dict:
    """Let the user draw a rectangle on screen to select the Zoom window."""
    print("\n📐  Region selection mode")
    print("   A fullscreen screenshot will appear.")
    print("   Click and drag to select the Zoom window area, then press ENTER.\n")

    sct = mss.mss()
    monitor = sct.monitors[1]
    shot = sct.grab(monitor)
    img = np.array(shot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Scale down for display if very large
    display_h, display_w = img.shape[:2]
    scale = 1.0
    max_dim = 1400
    if display_w > max_dim or display_h > max_dim:
        scale = max_dim / max(display_w, display_h)
        img_display = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_display = img.copy()

    roi = cv2.selectROI("Select Zoom Window — press ENTER when done", img_display, False, False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("No region selected, using full primary monitor.")
        return monitor

    x, y, w, h = roi
    return {
        "left":   int(x / scale) + monitor["left"],
        "top":    int(y / scale) + monitor["top"],
        "width":  int(w / scale),
        "height": int(h / scale),
    }


# ---------------------------------------------------------------------------
# Web Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Zoom Mood</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0;
    min-height: 100vh;
  }
  header {
    padding: 20px 32px;
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid #1e293b;
  }
  header h1 { font-size: 22px; font-weight: 600; }
  header .status {
    font-size: 13px; color: #94a3b8;
    margin-left: auto;
  }
  header .status .dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #4ade80;
    margin-right: 4px; vertical-align: middle;
  }
  .no-faces {
    text-align: center; padding: 80px 20px;
    color: #64748b; font-size: 16px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px; padding: 24px 32px;
  }
  .card {
    background: #1e293b; border-radius: 12px;
    padding: 20px; display: flex; gap: 16px;
    align-items: center; transition: transform 0.15s;
  }
  .card:hover { transform: translateY(-2px); }
  .card img {
    width: 64px; height: 64px; border-radius: 50%;
    object-fit: cover; border: 3px solid var(--mood-color);
  }
  .card .info { flex: 1; }
  .card .mood {
    font-size: 20px; font-weight: 700;
    color: var(--mood-color);
  }
  .card .emotion-detail {
    font-size: 12px; color: #94a3b8; margin-top: 2px;
  }
  .card .confidence-bar {
    margin-top: 8px; height: 4px; border-radius: 2px;
    background: #334155; overflow: hidden;
  }
  .card .confidence-bar .fill {
    height: 100%; border-radius: 2px;
    background: var(--mood-color);
    transition: width 0.4s ease;
  }
  .summary-bar {
    padding: 16px 32px;
    display: flex; gap: 8px; flex-wrap: wrap;
    border-bottom: 1px solid #1e293b;
  }
  .summary-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 500;
    background: #1e293b;
  }
  .summary-chip .dot {
    width: 10px; height: 10px; border-radius: 50%;
  }
</style>
</head>
<body>
  <header>
    <h1>Zoom Mood</h1>
    <div class="status"><span class="dot"></span><span id="fps">starting...</span></div>
  </header>
  <div class="summary-bar" id="summary"></div>
  <div id="content">
    <div class="no-faces">Waiting for faces...</div>
  </div>

<script>
const socket = io();
let lastUpdate = Date.now();

socket.on('update', data => {
  const now = Date.now();
  const dt = (now - lastUpdate) / 1000;
  lastUpdate = now;
  document.getElementById('fps').textContent =
    `${data.faces.length} face${data.faces.length !== 1 ? 's' : ''} · ${dt.toFixed(1)}s ago`;

  const content = document.getElementById('content');
  const summary = document.getElementById('summary');

  if (data.faces.length === 0) {
    content.innerHTML = '<div class="no-faces">No faces detected — is your Zoom window visible?</div>';
    summary.innerHTML = '';
    return;
  }

  // Summary chips
  const counts = {};
  data.faces.forEach(f => {
    const k = f.mood_label;
    if (!counts[k]) counts[k] = { count: 0, color: f.mood_color };
    counts[k].count++;
  });
  summary.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1].count - a[1].count)
    .map(([label, {count, color}]) =>
      `<div class="summary-chip">
        <span class="dot" style="background:${color}"></span>
        ${label}: ${count}
      </div>`
    ).join('');

  // Face cards
  content.innerHTML = '<div class="grid">' + data.faces.map((f, i) =>
    `<div class="card" style="--mood-color:${f.mood_color}">
      <img src="data:image/jpeg;base64,${f.thumbnail_b64}" alt="face ${i+1}">
      <div class="info">
        <div class="mood">${f.mood_label}</div>
        <div class="emotion-detail">${f.dominant_emotion} · ${(f.confidence * 100).toFixed(0)}%</div>
        <div class="confidence-bar"><div class="fill" style="width:${(f.confidence * 100).toFixed(0)}%"></div></div>
      </div>
    </div>`
  ).join('') + '</div>';
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class ZoomMoodApp:
    """Ties everything together: capture → detect → classify → dashboard."""

    def __init__(self, capture: ScreenCapture, interval: float = 0.8):
        self.capture = capture
        self.interval = interval
        self.engine = EmotionEngine()
        self.running = False

        # Flask + SocketIO
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "zoom-mood-local"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

    def _capture_loop(self):
        """Background loop: capture → process → emit."""
        print("🎬  Capture loop started")
        while self.running:
            t0 = time.time()
            try:
                frame = self.capture.grab()
                result = self.engine.process_frame(frame)
                payload = {
                    "timestamp": result.timestamp,
                    "faces": [
                        {
                            "face_id": f.face_id,
                            "thumbnail_b64": f.thumbnail_b64,
                            "dominant_emotion": f.dominant_emotion,
                            "mood_label": f.mood_label,
                            "mood_color": f.mood_color,
                            "confidence": round(f.confidence, 3),
                            "scores": {
                                k: round(v, 3) for k, v in f.scores.items()
                            },
                        }
                        for f in result.faces
                    ],
                }
                self.socketio.emit("update", payload)
            except Exception as e:
                print(f"⚠️  Frame error: {e}")

            elapsed = time.time() - t0
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)

    def run(self, host: str = "127.0.0.1", port: int = 5051):
        """Start the dashboard server and capture loop."""
        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

        print(f"\n🌐  Dashboard → http://{host}:{port}")
        print("   Open this URL in your browser to see live moods.")
        print("   Press Ctrl+C to stop.\n")

        try:
            self.socketio.run(self.app, host=host, port=port, allow_unsafe_werkzeug=True)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("\n👋  Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zoom Mood — live meeting mood dashboard")
    parser.add_argument(
        "--region", action="store_true",
        help="Interactively select the Zoom window region on screen",
    )
    parser.add_argument(
        "--monitor", type=int, default=None,
        help="Capture a specific monitor (1 = primary, 2 = secondary, etc.).",
    )
    parser.add_argument(
        "--interval", type=float, default=0.8,
        help="Seconds between captures (default: 0.8)",
    )
    parser.add_argument(
        "--port", type=int, default=5051,
        help="Dashboard port (default: 5051)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Zoom Mood — Real-time Meeting Mood Tracker")
    print("=" * 50)

    # Screen region
    if args.region:
        region = select_region_interactive()
        capture = ScreenCapture(region=region)
    elif args.monitor is not None:
        capture = ScreenCapture(monitor=args.monitor)
    else:
        capture = ScreenCapture()

    app = ZoomMoodApp(capture, interval=args.interval)
    app.run(port=args.port)


if __name__ == "__main__":
    main()
