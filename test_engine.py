"""Smoke test — verifies EmotionEngine initializes and can process a frame."""

import numpy as np

print("1/3  Importing EmotionEngine...")
from zoom_mood import EmotionEngine

print("2/3  Initializing (downloads model on first run)...")
engine = EmotionEngine()

print("3/3  Processing a dummy 640x480 frame...")
dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = engine.process_frame(dummy_frame)
print(f"     Detected {len(result.faces)} face(s) in random noise (expected: 0)")

print()
print("✅  Engine works! You're good to run: python zoom_mood.py --region")
