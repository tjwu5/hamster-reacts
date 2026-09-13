import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.detection.parser import parse_face

CALIBRATION_FILE = Path("calibration.json")
EPSILON = 0.005

class FaceBaseline:
    def __init__(self, means: Dict[str, float], stdevs: Dict[str, float]):
        self.means = means
        self.stdevs = stdevs

    def compute_z_scores(self, active_blendshapes: Dict[str, float]) -> Dict[str, float]:
        """Calculates sigma above baseline for all channels."""
        z_scores = {}
        for shape_name, val in active_blendshapes.items():
            mean = self.means.get(shape_name, 0.0)
            std = self.stdevs.get(shape_name, 0.02)
            z_scores[shape_name] = (val - mean) / (std + EPSILON)
        return z_scores

    def save(self, filepath: Path = CALIBRATION_FILE):
        data = {"means": self.means, "stdevs": self.stdevs}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path = CALIBRATION_FILE) -> Optional["FaceBaseline"]:
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(means=data["means"], stdevs=data["stdevs"])


def record_calibration(cap, face_detector, seconds: int = 5) -> FaceBaseline:
    """Collects neutral face blendshapes for N seconds and calculates mean/std."""
    import cv2
    import mediapipe as mp

    print(f"\n--- CALIBRATING: Look at the camera with a neutral face for {seconds}s ---")
    samples = {}
    start_time = time.time()
    
    while time.time() - start_time < seconds:
        ret, frame = cap.read()
        if not ret:
            continue

        remaining = int(seconds - (time.time() - start_time)) + 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face = parse_face(face_detector.detect(mp_image))

        if face:
            for name, score in face.blendshapes.items():
                samples.setdefault(name, []).append(score)

        # Calibration countdown HUD
        cv2.putText(frame, f"CALIBRATING: Hold neutral face ({remaining}s)", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.imshow("Hamster Reacts - Day 2/3 Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate statistics
    means = {k: float(np.mean(v)) for k, v in samples.items() if len(v) > 0}
    stdevs = {k: float(np.std(v)) for k, v in samples.items() if len(v) > 0}

    baseline = FaceBaseline(means, stdevs)
    baseline.save()
    print("--- Calibration Complete & Saved to calibration.json ---\n")
    return baseline