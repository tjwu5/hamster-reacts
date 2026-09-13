import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.calibration import FaceBaseline
from src.decide import ReactionDecider
from src.detection.geometry import FaceData, HandData, Point2D
from src.detection.parser import parse_face, parse_hands
from src.renderer import OverlayRenderer

ROOT = Path(__file__).resolve().parent
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
MAX_CAMERA_INDEX = 4
PREVIEW_WINDOW = "Hamster Reacts - Preview"

# OpenCV on macOS indexes AVFoundation devices sorted by uniqueID.
# On this machine that order is typically Continuity, OBS, then FaceTime — not 0=built-in.
OBS_HINTS = ("obs virtual", "obs-camera", "obs camera extension", "obs-virtualcam", "virtualcam")
EXTERNAL_HINTS = (
    "dji", "osmo", "pocket", "uvc", "usb camera", "usb video",
    "capture", "cam link", "elgato", "logitech", "insta360",
)
BUILTIN_HINTS = ("facetime", "built-in", "isight")
CONTINUITY_HINTS = ("iphone", "ipad", "continuity", "desk view")

KIND_PRIORITY = {
    "external": 0,     # DJI Osmo / UVC
    "other": 1,        # unknown live camera with valid frames
    "builtin": 2,      # FaceTime HD
    "continuity": 3,   # iPhone Continuity Camera
    "obs": 4,          # OBS Virtual Camera loopback
    "dead": 5,
}


class LetterboxMeta:
    """Maps normalized landmarks from a square-padded image back to the original frame."""

    def __init__(self, orig_w: int, orig_h: int, size: int, x0: int, y0: int):
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.size = size
        self.x0 = x0
        self.y0 = y0

    def xy(self, x: float, y: float) -> Tuple[float, float]:
        px = x * self.size - self.x0
        py = y * self.size - self.y0
        return px / max(self.orig_w, 1), py / max(self.orig_h, 1)

    def point(self, pt: Point2D) -> Point2D:
        nx, ny = self.xy(pt.x, pt.y)
        return Point2D(nx, ny)


def letterbox_square(frame: np.ndarray) -> Tuple[np.ndarray, LetterboxMeta]:
    """Pad a frame to square so MediaPipe's NORM_RECT projector has a square ROI."""
    h, w = frame.shape[:2]
    if h == w:
        return frame, LetterboxMeta(w, h, w, 0, 0)

    size = max(h, w)
    square = np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
    x0 = (size - w) // 2
    y0 = (size - h) // 2
    square[y0:y0 + h, x0:x0 + w] = frame
    return square, LetterboxMeta(w, h, size, x0, y0)


def remap_face(face: FaceData, meta: LetterboxMeta) -> FaceData:
    x_scale = meta.size / max(meta.orig_w, 1)
    y_scale = meta.size / max(meta.orig_h, 1)
    return FaceData(
        center=meta.point(face.center),
        top=meta.point(face.top),
        chin=meta.point(face.chin),
        mouth=meta.point(face.mouth),
        nose=meta.point(face.nose),
        width=face.width * x_scale,
        height=face.height * y_scale,
        blendshapes=face.blendshapes,
    )


def remap_hand(hand: HandData, meta: LetterboxMeta) -> HandData:
    return HandData(
        wrist=meta.point(hand.wrist),
        index_tip=meta.point(hand.index_tip),
        thumb_tip=meta.point(hand.thumb_tip),
        palm_center=meta.point(hand.palm_center),
        is_open=hand.is_open,
    )


class LandmarkEngine:
    """
    Face + hand landmarkers in VIDEO mode.

    Tasks FaceLandmarker is required for the 52 blendshapes used by Z-score
    calibration. The graph's landmark_projection_calculator only supports
    non-square ROIs when IMAGE_DIMENSIONS is provided (Python Tasks never
    wires that port), so every frame is letterboxed to square first.
    """

    def __init__(self, face_detector, hand_detector):
        self.face_detector = face_detector
        self.hand_detector = hand_detector
        self._ts_ms = 0

    def _next_timestamp_ms(self) -> int:
        ts = int(time.time() * 1000)
        if ts <= self._ts_ms:
            ts = self._ts_ms + 1
        self._ts_ms = ts
        return ts

    def infer(self, frame: np.ndarray) -> Tuple[Optional[FaceData], List[HandData]]:
        square, meta = letterbox_square(frame)
        rgb = np.ascontiguousarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = self._next_timestamp_ms()
        try:
            face_result = self.face_detector.detect_for_video(mp_image, ts_ms)
            hand_result = self.hand_detector.detect_for_video(mp_image, ts_ms)
        except Exception as exc:
            print(f"[Warning] MediaPipe inference failed: {exc}")
            return None, []

        face = parse_face(face_result)
        hands = parse_hands(hand_result)
        if face is not None:
            face = remap_face(face, meta)
        return face, [remap_hand(hand, meta) for hand in hands]

    def close(self) -> None:
        self.face_detector.close()
        self.hand_detector.close()


def _haystack(name: str, model: str = "") -> str:
    return f"{name} {model}".lower()


def classify_camera_kind(name: str, model: str = "", frame: Optional[np.ndarray] = None) -> str:
    h = _haystack(name, model)
    if any(hint in h for hint in OBS_HINTS) or ("obs" in h and "camera" in h):
        return "obs"
    if any(hint in h for hint in EXTERNAL_HINTS):
        return "external"
    if any(hint in h for hint in BUILTIN_HINTS):
        return "builtin"
    if any(hint in h for hint in CONTINUITY_HINTS):
        return "continuity"
    if frame is not None and is_obs_idle_frame(frame):
        return "obs"
    if frame is not None and has_valid_video(frame):
        return "other"
    return "dead"


def is_obs_idle_frame(frame: np.ndarray) -> bool:
    """OBS Virtual Camera idles on a flat blue logo / camera-slash screen."""
    if frame is None or frame.size == 0:
        return False
    small = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
    b, g, r = small.mean(axis=(0, 1))
    std = float(small.std())
    blue_dominant = b > 70 and b > (g + 20) and b > (r + 20)
    return blue_dominant and std < 30


def has_valid_video(frame: Optional[np.ndarray]) -> bool:
    if frame is None or frame.size == 0 or frame.shape[0] < 16 or frame.shape[1] < 16:
        return False
    if is_obs_idle_frame(frame):
        return False
    return float(frame.mean()) > 8.0


def macos_camera_catalog() -> Dict[int, Dict[str, str]]:
    """
    Best-effort OpenCV index → name map.

    OpenCV's AVFoundation backend uses video+muxed devices sorted by uniqueID,
    which matches system_profiler's `spcamera_unique-id` for ordinary webcams.
    """
    catalog: Dict[int, Dict[str, str]] = {}
    try:
        proc = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        cameras = json.loads(proc.stdout or "{}").get("SPCameraDataType", [])
        cameras = sorted(cameras, key=lambda cam: cam.get("spcamera_unique-id", ""))
        for index, cam in enumerate(cameras):
            catalog[index] = {
                "name": cam.get("_name", f"Camera {index}"),
                "model": cam.get("spcamera_model-id", ""),
                "unique_id": cam.get("spcamera_unique-id", ""),
            }
    except Exception:
        pass
    return catalog


def open_capture(index: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def grab_warmup_frame(cap: cv2.VideoCapture, reads: int = 8) -> Optional[np.ndarray]:
    frame = None
    for _ in range(reads):
        ok, grabbed = cap.read()
        if ok and grabbed is not None and grabbed.size > 0:
            frame = grabbed
    return frame


def discover_camera(manual_index: Optional[int] = None) -> Tuple[cv2.VideoCapture, int, str]:
    catalog = macos_camera_catalog()
    if catalog:
        print("Available cameras (OpenCV / AVFoundation order):")
        for index, info in catalog.items():
            extra = f" [{info['model']}]" if info.get("model") else ""
            print(f"  [{index}] {info['name']}{extra}")

    if manual_index is not None:
        cap = open_capture(manual_index)
        if cap is None:
            sys.exit(f"Could not open camera index {manual_index}.")
        info = catalog.get(manual_index, {})
        name = info.get("name", f"Camera {manual_index}")
        frame = grab_warmup_frame(cap)
        if frame is None:
            cap.release()
            sys.exit(f"Camera index {manual_index} opened but produced no frames.")
        print(f"Using --cam {manual_index} ({name})")
        return cap, manual_index, name

    ranked: List[Tuple[int, int, int, str, str]] = []
    deferred: List[Tuple[int, str, str]] = []  # (index, name, kind) for OBS / Continuity

    for index in range(MAX_CAMERA_INDEX + 1):
        info = catalog.get(index, {})
        name = info.get("name", f"Camera {index}")
        model = info.get("model", "")
        kind_by_name = classify_camera_kind(name, model, frame=None)

        # Don't open OBS (blue idle loopback) or Continuity (slow iPhone handshake)
        # unless no physical camera is available.
        if kind_by_name in ("obs", "continuity"):
            print(f"  probed [{index}] {name}: deferred ({kind_by_name})")
            deferred.append((index, name, kind_by_name))
            ranked.append((1, KIND_PRIORITY[kind_by_name], index, name, kind_by_name))
            continue

        cap = open_capture(index)
        if cap is None:
            print(f"  probed [{index}] {name}: not available")
            continue
        frame = grab_warmup_frame(cap)
        cap.release()
        time.sleep(0.05)

        kind = classify_camera_kind(name, model, frame)
        invalid = 0 if has_valid_video(frame) else 1
        if invalid:
            kind = "dead" if kind != "obs" else kind
        h, w = (frame.shape[:2] if frame is not None else (0, 0))
        print(f"  probed [{index}] {name}: kind={kind} valid={not bool(invalid)} {w}x{h}")
        ranked.append((invalid, KIND_PRIORITY.get(kind, 5), index, name, kind))

    ranked.sort()
    choice = next((item for item in ranked if item[0] == 0), None)

    if choice is None:
        for index, name, kind in sorted(deferred, key=lambda item: KIND_PRIORITY[item[2]]):
            print(f"[Warning] No preferred camera found; trying {name} ({kind})...")
            cap = open_capture(index)
            if cap is None:
                continue
            frame = grab_warmup_frame(cap)
            cap.release()
            time.sleep(0.05)
            if has_valid_video(frame) or kind == "obs":
                choice = (0 if has_valid_video(frame) else 1, KIND_PRIORITY[kind], index, name, kind)
                if has_valid_video(frame):
                    break
        if choice is not None and choice[4] == "obs":
            print("[Warning] Falling back to OBS Virtual Camera.")

    if choice is None:
        sys.exit("No working camera found. Plug in a camera or pass --cam <index>.")

    _, _, index, name, kind = choice
    cap = open_capture(index)
    if cap is None:
        sys.exit(f"Selected camera [{index}] {name} could not be reopened.")
    print(f"Selected camera [{index}] {name} ({kind})")
    return cap, index, name


def run_video_calibration(cap, engine: LandmarkEngine, seconds: int = 5) -> FaceBaseline:
    print(f"\n--- CALIBRATING: Look at the camera with a neutral face for {seconds}s ---")
    samples: Dict[str, List[float]] = {}
    start_t = time.time()

    while time.time() - start_t < seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        remaining = int(seconds - (time.time() - start_t)) + 1
        face, _ = engine.infer(frame)
        if face:
            for name, score in face.blendshapes.items():
                samples.setdefault(name, []).append(score)

        cv2.putText(
            frame,
            f"CALIBRATING: Hold neutral face ({remaining}s)",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
        )
        cv2.imshow(PREVIEW_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    means = {k: float(np.mean(v)) for k, v in samples.items() if v}
    stdevs = {k: float(np.std(v)) for k, v in samples.items() if v}
    baseline = FaceBaseline(means, stdevs)
    baseline.save()
    print("--- Calibration Complete & Saved to calibration.json ---\n")
    return baseline


def process_pipeline(cap, engine: LandmarkEngine, baseline: FaceBaseline, vcam=None) -> FaceBaseline:
    decider = ReactionDecider()
    renderer = OverlayRenderer(assets_dir=str(ROOT / "assets"))
    fail_count = 0

    print("Pipeline running! Press 'q' in the preview window to stop, or 'c' to recalibrate.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            fail_count += 1
            if fail_count == 1 or fail_count % 30 == 0:
                print(f"[Warning] Camera frame read failed (x{fail_count}), retrying...")
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
            continue
        fail_count = 0

        face, hands = engine.infer(frame)
        z_scores = baseline.compute_z_scores(face.blendshapes) if (face and baseline) else {}
        active_reaction, _ = decider.update(face, hands, z_scores)

        if active_reaction:
            renderer.draw_reaction(frame, face, active_reaction)

        if vcam is not None:
            vcam.send(frame)
            vcam.sleep_until_next_frame()

        reaction_label = f"[{active_reaction.upper()}]" if active_reaction else "None"
        cv2.putText(
            frame,
            f"Reaction: {reaction_label}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0) if active_reaction else (200, 200, 200),
            2,
        )
        cv2.imshow(PREVIEW_WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            baseline = run_video_calibration(cap, engine, seconds=5)

    return baseline


def start_virtual_camera(width: int, height: int):
    import pyvirtualcam

    return pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=30,
        fmt=pyvirtualcam.PixelFormat.BGR,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamster Reacts live overlay")
    parser.add_argument("--no-vcam", action="store_true", help="Preview window only, skip virtual camera")
    parser.add_argument("--cam", type=int, default=None, metavar="INDEX", help="Force a camera index (skip auto-discovery)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not FACE_MODEL.exists() or not HAND_MODEL.exists():
        sys.exit(f"Missing MediaPipe models. Expected:\n  {FACE_MODEL}\n  {HAND_MODEL}")

    # CPU delegate avoids Metal/GPU graph aborts on macOS while keeping blendshapes.
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(FACE_MODEL),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    engine = LandmarkEngine(
        vision.FaceLandmarker.create_from_options(face_options),
        vision.HandLandmarker.create_from_options(hand_options),
    )

    cap = None
    try:
        cap, cam_index, cam_name = discover_camera(manual_index=args.cam)
        ok, sample_frame = cap.read()
        if not ok or sample_frame is None:
            # Brief drop on first read is common with UVC; retry a few times.
            sample_frame = grab_warmup_frame(cap)
        if sample_frame is None:
            sys.exit("Failed to access camera.")

        height, width = sample_frame.shape[:2]
        print(f"Camera [{cam_index}] {cam_name} resolution: {width}x{height}")

        baseline = FaceBaseline.load()
        if not baseline:
            baseline = run_video_calibration(cap, engine, seconds=5)

        if args.no_vcam:
            print("Running in preview-only mode (--no-vcam)...")
            process_pipeline(cap, engine, baseline, vcam=None)
        else:
            try:
                with start_virtual_camera(width, height) as vcam:
                    print(f"Virtual camera live: {vcam.device}")
                    process_pipeline(cap, engine, baseline, vcam=vcam)
            except Exception as exc:
                print(f"\n[Warning] Could not start virtual camera: {exc}")
                print("Falling back to preview-only mode. Pass --no-vcam to skip this attempt.\n")
                process_pipeline(cap, engine, baseline, vcam=None)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        engine.close()


if __name__ == "__main__":
    main()
