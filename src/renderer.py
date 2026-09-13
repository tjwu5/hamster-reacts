import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from src.detection.geometry import FaceData

class OverlayRenderer:
    def __init__(self, assets_dir: str = "assets"):
        self.assets_dir = Path(assets_dir)
        self.cache: Dict[str, Optional[np.ndarray]] = {}

    def _load_asset(self, pose_name: str) -> Optional[np.ndarray]:
        if pose_name in self.cache:
            return self.cache[pose_name]

        # Look for .png, .jpg, or .jpeg
        for ext in [".png", ".jpg", ".jpeg"]:
            asset_path = self.assets_dir / f"{pose_name}{ext}"
            if asset_path.exists():
                # Read with alpha channel (UNCHANGED)
                img = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
                self.cache[pose_name] = img
                return img

        self.cache[pose_name] = None
        return None

    def draw_reaction(self, frame: np.ndarray, face: Optional[FaceData], pose_name: str):
        if not face:
            return

        h, w, _ = frame.shape
        img = self._load_asset(pose_name)

        # Scale overlay dynamically based on face width
        # Target overlay width = 1.2x the face width
        target_w = int(face.width * w * 1.2)
        if target_w <= 10:
            return

        # Position: Center horizontally over forehead, float slightly above
        center_x = int(face.top.x * w)
        # 0.4 face-heights above the forehead
        top_y = int((face.top.y - (face.height * 0.7)) * h)

        if img is None:
            # Fallback placeholder if asset file is missing
            cv2.putText(frame, f"[{pose_name.upper()}]", (center_x - 60, max(30, top_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return

        # Preserve aspect ratio
        aspect = img.shape[0] / img.shape[1]
        target_h = int(target_w * aspect)

        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Compute bounding box
        x1 = center_x - (target_w // 2)
        y1 = top_y
        x2 = x1 + target_w
        y2 = y1 + target_h

        # Screen boundary clamping
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = target_w - max(0, x2 - w)
        src_y2 = target_h - max(0, y2 - h)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(w, x2)
        dst_y2 = min(h, y2)

        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            return

        cropped_overlay = resized[src_y1:src_y2, src_x1:src_x2]

        # Alpha composite if PNG has 4 channels
        if cropped_overlay.shape[2] == 4:
            alpha = cropped_overlay[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            overlay_bgr = cropped_overlay[:, :, :3]
            frame_roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]

            frame[dst_y1:dst_y2, dst_x1:dst_x2] = (
                alpha * overlay_bgr + (1.0 - alpha) * frame_roi
            ).astype(np.uint8)
        else:
            # Fallback for 3-channel images (no transparency)
            frame[dst_y1:dst_y2, dst_x1:dst_x2] = cropped_overlay[:, :, :3]