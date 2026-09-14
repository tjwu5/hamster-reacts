from typing import Optional, List, Dict, Tuple
from src.detection.geometry import FaceData, HandData

class ReactionDecider:
    def __init__(self):
        # Frame debouncing buffers
        self.arm_counts: Dict[str, int] = {}
        self.active_pose: Optional[str] = None
        self.hold_timer: int = 0

        # Tuning configuration: (min_consecutive_frames_to_fire, frames_to_linger)
        self.POSE_CONFIG = {
            # Multi-modal / Hand + Face combinations
            "shocked":    (4, 16),
            "hearts":     (4, 16),
            "shh":        (3, 14),
            "thumbsup":   (4, 14),

            # Facial expressions (compound & single)
            "angry":      (4, 14),
            "crying":     (4, 15),
            "dirpy":      (3, 12),
            "queen":      (4, 14),
            "smirk":      (4, 14),
            "sad":        (4, 14),
            "bigmouth":   (4, 15),
            "happy":      (4, 14),
        }

    def _evaluate_raw_poses(
        self, face: Optional[FaceData], hands: List[HandData], z_scores: Dict[str, float]
    ) -> Optional[str]:
        """
        Ordered priority check. First matching candidate wins.
        High-priority / compound gestures are placed above generic expressions.
        """
        if not face:
            return None

        # Pre-compute common facial metric aggregates
        jaw_z = z_scores.get("jawOpen", 0.0)
        smile_l = z_scores.get("mouthSmileLeft", 0.0)
        smile_r = z_scores.get("mouthSmileRight", 0.0)
        avg_smile_z = (smile_l + smile_r) / 2.0
        brow_down_z = (z_scores.get("browDownLeft", 0.0) + z_scores.get("browDownRight", 0.0)) / 2.0
        frown_z = (z_scores.get("mouthFrownLeft", 0.0) + z_scores.get("mouthFrownRight", 0.0)) / 2.0
        blink_z = (z_scores.get("eyeBlinkLeft", 0.0) + z_scores.get("eyeBlinkRight", 0.0)) / 2.0
        pucker_z = z_scores.get("mouthPucker", 0.0)

        # -------------------------------------------------------------
        # 1. SHOCKED (2 hands near top/sides of head + open mouth)
        # -------------------------------------------------------------
        if len(hands) >= 2 and jaw_z > 4.5:
            h1_near_head = face.near(hands[0].wrist, face.top, 0.85) or (hands[0].wrist.y < face.center.y)
            h2_near_head = face.near(hands[1].wrist, face.top, 0.85) or (hands[1].wrist.y < face.center.y)
            if h1_near_head and h2_near_head:
                return "shocked"

        # -------------------------------------------------------------
        # 2. HEARTS (2 hands up, index tips & thumb tips close together)
        # -------------------------------------------------------------
        if len(hands) >= 2:
            h1, h2 = hands[0], hands[1]
            idx_close = face.near(h1.index_tip, h2.index_tip, 0.35)
            thumb_close = face.near(h1.thumb_tip, h2.thumb_tip, 0.35)
            if idx_close and thumb_close:
                return "hearts"

        # -------------------------------------------------------------
        # 3. SHH (Index finger covering or right in front of mouth)
        # -------------------------------------------------------------
        for h in hands:
            if face.near(h.index_tip, face.mouth, 0.28):
                return "shh"

        # -------------------------------------------------------------
        # 4. THUMBS UP (Thumb tip significantly higher than wrist & index)
        # -------------------------------------------------------------
        for h in hands:
            # Thumb tip must be elevated above wrist and index MCP by at least 0.2 face heights
            thumb_upward = (h.wrist.y - h.thumb_tip.y) > (face.height * 0.25)
            thumb_above_index = (h.index_tip.y - h.thumb_tip.y) > (face.height * 0.15)
            if thumb_upward and thumb_above_index:
                return "thumbsup"

        # -------------------------------------------------------------
        # 5. DIRPY (Tongue out)
        # MediaPipe provides jawOpen + tongueOut or mouthFunnel channel
        # -------------------------------------------------------------
        tongue_z = z_scores.get("tongueOut", 0.0)
        if tongue_z > 4.0 or (jaw_z > 4.0 and z_scores.get("mouthFunnel", 0.0) > 5.0):
            return "dirpy"

        # -------------------------------------------------------------
        # 6. ANGRY (Eyebrows furrowed down + mouth slightly/wide open)
        # -------------------------------------------------------------
        if brow_down_z > 4.5 and jaw_z > 3.5:
            return "angry"

        # -------------------------------------------------------------
        # 7. CRYING (Eyes squinched/closed + sad frown / inner brows up)
        # -------------------------------------------------------------
        inner_brow_up = z_scores.get("browInnerUp", 0.0)
        if blink_z > 5.0 and (frown_z > 5.5 or inner_brow_up > 4.0):
            return "crying"

        # -------------------------------------------------------------
        # 8. QUEEN (Kissy face / mouth pucker)
        # -------------------------------------------------------------
        if pucker_z > 5.5:
            return "queen"

        # -------------------------------------------------------------
        # 9. SMIRK (Look to the side / head yaw offset + asymmetric or subtle smile)
        # Head yaw can be measured by nose offset relative to face center X
        # -------------------------------------------------------------
        head_yaw_offset = abs(face.nose.x - face.center.x) / (face.width + 1e-5)
        is_looking_sideways = head_yaw_offset > 0.15
        asymmetric_smile = abs(smile_l - smile_r) > 3.0
        if is_looking_sideways and (asymmetric_smile or avg_smile_z > 3.5):
            return "smirk"

        # -------------------------------------------------------------
        # 10. SAD (Eyes open, sad frown / inner brow up)
        # -------------------------------------------------------------
        if blink_z < 2.5 and (frown_z > 4.5 or inner_brow_up > 5.0):
            return "sad"

        # -------------------------------------------------------------
        # 11. BIGMOUTH (Jaw dropped very wide)
        # -------------------------------------------------------------
        if jaw_z > 6.5:
            return "bigmouth"

        # -------------------------------------------------------------
        # 12. HAPPY (Broad smile with mouth mostly closed or slightly parted)
        # -------------------------------------------------------------
        if avg_smile_z > 5.0:
            return "happy"

        return None

    def update(
        self, face: Optional[FaceData], hands: List[HandData], z_scores: Dict[str, float]
    ) -> Tuple[Optional[str], float]:
        """
        Processes current frame candidates against arming counters and hold durations.
        Returns (active_pose_name_or_None, confidence_or_linger_ratio)
        """
        raw_candidate = self._evaluate_raw_poses(face, hands, z_scores)

        # Update arming counters for all registered poses
        for pose in self.POSE_CONFIG.keys():
            if pose == raw_candidate:
                self.arm_counts[pose] = self.arm_counts.get(pose, 0) + 1
            else:
                self.arm_counts[pose] = max(0, self.arm_counts.get(pose, 0) - 1)

        # Check if raw candidate has satisfied minimum arming frames
        if raw_candidate:
            arm_threshold, hold_duration = self.POSE_CONFIG[raw_candidate]
            if self.arm_counts[raw_candidate] >= arm_threshold:
                self.active_pose = raw_candidate
                self.hold_timer = hold_duration

        # Decrement active hold timer
        if self.hold_timer > 0:
            self.hold_timer -= 1
        else:
            self.active_pose = None

        return self.active_pose, self.hold_timer