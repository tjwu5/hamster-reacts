import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.detection.parser import parse_face, parse_hands
from src.calibration import FaceBaseline, record_calibration

# 1. Setup MediaPipe models
base_face_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=base_face_options,
    output_face_blendshapes=True,
    num_faces=1
)
base_hand_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(base_options=base_hand_options, num_hands=2)

face_detector = vision.FaceLandmarker.create_from_options(face_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(1)

# 2. Load or prompt baseline
baseline = FaceBaseline.load()
if not baseline:
    baseline = record_calibration(cap, face_detector, seconds=5)

print("\nControls:")
print("  'c' -> Recalibrate neutral face")
print("  'q' -> Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    face = parse_face(face_detector.detect(mp_image))
    hands = parse_hands(hand_detector.detect(mp_image))

    state_label = "Neutral"
    stats_hud = []

    if face and baseline:
        # Convert raw blendshapes to personal Z-scores
        z_scores = baseline.compute_z_scores(face.blendshapes)

        jaw_z = z_scores.get("jawOpen", 0.0)
        smile_z = (z_scores.get("mouthSmileLeft", 0.0) + z_scores.get("mouthSmileRight", 0.0)) / 2.0
        brow_up_z = z_scores.get("browInnerUp", 0.0)

        stats_hud.append(f"jawOpen Z: {jaw_z:+.1f}s")
        stats_hud.append(f"smile Z:   {smile_z:+.1f}s")

        # Dynamic anomaly triggers (Sigma > 6.0 standard deviations)
        if jaw_z > 6.0:
            state_label = f"GASP / JAW DROP (+{jaw_z:.1f}s)"
        elif smile_z > 5.0:
            state_label = f"SMILING (+{smile_z:.1f}s)"

        # Spatial check: Thinking pose
        for hand in hands:
            if face.near(hand.palm_center, face.chin, 0.35):
                state_label = "Thinking Pose (Hand at chin)"
                break

    # HUD rendering
    color = (0, 255, 0) if "Neutral" not in state_label else (200, 200, 200)
    cv2.putText(frame, f"State: {state_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    for i, stat in enumerate(stats_hud):
        cv2.putText(frame, stat, (30, 90 + (i * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("Hamster Reacts - Day 2/3 Preview", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        baseline = record_calibration(cap, face_detector, seconds=5)

cap.release()
cv2.destroyAllWindows()
face_detector.close()
hand_detector.close()