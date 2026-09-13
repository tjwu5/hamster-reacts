import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.detection.parser import parse_face, parse_hands
from src.calibration import FaceBaseline, record_calibration
from src.decide import ReactionDecider

# Setup MediaPipe models
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

baseline = FaceBaseline.load()
if not baseline:
    baseline = record_calibration(cap, face_detector, seconds=5)

print("Baseline initialized.")

decider = ReactionDecider()
    
print("\nRunning pipeline with debounced state machine. Press 'q' to quit, 'c' to recalibrate.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    face = parse_face(face_detector.detect(mp_image))
    hands = parse_hands(hand_detector.detect(mp_image))

    z_scores = baseline.compute_z_scores(face.blendshapes) if (face and baseline) else {}

    # State Machine update
    active_reaction, linger_frames = decider.update(face, hands, z_scores)

    # Render debounced state
    if active_reaction:
        status_text = f"REACTION: [{active_reaction.upper()}] (hold: {linger_frames})"
        box_color = (0, 255, 0)
    else:
        status_text = "REACTION: None"
        box_color = (150, 150, 150)

    cv2.putText(frame, status_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, box_color, 2)

    # Debug arming counters
    for idx, (pose_name, count) in enumerate(decider.arm_counts.items()):
        threshold = decider.POSE_CONFIG[pose_name][0]
        arm_text = f"{pose_name}: {count}/{threshold}"
        cv2.putText(frame, arm_text, (30, 90 + (idx * 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    cv2.imshow("Hamster Reacts - State Machine", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        baseline = record_calibration(cap, face_detector, seconds=5)

cap.release()
cv2.destroyAllWindows()
face_detector.close()
hand_detector.close()