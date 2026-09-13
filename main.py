import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize Landmarker options
base_face_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=base_face_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

base_hand_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=base_hand_options,
    num_hands=2
)

face_detector = vision.FaceLandmarker.create_from_options(face_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# 2. Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam live. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV loads BGR; MediaPipe requires RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect face & hands
    face_result = face_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    # Visual feedback: Overlay landmark counts and primary blendshapes
    hud_text = []
    if face_result.face_landmarks:
        hud_text.append(f"Face: 478 points detected")
        # Extract jawOpen blendshape as an example
        blendshapes = {b.category_name: b.score for b in face_result.face_blendshapes[0]}
        jaw_open = blendshapes.get("jawOpen", 0.0)
        hud_text.append(f"Jaw Open: {jaw_open:.2f}")
    else:
        hud_text.append("Face: None")

    if hand_result.hand_landmarks:
        hud_text.append(f"Hands: {len(hand_result.hand_landmarks)} detected")
    else:
        hud_text.append("Hands: 0")

    # Render HUD on frame
    for idx, text in enumerate(hud_text):
        cv2.putText(frame, text, (20, 40 + (idx * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hamster Reacts - Test Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()
hand_detector.close()