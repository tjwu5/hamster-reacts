import math
from typing import Optional, List, Tuple
from .geometry import Point2D, FaceData, HandData

# MediaPipe canonical face landmark indices
FACE_TEMPLE_LEFT = 454
FACE_TEMPLE_RIGHT = 234
FACE_CHIN = 152
FACE_FOREHEAD = 10
FACE_NOSE = 1
FACE_MOUTH_CENTER = 13

# Hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

def parse_face(face_result) -> Optional[FaceData]:
    if not face_result.face_landmarks or len(face_result.face_landmarks) == 0:
        return None

    landmarks = face_result.face_landmarks[0]
    
    # Calculate scale-invariant face width
    left_t = Point2D(landmarks[FACE_TEMPLE_LEFT].x, landmarks[FACE_TEMPLE_LEFT].y)
    right_t = Point2D(landmarks[FACE_TEMPLE_RIGHT].x, landmarks[FACE_TEMPLE_RIGHT].y)
    face_w = left_t.distance_to(right_t)

    # Face height and key landmarks
    top = Point2D(landmarks[FACE_FOREHEAD].x, landmarks[FACE_FOREHEAD].y)
    chin = Point2D(landmarks[FACE_CHIN].x, landmarks[FACE_CHIN].y)
    face_h = top.distance_to(chin)

    # Parse all 52 blendshapes into a dict
    shapes = {}
    if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
        for b in face_result.face_blendshapes[0]:
            shapes[b.category_name] = b.score

    return FaceData(
        center=Point2D((left_t.x + right_t.x) / 2.0, (top.y + chin.y) / 2.0),
        top=top,
        chin=chin,
        mouth=Point2D(landmarks[FACE_MOUTH_CENTER].x, landmarks[FACE_MOUTH_CENTER].y),
        nose=Point2D(landmarks[FACE_NOSE].x, landmarks[FACE_NOSE].y),
        width=face_w,
        height=face_h,
        blendshapes=shapes
    )

def parse_hands(hand_result) -> List[HandData]:
    if not hand_result.hand_landmarks:
        return []

    parsed = []
    for marks in hand_result.hand_landmarks:
        wrist = Point2D(marks[WRIST].x, marks[WRIST].y)
        index_tip = Point2D(marks[INDEX_TIP].x, marks[INDEX_TIP].y)
        thumb_tip = Point2D(marks[THUMB_TIP].x, marks[THUMB_TIP].y)
        index_mcp = Point2D(marks[INDEX_MCP].x, marks[INDEX_MCP].y)
        
        # Approximate palm center
        palm_center = Point2D((wrist.x + index_mcp.x) / 2.0, (wrist.y + index_mcp.y) / 2.0)

        # Simple heuristic: fingers extended away from wrist indicates open hand
        tips = [marks[INDEX_TIP], marks[MIDDLE_TIP], marks[RING_TIP], marks[PINKY_TIP]]
        avg_tip_dist = sum(Point2D(t.x, t.y).distance_to(wrist) for t in tips) / 4.0
        palm_len = wrist.distance_to(index_mcp)
        is_open = avg_tip_dist > (palm_len * 1.3)

        parsed.append(HandData(
            wrist=wrist,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            palm_center=palm_center,
            is_open=is_open
        ))
    return parsed