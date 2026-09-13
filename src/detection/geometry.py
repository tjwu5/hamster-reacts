import math
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class Point2D:
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

@dataclass
class FaceData:
    center: Point2D
    top: Point2D
    chin: Point2D
    mouth: Point2D
    nose: Point2D
    width: float
    height: float
    blendshapes: Dict[str, float]

    def near(self, pt: Point2D, target: Point2D, face_width_ratio: float) -> bool:
        """Returns True if pt is within (face_width_ratio * face.width) of target."""
        if self.width <= 0:
            return False
        return (pt.distance_to(target) / self.width) <= face_width_ratio

@dataclass
class HandData:
    wrist: Point2D
    index_tip: Point2D
    thumb_tip: Point2D
    palm_center: Point2D
    is_open: bool