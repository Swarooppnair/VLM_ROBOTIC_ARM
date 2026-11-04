"""Visual object detection using color segmentation and contour analysis."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ObjectDetector:
    """Performs visual object detection and bounding box extraction on camera images."""

    # HSV color thresholds for tabletop items
    COLOR_RANGES = {
        "red_cube": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
        "blue_cube": [
            (np.array([100, 150, 70]), np.array([130, 255, 255])),
        ],
        "green_cube": [
            (np.array([40, 100, 70]), np.array([85, 255, 255])),
        ],
        "bottle": [
            (np.array([20, 120, 100]), np.array([35, 255, 255])),
        ],
        "box": [
            (np.array([90, 80, 60]), np.array([110, 200, 200])),
        ],
    }

    def __init__(self, min_area: int = 150):
        self.min_area = min_area

    def detect_objects(self, rgb_image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Detect tabletop objects in the RGB image.
        Returns: Dict mapping object_name to detection metadata (bbox, centroid, confidence, etc.)
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        detections = {}

        for obj_name, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_c)
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(largest_c)
                    M = cv2.moments(largest_c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx = x + w // 2
                        cy = y + h // 2

                    detections[obj_name] = {
                        "name": obj_name,
                        "bbox": [x, y, w, h],
                        "centroid": (cx, cy),
                        "area": area,
                        "confidence": min(1.0, area / 2000.0 + 0.5),
                    }

        return detections

    def draw_detections(self, rgb_image: np.ndarray, detections: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and labels onto an image copy."""
        vis_img = rgb_image.copy()
        for name, det in detections.items():
            x, y, w, h = det["bbox"]
            cx, cy = det["centroid"]
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis_img, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                vis_img,
                f"{name} ({det['confidence']:.2f})",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return vis_img

