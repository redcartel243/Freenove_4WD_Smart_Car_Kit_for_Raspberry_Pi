"""
YOLO Vision Processor
Handles real-time object detection on video feed
"""

import numpy as np
import cv2
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple
import threading
import time


@dataclass
class Detection:
    """Represents a detected object"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center x, y
    area: int  # bounding box area in pixels

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox,
            "center": self.center,
            "area": self.area,
            "position": self._get_position_description()
        }

    def _get_position_description(self) -> str:
        """Get human-readable position (left/center/right)"""
        # Assuming 400x300 frame
        cx = self.center[0]
        if cx < 133:
            h_pos = "left"
        elif cx > 267:
            h_pos = "right"
        else:
            h_pos = "center"

        cy = self.center[1]
        if cy < 100:
            v_pos = "top"
        elif cy > 200:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        return f"{v_pos}-{h_pos}"


class YOLOProcessor:
    """Processes video frames with YOLO object detection"""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO processor.

        Args:
            model_path: Path to YOLO model (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.last_detections: List[Detection] = []
        self.last_frame: Optional[np.ndarray] = None
        self.last_annotated_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.processing = False
        self.frame_count = 0
        self.fps = 0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Process a single frame and return detections.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of Detection objects
        """
        self.processing = True
        self.frame_count += 1
        self._update_fps()

        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        detections = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # Calculate center and area
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                detection = Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    area=area
                )
                detections.append(detection)

                # Draw on annotated frame
                color = self._get_color_for_label(label)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add FPS counter
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Store results
        with self.lock:
            self.last_detections = detections
            self.last_frame = frame.copy()
            self.last_annotated_frame = annotated_frame

        self.processing = False
        return detections

    def _update_fps(self):
        """Update FPS calculation"""
        self._fps_frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

    def _get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """Get consistent color for a label"""
        # Hash the label to get a consistent color
        hash_val = hash(label) % 360
        # Convert HSV to BGR
        hsv = np.uint8([[[hash_val // 2, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(map(int, bgr[0][0]))

    def get_detections(self) -> List[Detection]:
        """Get last detections thread-safely"""
        with self.lock:
            return self.last_detections.copy()

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Get last annotated frame thread-safely"""
        with self.lock:
            if self.last_annotated_frame is not None:
                return self.last_annotated_frame.copy()
            return None

    def get_detection_summary(self) -> dict:
        """
        Get a summary of current detections for the LLM.
        Returns structured data about what's visible.
        """
        with self.lock:
            detections = self.last_detections.copy()

        if not detections:
            return {
                "objects_detected": 0,
                "objects": [],
                "summary": "No objects detected in view"
            }

        # Group by label
        label_counts = {}
        for det in detections:
            if det.label not in label_counts:
                label_counts[det.label] = []
            label_counts[det.label].append(det)

        objects_list = []
        for label, dets in label_counts.items():
            for i, det in enumerate(dets):
                obj = det.to_dict()
                if len(dets) > 1:
                    obj["instance"] = i + 1
                objects_list.append(obj)

        # Create natural language summary
        summary_parts = []
        for label, dets in label_counts.items():
            count = len(dets)
            if count == 1:
                pos = dets[0]._get_position_description()
                summary_parts.append(f"1 {label} ({pos})")
            else:
                summary_parts.append(f"{count} {label}s")

        summary = "Detected: " + ", ".join(summary_parts)

        return {
            "objects_detected": len(detections),
            "objects": objects_list,
            "summary": summary,
            "fps": round(self.fps, 1)
        }

    def find_object(self, target_label: str) -> Optional[Detection]:
        """
        Find the largest instance of a specific object.

        Args:
            target_label: Object class to find (e.g., "person", "chair")

        Returns:
            Detection if found, None otherwise
        """
        with self.lock:
            detections = self.last_detections.copy()

        matching = [d for d in detections if d.label.lower() == target_label.lower()]
        if not matching:
            return None

        # Return the largest one (closest to camera usually)
        return max(matching, key=lambda d: d.area)

    def is_object_in_path(self, min_area: int = 5000) -> Tuple[bool, Optional[Detection]]:
        """
        Check if there's an object directly in the robot's path (center of frame).

        Args:
            min_area: Minimum bounding box area to consider

        Returns:
            (is_blocked, blocking_object)
        """
        with self.lock:
            detections = self.last_detections.copy()

        for det in detections:
            # Check if object is in center third of frame and large enough
            cx = det.center[0]
            if 133 < cx < 267 and det.area > min_area:
                return True, det

        return False, None

    def get_detections_raw(self) -> List[dict]:
        """
        Get detections in raw dict format for the autonomous system.
        Returns list of dicts with class_name, confidence, box fields.
        """
        with self.lock:
            detections = self.last_detections.copy()

        return [
            {
                "class_name": det.label,
                "confidence": det.confidence,
                "box": list(det.bbox)  # [x1, y1, x2, y2]
            }
            for det in detections
        ]


# Simple test
if __name__ == "__main__":
    processor = YOLOProcessor()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = processor.process_frame(frame)
        print(processor.get_detection_summary())

        annotated = processor.get_annotated_frame()
        if annotated is not None:
            cv2.imshow("YOLO", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
