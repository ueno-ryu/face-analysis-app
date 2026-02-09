"""
Face Detection Module

Detects faces in images and extracts face embeddings using DeepFace.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Face:
    """Represents a detected face with its metadata."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: np.ndarray
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceDetector:
    """
    Face detector using DeepFace library.

    Supports multiple backends: VGG-Face, GoogleFace, ArcFace, Facenet.
    """

    def __init__(self, model_name: str = "VGG-Face",
                 detector_backend: str = "retinaface",
                 enforce_detection: bool = True):
        """
        Initialize the face detector.

        Args:
            model_name: DeepFace model name (default: VGG-Face)
            detector_backend: Face detector backend (default: retinaface)
            enforce_detection: Whether to enforce face detection
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

        logger.info(f"Initializing FaceDetector with model: {model_name}")
        logger.info(f"Detector backend: {detector_backend}")

        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            logger.info("FaceDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {e}")
            raise

    def detect_faces(self, image: np.ndarray,
                    min_confidence: float = 0.5) -> List[Face]:
        """
        Detect all faces in an image.

        Args:
            image: Input image (numpy array from cv2.imread)
            min_confidence: Minimum confidence threshold for detection

        Returns:
            List of Face objects with bounding boxes and embeddings
        """
        if image is None:
            logger.warning("Received None image")
            return []

        try:
            # Import DeepFace module directly
            from deepface import DeepFace

            # Detect faces using DeepFace
            detections = DeepFace.extract_faces(
                image,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection
            )

            if not isinstance(detections, list):
                detections = [detections] if detections is not None else []

            results = []
            for det in detections:
                # Extract confidence if available
                confidence = det.get('confidence', 0.0)
                if confidence < min_confidence:
                    continue

                # Extract bounding box
                # DeepFace returns: x, y, w, h (top-left corner)
                x, y, w, h = det['facial_area']['x'], det['facial_area']['y'], det['facial_area']['w'], det['facial_area']['h']
                bbox = [x, y, x + w, y + h]
                # Ensure bbox is within image bounds
                bbox = self._clamp_bbox(bbox, image.shape)

                # Extract embedding for this face
                embedding = self._extract_embedding(image, bbox)

                if embedding is None:
                    continue

                # Create Face object
                face_obj = Face(
                    bbox=tuple(bbox),
                    embedding=embedding,
                    confidence=float(confidence),
                    landmarks=None  # DeepFace doesn't provide landmarks by default
                )
                results.append(face_obj)

            logger.debug(f"Detected {len(results)} faces in image")
            return results

        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def _extract_embedding(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract embedding for a specific face region.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Face embedding vector or None if extraction fails
        """
        try:
            from deepface import DeepFace

            x1, y1, x2, y2 = bbox
            face_region = image[y1:y2, x1:x2]

            # Extract embedding using DeepFace
            embeddings = DeepFace.represent(
                face_region,
                model_name=self.model_name,
                enforce_detection=False
            )

            if embeddings and len(embeddings) > 0:
                return np.array(embeddings[0]['embedding'])
            else:
                return None

        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None

    def _clamp_bbox(self, bbox: List[int],
                   image_shape: Tuple[int, int, int]) -> List[int]:
        """
        Clamp bounding box coordinates to image bounds.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_shape: Image shape (height, width, channels)

        Returns:
            Clamped bounding box
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        return [x1, y1, x2, y2]

    def detect_from_file(self, file_path: str,
                        min_confidence: float = 0.5) -> List[Face]:
        """
        Detect faces from an image file.

        Args:
            file_path: Path to image file
            min_confidence: Minimum confidence threshold

        Returns:
            List of Face objects
        """
        try:
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"Failed to read image: {file_path}")
                return []

            return self.detect_faces(image, min_confidence)

        except Exception as e:
            logger.error(f"Error detecting faces from file {file_path}: {e}")
            return []

    def extract_face_image(self, image: np.ndarray,
                          face: Face) -> np.ndarray:
        """
        Extract the face region from an image.

        Args:
            image: Original image
            face: Face object with bounding box

        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = face.bbox
        return image[y1:y2, x1:x2].copy()

    def get_embedding(self, image: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get embedding for a specific face region.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Face embedding vector or None
        """
        bbox_list = list(bbox)
        return self._extract_embedding(image, bbox_list)


def test_detector():
    """Test the face detector."""

    detector = FaceDetector()

    # Test with a sample image if available
    test_image_path = "./samples/person_01/sample_01.jpg"
    if Path(test_image_path).exists():
        faces = detector.detect_from_file(test_image_path)
        print(f"Detected {len(faces)} faces")
        for i, face in enumerate(faces):
            print(f"Face {i+1}: bbox={face.bbox}, confidence={face.confidence}")
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    test_detector()
