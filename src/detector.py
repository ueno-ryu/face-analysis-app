"""
Face Detection Module

Detects faces in images and extracts face embeddings using InsightFace.
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
    Face detector using InsightFace library.

    Supports Apple M1 Metal acceleration via CoreMLExecutionProvider.
    """

    def __init__(self, model_name: str = "buffalo_l",
                 det_size: Tuple[int, int] = (640, 640),
                 providers: List[str] = None):
        """
        Initialize the face detector.

        Args:
            model_name: InsightFace model name (default: buffalo_l)
            det_size: Detection size (width, height)
            providers: ONNX execution providers (e.g., ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        """
        self.model_name = model_name
        self.det_size = det_size
        self.providers = providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        logger.info(f"Initializing FaceDetector with model: {model_name}")
        logger.info(f"Detection size: {det_size}")
        logger.info(f"Execution providers: {providers}")

        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name=model_name,
                providers=self.providers
            )
            self.app.prepare(ctx_id=-1, det_size=det_size)  # ctx_id=-1 for CPU/Metal
            logger.info("FaceDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {e}")
            # Try fallback to CPU only
            if "CoreMLExecutionProvider" in self.providers:
                logger.warning("Attempting fallback to CPU execution")
                self.providers = ["CPUExecutionProvider"]
                try:
                    self.app = insightface.app.FaceAnalysis(
                        name=model_name,
                        providers=self.providers
                    )
                    self.app.prepare(ctx_id=-1, det_size=det_size)
                    logger.info("FaceDetector initialized with CPU fallback")
                except Exception as e2:
                    logger.error(f"Failed to initialize FaceDetector even with CPU fallback: {e2}")
                    raise
            else:
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
            # Detect faces
            faces = self.app.get(image)

            results = []
            for face in faces:
                if face.det_score < min_confidence:
                    continue

                # Extract bounding box
                bbox = face.bbox.astype(int).tolist()
                # Ensure bbox is within image bounds
                bbox = self._clamp_bbox(bbox, image.shape)

                # Extract embedding
                embedding = face.embedding

                # Create Face object
                face_obj = Face(
                    bbox=tuple(bbox),
                    embedding=embedding,
                    confidence=float(face.det_score),
                    landmarks=face.landmark
                )
                results.append(face_obj)

            logger.debug(f"Detected {len(results)} faces in image")
            return results

        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

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
                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Get embedding for a specific face region.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Face embedding vector
        """
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]

        faces = self.detect_faces(face_region)
        if faces:
            return faces[0].embedding
        else:
            logger.warning("No face detected in specified region")
            return None


def test_detector():
    """Test the face detector."""
    logging.basicConfig(level=logging.DEBUG)

    detector = FaceDetector()

    # Test with a sample image if available
    test_image_path = "./samples/person_01/test_image.jpg"
    if Path(test_image_path).exists():
        faces = detector.detect_from_file(test_image_path)
        print(f"Detected {len(faces)} faces")
        for i, face in enumerate(faces):
            print(f"Face {i+1}: bbox={face.bbox}, confidence={face.confidence}")
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    test_detector()
