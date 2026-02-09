"""
Face Classifier Module

Main processing pipeline for detecting and classifying faces in images and videos.
"""

import cv2
import numpy as np
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import multiprocessing as mp

from detector import FaceDetector, Face
from recognizer import FaceRecognizer, MatchResult
from database import DatabaseManager
from checkpoint import CheckpointManager, CheckpointTracker

logger = logging.getLogger(__name__)


class FaceClassifier:
    """
    Main classifier for processing files and classifying faces.
    """

    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 samples_dir: str,
                 embeddings_dir: str,
                 database_path: str,
                 model_name: str = "buffalo_l",
                 confidence_threshold: float = 0.75,
                 det_size: Tuple[int, int] = (640, 640),
                 providers: List[str] = None,
                 parallel_workers: int = None,
                 video_sample_fps: int = 2,
                 checkpoint_interval: int = 100):
        """
        Initialize the face classifier.

        Args:
            source_dir: Source directory with files to process
            output_dir: Output directory for classified files
            samples_dir: Directory containing sample images
            embeddings_dir: Directory containing embedding files
            database_path: Path to SQLite database
            model_name: InsightFace model name
            confidence_threshold: Initial confidence threshold
            det_size: Detection size
            providers: ONNX execution providers
            parallel_workers: Number of parallel workers
            video_sample_fps: Video sampling FPS
            checkpoint_interval: Checkpoint save interval
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.samples_dir = Path(samples_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.database_path = database_path

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing FaceClassifier components...")

        self.detector = FaceDetector(
            model_name=model_name,
            det_size=det_size,
            providers=providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )

        self.recognizer = FaceRecognizer(
            embeddings_dir=str(self.embeddings_dir),
            num_persons=35
        )

        self.database = DatabaseManager(database_path)
        self.checkpoint_manager = CheckpointManager(str(Path(database_path).parent))

        # Settings
        self.confidence_threshold = confidence_threshold
        self.parallel_workers = parallel_workers or mp.cpu_count()
        self.video_sample_fps = video_sample_fps
        self.checkpoint_interval = checkpoint_interval

        logger.info(f"FaceClassifier initialized with {self.parallel_workers} workers")

    def generate_sample_embeddings(self) -> bool:
        """
        Generate face embeddings from sample images.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Generating sample embeddings...")

        for person_id in range(1, 36):  # 1 to 35
            person_dir = self.samples_dir / f"person_{person_id:02d}"

            if not person_dir.exists():
                logger.error(f"Sample directory not found: {person_dir}")
                return False

            # Get all image files
            image_files = self._get_image_files(person_dir)

            if len(image_files) == 0:
                logger.error(f"No images found in {person_dir}")
                return False

            logger.info(f"Processing {len(image_files)} samples for person_{person_id:02d}")

            embeddings = []

            for image_file in tqdm(image_files, desc=f"Person {person_id:02d}"):
                # Detect faces
                faces = self.detector.detect_from_file(str(image_file))

                if len(faces) == 0:
                    logger.warning(f"No face detected in {image_file}")
                    continue

                if len(faces) > 1:
                    logger.warning(f"Multiple faces detected in {image_file}, using first one")

                # Extract embedding
                embedding = faces[0].embedding
                embeddings.append(embedding)

            if len(embeddings) == 0:
                logger.error(f"No valid embeddings for person_{person_id:02d}")
                return False

            # Save embeddings
            embedding_array = np.array(embeddings)
            embedding_file = self.embeddings_dir / f"person_{person_id:02d}.npy"
            np.save(embedding_file, embedding_array)

            logger.info(f"Saved {len(embeddings)} embeddings for person_{person_id:02d}")

        logger.info("Sample embedding generation completed")
        return True

    def process_file(self, file_path: str) -> Dict:
        """
        Process a single file (image or video).

        Args:
            file_path: Path to file

        Returns:
            Processing result dictionary
        """
        file_path = Path(file_path)
        file_type = self._get_file_type(file_path)

        # Save file record
        file_id = self.database.save_file(
            original_path=str(file_path),
            file_type=file_type,
            status="processing"
        )

        result = {
            "file_path": str(file_path),
            "file_id": file_id,
            "file_type": file_type,
            "detections": [],
            "error": None
        }

        try:
            if file_type == "image":
                detections = self._process_image(file_path, file_id)
            elif file_type == "video":
                detections = self._process_video(file_path, file_id)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            result["detections"] = detections

            # Update file status
            if detections:
                self.database.update_file_status(file_id, "completed")
            else:
                self.database.update_file_status(file_id, "no_faces")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            result["error"] = str(e)
            self.database.update_file_status(file_id, "error")

        return result

    def _process_image(self, image_path: Path, file_id: int) -> List[Dict]:
        """
        Process an image file.

        Args:
            image_path: Path to image
            file_id: Database file ID

        Returns:
            List of detection results
        """
        # Detect faces
        faces = self.detector.detect_from_file(str(image_path))

        if len(faces) == 0:
            logger.debug(f"No faces detected in {image_path}")
            return []

        detections = []

        for face in faces:
            # Match face
            match_result = self.recognizer.match_face(
                face.embedding,
                threshold=self.confidence_threshold
            )

            # Save detection to database
            detection_id = self.database.save_detection(
                file_id=file_id,
                person_id=match_result.person_id,
                confidence=match_result.confidence,
                bbox=face.bbox,
                needs_review=match_result.needs_review
            )

            detection = {
                "detection_id": detection_id,
                "person_id": match_result.person_id,
                "confidence": match_result.confidence,
                "bbox": face.bbox,
                "needs_review": match_result.needs_review
            }

            detections.append(detection)

            # Copy to output if auto-classified
            if not match_result.needs_review and match_result.person_id:
                self._copy_file_to_output(image_path, match_result.person_id, file_id)

        return detections

    def _process_video(self, video_path: Path, file_id: int) -> List[Dict]:
        """
        Process a video file by sampling frames.

        Args:
            video_path: Path to video
            file_id: Database file ID

        Returns:
            List of aggregated detection results
        """
        logger.debug(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval
        frame_interval = int(fps / self.video_sample_fps) if fps > 0 else 15

        logger.debug(f"Video FPS: {fps}, Total frames: {total_frames}, "
                    f"Sampling interval: {frame_interval}")

        # Aggregate detections
        person_detections = {}  # person_id -> [detections]

        frame_count = 0
        sampled_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Sample frames at interval
            if frame_count % frame_interval == 0:
                # Detect faces in frame
                faces = self.detector.detect_faces(frame)

                for face in faces:
                    # Match face
                    match_result = self.recognizer.match_face(
                        face.embedding,
                        threshold=self.confidence_threshold
                    )

                    if match_result.person_id and not match_result.needs_review:
                        # Aggregate detections by person
                        if match_result.person_id not in person_detections:
                            person_detections[match_result.person_id] = []

                        person_detections[match_result.person_id].append({
                            "confidence": match_result.confidence,
                            "frame": frame_count
                        })

                sampled_count += 1

            frame_count += 1

        cap.release()

        logger.debug(f"Processed {sampled_count} frames from {video_path}")

        # Save aggregated detections
        detections = []

        for person_id, person_dets in person_detections.items():
            # Calculate average confidence
            avg_confidence = np.mean([d["confidence"] for d in person_dets])

            # Save detection (no bbox for video)
            detection_id = self.database.save_detection(
                file_id=file_id,
                person_id=person_id,
                confidence=float(avg_confidence),
                bbox=(0, 0, 0, 0),  # No bbox for videos
                needs_review=False
            )

            detection = {
                "detection_id": detection_id,
                "person_id": person_id,
                "confidence": float(avg_confidence),
                "bbox": (0, 0, 0, 0),
                "needs_review": False
            }

            detections.append(detection)

            # Copy to output for each detected person
            self._copy_file_to_output(video_path, person_id, file_id)

        return detections

    def _copy_file_to_output(self, file_path: Path, person_id: int, file_id: int):
        """
        Copy a file to the output directory for a specific person.

        Args:
            file_path: Source file path
            person_id: Target person ID
            file_id: Database file ID
        """
        # Create person directory
        person_dir = self.output_dir / f"person_{person_id:02d}"
        person_dir.mkdir(parents=True, exist_ok=True)

        # Target path
        target_path = person_dir / file_path.name

        # Copy file
        try:
            shutil.copy2(file_path, target_path)

            # Save copy record to database
            self.database.save_copy(
                file_id=file_id,
                person_id=person_id,
                target_path=str(target_path)
            )

            logger.debug(f"Copied {file_path.name} to person_{person_id:02d}")

        except Exception as e:
            logger.error(f"Failed to copy {file_path} to {target_path}: {e}")

    def process_batch(self, file_paths: List[str],
                     progress_callback=None) -> Dict:
        """
        Process a batch of files.

        Args:
            file_paths: List of file paths
            progress_callback: Optional callback for progress updates

        Returns:
            Batch processing results
        """
        results = {
            "total": len(file_paths),
            "processed": 0,
            "errors": 0,
            "no_faces": 0,
            "classifications": 0
        }

        for file_path in tqdm(file_paths, desc="Processing batch"):
            result = self.process_file(file_path)

            results["processed"] += 1

            if result["error"]:
                results["errors"] += 1
            elif len(result["detections"]) == 0:
                results["no_faces"] += 1
            else:
                results["classifications"] += len(result["detections"])

            if progress_callback:
                progress_callback(results["processed"], results["total"])

        return results

    def process_all(self, resume: bool = False) -> Dict:
        """
        Process all files in the source directory.

        Args:
            resume: Resume from checkpoint

        Returns:
            Processing results
        """
        # Get all files
        all_files = self._get_all_files()

        logger.info(f"Found {len(all_files)} files to process")

        # Check for checkpoint
        if resume:
            checkpoint_state = self.checkpoint_manager.resume_from_checkpoint()
            if checkpoint_state:
                logger.info(f"Resuming from checkpoint: "
                          f"{checkpoint_state.processed_files_count}/{checkpoint_state.total_files} files processed")
                self.confidence_threshold = checkpoint_state.current_threshold
                all_files = self.checkpoint_manager.get_remaining_files(
                    checkpoint_state, all_files
                )

        # Initialize checkpoint tracker
        tracker = CheckpointTracker(
            checkpoint_manager=self.checkpoint_manager,
            total_files=len(all_files),
            checkpoint_interval=self.checkpoint_interval
        )

        # Process files
        results = {
            "total": len(all_files),
            "processed": 0,
            "errors": 0,
            "no_faces": 0,
            "classifications": 0
        }

        for file_path in tqdm(all_files, desc="Processing files"):
            result = self.process_file(file_path)

            results["processed"] += 1

            if result["error"]:
                results["errors"] += 1
            elif len(result["detections"]) == 0:
                results["no_faces"] += 1
            else:
                results["classifications"] += len(result["detections"])

            # Update checkpoint
            tracker.update(file_path)

            # Adjust threshold periodically
            if tracker.processed_files % 500 == 0:
                self._adjust_threshold()

        # Mark completed
        tracker.mark_completed()

        logger.info(f"Processing completed: {results}")
        return results

    def _adjust_threshold(self):
        """
        Dynamically adjust confidence threshold based on review ratio.
        """
        stats = self.database.get_statistics()

        total_processed = stats.get("files_by_status", {}).get("completed", 0)
        needs_review = stats.get("needs_review", 0)

        if total_processed == 0:
            return

        review_ratio = needs_review / total_processed

        logger.info(f"Review ratio: {review_ratio:.2%} "
                   f"({needs_review}/{total_processed} files)")

        if review_ratio < 0.10:
            # Too few reviews, be more strict
            old_threshold = self.confidence_threshold
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
            logger.info(f"Adjusted threshold: {old_threshold:.2f} -> {self.confidence_threshold:.2f} "
                       f"(more strict)")
        elif review_ratio > 0.30:
            # Too many reviews, be more lenient
            old_threshold = self.confidence_threshold
            self.confidence_threshold = max(0.50, self.confidence_threshold - 0.05)
            logger.info(f"Adjusted threshold: {old_threshold:.2f} -> {self.confidence_threshold:.2f} "
                       f"(more lenient)")

    def _get_file_type(self, file_path: Path) -> str:
        """Determine if file is image or video."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV'}

        ext = file_path.suffix

        if ext in image_extensions:
            return "image"
        elif ext in video_extensions:
            return "video"
        else:
            return "unknown"

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

        files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix in image_extensions:
                files.append(file_path)

        return sorted(files)

    def _get_all_files(self) -> List[str]:
        """Get all processable files from source directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV'}

        valid_extensions = image_extensions | video_extensions

        files = []

        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in valid_extensions:
                files.append(str(file_path))

        return sorted(files)
