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
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
import yaml

from .detector import FaceDetector, Face
from .recognizer import FaceRecognizer, MatchResult
from .database import DatabaseManager
from .checkpoint import CheckpointManager, CheckpointTracker

logger = logging.getLogger(__name__)


class FaceClassifier:
    """
    Main classifier for processing files and classifying faces.
    """

    def __init__(self, config_path: str = "./config.yaml"):
        """
        Initialize the face classifier from config file.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract paths
        paths = self.config['paths']
        self.source_dir = Path(paths['source_directory'])
        self.output_dir = Path(paths['output_directory'])
        self.samples_dir = Path(paths['samples_directory'])
        self.embeddings_dir = Path(paths['embeddings_directory'])
        self.database_path = paths['database_path']

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

        # Extract recognition settings
        rec = self.config['recognition']
        self.model_name = rec['model_name']
        self.detector_backend = rec['detector_backend']
        self.enforce_detection = rec['enforce_detection']
        # Legacy fields for compatibility
        self.det_size = tuple(rec.get('det_size', [640, 640]))
        self.providers = rec.get('providers', ['CPUExecutionProvider'])

        # Extract processing settings
        proc = self.config['processing']
        self.batch_size = proc['batch_size']
        self.parallel_workers = proc['parallel_workers']
        self.video_sample_fps = proc['video_sample_fps']
        self.checkpoint_interval = proc['checkpoint_interval']
        self.threshold_adjust_interval = proc['threshold_adjust_interval']

        # Extract threshold settings
        thresh = self.config['threshold']
        self.confidence_threshold = thresh['initial']
        self.threshold_min = thresh['min']
        self.threshold_max = thresh['max']
        self.threshold_step = thresh['step']
        self.target_ratio_min = thresh['target_ratio_min']
        self.target_ratio_max = thresh['target_ratio_max']

        # Extract file extensions
        self.image_extensions = set(self.config['file_extensions']['images'])
        self.video_extensions = set(self.config['file_extensions']['videos'])

        # Initialize components
        logger.info("Initializing FaceClassifier components...")

        self.detector = FaceDetector(
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection
        )

        self.recognizer = FaceRecognizer(
            embeddings_dir=str(self.embeddings_dir),
            num_persons=self.config['persons']['count']
        )

        self.database = DatabaseManager(self.database_path)
        self.checkpoint_manager = CheckpointManager(str(Path(self.database_path).parent))

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
        Process a batch of files using multiprocessing.

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

        # Process files in parallel using multiprocessing Pool
        with Pool(processes=self.parallel_workers) as pool:
            # Prepare arguments for each file
            args_list = [(fp, self.confidence_threshold) for fp in file_paths]

            # Process files asynchronously with progress bar
            imap_results = pool.imap_unordered(
                _process_file_wrapper,
                args_list
            )

            for result in tqdm(imap_results, total=len(file_paths), desc="Processing batch"):
                results["processed"] += 1

                if result.get("error"):
                    results["errors"] += 1
                elif len(result.get("detections", [])) == 0:
                    results["no_faces"] += 1
                else:
                    results["classifications"] += len(result.get("detections", []))

                if progress_callback:
                    progress_callback(results["processed"], results["total"])

        return results

    def process_all(self, resume: bool = False) -> Dict:
        """
        Process all files in the source directory using parallel batch processing.

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

        # Split files into batches
        batches = [
            all_files[i:i + self.batch_size]
            for i in range(0, len(all_files), self.batch_size)
        ]

        logger.info(f"Processing {len(all_files)} files in {len(batches)} batches "
                   f"(batch_size={self.batch_size}, workers={self.parallel_workers})")

        # Process batches
        results = {
            "total": len(all_files),
            "processed": 0,
            "errors": 0,
            "no_faces": 0,
            "classifications": 0
        }

        for batch_idx, batch_files in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                       f"({len(batch_files)} files)")

            # Process batch with current threshold
            batch_args = [(fp, self.confidence_threshold) for fp in batch_files]

            # Use sequential processing for single worker to avoid Pool overhead
            if self.parallel_workers == 1:
                # Sequential processing - reuse detector/recognizer instances
                for file_path, threshold in tqdm(batch_args, total=len(batch_files),
                                                 desc=f"Batch {batch_idx + 1}"):
                    result = self._process_single_file(file_path, threshold)
                    results["processed"] += 1

                    if result.get("error"):
                        results["errors"] += 1
                    elif len(result.get("detections", [])) == 0:
                        results["no_faces"] += 1
                    else:
                        results["classifications"] += len(result.get("detections", []))

                    # Update checkpoint for each file
                    tracker.update(result.get("file_path", ""))
            else:
                # Multiprocessing for multiple workers
                with Pool(processes=self.parallel_workers) as pool:
                    imap_results = pool.imap_unordered(
                        _process_file_wrapper,
                        batch_args
                    )

                    for result in tqdm(imap_results, total=len(batch_files),
                                      desc=f"Batch {batch_idx + 1}"):
                        results["processed"] += 1

                        if result.get("error"):
                            results["errors"] += 1
                        elif len(result.get("detections", [])) == 0:
                            results["no_faces"] += 1
                        else:
                            results["classifications"] += len(result.get("detections", []))

                        # Update checkpoint for each file
                        tracker.update(result.get("file_path", ""))

            # Save checkpoint after each batch
            logger.info(f"Batch {batch_idx + 1} completed. Saving checkpoint...")
            tracker.save()

            # Adjust threshold periodically
            if results["processed"] % self.threshold_adjust_interval == 0:
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

        if review_ratio < self.target_ratio_min:
            # Too few reviews, be more strict
            old_threshold = self.confidence_threshold
            self.confidence_threshold = min(
                self.threshold_max,
                self.confidence_threshold + self.threshold_step
            )
            logger.info(f"Adjusted threshold: {old_threshold:.2f} -> {self.confidence_threshold:.2f} "
                       f"(more strict)")
        elif review_ratio > self.target_ratio_max:
            # Too many reviews, be more lenient
            old_threshold = self.confidence_threshold
            self.confidence_threshold = max(
                self.threshold_min,
                self.confidence_threshold - self.threshold_step
            )
            logger.info(f"Adjusted threshold: {old_threshold:.2f} -> {self.confidence_threshold:.2f} "
                       f"(more lenient)")

    def _get_file_type(self, file_path: Path) -> str:
        """Determine if file is image or video."""
        ext = file_path.suffix

        if ext in self.image_extensions:
            return "image"
        elif ext in self.video_extensions:
            return "video"
        else:
            return "unknown"

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory."""
        files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix in self.image_extensions:
                files.append(file_path)

        return sorted(files)

    def _get_all_files(self) -> List[str]:
        """Get all processable files from source directory."""
        valid_extensions = self.image_extensions | self.video_extensions

        files = []

        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in valid_extensions:
                files.append(str(file_path))

        return sorted(files)

    def _process_single_file(self, file_path: str, threshold: float) -> Dict:
        """
        Process a single file using existing detector/recognizer instances.
        Used for sequential processing (parallel_workers=1).

        Args:
            file_path: Path to file
            threshold: Confidence threshold

        Returns:
            Processing result dictionary
        """
        result = {
            "file_path": file_path,
            "file_id": None,
            "file_type": None,
            "detections": [],
            "error": None
        }

        try:
            file_path_obj = Path(file_path)
            file_type = self._get_file_type(file_path_obj)
            result["file_type"] = file_type

            # Save file record
            file_id = self.database.save_file(
                original_path=file_path,
                file_type=file_type,
                status="processing"
            )
            result["file_id"] = file_id

            if file_type == "image":
                # Process image
                img = cv2.imread(str(file_path))
                if img is None:
                    result["error"] = "Failed to read image"
                    return result

                # Detect faces
                faces = self.detector.detect_faces(img)
                if not faces:
                    result["error"] = "No faces detected"
                    self.database.update_file_status(file_id, "completed", 0, [])
                    return result

                # Recognize faces using Face objects
                for face in faces:
                    # Face object has embedding, confidence, landmarks
                    match_result = self.recognizer.match_face(face.embedding, threshold)
                    if match_result.person_id:
                        result["detections"].append({
                            "person_id": match_result.person_id,
                            "confidence": float(match_result.confidence),
                            "similarity": float(match_result.similarity),
                            "detection_confidence": float(face.confidence),
                            "needs_review": match_result.needs_review
                        })

                # Save detections to database
                self.database.save_detections(file_id, result["detections"])

                # Determine if needs review
                needs_review = any(
                    d["confidence"] < threshold for d in result["detections"]
                )
                status = "needs_review" if needs_review else "completed"
                self.database.update_file_status(
                    file_id, status, len(result["detections"]),
                    [d["person_id"] for d in result["detections"]]
                )

            elif file_type == "video":
                # Process video (sample frames)
                result["error"] = "Video processing not implemented in sequential mode"
                return result

            else:
                result["error"] = f"Unknown file type: {file_type}"
                return result

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            result["error"] = str(e)

        return result


# =============================================================================
# Multiprocessing Helper Functions
# =============================================================================

def _process_file_wrapper(args: Tuple[str, float]) -> Dict:
    """
    Wrapper function for multiprocessing pool.

    Args:
        args: Tuple of (file_path, threshold)

    Returns:
        Processing result dictionary
    """
    file_path, threshold = args

    # Import inside function to avoid pickling issues
    from pathlib import Path
    import cv2
    import numpy as np
    import shutil
    import logging
    from src.detector import FaceDetector
    from src.recognizer import FaceRecognizer
    from src.database import DatabaseManager
    import yaml

    # Load config
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    logger = logging.getLogger(__name__)

    # Initialize components for this worker
    detector = FaceDetector(
        model_name=config['recognition']['model_name'],
        detector_backend=config['recognition']['detector_backend'],
        enforce_detection=config['recognition']['enforce_detection']
    )

    recognizer = FaceRecognizer(
        embeddings_dir=config['paths']['embeddings_directory'],
        num_persons=config['persons']['count']
    )
    recognizer.load_sample_embeddings()

    database = DatabaseManager(config['paths']['database_path'])

    # Process file
    file_path_obj = Path(file_path)
    file_type = _get_file_type_static(file_path_obj, config)

    result = {
        "file_path": file_path,
        "file_id": None,
        "file_type": file_type,
        "detections": [],
        "error": None
    }

    try:
        # Save file record
        file_id = database.save_file(
            original_path=file_path,
            file_type=file_type,
            status="processing"
        )
        result["file_id"] = file_id

        # Process based on type
        if file_type == "image":
            detections = _process_image_static(
                file_path_obj, file_id, detector, recognizer,
                database, threshold, config
            )
        elif file_type == "video":
            detections = _process_video_static(
                file_path_obj, file_id, detector, recognizer,
                database, threshold, config
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        result["detections"] = detections

        # Update file status
        if detections:
            database.update_file_status(file_id, "completed")
        else:
            database.update_file_status(file_id, "no_faces")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        result["error"] = str(e)
        if result["file_id"]:
            database.update_file_status(result["file_id"], "error")

    return result


def _get_file_type_static(file_path: Path, config: Dict) -> str:
    """Determine file type from config."""
    image_extensions = set(config['file_extensions']['images'])
    video_extensions = set(config['file_extensions']['videos'])

    ext = file_path.suffix

    if ext in image_extensions:
        return "image"
    elif ext in video_extensions:
        return "video"
    else:
        return "unknown"


def _process_image_static(image_path: Path, file_id: int,
                         detector: 'FaceDetector',
                         recognizer: 'FaceRecognizer',
                         database: 'DatabaseManager',
                         threshold: float,
                         config: Dict) -> List[Dict]:
    """Process an image file (static method for multiprocessing)."""
    from pathlib import Path
    import shutil

    # Detect faces
    faces = detector.detect_from_file(str(image_path))

    if len(faces) == 0:
        return []

    detections = []
    output_dir = Path(config['paths']['output_directory'])

    for face in faces:
        # Match face
        match_result = recognizer.match_face(
            face.embedding,
            threshold=threshold
        )

        # Save detection to database
        detection_id = database.save_detection(
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
            _copy_file_to_output_static(
                image_path, match_result.person_id, file_id,
                database, output_dir
            )

    return detections


def _process_video_static(video_path: Path, file_id: int,
                         detector: 'FaceDetector',
                         recognizer: 'FaceRecognizer',
                         database: 'DatabaseManager',
                         threshold: float,
                         config: Dict) -> List[Dict]:
    """Process a video file by sampling frames (static method for multiprocessing)."""
    from pathlib import Path
    import shutil

    video_sample_fps = config['processing']['video_sample_fps']
    output_dir = Path(config['paths']['output_directory'])

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval
    frame_interval = int(fps / video_sample_fps) if fps > 0 else 15

    # Aggregate detections
    person_detections = {}  # person_id -> [detections]

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Sample frames at interval
        if frame_count % frame_interval == 0:
            # Detect faces in frame
            faces = detector.detect_faces(frame)

            for face in faces:
                # Match face
                match_result = recognizer.match_face(
                    face.embedding,
                    threshold=threshold
                )

                if match_result.person_id and not match_result.needs_review:
                    # Aggregate detections by person
                    if match_result.person_id not in person_detections:
                        person_detections[match_result.person_id] = []

                    person_detections[match_result.person_id].append({
                        "confidence": match_result.confidence,
                        "frame": frame_count
                    })

        frame_count += 1

    cap.release()

    # Save aggregated detections
    detections = []

    for person_id, person_dets in person_detections.items():
        # Calculate average confidence
        avg_confidence = np.mean([d["confidence"] for d in person_dets])

        # Save detection (no bbox for video)
        detection_id = database.save_detection(
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
        _copy_file_to_output_static(
            video_path, person_id, file_id, database, output_dir
        )

    return detections


def _copy_file_to_output_static(file_path: Path, person_id: int,
                                file_id: int, database: 'DatabaseManager',
                                output_dir: Path):
    """Copy a file to the output directory (static method for multiprocessing)."""
    import shutil

    # Create person directory
    person_dir = output_dir / f"person_{person_id:02d}"
    person_dir.mkdir(parents=True, exist_ok=True)

    # Target path
    target_path = person_dir / file_path.name

    # Copy file
    try:
        shutil.copy2(file_path, target_path)

        # Save copy record to database
        database.save_copy(
            file_id=file_id,
            person_id=person_id,
            target_path=str(target_path)
        )

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to copy {file_path} to {target_path}: {e}")


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    # Ensure both are numpy arrays
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()

    # Calculate cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, similarity)))


def add_to_review_queue(file_path: str, detections: List[Dict],
                       database: 'DatabaseManager'):
    """
    Add file and its detections to review queue.

    Args:
        file_path: Path to the file
        detections: List of detection dictionaries
        database: Database manager instance
    """
    logger = logging.getLogger(__name__)

    # Get file_id from database
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_id FROM files WHERE original_path = ?
        """, (file_path,))
        row = cursor.fetchone()
        if not row:
            logger.error(f"File not found in database: {file_path}")
            return
        file_id = row["file_id"]

    # Mark detections as needing review
    for detection in detections:
        if detection.get("needs_review", False):
            with database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE detections
                    SET needs_review = 1
                    WHERE detection_id = ?
                """, (detection["detection_id"],))

    logger.info(f"Added {len([d for d in detections if d.get('needs_review')])} "
               f"detections to review queue for {file_path}")


def should_copy(file_path: str, person_id: int, output_dir: Path) -> bool:
    """
    Determine if a file should be copied to classified output.

    Args:
        file_path: Source file path
        person_id: Target person ID
        output_dir: Output directory path

    Returns:
        True if file should be copied, False otherwise
    """
    # Check if file already exists in target directory
    person_dir = output_dir / f"person_{person_id:02d}"
    target_path = person_dir / Path(file_path).name

    if target_path.exists():
        return False

    return True
