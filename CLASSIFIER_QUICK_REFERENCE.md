# FaceClassifier - Quick Reference Guide

## File Location
`/Volumes/01022610461/_PRJ/face-analysis-app/src/classifier.py`

## Key Code Sections

### 1. Initialization (Lines 24-93)
```python
classifier = FaceClassifier(config_path="./config.yaml")
```
Loads config.yaml and initializes:
- FaceDetector (InsightFace buffalo_l model)
- FaceRecognizer (embeddings matching)
- DatabaseManager (SQLite operations)
- CheckpointManager (resume capability)

### 2. Main Processing Pipeline (Lines 425-541)
```python
results = classifier.process_all(resume=False)
```
- Scans source directory for files
- Splits into batches (100 files/batch)
- Processes batches with multiprocessing
- Saves checkpoints after each batch
- Adjusts threshold every 500 files

### 3. Batch Processing with Multiprocessing (Lines 388-424)
```python
with Pool(processes=self.parallel_workers) as pool:
    imap_results = pool.imap_unordered(
        _process_file_wrapper,
        args_list
    )
```
- Creates Pool with N workers (configurable)
- Uses imap_unordered for parallel execution
- Processes 100 files concurrently

### 4. File Processing Wrapper (Lines 564-630)
```python
def _process_file_wrapper(args: Tuple[str, float]) -> Dict:
    file_path, threshold = args
    # Loads detector/recognizer per worker
    # Processes image or video
    # Returns results dictionary
```
- Entry point for multiprocessing
- Initializes components per worker
- Handles both images and videos

### 5. Image Processing (Lines 632-677)
```python
def _process_image_static(image_path, file_id, detector, recognizer,
                         database, threshold, config):
    # Detect faces
    faces = detector.detect_from_file(str(image_path))
    # Match each face
    for face in faces:
        match_result = recognizer.match_face(face.embedding, threshold)
        # Save detection and copy if classified
```
- Detects all faces in image
- Extracts embeddings (512-dim vectors)
- Matches against 35 person samples
- Copies to person_XX if confident

### 6. Video Processing (Lines 679-745)
```python
def _process_video_static(video_path, file_id, detector, recognizer,
                         database, threshold, config):
    cap = cv2.VideoCapture(str(video_path))
    frame_interval = int(fps / video_sample_fps)  # 2fps
    # Sample frames at interval
    # Aggregate detections by person
    # Calculate average confidence
```
- Opens video file
- Samples frames at 2fps
- Detects faces in each frame
- Aggregates by person_id
- Copies if any confident detection

### 7. Cosine Similarity (Lines 768-786)
```python
def cosine_similarity(embedding1: np.ndarray,
                     embedding2: np.ndarray) -> float:
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm1 * norm2)
    return float(max(0.0, min(1.0, similarity)))
```
- Calculates cosine of angle between vectors
- Returns value in [0, 1] range
- Higher = more similar

### 8. Dynamic Threshold Adjustment (Lines 492-521)
```python
def _adjust_threshold(self):
    stats = self.database.get_statistics()
    review_ratio = needs_review / total_processed

    if review_ratio < 0.10:
        # Too few reviews, be more strict
        self.confidence_threshold += 0.05
    elif review_ratio > 0.30:
        # Too many reviews, be more lenient
        self.confidence_threshold -= 0.05
```
- Calculates review ratio (0-30% target)
- Adjusts threshold by 0.05 increments
- Range: 0.50 to 0.95

### 9. File Copy Logic (Lines 748-765)
```python
def _copy_file_to_output_static(file_path, person_id, file_id,
                                database, output_dir):
    person_dir = output_dir / f"person_{person_id:02d}"
    target_path = person_dir / file_path.name
    shutil.copy2(file_path, target_path)
    database.save_copy(file_id, person_id, str(target_path))
```
- Creates person_XX directory
- Copies file with metadata
- Records in database

### 10. Review Queue (Lines 789-808)
```python
def add_to_review_queue(file_path, detections, database):
    for detection in detections:
        if detection.get("needs_review", False):
            # Flag for manual review
            database.update_detection_needs_review(detection_id)
```
- Flags low-confidence detections
- Queryable via database.get_detections_needing_review()
- Human can review and correct

## Configuration (config.yaml)

### Paths
```yaml
paths:
  source_directory: "/path/to/files"
  output_directory: "./classified_output/"
  database_path: "./data/metadata.db"
  samples_directory: "./samples/"
  embeddings_directory: "./embeddings/"
```

### Processing
```yaml
processing:
  batch_size: 100              # Files per batch
  parallel_workers: 4          # Multiprocessing workers
  video_sample_fps: 2          # Video frame sampling
  checkpoint_interval: 100     # Checkpoint frequency
  threshold_adjust_interval: 500  # Threshold adjustment
```

### Threshold
```yaml
threshold:
  initial: 0.75        # Starting confidence threshold
  min: 0.50           # Minimum threshold
  max: 0.95           # Maximum threshold
  step: 0.05          # Adjustment increment
  target_ratio_min: 0.10   # Target review ratio (min)
  target_ratio_max: 0.30   # Target review ratio (max)
```

## Database Schema

### files
```sql
file_id, original_path, file_type, status,
created_at, updated_at
```

### detections
```sql
detection_id, file_id, person_id, confidence,
bbox_x1, bbox_y1, bbox_x2, bbox_y2,
needs_review, created_at
```

### copies
```sql
copy_id, file_id, person_id, target_path, created_at
```

### checkpoints
```sql
checkpoint_id, batch_number, processed_files_count,
status, metadata, created_at
```

## Usage Examples

### Basic Usage
```python
from src.classifier import FaceClassifier

# Initialize
classifier = FaceClassifier(config_path="./config.yaml")

# Generate sample embeddings (first time)
classifier.generate_sample_embeddings()

# Process all files
results = classifier.process_all(resume=False)

# Resume from checkpoint
results = classifier.process_all(resume=True)
```

### Process Single File
```python
result = classifier.process_file("/path/to/image.jpg")
# result['detections']: List of face detections
# result['error']: None or error message
```

### Process Batch
```python
file_list = ["/path/to/file1.jpg", "/path/to/file2.jpg"]
results = classifier.process_batch(file_list)
# results['processed']: Number of files processed
# results['classifications']: Total detections
# results['errors']: Error count
```

### Get Statistics
```python
stats = classifier.database.get_statistics()
# stats['total_files']: Total files in database
# stats['needs_review']: Detections needing review
# stats['detections_by_person']: Count per person
```

## Performance Notes

- **Workers**: 4 workers by default, can increase to 8
- **Batch Size**: 100 files balances memory and throughput
- **Checkpoints**: Every 100 files (every batch)
- **Video Sampling**: 2fps significantly reduces processing time
- **GPU Acceleration**: CoreMLExecutionProvider for Metal support

## Error Handling

All functions include try-except blocks:
- File processing errors logged
- Database errors handled with rollback
- Checkpoint system prevents data loss
- Individual file failures don't stop batch

## Monitoring Progress

- **Progress bars**: tqdm shows batch and file progress
- **Logging**: All operations logged to file
- **Checkpoints**: Resume from interruption
- **Statistics**: Real-time stats via database.get_statistics()
