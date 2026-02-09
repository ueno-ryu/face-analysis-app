# Checkpoint System Implementation

## Overview

Implemented a comprehensive checkpoint system for resumable face analysis processing. The system supports both file-based (legacy) and database-backed checkpointing to match the handover documentation requirements.

## Database Schema

### checkpoints table
```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_number INTEGER NOT NULL,
    processed_files_count INTEGER NOT NULL,
    status TEXT DEFAULT 'in_progress',
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### processed_files table
```sql
CREATE TABLE IF NOT EXISTS processed_files (
    file_id INTEGER PRIMARY KEY,
    worker_id INTEGER,
    status TEXT,
    processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
)
```

## Implementation Components

### 1. CheckpointState (dataclass)
Represents a processing checkpoint state with:
- `batch_number`: Current batch number
- `processed_files_count`: Number of files processed
- `total_files`: Total files to process
- `current_threshold`: Current confidence threshold
- `last_processed_file`: Last successfully processed file path
- `timestamp`: ISO format timestamp
- `status`: Processing status (in_progress, completed, error)

### 2. CheckpointManager (File-based)
Legacy file-based checkpoint system for backward compatibility.

**Key Methods:**
- `save_checkpoint(state)`: Save checkpoint to JSON file
- `load_checkpoint()`: Load checkpoint from file
- `has_checkpoint()`: Check if checkpoint exists
- `create_checkpoint(...)`: Create and save new checkpoint
- `get_progress_percentage(state)`: Calculate progress (0-100%)
- `get_remaining_files(state, all_files)`: Get list of unprocessed files
- `resume_from_checkpoint()`: Resume from valid checkpoint
- `mark_completed()`: Mark processing as completed

**Usage:**
```python
from src.checkpoint import CheckpointManager

manager = CheckpointManager("./data/")
state = manager.create_checkpoint(
    batch_number=1,
    processed_files_count=100,
    total_files=17000,
    current_threshold=0.75,
    last_processed_file="/path/to/file.jpg",
    status="in_progress"
)
```

### 3. DatabaseCheckpointManager (Database-backed)
New database-backed checkpoint manager integrating with DatabaseManager.

**Key Methods:**
- `save_checkpoint(batch_number, processed_count, status)`: Save to database
- `load_checkpoint()`: Get last checkpoint from database
- `get_remaining_files(checkpoint_batch)`: Return files not yet processed
- `update_processed_files(file_id, worker_id, status)`: Track processed files
- `get_checkpoint_with_files()`: Get checkpoint with remaining files
- `mark_checkpoint_completed(checkpoint_id)`: Mark as completed
- `get_statistics()`: Get checkpoint statistics

**Usage:**
```python
from src.checkpoint import DatabaseCheckpointManager
from src.database import DatabaseManager

db = DatabaseManager("./data/metadata.db")
checkpoint_mgr = DatabaseCheckpointManager(db)

# Save checkpoint
checkpoint_id = checkpoint_mgr.save_checkpoint(
    batch_number=1,
    processed_count=100,
    status='in_progress'
)

# Update processed file
checkpoint_mgr.update_processed_files(
    file_id=42,
    worker_id=1,
    status='processed'
)

# Load checkpoint and get remaining files
checkpoint, remaining = checkpoint_mgr.get_checkpoint_with_files()
```

### 4. CheckpointTracker
Tracks file processing progress for automatic checkpointing.

**Key Methods:**
- `update(file_path, current_threshold)`: Update progress after processing
- `save()`: Manually save checkpoint
- `mark_completed()`: Mark processing complete
- `get_progress()`: Get current progress percentage
- `should_save_checkpoint()`: Check if checkpoint should be saved

**Usage:**
```python
from src.checkpoint import CheckpointTracker, CheckpointManager

manager = CheckpointManager("./data/")
tracker = CheckpointTracker(
    checkpoint_manager=manager,
    total_files=17000,
    checkpoint_interval=100
)

# After processing each file
tracker.update("/path/to/file.jpg")

# Automatically saves checkpoint every 100 files
```

## Resume Mode Support

The checkpoint system supports `python main.py --mode resume`:

```python
# In main.py mode_resume()
def mode_resume(config: dict):
    checkpoint_manager = CheckpointManager("./data/")

    if not checkpoint_manager.has_checkpoint():
        logger.error("No checkpoint found. Run --mode scan first.")
        sys.exit(1)

    # Resume processing
    mode_scan(config, resume=True)
```

## Test Results

All tests passed successfully:

### TEST 1: File-Based Checkpoint System ✓
- Checkpoint creation and loading
- Progress calculation
- Remaining files calculation
- Resume functionality

### TEST 2: Database Schema ✓
- checkpoints table creation
- processed_files table creation
- CRUD operations
- Foreign key constraints

### TEST 3: DatabaseCheckpointManager ✓
- save_checkpoint()
- load_checkpoint()
- update_processed_files()
- get_remaining_files()
- mark_checkpoint_completed()
- get_statistics()

### TEST 4: Real-World Scenarios ✓
- Initial processing (batch 1)
- Resume and continue (batch 2)
- Processing completion (100%)
- Partial batch interruption

## File Structure

```
src/
├── checkpoint.py          # Main checkpoint implementation
│   ├── CheckpointState          # Dataclass for checkpoint state
│   ├── CheckpointManager        # File-based checkpoint manager
│   ├── DatabaseCheckpointManager # Database-backed checkpoint manager
│   └── CheckpointTracker        # Progress tracking helper
│
└── database.py           # Database manager with checkpoint tables
    ├── init_db()                # Creates checkpoints and processed_files tables
    ├── save_checkpoint()         # Saves checkpoint to database
    └── get_checkpoint()          # Retrieves latest checkpoint
```

## Requirements Met

✓ `save_checkpoint(batch_number, processed_count, status)` - Save processing checkpoints
✓ `load_checkpoint()` - Get last completed checkpoint
✓ `get_remaining_files(checkpoint_batch)` - Return files not yet processed
✓ `update_processed_files(file_id, worker_id, status)` - Track processed files by worker
✓ Database schema matches handover documentation
✓ Supports: `python main.py --mode resume`

## Integration Points

The checkpoint system integrates with:

1. **FaceClassifier**: Uses CheckpointTracker to save progress during batch processing
2. **DatabaseManager**: Stores checkpoints in SQLite database
3. **main.py**: Supports `--mode resume` for resumable execution

## Usage Example

```python
# Initial processing
python main.py --mode scan

# If interrupted, resume from checkpoint
python main.py --mode resume
```

## Logging

All checkpoint operations are logged:

```
INFO - Checkpoint saved: batch=1, processed=100/17000
INFO - Checkpoint loaded: batch=1, processed=100/17000, status=in_progress
INFO - Database checkpoint saved: batch=1, processed=100, status=in_progress
INFO - Found 8500 remaining files from checkpoint batch 1
```

## Error Handling

- File I/O errors during checkpoint save/load are caught and logged
- Database errors are caught and logged with context
- Invalid checkpoints (completed, all files processed) are detected
- Missing checkpoint files are handled gracefully

## Performance Considerations

- Checkpoint interval can be configured (default: every 100 files)
- Database checkpoints use JSON metadata for flexibility
- File-based checkpoints use JSON for human readability
- Both systems support incremental progress tracking

## Future Enhancements

Possible improvements:
- Add checkpoint compression for large file lists
- Support for distributed processing checkpoints
- Checkpoint validation and recovery
- Automatic cleanup of old checkpoints
- Checkpoint export/import for migration
