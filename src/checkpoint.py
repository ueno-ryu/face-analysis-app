"""
Checkpoint Module

Manages processing checkpoints for resumable execution using database backend.
Supports both file-based (legacy) and database-based checkpointing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from src.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Represents a processing checkpoint state."""
    batch_number: int
    processed_files_count: int
    total_files: int
    current_threshold: float
    last_processed_file: Optional[str]
    timestamp: str
    status: str  # in_progress, completed, error

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointState':
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages checkpoint files for resumable processing.
    Legacy file-based implementation for backward compatibility.
    """

    def __init__(self, checkpoint_dir: str = "./data/"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"

        logger.info(f"CheckpointManager initialized: {self.checkpoint_file}")

    def save_checkpoint(self, state: CheckpointState):
        """
        Save checkpoint state to file.

        Args:
            state: Checkpoint state to save
        """
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)

            logger.info(f"Checkpoint saved: batch={state.batch_number}, "
                       f"processed={state.processed_files_count}/{state.total_files}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self) -> Optional[CheckpointState]:
        """
        Load checkpoint state from file.

        Returns:
            Checkpoint state or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found")
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)

            state = CheckpointState.from_dict(data)
            logger.info(f"Checkpoint loaded: batch={state.batch_number}, "
                       f"processed={state.processed_files_count}/{state.total_files}, "
                       f"status={state.status}")

            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def has_checkpoint(self) -> bool:
        """
        Check if a checkpoint exists.

        Returns:
            True if checkpoint file exists
        """
        return self.checkpoint_file.exists()

    def delete_checkpoint(self):
        """Delete checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Checkpoint deleted")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
                raise

    def create_checkpoint(self,
                         batch_number: int,
                         processed_files_count: int,
                         total_files: int,
                         current_threshold: float,
                         last_processed_file: Optional[str] = None,
                         status: str = "in_progress") -> CheckpointState:
        """
        Create and save a checkpoint.

        Args:
            batch_number: Current batch number
            processed_files_count: Number of files processed so far
            total_files: Total number of files to process
            current_threshold: Current confidence threshold
            last_processed_file: Last successfully processed file
            status: Processing status

        Returns:
            Checkpoint state
        """
        state = CheckpointState(
            batch_number=batch_number,
            processed_files_count=processed_files_count,
            total_files=total_files,
            current_threshold=current_threshold,
            last_processed_file=last_processed_file,
            timestamp=datetime.now().isoformat(),
            status=status
        )

        self.save_checkpoint(state)
        return state

    def resume_from_checkpoint(self) -> Optional[CheckpointState]:
        """
        Resume processing from checkpoint.

        Returns:
            Checkpoint state or None if no valid checkpoint
        """
        state = self.load_checkpoint()

        if state is None:
            return None

        # Check if checkpoint is valid
        if state.status == "completed":
            logger.warning("Checkpoint shows processing already completed")
            return None

        if state.processed_files_count >= state.total_files:
            logger.warning("Checkpoint shows all files already processed")
            return None

        return state

    def mark_completed(self):
        """Mark processing as completed in checkpoint."""
        if self.has_checkpoint():
            state = self.load_checkpoint()
            if state:
                state.status = "completed"
                state.timestamp = datetime.now().isoformat()
                self.save_checkpoint(state)
                logger.info("Checkpoint marked as completed")

    def get_progress_percentage(self, state: CheckpointState) -> float:
        """
        Calculate progress percentage from checkpoint state.

        Args:
            state: Checkpoint state

        Returns:
            Progress percentage (0-100)
        """
        if state.total_files == 0:
            return 0.0

        return (state.processed_files_count / state.total_files) * 100

    def get_remaining_files(self, state: CheckpointState,
                           all_files: List[str]) -> List[str]:
        """
        Get list of remaining files to process based on checkpoint.

        Args:
            state: Checkpoint state
            all_files: List of all file paths

        Returns:
            List of remaining file paths
        """
        if state.last_processed_file is None:
            return all_files

        try:
            # Find index of last processed file
            last_index = all_files.index(state.last_processed_file)
            # Return files after that index
            return all_files[last_index + 1:]
        except ValueError:
            # Last processed file not in list, return all files
            logger.warning(f"Last processed file not found: {state.last_processed_file}")
            return all_files


class DatabaseCheckpointManager:
    """
    Database-backed checkpoint manager for resumable processing.
    Integrates with DatabaseManager for persistent checkpoint storage.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize database checkpoint manager.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        logger.info("DatabaseCheckpointManager initialized")

    def save_checkpoint(self, batch_number: int, processed_count: int,
                       status: str = 'in_progress') -> int:
        """
        Save processing checkpoint to database.

        Args:
            batch_number: Current batch number
            processed_count: Number of files processed so far
            status: Checkpoint status (in_progress, completed, error)

        Returns:
            checkpoint_id
        """
        try:
            # Convert state to JSON metadata
            metadata = json.dumps({
                'batch_number': batch_number,
                'processed_files_count': processed_count,
                'status': status,
                'timestamp': datetime.now().isoformat()
            })

            checkpoint_id = self.db.save_checkpoint(
                batch_number=batch_number,
                processed_files_count=processed_count,
                metadata=metadata
            )

            logger.info(f"Database checkpoint saved: batch={batch_number}, "
                       f"processed={processed_count}, status={status}")

            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save database checkpoint: {e}")
            raise

    def load_checkpoint(self) -> Optional[Dict]:
        """
        Get last completed checkpoint from database.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        try:
            checkpoint = self.db.get_checkpoint()

            if checkpoint is None:
                logger.info("No checkpoint found in database")
                return None

            # Parse metadata if available
            if checkpoint.get('metadata'):
                try:
                    metadata = json.loads(checkpoint['metadata'])
                    checkpoint['metadata_parsed'] = metadata
                except json.JSONDecodeError:
                    logger.warning("Failed to parse checkpoint metadata")

            logger.info(f"Database checkpoint loaded: batch={checkpoint['batch_number']}, "
                       f"processed={checkpoint['processed_files_count']}, "
                       f"status={checkpoint.get('status', 'unknown')}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load database checkpoint: {e}")
            return None

    def get_remaining_files(self, checkpoint_batch: int) -> List[Dict]:
        """
        Return files not yet processed since the checkpoint.

        Args:
            checkpoint_batch: Batch number from checkpoint

        Returns:
            List of file records that haven't been processed
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get files that are still pending or haven't been processed
                # after the checkpoint batch
                cursor.execute("""
                    SELECT f.*
                    FROM files f
                    WHERE f.status = 'pending'
                       OR f.file_id > (
                           SELECT COALESCE(MAX(cp.processed_files_count), 0)
                           FROM checkpoints cp
                           WHERE cp.batch_number <= ?
                       )
                    ORDER BY f.file_id
                """, (checkpoint_batch,))

                rows = cursor.fetchall()
                remaining_files = [dict(row) for row in rows]

                logger.info(f"Found {len(remaining_files)} remaining files "
                           f"from checkpoint batch {checkpoint_batch}")

                return remaining_files

        except Exception as e:
            logger.error(f"Failed to get remaining files: {e}")
            return []

    def update_processed_files(self, file_id: int, worker_id: int,
                              status: str = 'processed') -> bool:
        """
        Update processed file status and record worker.

        Args:
            file_id: File ID to update
            worker_id: Worker process ID
            status: Processing status

        Returns:
            True if successful
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Update file status
                cursor.execute("""
                    UPDATE files
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE file_id = ?
                """, (status, file_id))

                # Check if processed_files record exists
                cursor.execute("""
                    SELECT file_id FROM processed_files WHERE file_id = ?
                """, (file_id,))

                if cursor.fetchone():
                    # Update existing record
                    cursor.execute("""
                        UPDATE processed_files
                        SET worker_id = ?, status = ?,
                            processed_timestamp = CURRENT_TIMESTAMP
                        WHERE file_id = ?
                    """, (worker_id, status, file_id))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO processed_files (file_id, worker_id, status)
                        VALUES (?, ?, ?)
                    """, (file_id, worker_id, status))

                logger.debug(f"Updated processed file: file_id={file_id}, "
                            f"worker_id={worker_id}, status={status}")

                return True

        except Exception as e:
            logger.error(f"Failed to update processed file: {e}")
            return False

    def get_checkpoint_with_files(self) -> Optional[Tuple[Dict, List[Dict]]]:
        """
        Get checkpoint with remaining files to process.

        Returns:
            Tuple of (checkpoint_data, remaining_files) or None
        """
        checkpoint = self.load_checkpoint()

        if checkpoint is None:
            return None

        checkpoint_batch = checkpoint['batch_number']
        remaining_files = self.get_remaining_files(checkpoint_batch)

        return checkpoint, remaining_files

    def mark_checkpoint_completed(self, checkpoint_id: Optional[int] = None):
        """
        Mark checkpoint as completed.

        Args:
            checkpoint_id: Checkpoint ID (uses latest if None)
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if checkpoint_id:
                    cursor.execute("""
                        UPDATE checkpoints
                        SET status = 'completed'
                        WHERE checkpoint_id = ?
                    """, (checkpoint_id,))
                else:
                    # Mark latest checkpoint as completed
                    cursor.execute("""
                        UPDATE checkpoints
                        SET status = 'completed'
                        WHERE checkpoint_id = (
                            SELECT checkpoint_id FROM checkpoints
                            ORDER BY checkpoint_id DESC
                            LIMIT 1
                        )
                    """)

                logger.info("Checkpoint marked as completed")

        except Exception as e:
            logger.error(f"Failed to mark checkpoint completed: {e}")
            raise

    def get_statistics(self) -> Dict:
        """
        Get checkpoint statistics.

        Returns:
            Dictionary with checkpoint statistics
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get latest checkpoint
                cursor.execute("""
                    SELECT * FROM checkpoints
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()

                # Count processed files
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM processed_files
                    WHERE status = 'processed'
                """)
                processed_count = cursor.fetchone()['count']

                # Count pending files
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM files
                    WHERE status = 'pending'
                """)
                pending_count = cursor.fetchone()['count']

                return {
                    'latest_checkpoint': dict(latest) if latest else None,
                    'processed_files_count': processed_count,
                    'pending_files_count': pending_count
                }

        except Exception as e:
            logger.error(f"Failed to get checkpoint statistics: {e}")
            return {}


class CheckpointTracker:
    """
    Tracks file processing progress for checkpointing.
    """

    def __init__(self, checkpoint_manager: CheckpointManager,
                 total_files: int,
                 checkpoint_interval: int = 100):
        """
        Initialize checkpoint tracker.

        Args:
            checkpoint_manager: CheckpointManager instance
            total_files: Total number of files to process
            checkpoint_interval: Save checkpoint every N files
        """
        self.checkpoint_manager = checkpoint_manager
        self.total_files = total_files
        self.checkpoint_interval = checkpoint_interval

        self.processed_files = 0
        self.batch_number = 0
        self.current_threshold = 0.75
        self.last_processed_file = None

        logger.info(f"CheckpointTracker initialized: total={total_files}, "
                   f"interval={checkpoint_interval}")

    def update(self, file_path: str, current_threshold: float = None):
        """
        Update progress after processing a file.

        Args:
            file_path: Path to processed file
            current_threshold: Current confidence threshold
        """
        self.processed_files += 1
        self.last_processed_file = file_path

        if current_threshold is not None:
            self.current_threshold = current_threshold

        # Check if we should save a checkpoint
        if self.processed_files % self.checkpoint_interval == 0:
            self.batch_number += 1
            self.save()

    def save(self):
        """Save current checkpoint."""
        self.checkpoint_manager.create_checkpoint(
            batch_number=self.batch_number,
            processed_files_count=self.processed_files,
            total_files=self.total_files,
            current_threshold=self.current_threshold,
            last_processed_file=self.last_processed_file,
            status="in_progress"
        )

    def mark_completed(self):
        """Mark processing as completed."""
        self.checkpoint_manager.mark_completed()

    def get_progress(self) -> float:
        """
        Get current progress percentage.

        Returns:
            Progress (0-100)
        """
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    def should_save_checkpoint(self) -> bool:
        """
        Check if it's time to save a checkpoint.

        Returns:
            True if checkpoint should be saved
        """
        return self.processed_files % self.checkpoint_interval == 0


def test_checkpoint():
    """Test checkpoint functionality."""

    # Test file-based checkpoint
    print("Testing file-based checkpoint...")
    manager = CheckpointManager("./data/")

    # Create test checkpoint
    state = manager.create_checkpoint(
        batch_number=1,
        processed_files_count=100,
        total_files=17000,
        current_threshold=0.75,
        last_processed_file="/test/image_100.jpg",
        status="in_progress"
    )

    print(f"Created checkpoint: {state}")

    # Load checkpoint
    loaded_state = manager.load_checkpoint()
    print(f"Loaded checkpoint: {loaded_state}")

    # Get progress
    progress = manager.get_progress_percentage(loaded_state)
    print(f"Progress: {progress:.2f}%")

    # Test remaining files
    all_files = [f"/test/image_{i}.jpg" for i in range(1, 201)]
    remaining = manager.get_remaining_files(loaded_state, all_files)
    print(f"Remaining files: {len(remaining)}")

    # Cleanup
    manager.delete_checkpoint()


def test_database_checkpoint():
    """Test database checkpoint functionality."""
    from src.database import DatabaseManager

    print("\nTesting database-based checkpoint...")

    db = DatabaseManager("./data/test_checkpoint.db")
    manager = DatabaseCheckpointManager(db)

    # Save checkpoint
    checkpoint_id = manager.save_checkpoint(
        batch_number=1,
        processed_count=100,
        status='in_progress'
    )
    print(f"Saved checkpoint ID: {checkpoint_id}")

    # Load checkpoint
    checkpoint = manager.load_checkpoint()
    print(f"Loaded checkpoint: {checkpoint}")

    # Get statistics
    stats = manager.get_statistics()
    print(f"Statistics: {stats}")

    # Mark completed
    manager.mark_checkpoint_completed(checkpoint_id)
    print("Checkpoint marked as completed")


if __name__ == "__main__":
    test_checkpoint()
    test_database_checkpoint()
