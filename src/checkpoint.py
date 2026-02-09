"""
Checkpoint Module

Manages processing checkpoints for resumable execution.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

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


if __name__ == "__main__":
    test_checkpoint()
