"""
Database Module

Handles SQLite database operations for storing metadata about
processed files, detections, and copies.
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database for face classification metadata.
    """

    def __init__(self, db_path: str = "./data/metadata.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing database at: {db_path}")

        # Initialize database schema
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def init_db(self):
        """Initialize database schema."""
        logger.info("Initializing database schema...")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_path TEXT UNIQUE NOT NULL,
                    file_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Detections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    person_id INTEGER,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    needs_review BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(file_id)
                )
            """)

            # Copies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS copies (
                    copy_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    target_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(file_id)
                )
            """)

            # Checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_number INTEGER NOT NULL,
                    processed_files_count INTEGER NOT NULL,
                    status TEXT DEFAULT 'in_progress',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Processed files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_id INTEGER PRIMARY KEY,
                    worker_id INTEGER,
                    status TEXT,
                    processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(file_id)
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_status
                ON files(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_person
                ON detections(person_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_needs_review
                ON detections(needs_review)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_copies_person
                ON copies(person_id)
            """)

            logger.info("Database schema initialized successfully")

    def save_file(self, original_path: str, file_type: str,
                  status: str = "pending") -> int:
        """
        Save file record to database.

        Args:
            original_path: Original file path
            file_type: File type (image/video)
            status: Processing status

        Returns:
            file_id
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO files (original_path, file_type, status)
                    VALUES (?, ?, ?)
                """, (original_path, file_type, status))

                file_id = cursor.lastrowid
                logger.debug(f"Saved file record: {file_id} - {original_path}")
                return file_id

            except sqlite3.IntegrityError:
                # File already exists, update status
                cursor.execute("""
                    UPDATE files
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE original_path = ?
                """, (status, original_path))

                cursor.execute("""
                    SELECT file_id FROM files WHERE original_path = ?
                """, (original_path,))

                row = cursor.fetchone()
                file_id = row["file_id"] if row else None
                logger.debug(f"Updated file record: {file_id} - {original_path}")
                return file_id

    def save_detection(self, file_id: int, person_id: Optional[int],
                      confidence: float, bbox: Tuple[int, int, int, int],
                      needs_review: bool = False) -> int:
        """
        Save face detection result to database.

        Args:
            file_id: File ID
            person_id: Matched person ID (None if unknown)
            confidence: Confidence score
            bbox: Bounding box (x1, y1, x2, y2)
            needs_review: Whether this detection needs manual review

        Returns:
            detection_id
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO detections
                (file_id, person_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, needs_review)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, person_id, confidence, bbox[0], bbox[1], bbox[2], bbox[3],
                  int(needs_review)))

            detection_id = cursor.lastrowid
            logger.debug(f"Saved detection: {detection_id} - person_id={person_id}")
            return detection_id

    def save_detections(self, file_id: int, detections: List[Dict]) -> None:
        """
        Save multiple face detections for a file.

        Args:
            file_id: File ID
            detections: List of detection dictionaries with keys:
                       - person_id: Matched person ID
                       - confidence: Confidence score
                       - needs_review: Whether this detection needs manual review
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            for detection in detections:
                cursor.execute("""
                    INSERT INTO detections
                    (file_id, person_id, confidence, needs_review)
                    VALUES (?, ?, ?, ?)
                """, (
                    file_id,
                    detection["person_id"],
                    detection["confidence"],
                    int(detection.get("needs_review", False))
                ))

            logger.debug(f"Saved {len(detections)} detections for file_id={file_id}")

    def save_copy(self, file_id: int, person_id: int, target_path: str) -> int:
        """
        Save copy record to database.

        Args:
            file_id: File ID
            person_id: Target person ID
            target_path: Target file path

        Returns:
            copy_id
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO copies (file_id, person_id, target_path)
                VALUES (?, ?, ?)
            """, (file_id, person_id, target_path))

            copy_id = cursor.lastrowid
            logger.debug(f"Saved copy: {copy_id} - {target_path}")
            return copy_id

    def update_file_status(self, file_id: int, status: str, num_faces: int = None,
                          person_ids: List[int] = None):
        """
        Update file processing status.

        Args:
            file_id: File ID
            status: New status
            num_faces: Number of faces detected (optional)
            person_ids: List of matched person IDs (optional)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE files
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_id = ?
            """, (status, file_id))

            logger.debug(f"Updated file {file_id} status to {status}")
            if num_faces is not None:
                logger.debug(f"  - Faces detected: {num_faces}")
            if person_ids is not None:
                logger.debug(f"  - Matched persons: {person_ids}")

    def get_files_by_status(self, status: str) -> List[Dict]:
        """
        Get all files with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of file records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM files WHERE status = ?
            """, (status,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_detections_needing_review(self) -> List[Dict]:
        """
        Get all detections that need manual review.

        Returns:
            List of detection records with file paths
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT d.*, f.original_path
                FROM detections d
                JOIN files f ON d.file_id = f.file_id
                WHERE d.needs_review = 1
                ORDER BY d.created_at DESC
            """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_checkpoint(self) -> Optional[Dict]:
        """
        Get the latest checkpoint.

        Returns:
            Checkpoint data or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM checkpoints
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            return dict(row) if row else None

    def save_checkpoint(self, batch_number: int,
                       processed_files_count: int,
                       metadata: str = None) -> int:
        """
        Save processing checkpoint.

        Args:
            batch_number: Current batch number
            processed_files_count: Total files processed
            metadata: Additional metadata as JSON string

        Returns:
            checkpoint_id
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO checkpoints (batch_number, processed_files_count, metadata)
                VALUES (?, ?, ?)
            """, (batch_number, processed_files_count, metadata))

            checkpoint_id = cursor.lastrowid
            logger.info(f"Saved checkpoint: batch={batch_number}, "
                       f"processed={processed_files_count}")
            return checkpoint_id

    def get_statistics(self) -> Dict:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total files
            cursor.execute("SELECT COUNT(*) as count FROM files")
            total_files = cursor.fetchone()["count"]

            # Files by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM files
                GROUP BY status
            """)
            files_by_status = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Total detections
            cursor.execute("SELECT COUNT(*) as count FROM detections")
            total_detections = cursor.fetchone()["count"]

            # Detections by person
            cursor.execute("""
                SELECT person_id, COUNT(*) as count
                FROM detections
                WHERE person_id IS NOT NULL
                GROUP BY person_id
                ORDER BY count DESC
            """)
            detections_by_person = {row["person_id"]: row["count"]
                                   for row in cursor.fetchall()}

            # Review queue
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM detections
                WHERE needs_review = 1
            """)
            needs_review = cursor.fetchone()["count"]

            return {
                "total_files": total_files,
                "files_by_status": files_by_status,
                "total_detections": total_detections,
                "detections_by_person": detections_by_person,
                "needs_review": needs_review
            }

    def get_person_files(self, person_id: int) -> List[Dict]:
        """
        Get all files classified for a specific person.

        Args:
            person_id: Person ID

        Returns:
            List of file records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT f.*
                FROM files f
                JOIN detections d ON f.file_id = d.file_id
                WHERE d.person_id = ?
                ORDER BY f.original_path
            """, (person_id,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]


def test_database():
    """Test database functionality."""

    db = DatabaseManager("./data/test_metadata.db")

    # Test saving a file
    file_id = db.save_file("/test/path/image.jpg", "image", "pending")
    print(f"Saved file with ID: {file_id}")

    # Test saving a detection
    detection_id = db.save_detection(
        file_id=file_id,
        person_id=1,
        confidence=0.85,
        bbox=(100, 100, 200, 200),
        needs_review=False
    )
    print(f"Saved detection with ID: {detection_id}")

    # Test statistics
    stats = db.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    test_database()
