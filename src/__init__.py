"""
Face Analysis App - Source Modules

This package contains all core modules for the face classification pipeline.
"""

__version__ = "1.0.0"
__author__ = "ueno-ryu"

from .detector import FaceDetector, Face
from .recognizer import FaceRecognizer, MatchResult
from .database import DatabaseManager
from .checkpoint import CheckpointManager, CheckpointTracker, CheckpointState
from .classifier import FaceClassifier
from .reviewer import FaceReviewerGUI, launch_reviewer

__all__ = [
    "FaceDetector",
    "Face",
    "FaceRecognizer",
    "MatchResult",
    "DatabaseManager",
    "CheckpointManager",
    "CheckpointTracker",
    "CheckpointState",
    "FaceClassifier",
    "FaceReviewerGUI",
    "launch_reviewer",
]
