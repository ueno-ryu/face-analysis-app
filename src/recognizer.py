"""
Face Recognition Module

Matches detected faces against known sample embeddings using cosine similarity.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of face matching."""
    person_id: Optional[int]
    confidence: float
    similarity: float
    needs_review: bool


class FaceRecognizer:
    """
    Face recognizer using cosine similarity on embeddings.
    """

    def __init__(self, embeddings_dir: str = "./embeddings/",
                 num_persons: int = 35):
        """
        Initialize the face recognizer.

        Args:
            embeddings_dir: Directory containing .npy embedding files
            num_persons: Number of persons to classify (1-35)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.num_persons = num_persons
        self.sample_embeddings: Dict[int, np.ndarray] = {}

        logger.info(f"Initializing FaceRecognizer with {num_persons} persons")
        logger.info(f"Embeddings directory: {embeddings_dir}")

    def load_sample_embeddings(self) -> bool:
        """
        Load sample embeddings for all persons from disk.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Loading sample embeddings...")

        for person_id in range(1, self.num_persons + 1):
            embedding_file = self.embeddings_dir / f"person_{person_id:02d}.npy"

            if not embedding_file.exists():
                logger.error(f"Embedding file not found: {embedding_file}")
                return False

            try:
                embeddings = np.load(embedding_file)
                self.sample_embeddings[person_id] = embeddings
                logger.debug(f"Loaded {len(embeddings)} embeddings for person_{person_id:02d}")
            except Exception as e:
                logger.error(f"Failed to load embeddings for person_{person_id:02d}: {e}")
                return False

        logger.info(f"Successfully loaded embeddings for {len(self.sample_embeddings)} persons")
        return True

    def match_face(self, embedding: np.ndarray,
                   threshold: float = 0.75) -> MatchResult:
        """
        Match a face embedding against all sample embeddings.

        Args:
            embedding: Face embedding vector
            threshold: Minimum similarity threshold for classification

        Returns:
            MatchResult with person_id, confidence, and needs_review flag
        """
        if not self.sample_embeddings:
            logger.error("No sample embeddings loaded. Call load_sample_embeddings() first.")
            return MatchResult(
                person_id=None,
                confidence=0.0,
                similarity=0.0,
                needs_review=True
            )

        best_person_id = None
        best_similarity = 0.0

        # Compare with all person embeddings
        for person_id, sample_embeddings in self.sample_embeddings.items():
            # Calculate cosine similarity with all samples for this person
            similarities = cosine_similarity(
                embedding.reshape(1, -1),
                sample_embeddings
            )[0]

            # Take maximum similarity for this person
            max_similarity = np.max(similarities)

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_person_id = person_id

        # Determine if result needs review
        needs_review = best_similarity < threshold

        result = MatchResult(
            person_id=best_person_id,
            confidence=float(best_similarity),
            similarity=float(best_similarity),
            needs_review=needs_review
        )

        logger.debug(f"Match result: person_id={best_person_id}, "
                    f"similarity={best_similarity:.4f}, "
                    f"needs_review={needs_review}")

        return result

    def match_faces(self, embeddings: List[np.ndarray],
                    threshold: float = 0.75) -> List[MatchResult]:
        """
        Match multiple face embeddings.

        Args:
            embeddings: List of face embedding vectors
            threshold: Minimum similarity threshold

        Returns:
            List of MatchResult objects
        """
        results = []
        for embedding in embeddings:
            result = self.match_face(embedding, threshold)
            results.append(result)
        return results

    def get_top_k_matches(self, embedding: np.ndarray,
                         k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k matching persons for an embedding.

        Args:
            embedding: Face embedding vector
            k: Number of top matches to return

        Returns:
            List of (person_id, similarity) tuples, sorted by similarity
        """
        if not self.sample_embeddings:
            logger.error("No sample embeddings loaded")
            return []

        similarities = []

        for person_id, sample_embeddings in self.sample_embeddings.items():
            # Calculate max similarity for this person
            sim_scores = cosine_similarity(
                embedding.reshape(1, -1),
                sample_embeddings
            )[0]
            max_similarity = np.max(sim_scores)
            similarities.append((person_id, float(max_similarity)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def compute_similarity_matrix(self, embeddings1: np.ndarray,
                                  embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (n x d)
            embeddings2: Second set of embeddings (m x d)

        Returns:
            Similarity matrix (n x m)
        """
        return cosine_similarity(embeddings1, embeddings2)

    def validate_embeddings(self) -> Dict[int, bool]:
        """
        Validate that all embedding files exist and are valid.

        Returns:
            Dictionary mapping person_id to validity
        """
        validation_results = {}

        for person_id in range(1, self.num_persons + 1):
            embedding_file = self.embeddings_dir / f"person_{person_id:02d}.npy"

            if not embedding_file.exists():
                logger.error(f"Embedding file missing: {embedding_file}")
                validation_results[person_id] = False
                continue

            try:
                embeddings = np.load(embedding_file)
                is_valid = len(embeddings) > 0 and embeddings.shape[1] > 0
                validation_results[person_id] = is_valid

                if not is_valid:
                    logger.error(f"Invalid embeddings in {embedding_file}")

            except Exception as e:
                logger.error(f"Error validating {embedding_file}: {e}")
                validation_results[person_id] = False

        return validation_results


def test_recognizer():
    """Test the face recognizer."""
    logging.basicConfig(level=logging.DEBUG)

    recognizer = FaceRecognizer(
        embeddings_dir="./embeddings/",
        num_persons=35
    )

    # Test loading embeddings
    success = recognizer.load_sample_embeddings()
    if success:
        print("Successfully loaded sample embeddings")

        # Test with dummy embedding
        dummy_embedding = np.random.rand(512)
        result = recognizer.match_face(dummy_embedding, threshold=0.75)
        print(f"Match result: {result}")

        # Test top-k matches
        top_k = recognizer.get_top_k_matches(dummy_embedding, k=5)
        print(f"Top 5 matches: {top_k}")
    else:
        print("Failed to load embeddings")


if __name__ == "__main__":
    test_recognizer()
