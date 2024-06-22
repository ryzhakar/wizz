import numpy as np
from scipy.spatial.distance import cdist

from wizz.extraction import constants

_QUARTILES = (25, 75)


class OutlierDetector:
    """Detect outlier sections based on distances to the document."""

    def __init__(self, iqr_multiplier: float = constants.PHI) -> None:
        """Initialize the detector with a configurable IQR multiplier."""
        self.iqr_multiplier = iqr_multiplier

    def __call__(
        self,
        document_embedding: np.ndarray,
        section_embeddings: dict[int, np.ndarray],
    ) -> dict[int, float]:
        """Detect outliers using the IQR method."""
        section_ids, embeddings = zip(*section_embeddings.items())
        distances = self._calculate_distances(document_embedding, embeddings)

        return {
            section_id: distance
            for section_id, distance in zip(section_ids, distances)
            if distance > self._calculate_threshold(distances)
        }

    def _calculate_distances(
        self,
        document_embedding: np.ndarray,
        section_embeddings: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Calculate cosine distances."""
        return cdist(
            [document_embedding],
            section_embeddings,
            metric='cosine',
        )[0]

    def _calculate_threshold(self, distances: np.ndarray) -> float:
        """Calculate the threshold for outlier detection."""
        first_quartile, third_quartile = np.percentile(distances, _QUARTILES)
        iqr = third_quartile - first_quartile
        return third_quartile + self.iqr_multiplier * iqr
