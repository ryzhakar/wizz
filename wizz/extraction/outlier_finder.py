import numpy as np
from scipy.spatial.distance import cdist

from wizz.extraction import constants
from wizz.extraction import converters
from wizz.models import knowledge

_QUARTILES = (25, 75)


async def find_outliers_for(  # noqa: WPS210
    source: knowledge.Source,
) -> dict[int, tuple[knowledge.Blob, np.ndarray, float]]:
    """Find outlier sections against the whole document.

    Can find no outliers in some cases, which is expected.
    """
    ofinder = OutlierFinder()
    blobmap = {
        blob.id: blob
        for blob in await source.awaitable_attrs.blobs
    }
    blob_embeddings = {
        blob_id: converters.hex_to_vector(blob.vector_hex)
        for blob_id, blob in blobmap.items()
    }
    outliers = ofinder(
        converters.hex_to_vector(source.vector_hex),
        blob_embeddings,
    )
    return {
        blob_id: (
            blobmap[blob_id],
            blob_embeddings[blob_id],
            distance,
        )
        for blob_id, distance in outliers.items()
    }


class OutlierFinder:
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
