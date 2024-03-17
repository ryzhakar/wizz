from logging import getLogger
from typing import cast

from numpy import ndarray
from sentence_transformers import SentenceTransformer

from wizz.extraction.constants import EMBEDDING_CACHE_PATH
from wizz.extraction.constants import EMBEDDING_MODEL

logger = getLogger('wizz')


class Embedder(SentenceTransformer):
    """Embedding model with type and option overrides."""

    def __init__(self):
        """Initialize without options."""
        logger.info('Initializing embedder with model: %s', EMBEDDING_MODEL)
        super().__init__(
            EMBEDDING_MODEL,
            cache_folder=EMBEDDING_CACHE_PATH,
        )

    def __call__(self, section: str) -> ndarray:
        """Encode on call."""
        return self.encode(section)

    def encode(self, section: str) -> ndarray:
        """Only support string encodings into ndarrays."""
        logger.info('Encoding section of length: %s', len(section))
        return cast(ndarray, super().encode(section))
