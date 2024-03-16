from __future__ import annotations

from functools import lru_cache
from os import path
from typing import Self

import numpy as np
from aiorwlock import RWLock
from annoy import AnnoyIndex
from numpy import ndarray

from wizz.extraction import constants


class AnnoyIndexManager:
    """Manages access to hard disk-based Annoy indices."""

    def __init__(self, user_id: str):
        """Initialize a user-specific Annoy index."""
        self.user_id = user_id
        self.index = AnnoyIndex(
            constants.EMBEDDING_DIM,
            constants.ANNOY_METRIC,
        )
        self.lock = RWLock()
        self.refresh_index_connection()

    def refresh_index_connection(self) -> None:
        """Load the index memory-mapping from disk."""
        self.index.load(f'{self.user_id}.ann')

    @property
    def path(self) -> str:
        """Get the path to the user-specific Annoy index."""
        return path.join(
            constants.ANNOY_INDICES_STORE_PATH,
            f'{self.user_id}.ann',
        )


@lru_cache(maxsize=None)
def get_index_manager(user_id):
    """Get a shared index manager for a user."""
    return AnnoyIndexManager(user_id)


class AnnoyReader:
    """A context manager that allows read access to an Annoy index."""

    def __init__(self, user_id: str):
        """Initialize the index manager."""
        self.manager = get_index_manager(user_id)

    async def __aenter__(self) -> Self:
        """Acquire a read lock."""
        await self.manager.lock.reader_lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Release a read lock."""
        self.manager.lock.reader_lock.release()

    async def get_neighbours_for(
        self,
        *,
        vector: ndarray,
        n: int = 10,  # noqa: WPS111
    ) -> list[int]:
        """Get the indices of the n nearest neighbors to a vector."""
        return self.manager.index.get_nns_by_vector(vector, n)

    async def get_vector_by(self, index: int) -> ndarray:
        """Get the vector at a given index."""
        return np.array(
            self.manager.index.get_item_vector(index),
            dtype=constants.DTYPE,
        )


class AnnoyWriter:
    """A context manager that allows write access to an Annoy index."""

    def __init__(self, user_id: str):
        """Initialize the index manager."""
        self.manager = get_index_manager(user_id)

    async def __aenter__(self) -> Self:
        """Acquire a write lock."""
        await self.manager.lock.writer_lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Release a write lock and refresh the index memory-mapping."""
        self.manager.index.on_disk_build(self.manager.path)
        self.manager.refresh_index_connection()
        self.manager.lock.writer_lock.release()

    async def add_item(self, index: int, vector: ndarray) -> None:
        """Add an item to the index."""
        self.manager.index.add_item(index, vector)
