import math

import numpy as np

PHI = (1 + math.sqrt(5)) / 2

EMBEDDING_CACHE_PATH = 'embeddings_cache'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

ANNOY_METRIC = 'angular'
ANNOY_INDICES_STORE_PATH = 'annoy_indices'

DTYPE = np.float32


CHUNK_SIZE = 300
CHUNK_OVERLAP = int(CHUNK_SIZE // PHI ** 6)
