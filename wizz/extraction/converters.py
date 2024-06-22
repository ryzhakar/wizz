from numpy import float32
from numpy import frombuffer
from numpy import ndarray


def vector_to_hex(vector: ndarray) -> str:
    """Converts a numpy array to a hexadecimal string."""
    return vector.tobytes().hex()


def hex_to_vector(
    hex_str: str,
    dtype: type = float32,
) -> ndarray:
    """Converts a hexadecimal string to a numpy array."""
    return frombuffer(bytes.fromhex(hex_str), dtype=dtype)


def to_blob_ix_name(context_name: str) -> str:
    """Converts a context name to a blob index name."""
    return f'{context_name}_blobs'


def to_source_ix_name(context_name: str) -> str:
    """Converts a context name to a source index name."""
    return f'{context_name}_source'
