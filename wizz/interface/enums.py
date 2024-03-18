from enum import auto
from enum import StrEnum


class MessageRole(StrEnum):
    """Chat message role."""
    user = auto()
    system = auto()
    assistant = auto()
