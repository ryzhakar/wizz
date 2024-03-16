from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

from wizz.models.base import Base


class Context(Base):
    """A space for shared knowledge."""
    name = Column(
        String,
        nullable=False,
        unique=True,
        index=True,
    )


class Source(Base):
    """A metadata about a source text."""
    context = ForeignKey('context.id')
    name = Column(String, nullable=False)
    hash = Column(String, nullable=False)


class Blob(Base):
    """A vectorized source text section."""
    source = ForeignKey('source.id')
    text = Column(String, nullable=False)
    blob_index = Column(Integer, nullable=False)
    vector_hex = Column(String, nullable=False)
