from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

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
    context_id = Column(
        Integer,
        ForeignKey('context.id'),
        nullable=False,
    )
    context = relationship('Context', back_populates='sources')
    name = Column(String, nullable=False)
    hash = Column(String, nullable=False)


class Blob(Base):
    """A vectorized source text section."""
    source_id = Column(
        Integer,
        ForeignKey('source.id'),
        nullable=False,
    )
    source = relationship('Source', back_populates='blobs')
    text = Column(String, nullable=False)
    blob_index = Column(Integer, nullable=False)
    vector_hex = Column(String, nullable=False)


Context.sources = relationship(
    'Source', order_by=Source.id, back_populates='context',
)
Source.blobs = relationship('Blob', order_by=Blob.id, back_populates='source')
