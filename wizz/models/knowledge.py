from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from wizz.models.base import Base


class Context(Base):
    """A space for shared knowledge."""
    name = Column(String, nullable=False, unique=True, index=True)

    sources = relationship('Source', back_populates='context')


class Source(Base):
    """A metadata about a source text."""
    context_id = Column(Integer, ForeignKey('context.id'), nullable=False)
    name = Column(String, nullable=False)
    hash = Column(String, nullable=False)
    vector_hex = Column(String, nullable=False)

    context = relationship('Context', back_populates='sources')
    blobs = relationship('Blob', back_populates='source')
    links = relationship('Link', back_populates='target_source')


class Blob(Base):
    """A vectorized source text section."""
    source_id = Column(Integer, ForeignKey('source.id'), nullable=False)
    text = Column(String, nullable=False)
    blob_index = Column(Integer, nullable=False)
    vector_hex = Column(String, nullable=False)

    source = relationship('Source', back_populates='blobs')
    links = relationship('Link', back_populates='blob')


class Link(Base):
    """A blob-based inter-source relationship."""
    blob_id = Column(Integer, ForeignKey('blob.id'), nullable=False)
    target_source_id = Column(Integer, ForeignKey('source.id'), nullable=False)
    origin_distance = Column(Float, nullable=False)
    destination_distance = Column(Float, nullable=False)

    blob = relationship('Blob', back_populates='links')
    target_source = relationship('Source', back_populates='links')
