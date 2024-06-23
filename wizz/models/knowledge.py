from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from wizz.models.base import Base


class Context(Base):
    """A space for shared knowledge."""

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        nullable=False,
        unique=True,
        index=True,
    )

    sources: Mapped[list['Source']] = relationship(
        'Source',
        back_populates='context',
        cascade='all, delete-orphan',
    )


class Source(Base):
    """A metadata about a source text."""

    id: Mapped[int] = mapped_column(primary_key=True)
    context_id: Mapped[int] = mapped_column(
        ForeignKey('context.id', ondelete='CASCADE'),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(nullable=False)
    hash: Mapped[str] = mapped_column(nullable=False)
    vector_hex: Mapped[str] = mapped_column(nullable=False)

    context: Mapped['Context'] = relationship(
        'Context', back_populates='sources',
    )
    blobs: Mapped[list['Blob']] = relationship(
        'Blob',
        back_populates='source',
        cascade='all, delete-orphan',
    )
    links: Mapped[list['Link']] = relationship(
        'Link',
        back_populates='target_source',
        cascade='all, delete-orphan',
    )


class Blob(Base):
    """A vectorized source text section."""

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey('source.id', ondelete='CASCADE'),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(nullable=False)
    blob_index: Mapped[int] = mapped_column(nullable=False)
    vector_hex: Mapped[str] = mapped_column(nullable=False)

    source: Mapped['Source'] = relationship('Source', back_populates='blobs')
    links: Mapped[list['Link']] = relationship(
        'Link',
        back_populates='blob',
        cascade='all, delete-orphan',
    )


class Link(Base):
    """A blob-based inter-source relationship."""

    id: Mapped[int] = mapped_column(primary_key=True)
    blob_id: Mapped[int] = mapped_column(
        ForeignKey('blob.id', ondelete='CASCADE'),
        nullable=False,
    )
    target_source_id: Mapped[int] = mapped_column(
        ForeignKey('source.id', ondelete='CASCADE'),
        nullable=False,
    )
    origin_distance: Mapped[float] = mapped_column(nullable=False)
    destination_distance: Mapped[float] = mapped_column(nullable=False)

    blob: Mapped['Blob'] = relationship('Blob', back_populates='links')
    target_source: Mapped['Source'] = relationship(
        'Source', back_populates='links',
    )
