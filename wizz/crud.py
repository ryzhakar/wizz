from collections.abc import Iterable
from functools import wraps

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from wizz.models import knowledge


def optional_commit(func):
    """Decorator to optionally commit the session after the function call."""
    @wraps(func)
    async def wrapper(
        *args,
        commit: bool = True,
        **kwargs,
    ):
        original_result = await func(*args, **kwargs)
        if commit:
            await args[0].commit()
        return original_result
    return wrapper


@optional_commit
async def get_or_create_context(
    session: AsyncSession,
    *,
    name: str,
) -> knowledge.Context:
    """Get an existing Context by its name or create a new one."""
    try:
        return (
            await session.execute(
                select(knowledge.Context).filter_by(name=name),
            )
        ).scalar_one()
    except NoResultFound:
        context_instance = knowledge.Context(name=name)
        session.add(context_instance)
    return context_instance


@optional_commit
async def cascade_delete_context(
    session: AsyncSession,
    *,
    context: knowledge.Context,
) -> None:
    """Delete a Context and all its associated Sources and Blobs."""
    await session.execute(
        knowledge.Blob.__table__.delete().where(
            knowledge.Blob.source_id.in_(
                select(knowledge.Source.id).filter_by(context=context),
            ),
        ),
    )
    await session.execute(
        knowledge.Source.__table__.delete().where(
            knowledge.Source.context == context,
        ),
    )
    await session.execute(
        knowledge.Context.__table__.delete().where(
            knowledge.Context.id == context.id,
        ),
    )


@optional_commit
async def create_source(
    session: AsyncSession,
    *,
    context: knowledge.Context,
    name: str,
    content_hash: str,
) -> knowledge.Source:
    """Create a new Source."""
    source_instance = knowledge.Source(
        context=context,
        name=name,
        hash=content_hash,
    )
    session.add(source_instance)
    return source_instance


async def stream_blobs(
    session: AsyncSession,
    *,
    context: knowledge.Context,
) -> Iterable[knowledge.Blob]:
    """List all Blobs in a given Context."""
    query_result = await session.execute(
        select(
            knowledge.Blob,
        ).join(
            knowledge.Source,
            knowledge.Blob.source_id == knowledge.Source.id,
        ).join(
            knowledge.Context,
            knowledge.Source.context_id == knowledge.Context.id,
        ).filter(
            knowledge.Context.id == context.id,
        ),
    )
    return query_result.scalars()


async def load_set_of_blobs(
    session: AsyncSession,
    *,
    blob_ids: set[int],
) -> list[knowledge.Blob]:
    """Load a set of Blobs by their IDs."""
    query_result = await session.execute(
        select(knowledge.Blob).filter(knowledge.Blob.id.in_(blob_ids)),
    )
    return query_result.scalars().all()


async def does_source_exist(
    session: AsyncSession,
    *,
    hashstring: str,
) -> bool:
    """Check if a Source with the given hash exists."""
    query_result = await session.execute(
        select(knowledge.Source).filter_by(hash=hashstring),
    )
    return query_result.scalar() is not None


@optional_commit
async def create_blob(
    session: AsyncSession,
    *,
    source: knowledge.Source,
    text: str,
    index: int,
    vector_hex: str,
) -> knowledge.Blob:
    """Create a new Blob record."""
    blob_instance = knowledge.Blob(
        source=source,
        text=text,
        blob_index=index,
        vector_hex=vector_hex,
    )
    session.add(blob_instance)
    return blob_instance
