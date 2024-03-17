from logging import getLogger

import typer

from wizz import crud
from wizz.database import get_db_session
from wizz.extraction import converters
from wizz.extraction.batcher import TextBatcher
from wizz.extraction.embedder import Embedder
from wizz.extraction.indexer import AnnoyWriter
from wizz.filesystem import stream_files_from
from wizz.syncer import synchronize_async_command

logger = getLogger('wizz')

app = typer.Typer(invoke_without_command=False)


@synchronize_async_command(app)
async def load(  # noqa: WPS210, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the context to bind the knowledge to.',
    ),
    load_path: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The path to the directory to load.',
        is_flag=True,
        readable=True,
        dir_okay=True,
    ),
) -> None:
    """Read a directory and load its contents into the knowledge base."""
    embedder = Embedder()
    async with get_db_session() as session:
        context_instance = await crud.get_or_create_context(
            session,
            name=context_name,
        )
        for filename, filecontent, hashstr in stream_files_from(load_path):
            if await crud.does_source_exist(session, hashstring=hashstr):
                logger.info(
                    f'Skipping {filename} because it is already loaded.',
                )
                continue
            logger.info(f'Loading {filename} into the knowledge base.')
            source_instance = await crud.create_source(
                session,
                context=context_instance,
                name=filename,
                content_hash=hashstr,
                commit=False,  # type: ignore
            )
            await session.commit()
            for ix, textblob in enumerate(TextBatcher(filecontent)):
                vector_hex = converters.vector_to_hex(embedder(textblob))
                await crud.create_blob(
                    session,
                    source=source_instance,
                    text=textblob,
                    index=ix,
                    vector_hex=vector_hex,
                    commit=False,  # type: ignore
                )
            await session.commit()
        async with AnnoyWriter(context_name) as writer:
            logger.info('Building the index.')
            blob_stream = await crud.stream_blobs(
                session,
                context=context_instance,
            )
            for blob in blob_stream:
                await writer.add_item(
                    blob.id,
                    converters.hex_to_vector(blob.vector_hex),
                )
