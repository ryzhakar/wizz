import asyncio
from logging import getLogger

import typer
from rich import print as rich_print
from rich.progress import Progress
from rich.prompt import Confirm
from rich.prompt import Prompt

from wizz import crud
from wizz.database import get_db_session
from wizz.extraction import converters
from wizz.extraction.batcher import TextBatcher
from wizz.extraction.embedder import Embedder
from wizz.extraction.indexer import AnnoyReader
from wizz.extraction.indexer import AnnoyWriter
from wizz.filesystem import get_file_streamer
from wizz.filesystem import shorten_filename
from wizz.syncer import synchronize_async_command

logger = getLogger('wizz')

app = typer.Typer(invoke_without_command=False)


@synchronize_async_command(app)
async def delete(  # noqa: WPS210, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the context to bind the knowledge to.',
    ),
):
    """Delete a context and all its associated sources and blobs."""
    confirmed = Confirm.ask(
        f'This means deleting the knowledge base for {context_name}. '
        'Are you sure?',
    )
    if not confirmed:
        return rich_print('Aborted.')
    async with get_db_session() as session:
        context_instance = await crud.get_or_create_context(
            session,
            name=context_name,
        )
        await crud.cascade_delete_context(
            session,
            context=context_instance,
        )
        # No need to delete the Annoy index,
        # because it will be overwritten on the next load.


@synchronize_async_command(app)
async def search(  # noqa: WPS210, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the context to bind the knowledge to.',
    ),
):
    """Search the knowledge base for a query."""
    embedder = Embedder()
    async with get_db_session() as session:
        async with AnnoyReader(context_name) as reader:
            while query := Prompt.ask('Enter a query'):
                lookup_indices = await reader.get_neighbours_for(
                    vector=embedder(query),
                    n=5,
                )
                multiple_blobs = await crud.load_set_of_blobs(
                    session,
                    blob_ids=set(lookup_indices),
                )
                sources = await asyncio.gather(
                    *(blob.awaitable_attrs.source for blob in multiple_blobs),
                )
                ellipted_texts = [
                    f'{source.name}:\n"""...{blob.text}..."""'
                    for source, blob in zip(sources, multiple_blobs)
                ]
                rich_print(*ellipted_texts, sep='\n\n')


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
    number_of_files, file_stream = get_file_streamer(load_path)
    number_of_existing_files = 0
    with Progress(
        transient=True,
        refresh_per_second=2,
    ) as progress:
        file_task = progress.add_task(
            'Processing files...', total=number_of_files,
        )
        async with get_db_session() as session:
            context_instance = await crud.get_or_create_context(
                session,
                name=context_name,
            )
            for filename, filecontent, hashstr in file_stream:
                if await crud.does_source_exist(session, hashstring=hashstr):
                    progress.update(file_task, advance=1)
                    number_of_existing_files += 1
                    continue
                blob_task = progress.add_task(
                    shorten_filename(filename),
                    total=None,
                )
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
                progress.update(blob_task, visible=False)
                progress.update(file_task, advance=1)
            async with AnnoyWriter(context_name) as writer:
                blob_stream = await crud.stream_blobs(
                    session,
                    context=context_instance,
                )
                for blob in blob_stream:
                    await writer.add_item(
                        blob.id,
                        converters.hex_to_vector(blob.vector_hex),
                    )
    rich_print(
        'Skipped {skipped} already loaded files.'.format(
            skipped=number_of_existing_files,
        ),
        'Loaded {delta} new files into the knowledge base.'.format(
            delta=number_of_files - number_of_existing_files,
        ),
        sep='\n',
    )
