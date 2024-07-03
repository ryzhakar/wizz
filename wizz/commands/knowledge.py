import asyncio
from logging import getLogger

import typer
from async_annoy import AsyncAnnoy
from dotenv import load_dotenv
from rich import print as rich_print
from rich import prompt as rich_prompt
from rich.progress import Progress

from wizz import crud
from wizz.agent.retriever import Retriever
from wizz.database import get_db_session
from wizz.extraction import converters
from wizz.extraction.batcher import TextBatcher
from wizz.extraction.embedder import Embedder
from wizz.extraction.outlier_finder import find_outliers_for
from wizz.filesystem import get_file_streamer
from wizz.filesystem import shorten_filename
from wizz.models import knowledge as knowledge_models
from wizz.syncer import synchronize_async_command

load_dotenv()

logger = getLogger('wizz')

app = typer.Typer(invoke_without_command=False)


@synchronize_async_command(app)
async def load(  # noqa: WPS210, WPS213, WPS217
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
                source_vector_hex = converters.vector_to_hex(
                    embedder(filecontent),
                )
                source_instance = await crud.create_source(
                    session,
                    context=context_instance,
                    name=filename,
                    content_hash=hashstr,
                    vector_hex=source_vector_hex,
                    commit=False,  # type: ignore
                )
                await session.commit()
                for ix, textblob in TextBatcher(filecontent):
                    blob_vector_hex = converters.vector_to_hex(
                        embedder(textblob),
                    )
                    await crud.create_blob(
                        session,
                        source=source_instance,
                        text=textblob,
                        index=ix,
                        vector_hex=blob_vector_hex,
                        commit=False,  # type: ignore
                    )
                await session.commit()
                progress.update(blob_task, visible=False)
                progress.update(file_task, advance=1)
    rich_print(
        'Skipped {skipped} already loaded files.'.format(
            skipped=number_of_existing_files,
        ),
        'Loaded {delta} new files into the knowledge base.'.format(
            delta=number_of_files - number_of_existing_files,
        ),
        sep='\n',
    )


@synchronize_async_command(app)
async def index(  # noqa: WPS210, WPS213, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the knowledge context.',
    ),
) -> None:
    """Add semantic coordinates to indices and build links."""
    with Progress(transient=True, refresh_per_second=2) as progress:
        async with get_db_session() as session:
            context_instance = await crud.get_or_create_context(
                session,
                name=context_name,
            )

            # Index sources
            total_sources = await crud.count_objects(
                session,
                model=knowledge_models.Source,
            )
            source_indexing_task = progress.add_task(
                'Indexing sources...',
                total=total_sources,
            )
            source_index_name = converters.to_source_ix_name(context_name)
            async with AsyncAnnoy(source_index_name).writer() as swriter:
                source_stream = await crud.stream_sources(
                    session,
                    context=context_instance,
                )
                for source in source_stream:
                    await swriter.add_item(
                        source.id,
                        converters.hex_to_vector(source.vector_hex),
                    )
                    progress.update(source_indexing_task, advance=1)
                rich_print(f'Indexed {total_sources} sources.')

            # Index blobs
            total_blobs = await crud.count_objects(
                session,
                model=knowledge_models.Blob,
            )
            blob_indexing_task = progress.add_task(
                'Indexing blobs...',
                total=total_blobs,
            )
            blob_index_name = converters.to_blob_ix_name(context_name)
            async with AsyncAnnoy(blob_index_name).writer() as bwriter:
                blob_stream = await crud.stream_blobs(
                    session,
                    context=context_instance,
                )
                for blob in blob_stream:
                    await bwriter.add_item(
                        blob.id,
                        converters.hex_to_vector(blob.vector_hex),
                    )
                    progress.update(blob_indexing_task, advance=1)
                rich_print(f'Indexed {total_blobs} blobs.')

            # Find links
            rich_print('Flushing all existing links...')
            await crud.remove_all_links_for(
                session,
                context=context_instance,
            )
            rich_print('Finding semantic outliers for linking...')
            sources = await crud.stream_sources(
                session,
                context=context_instance,
            )
            outliers: dict[int, tuple] = {}
            for src in sources:
                outliers.update(await find_outliers_for(src))
            total_outliers = len(outliers)
            rich_print(f'Found {total_outliers} outliers.')
            linking_task = progress.add_task(
                'Linking outliers...',
                total=total_outliers,
            )
            async with AsyncAnnoy(source_index_name).reader() as source_reader:
                for blb, vector, origin_distance in outliers.values():
                    ranked_destinations = (
                        await source_reader.get_ranked_neighbours_for(
                            vector=vector,
                            n=1,
                        )
                    )
                    destination_id, destination_dist = ranked_destinations[0]
                    destination = await crud.get_single_object(
                        session,
                        model=knowledge_models.Source,
                        object_id=destination_id,
                    )
                    await crud.create_link(
                        session,
                        blob=blb,
                        target_source=destination,
                        origin_distance=origin_distance,
                        destination_distance=destination_dist,
                    )
                    progress.update(linking_task, advance=1)
                await session.commit()
                rich_print('Linked all outliers.')
    rich_print('Done!')


@synchronize_async_command(app)
async def search(  # noqa: WPS210, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the context to bind the knowledge to.',
    ),
):
    """Search the knowledge base for a query."""
    embedder = Embedder()
    retriever = Retriever()
    async with get_db_session() as session:
        blob_ix_name = converters.to_blob_ix_name(context_name)
        async with AsyncAnnoy(blob_ix_name).reader() as reader:
            while query := rich_prompt.Prompt.ask('Enter a query'):
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
                    retriever.wrap_result(source.name, blob.text)
                    for source, blob in zip(sources, multiple_blobs)
                ]
                rich_print(*ellipted_texts, sep='\n\n')
    rich_print('Goodbye!')


@synchronize_async_command(app)
async def interact(  # noqa: WPS210, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the context to bind the knowledge to.',
    ),
):
    """Interact with LLM that has access to the knowledge base."""
    retriever = Retriever()
    embedder = Embedder()
    async with get_db_session() as session:
        blob_ix_name = converters.to_blob_ix_name(context_name)
        async with AsyncAnnoy(blob_ix_name).reader() as reader:
            while query := rich_prompt.Prompt.ask('\n\n'):
                query = retriever.construct_query(query)
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
                    retriever.wrap_result(source.name, blob.text)
                    for source, blob in zip(sources, multiple_blobs)
                ]
                answer = retriever.request_answer_based_on(
                    *ellipted_texts,
                    query=query,
                )
                rich_print(answer, sep='\n\n')
    rich_print('Goodbye!')


@synchronize_async_command(app)
async def delete(  # noqa: WPS210, WPS217
    context_name: str = typer.Option(  # noqa: WPS404, B008
        ...,
        help='The name of the context to bind the knowledge to.',
    ),
):
    """Delete a context and all its associated sources and blobs."""
    confirmed = rich_prompt.Confirm.ask(
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
