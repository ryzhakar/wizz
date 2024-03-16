import asyncio
from collections.abc import Callable
from functools import wraps

import typer


@wraps(typer.Typer.command)
def synchronize_async_command(
    app: typer.Typer,
    *decorator_args,
    **decorator_kwargs,
):
    """Decorator to synchronize async commands with the main event loop."""
    def _decorator(async_func: Callable):  # noqa: WPS430
        @wraps(async_func)
        def _sync_wrapper(*args, **kwargs):  # noqa: WPS430
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.ensure_future(async_func(*args, **kwargs))
            return loop.run_until_complete(async_func(*args, **kwargs))
        return app.command(*decorator_args, **decorator_kwargs)(_sync_wrapper)
    return _decorator
