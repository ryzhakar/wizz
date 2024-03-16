from typer import Typer

from wizz.syncer import synchronize_async_command

app = Typer()


@synchronize_async_command(app)
async def main():
    """Test the Typer app."""
    print('Hello, world!')  # noqa: WPS421


if __name__ == '__main__':
    app()
