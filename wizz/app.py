import logging

from typer import Typer

from wizz.commands import knowledge

# Set loglevel to DEBUG
logging.basicConfig(level=logging.INFO)

app = Typer()
app.add_typer(knowledge.app, name='knowledge')

if __name__ == '__main__':
    app()
