from typer import Typer

from wizz.commands import knowledge


app = Typer()
app.add_typer(knowledge.app, name='knowledge')

if __name__ == '__main__':
    app()
