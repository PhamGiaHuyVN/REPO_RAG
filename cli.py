import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer()
console = Console()

@app.command()
def index(full: bool = False):
    from indexer import index_repo
    index_repo(full)

@app.command()
def ask():
    from chat import ask as ask_fn
    while True:
        q = console.input("\n[bold cyan]❓ Hỏi:[/bold cyan] ").strip()
        if not q:
            continue
        if q.strip() in {"exit", "quit"}:
            break
        with console.status("[yellow]Đang tìm và trả lời..."):
            answer = ask_fn(q)
        console.print(Markdown(answer))

if __name__ == "__main__":
    app()