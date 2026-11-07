"""Điểm vào CLI (Typer): index / ask / db."""
import logging

import typer
from rich.markdown import Markdown
from rich.markup import escape

from rag_cli.cli.commands import CommandContext, dispatch, run_db
from rag_cli.cli.render import console

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context,
         verbose: bool = typer.Option(False, "--verbose", "-v",
                                      help="In thời gian từng bước và thông tin debug")):
    """Không truyền lệnh con -> mặc định vào khung chat chính (ask)."""
    logging.basicConfig(format="%(message)s")
    logging.getLogger("rag_cli").setLevel(logging.DEBUG if verbose else logging.WARNING)
    if ctx.invoked_subcommand is None:
        ctx.invoke(ask)


@app.command()
def index(full: bool = False):
    """Lập chỉ mục repo (mặc định chỉ file đã đổi; --full để làm lại tất cả)."""
    from rag_cli.ingest.indexer import index_repo
    stats = index_repo(full, report=lambda m: console.print(m, markup=False))
    console.print(f"Xong: {stats['indexed']} file mới/đổi, {stats['unchanged']} không đổi, "
                  f"{stats['removed']} đã xóa, {stats['commits']} commit.")


@app.command()
def db():
    """Khung chat riêng cho DB (giữ lại để dùng độc lập)."""
    while True:
        try:
            q = console.input("\n[bold green]🗄 Hỏi DB:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q in {"exit", "quit"}:
            break
        answer, _ = run_db(q)
        console.print(Markdown(answer))


@app.command()
def ask():
    """Khung chat chính: hỏi về codebase; /db, /g, /help là lệnh."""
    try:
        with console.status("[yellow]Đang nạp chỉ mục..."):
            from rag_cli.chat.pipeline import build_pipeline
            pipeline = build_pipeline()
    except Exception as e:
        console.print(f"[red]Không khởi động được: {escape(str(e))}[/red]")
        raise typer.Exit(1)

    ctx = CommandContext(history=pipeline.history)
    console.print("[dim]Gõ /help để xem lệnh. /db <câu hỏi> để truy vấn database.[/dim]")
    while True:
        try:
            q = console.input("\n[bold cyan]❓ Hỏi:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q in {"exit", "quit"}:
            break
        if dispatch(q, ctx):
            continue
        try:
            with console.status("[yellow]Đang tìm và trả lời..."):
                answer = pipeline.ask(q)
        except Exception as e:  # lỗi mạng/Gemini không làm thoát khung chat
            console.print(f"[red]Lỗi: {escape(str(e))}[/red]")
            continue
        console.print(Markdown(answer))
