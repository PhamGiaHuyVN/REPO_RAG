import re
import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer()
console = Console()

# "/db câu hỏi", "/help"  -> là lệnh.   "/src/app.py làm gì" -> KHÔNG phải lệnh
CMD_RE = re.compile(r"^/([A-Za-z]+)(?:\s+(.*))?$", re.S)

HELP_TEXT = (
    "**Lệnh có sẵn**\n\n"
    "- `/db <câu hỏi>`: truy vấn database bằng ngôn ngữ tự nhiên\n"
    "- `/help`: hiện trợ giúp này\n"
    "- `exit` / `quit`: thoát\n\n"
    "Không có `/` ở đầu thì hỏi về codebase như bình thường."
)


def run_db(question: str) -> tuple[str, bool]:
    """Chạy truy vấn DB. Trả về (nội dung, thành công?)."""
    from db_query import ask_db  # import trễ: chỉ nạp khi thật sự dùng /db
    with console.status("[yellow]Đang truy vấn DB..."):
        try:
            return ask_db(question), True
        except Exception as e:
            return f"Lỗi: {e}", False


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Không truyền lệnh con -> mặc định vào khung chat chính (ask)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(ask)


@app.command()
def index(full: bool = False):
    from indexer import index_repo
    index_repo(full)


@app.command()
def db():
    """Khung chat riêng cho DB (giữ lại để dùng độc lập)."""
    while True:
        q = console.input("\n[bold green]🗄 Hỏi DB:[/bold green] ").strip()
        if not q:
            continue
        if q in {"exit", "quit"}:
            break
        answer, _ = run_db(q)
        console.print(Markdown(answer))


@app.command()
def ask():
    from chat import ask as ask_fn, remember
    console.print("[dim]Gõ /help để xem lệnh. /db <câu hỏi> để truy vấn database.[/dim]")
    while True:
        q = console.input("\n[bold cyan]❓ Hỏi:[/bold cyan] ").strip()
        if not q:
            continue
        if q in {"exit", "quit"}:
            break

        m = CMD_RE.match(q)
        if m:
            name, arg = m.group(1).lower(), (m.group(2) or "").strip()
            if name == "db":
                if not arg:
                    console.print("Cách dùng: `/db <câu hỏi>`  ví dụ: `/db hôm nay có bao nhiêu biển số?`")
                    continue
                answer, ok = run_db(arg)
                console.print(Markdown(answer))
                if ok:  # cho phép hỏi nối tiếp ở khung chat chính
                    remember(f"/db {arg}", answer)
            elif name == "help":
                console.print(Markdown(HELP_TEXT))
            else:
                console.print(f"[red]Lệnh /{name} không tồn tại.[/red] Gõ /help để xem danh sách.")
            continue

        with console.status("[yellow]Đang tìm và trả lời..."):
            answer = ask_fn(q)
        console.print(Markdown(answer))


if __name__ == "__main__":
    app()