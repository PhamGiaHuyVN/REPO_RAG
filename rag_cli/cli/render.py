"""Mọi thứ liên quan tới hiển thị trên terminal."""
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

console = Console()


def print_markdown(text: str) -> None:
    console.print(Markdown(text))


def commit_table(commits: list[tuple], keyword: str | None = None) -> Table:
    title = f"Commit chứa '{keyword}'" if keyword else f"{len(commits)} commit gần nhất"
    table = Table(title=title, title_justify="left")
    for col in ("Hash", "Tác giả", "Ngày", "Nội dung"):
        table.add_column(col, overflow="fold")
    for row in commits:
        table.add_row(*row)
    return table


def commits_as_text(commits: list[tuple]) -> str:
    """Dạng văn bản thuần để ghi vào lịch sử chat."""
    return "Lịch sử commit:\n" + "\n".join(f"{h} | {a} | {d} | {m}" for h, a, d, m in commits)
