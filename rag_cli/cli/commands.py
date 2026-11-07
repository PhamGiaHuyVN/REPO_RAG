"""Các lệnh slash (/db, /g, /help). Thêm lệnh mới = thêm một hàm có @command."""
from dataclasses import dataclass
from typing import Callable, Optional

from rich.markup import escape

from rag_cli.chat.history import ChatHistory
from rag_cli.cli.parser import parse_command
from rag_cli.cli.render import commit_table, commits_as_text, console, print_markdown
from rag_cli.settings import REPO
from rag_cli.tools.git_log import get_commits, parse_git_args


@dataclass
class CommandContext:
    history: Optional[ChatHistory] = None   # None: chạy độc lập, không ghi vào lịch sử chat


Handler = Callable[[str, CommandContext], None]


@dataclass
class Command:
    names: tuple
    usage: str
    description: str
    handler: Handler


_COMMANDS: list = []
_BY_NAME: dict = {}


def command(*names: str, usage: str, description: str):
    """Đăng ký handler cho lệnh /<names>."""
    def deco(fn: Handler) -> Handler:
        cmd = Command(names, usage, description, fn)
        _COMMANDS.append(cmd)
        for n in names:
            _BY_NAME[n.lower()] = cmd
        return fn
    return deco


def dispatch(line: str, ctx: CommandContext) -> bool:
    """True nếu `line` là lệnh slash (đã xử lý xong, kể cả khi báo lỗi); False -> không phải lệnh."""
    parsed = parse_command(line)
    if parsed is None:
        return False
    name, arg = parsed
    cmd = _BY_NAME.get(name)
    if cmd is None:
        console.print(f"[red]Lệnh /{escape(name)} không tồn tại.[/red] Gõ /help để xem danh sách.")
        return True
    try:
        cmd.handler(arg, ctx)
    except Exception as e:  # một lệnh lỗi không được làm sập cả CLI
        console.print(f"[red]Lỗi khi chạy /{name}: {escape(str(e))}[/red]")
    return True


# ----------------------------- /db -----------------------------------

def run_db(question: str) -> tuple[str, bool]:
    """Chạy truy vấn DB. Trả về (nội dung, thành công?). Dùng chung cho /db và lệnh `db`."""
    from rag_cli.tools.db_query import ask_db  # import trễ: chỉ nạp khi thật sự dùng DB
    with console.status("[yellow]Đang truy vấn DB..."):
        try:
            return ask_db(question), True
        except Exception as e:
            return f"Lỗi: {e}", False


@command("db", usage="/db <câu hỏi>", description="truy vấn database bằng ngôn ngữ tự nhiên")
def cmd_db(arg: str, ctx: CommandContext) -> None:
    if not arg:
        console.print("Cách dùng: `/db <câu hỏi>`  ví dụ: `/db hôm nay có bao nhiêu biển số?`")
        return
    answer, ok = run_db(arg)
    print_markdown(answer)
    if ok and ctx.history is not None:  # cho phép hỏi nối tiếp ở khung chat chính
        ctx.history.add(f"/db {arg}", answer, max_chars=1500)


# ----------------------------- /g ------------------------------------

@command("g", "git", usage="/g [số | từ khóa]",
         description="lịch sử commit (mặc định 10; `/g 30`; `/g fix` lọc theo từ khóa)")
def cmd_git(arg: str, ctx: CommandContext) -> None:
    limit, keyword = parse_git_args(arg)
    commits = get_commits(REPO, limit, keyword)
    if not commits:
        console.print(f"Không có commit nào chứa '{escape(keyword)}'." if keyword
                      else "Repo chưa có commit nào.")
        return
    console.print(commit_table(commits, keyword))
    if ctx.history is not None:
        ctx.history.add(f"/g {arg}".strip(), commits_as_text(commits))


# ----------------------------- /help ---------------------------------

@command("help", usage="/help", description="hiện trợ giúp này")
def cmd_help(arg: str, ctx: CommandContext) -> None:
    lines = ["**Lệnh có sẵn**", ""]
    lines += [f"- `{c.usage}`: {c.description}" for c in _COMMANDS]
    lines += ["- `exit` / `quit`: thoát", "",
              "Không có `/` ở đầu thì hỏi về codebase như bình thường."]
    print_markdown("\n".join(lines))
