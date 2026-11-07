"""Đọc lịch sử commit bằng `git log` (không qua GitHub, không qua index)."""
import re
import subprocess

SEP = "\x1f"  # ký tự phân tách an toàn: tiêu đề commit có thể chứa dấu | hay dấu phẩy


def get_commits(repo, limit: int = 10, keyword: str | None = None) -> list[tuple]:
    """Trả về [(hash, tác giả, ngày, tiêu đề), ...], mới nhất trước.

    keyword: nếu có, chỉ lấy commit mà tiêu đề/mô tả chứa từ khóa (không phân biệt hoa thường).
    """
    cmd = ["git", "-C", str(repo), "log", f"-{limit}", "--date=short",
           f"--pretty=format:%h{SEP}%an{SEP}%ad{SEP}%s"]
    if keyword:
        # dạng --grep=<giá trị> dính liền để từ khóa bắt đầu bằng "-" không bị hiểu là tùy chọn
        cmd += ["-i", f"--grep={keyword}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=15)
    except FileNotFoundError:
        raise RuntimeError("Không tìm thấy lệnh git. Hãy cài git và thêm vào PATH.")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "git log báo lỗi")
    rows = (line.split(SEP) for line in r.stdout.splitlines() if line)
    return [tuple(x) for x in rows if len(x) == 4]


def parse_git_args(arg: str) -> tuple[int, str | None]:
    """'' -> (10, None) | '30' -> (30, None) | 'fix' -> (30, 'fix'). Số commit giới hạn 1..100."""
    arg = arg.strip()
    if re.fullmatch(r"[0-9]+", arg):
        return max(1, min(int(arg), 100)), None
    if arg:
        return 30, arg
    return 10, None
