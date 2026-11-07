"""Nhận diện lệnh slash (hàm thuần)."""
import re

# "/db câu hỏi", "/help"  -> là lệnh.   "/src/app.py làm gì" -> KHÔNG phải lệnh
CMD_RE = re.compile(r"^/([A-Za-z]+)(?:\s+(.*))?$", re.S)


def parse_command(line: str) -> tuple[str, str] | None:
    """'/db  có bao nhiêu xe?' -> ('db', 'có bao nhiêu xe?'). Không phải lệnh -> None."""
    m = CMD_RE.match(line.strip())
    if not m:
        return None
    return m.group(1).lower(), (m.group(2) or "").strip()
