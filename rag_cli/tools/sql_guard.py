"""Kiểm tra SQL do LLM sinh ra (hàm thuần, không gọi mạng/DB)."""
import re

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy)\b", re.I)


def clean_sql(text: str) -> str:
    """Bỏ rào markdown, chỉ chấp nhận đúng một câu SELECT/WITH không chứa lệnh ghi."""
    text = re.sub(r"```(?:sql)?|```", "", text).strip().rstrip(";")
    if ";" in text or not re.match(r"^(select|with)\b", text, re.I) \
            or FORBIDDEN.search(text):
        raise ValueError(f"SQL không an toàn hoặc không hợp lệ: {text}")
    return text
