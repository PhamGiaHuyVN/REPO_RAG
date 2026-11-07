"""Bảng hash file của lần index trước, để chỉ index lại file đã đổi."""
import hashlib
import json
from pathlib import Path

from rag_cli.settings import DB_DIR


def state_path(db_dir=DB_DIR) -> Path:
    return Path(db_dir) / "hashes.json"


def file_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_state(db_dir=DB_DIR) -> dict:
    path = state_path(db_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}  # file hỏng -> coi như chưa có, index lại từ đầu


def save_state(state: dict, db_dir=DB_DIR) -> None:
    path = state_path(db_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def removed_files(previous: dict, current: dict) -> set:
    """File có ở lần index trước nhưng không còn ở lần này (đã bị xóa khỏi repo)."""
    return set(previous) - set(current)
