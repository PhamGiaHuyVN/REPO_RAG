"""Hạ điểm chunk từ test/migration trừ khi câu hỏi nhắc tới chúng (hàm thuần)."""
import re

# hệ số nhân điểm
PENALTY = {"test": 0.5, "migration": 0.5}

# câu hỏi chứa các từ này thì không hạ điểm tương ứng
INTENT = {
    "test": re.compile(r"\b(test|tests|pytest|kiểm thử|unit test)\b|test nào", re.I),
    "migration": re.compile(r"\b(migration|migrations|alembic|revision)\b|di trú", re.I),
}


def kind_of(source: str) -> str:
    s = (source or "").replace("\\", "/").lower()
    if s.startswith("tests/") or "/tests/" in s or s.rsplit("/", 1)[-1].startswith("test_"):
        return "test"
    if s.startswith("migrations/") or "/migrations/" in s:
        return "migration"
    return "code"


def active_penalties(query: str) -> dict:
    return {kind: factor for kind, factor in PENALTY.items()
            if not INTENT[kind].search(query)}


def penalize(scores: dict, store: dict, query: str) -> dict:
    """Trả về bản sao của scores đã nhân hệ số phạt cho chunk test/migration."""
    pen = active_penalties(query)
    out = dict(scores)
    for key, doc in store.items():
        factor = pen.get(kind_of(doc.metadata.get("source")))
        if factor:
            out[key] *= factor
    return out
