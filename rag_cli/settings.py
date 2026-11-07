"""Cấu hình tập trung (thay cho config.py cũ).

Các module khác import tường minh từ đây, không dùng `from config import *`.
"""
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_text_splitters import Language

# Nạp .env (tìm từ thư mục đang chạy trở lên). Nạp một lần, ở đây.
load_dotenv(find_dotenv(usecwd=True))

REPO = Path(r"C:\Users\Huy\PycharmProjects\WelcomeScreen\plate-api")
DB_DIR = Path("./.repo_index")   # giữ nguyên: đổi sẽ mất dữ liệu đã index
COLLECTION = "repo"              # giữ nguyên: đổi sẽ mất dữ liệu đã index

EMBED_MODEL = "bge-m3"
LLM_MODEL = "gemini-3.1-flash-lite"

IGNORE_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__",
               "dist", "build", ".idea", ".next", ".repo_index", ".pytest_cache", "models", "easyocr_cache"}
IGNORE_FILES = {".env", "id_rsa", "package-lock.json", "yarn.lock", "config.py", "indexer.py", "retriever.py", "chat.py", "cli.py"}
IGNORE_SUFFIX = {".pem", ".key", ".pfx", ".min.js"}

EXT_LANG = {".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS,
            ".java": Language.JAVA, ".go": Language.GO, ".md": Language.MARKDOWN}
TEXT_EXT = set(EXT_LANG) | {".json", ".yaml", ".yml", ".toml", ".txt", ".sql"}

MAX_FILE_BYTES = 300_000
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 150
