"""Duyệt repo và lọc ra các file đáng index."""
import os
from pathlib import Path
from typing import Iterator

from rag_cli.settings import (IGNORE_DIRS, IGNORE_FILES, IGNORE_SUFFIX,
                              MAX_FILE_BYTES, REPO, TEXT_EXT)


def is_indexable(path: Path, repo: Path) -> bool:
    # chỉ xét phần đường dẫn TƯƠNG ĐỐI so với repo: repo nằm trong thư mục tên "build"
    # chẳng hạn sẽ không làm cả repo bị bỏ qua
    if set(path.relative_to(repo).parts) & IGNORE_DIRS or path.name in IGNORE_FILES:
        return False
    # endswith để bắt cả hậu tố kép như ".min.js" (path.suffix chỉ cho ".js")
    if path.name.endswith(tuple(IGNORE_SUFFIX)) or path.suffix not in TEXT_EXT:
        return False
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:  # symlink hỏng, file vừa bị xóa...
        return False


def iter_files(repo: Path = REPO) -> Iterator[Path]:
    for root, dirs, names in os.walk(repo):
        # cắt nhánh ngay từ đầu: không đi vào node_modules, .git... cho nhanh
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for name in sorted(names):
            p = Path(root) / name
            if is_indexable(p, repo):
                yield p
