"""Điều phối việc index: duyệt file -> cắt chunk -> lưu vào vector store."""
from rag_cli.ingest.chunker import split_file
from rag_cli.ingest.commits import commit_documents
from rag_cli.ingest.files import iter_files
from rag_cli.ingest.state import (file_hash, load_state, removed_files,
                                  save_state)
from rag_cli.settings import DB_DIR, REPO


def index_repo(full: bool = False, repo=REPO, store=None, db_dir=DB_DIR,
               report=print) -> dict:
    """Index repo. Trả về thống kê {indexed, unchanged, removed, commits}.

    full=True: index lại mọi file. File đã bị xóa khỏi repo vẫn được dọn khỏi store.
    store/db_dir/report có thể truyền vào (để test hoặc đổi nơi lưu).
    """
    if store is None:
        from rag_cli.ingest.store import get_vectorstore
        store = get_vectorstore()

    previous = load_state(db_dir)
    old = {} if full else previous
    new: dict[str, str] = {}
    stats = {"indexed": 0, "unchanged": 0, "removed": 0, "commits": 0}

    for path in iter_files(repo):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(repo))
        digest = file_hash(text)
        if old.get(rel) == digest:          # không đổi -> bỏ qua
            new[rel] = digest
            stats["unchanged"] += 1
            continue
        docs = split_file(rel, text)
        if not docs:
            continue
        store.delete(where={"source": rel})  # xóa chunk cũ của file này rồi thêm mới
        store.add_documents(docs)
        new[rel] = digest
        stats["indexed"] += 1
        report(f"indexed: {rel}")

    for rel in sorted(removed_files(previous, new)):
        store.delete(where={"source": rel})
        stats["removed"] += 1
        report(f"removed: {rel}")

    try:
        commits = commit_documents(repo)
    except RuntimeError as e:
        commits = None
        report(f"bỏ qua lịch sử git: {e}")
    if commits is not None:                  # chỉ thay commit cũ khi đọc git thành công
        store.delete(where={"type": "commit"})
        if commits:
            store.add_documents(commits)
        stats["commits"] = len(commits)

    save_state(new, db_dir)
    return stats
