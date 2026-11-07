"""Biến lịch sử git thành Document để index cùng với code."""
from langchain_core.documents import Document

from rag_cli.tools.git_log import get_commits


def commit_documents(repo, limit: int = 200) -> list[Document]:
    """Mỗi commit một Document: 'hash|tác giả|ngày|tiêu đề'. Lỗi git -> RuntimeError."""
    return [Document(page_content="|".join(row),
                     metadata={"source": "git:log", "type": "commit"})
            for row in get_commits(repo, limit)]
