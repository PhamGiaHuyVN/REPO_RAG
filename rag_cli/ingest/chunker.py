"""Cắt nội dung file thành các chunk (Document) để đưa vào vector DB."""
from pathlib import PurePath

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_cli.settings import CHUNK_OVERLAP, CHUNK_SIZE, EXT_LANG


def make_splitter(suffix: str) -> RecursiveCharacterTextSplitter:
    lang = EXT_LANG.get(suffix)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def split_file(rel_path: str, text: str) -> list[Document]:
    """rel_path: đường dẫn tương đối so với repo (dùng làm metadata `source`)."""
    suffix = PurePath(rel_path).suffix
    docs = make_splitter(suffix).create_documents([text], metadatas=[{
        "source": rel_path, "ext": suffix, "type": "code"}])
    for i, d in enumerate(docs):
        d.page_content = f"# File: {rel_path}\n{d.page_content}"
        d.metadata["chunk"] = i
    return docs
