"""Truy hồi lai: BM25 (từ khóa) + vector (ngữ nghĩa) -> RRF -> phạt test/migration."""
from rag_cli.retrieval.fusion import rrf_fuse
from rag_cli.retrieval.penalties import penalize

CANDIDATES = 24                   # số ứng viên BM25 lấy về
VECTOR_K, VECTOR_FETCH_K = 16, 30  # MMR: lấy 30 ứng viên, chọn 16 vừa liên quan vừa đa dạng
W_BM25, W_VECTOR = 0.4, 0.6        # trọng số khởi điểm


class HybridRetriever:
    """Nhận sẵn hai retriever (có .invoke(query) -> list[Document]) nên test được bằng đối tượng giả."""

    def __init__(self, keyword, vector, k: int = 8, weights=(W_BM25, W_VECTOR)):
        self.keyword = keyword
        self.vector = vector
        self.k = k
        self.weights = weights

    def invoke(self, query: str) -> list:
        w_kw, w_vec = self.weights
        scores, store = rrf_fuse([
            (w_kw, self.keyword.invoke(query)),
            (w_vec, self.vector.invoke(query)),
        ])
        scores = penalize(scores, store, query)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [store[key] for key in ranked[:self.k]]


def build_retriever(k: int = 8, store=None) -> HybridRetriever:
    # import trễ: chỉ nạp khi thật sự dựng retriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    if store is None:
        from rag_cli.ingest.store import get_vectorstore
        store = get_vectorstore()

    # lấy chunk trong Chroma (không lấy vector), dựng chỉ mục BM25 trong RAM
    data = store.get()
    docs = [Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])]
    if not docs:
        raise RuntimeError("Chưa có dữ liệu index. Hãy chạy `python cli.py index` trước.")

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = CANDIDATES
    vec = store.as_retriever(search_type="mmr",
                             search_kwargs={"k": VECTOR_K, "fetch_k": VECTOR_FETCH_K})
    return HybridRetriever(bm25, vec, k=k)
