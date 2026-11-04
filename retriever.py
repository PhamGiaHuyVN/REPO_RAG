from langchain_core.documents import Document
# tìm kiếm theo từ khóa (thuật toán BM25)
from langchain_community.retrievers import BM25Retriever
# kết nối db không cần tạo lại
from indexer import db

import re
import time

from sentence_transformers import CrossEncoder

CANDIDATES = 24
RERANK_TOP_N = 10
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_CHARS = 800
RRF_K = 60
W_HYBRID , W_RERANK = 0.4, 0.6

# hệ số nhân điểm
PENALTY = {"test": 0.5, "migration": 0.5}

# câu hỏi chức các ừ này thì khng hạ đim tương ứng
INTENT = {
    "test": re.compile(r"\b(test|tests|pytest|kiểm thử|unit test)\b|test nào", re.I),
    "migration": re.compile(r"\b(migration|migrations|alembic|revision)\b|di trú", re.I),
}

_reranker = CrossEncoder(RERANK_MODEL, max_length=512)

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

def get_reranker():
    # chi load model khi can, chi load mot lan
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker

def build_retriever(k=8, use_rerank=True):
    # nạp dữ liệu cho BM25
    # lấy chunk trong chroma không lấy vector
    # dựng thành doc de BM25 xây chỉ mục từ khóa trong RAM
    data = db.get()
    docs = [Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])]

    # bm25 tìm khớp ừ khóa
    # hỏi tên hàm , biến, file chính xác
    # yếu khi tìm tu khác nhưng cùng nghĩa

    #vector(chroma + ollama)
    # tìm gần nghĩa
    # hỏi theo ý
    # yếu khi tìm tên ring, mã định danh hiếm
    # mmr (Maximal Marginal Relevance) lấy 30 ứng viên (fetch_k) rồi chọn 8 chunk vừa
    # liên quan vừa đa dạng tránh 8 chunk gần nhau từ cùng file

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = CANDIDATES
    vec = db.as_retriever(search_type="mmr",
                          search_kwargs={"k": 16, "fetch_k": 30})

    class HybridRetriever:

        def _hybrid(self, query):
            # gộp kết quả bằng RRF(Reciprocal Rank Fusion)
            # mỗi chunk nhận điểm trọng số / (60 + hạng )
            # hạng cao -> điểm lớn
            # chunk xuất hiện ở cả hai danh sách được cộng điểm
            # 60 làm muojnwt chênh lệch hạng , giá trị chuẩn
            # key nhận diện chung cùng một đoạn file + so chunk + 50 ký tự đầu ể cộng điểm đúng ch
            # trọng số 0.4 cho BM25 và 0.6 cho vector là giá trị khởi điểm

            # gop diem bang Reciprocal Rank Fusion
            scores, store = {}, {}
            for weight, retriever in ((0.4, bm25), (0.6, vec)):
                for rank, d in enumerate(retriever.invoke(query)):
                    key = (d.metadata.get("source"), d.metadata.get("chunk"),
                           d.page_content[:50])
                    scores[key] = scores.get(key, 0) + weight / (RRF_K + rank)
                    store[key] = d
            pen = active_penalties(query)
            for key, d in store.items():
                f = pen.get(kind_of(d.metadata.get("source")))
                if f:
                    scores[key] *= f
            ranked = sorted(scores, key=scores.get, reverse=True)
            return ranked, store, pen

        def _rerank(self, query, keys, store, pen):
            pairs = [(query, store[key].page_content[:RERANK_CHARS]) for key in keys]
            ce = get_reranker().predict(pairs, batch_size=8)
            t = time.perf_counter()
            print(f"[rerank] {len(pairs)} cặp: {time.perf_counter() - t:.2f}s")

            # hạ điểm test/migration cả ở bước rerank (điểm CE có thể âm
            # nên không nhân hệ số; dùng trừ điểm thay vì nhân)
            adj = []
            for key, s in zip(keys, ce):
                if pen.get(kind_of(store[key].metadata.get("source"))):
                    s = s - 2.0
                adj.append(s)

            order = sorted(range(len(keys)), key=lambda i: adj[i], reverse=True)
            rerank_rank = {keys[i]: r for r, i in enumerate(order)}

            # RRF lần hai: hạng hybrid + hạng reranker
            final = {}
            for r, key in enumerate(keys):
                final[key] = (W_HYBRID / (RRF_K + r)
                              + W_RERANK / (RRF_K + rerank_rank[key]))
            return sorted(final, key=final.get, reverse=True)

        """Gộp kết quả BM25 + vector bằng Reciprocal Rank Fusion."""
        def invoke(self, query: str):
            ranked, store, pen = self._hybrid(query)
            if use_rerank:
                cand = ranked[:RERANK_TOP_N]
                ranked = self._rerank(query, cand, store, pen)
            return [store[key] for key in ranked[:k]]



    return HybridRetriever()