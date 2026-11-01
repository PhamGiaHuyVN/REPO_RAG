from langchain_core.documents import Document
# tìm kiếm theo từ khóa (thuật toán BM25)
from langchain_community.retrievers import BM25Retriever
# kết nối db không cần tạo lại
from indexer import db

def build_retriever(k=8):
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
    bm25.k = k
    vec = db.as_retriever(search_type="mmr",
                          search_kwargs={"k": k, "fetch_k": 30})

    class HybridRetriever:
        """Gộp kết quả BM25 + vector bằng Reciprocal Rank Fusion."""
        def invoke(self, query: str):
            scores, store = {}, {}
            # gộp kết quả bằng RRF(Reciprocal Rank Fusion)
            # mỗi chunk nhận điểm trọng số / (60 + hạng )
            # hạng cao -> điểm lớn
            # chunk xuất hiện ở cả hai danh sách được cộng điểm
            # 60 làm muojnwt chênh lệch hạng , giá trị chuẩn
            # key nhận diện chung cùng một đoạn file + so chunk + 50 ký tự đầu ể cộng điểm đúng ch
            # trọng số 0.4 cho BM25 và 0.6 cho vector là giá trị khởi điểm
            for weight, retriever in ((0.4, bm25), (0.6, vec)):
                for rank, d in enumerate(retriever.invoke(query)):
                    key = (d.metadata.get("source"), d.metadata.get("chunk"),
                           d.page_content[:50])
                    scores[key] = scores.get(key, 0) + weight / (60 + rank)
                    store[key] = d
            top = sorted(scores, key=scores.get, reverse=True)[:k]
            return [store[key] for key in top]

    return HybridRetriever()