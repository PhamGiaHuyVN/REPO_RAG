"""Gộp nhiều danh sách xếp hạng bằng Reciprocal Rank Fusion (hàm thuần)."""

RRF_K = 60


def doc_key(doc) -> tuple:
    """Khóa nhận diện cùng một đoạn: file + số chunk + 50 ký tự đầu."""
    return (doc.metadata.get("source"), doc.metadata.get("chunk"),
            doc.page_content[:50])


def rrf_fuse(weighted_lists, rrf_k: int = RRF_K):
    """weighted_lists: [(trọng_số, [doc theo thứ hạng]), ...].

    Mỗi doc nhận điểm trọng_số / (rrf_k + hạng); doc xuất hiện ở nhiều danh sách
    được cộng dồn. Trả về (scores, store) với khóa là doc_key.
    """
    scores, store = {}, {}
    for weight, docs in weighted_lists:
        for rank, doc in enumerate(docs):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            store[key] = doc
    return scores, store
