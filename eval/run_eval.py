import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# cho phép import retriever.py / indexer.py ở thư mục cha (cấu trúc phẳng hiện tại)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indexer import db                    # noqa: E402
from retriever import build_retriever     # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"
HISTORY = ROOT / "eval" / "history.log"


def norm(path: str) -> str:
    """Chuẩn hoá đường dẫn Windows về dạng app/db/session.py."""
    return (path or "").replace("\\", "/")


def load_items():
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for it in items:
        it["expected"] = [norm(e) for e in it["expected"]]
    return items


def check_expected(items):
    """Cảnh báo câu hỏi mà không file đáp án nào có trong index."""
    known = {norm(m["source"]) for m in db.get()["metadatas"]}
    bad = [it for it in items if not any(e in known for e in it["expected"])]
    for it in bad:
        print(f"⚠ không file nào trong index: {it['q']}  {it['expected']}")
    return bad


def evaluate(retriever, items):
    results = []
    for it in items:
        t = time.perf_counter()
        docs = retriever.invoke(it["q"])
        dt = time.perf_counter() - t

        # giữ thứ tự, bỏ file trùng để hạng tính theo file chứ không theo chunk
        sources = []
        for d in docs:
            s = norm(d.metadata.get("source"))
            if s not in sources:
                sources.append(s)

        rank = next((i + 1 for i, s in enumerate(sources)
                     if s in it["expected"]), None)
        results.append({
            "q": it["q"], "kind": it.get("kind", "other"),
            "expected": it["expected"], "rank": rank,
            "time": dt, "got": sources,
        })
    return results


def summarize(results):
    n = len(results)
    hits = sum(1 for r in results if r["rank"])
    mrr = sum(1 / r["rank"] for r in results if r["rank"]) / n
    avg_t = sum(r["time"] for r in results) / n
    return {"n": n, "hit": hits / n, "mrr": mrr, "time": avg_t}


def print_report(results, k):
    print("\n" + "=" * 70)
    print(f"{'Kết quả đánh giá truy hồi (k=' + str(k) + ')':^70}")
    print("=" * 70)

    for r in results:
        mark = "✓" if r["rank"] else "✗"
        rk = f"hạng {r['rank']}" if r["rank"] else "trượt"
        print(f"{mark} [{r['kind']:<8}] {rk:<8} {r['time']:.2f}s  {r['q']}")

    total = summarize(results)
    print("\n--- Tổng ---")
    print(f"Số câu: {total['n']} | Hit rate: {total['hit']:.2%} | "
          f"MRR: {total['mrr']:.3f} | Thời gian TB: {total['time']:.2f}s")

    print("\n--- Theo loại câu hỏi ---")
    for kind in sorted({r["kind"] for r in results}):
        sub = [r for r in results if r["kind"] == kind]
        s = summarize(sub)
        print(f"{kind:<9} n={s['n']:<3} hit={s['hit']:.2%}  mrr={s['mrr']:.3f}")

    missed = [r for r in results if not r["rank"]]
    if missed:
        print("\n--- Các câu trượt (xem vì sao sai) ---")
        for r in missed:
            print(f"\n❓ {r['q']}")
            print(f"   đáp án : {r['expected']}")
            print(f"   nhận về: {r['got'][:5]}")
    return total


def append_history(label, k, total):
    HISTORY.parent.mkdir(exist_ok=True)
    line = (f"{datetime.now():%Y-%m-%d %H:%M} | {label:<24} | k={k} | "
            f"hit {total['hit']:.3f} | mrr {total['mrr']:.3f} | "
            f"{total['time']:.2f}s\n")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"\nĐã ghi log: {HISTORY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="số chunk truy hồi")
    ap.add_argument("--label", default="baseline",
                    help="tên lần đo, ví dụ 'tokenizer' hoặc 'reranker'")
    ap.add_argument("--no-log", action="store_true", help="không ghi history.log")
    args = ap.parse_args()

    items = load_items()
    check_expected(items)

    print("Đang dựng retriever...")
    retriever = build_retriever(k=args.k)

    # khởi động nóng: lần gọi đầu nạp model Ollama nên chậm, không tính vào số đo
    retriever.invoke("warm up")

    results = evaluate(retriever, items)
    total = print_report(results, args.k)
    if not args.no_log:
        append_history(args.label, args.k, total)


if __name__ == "__main__":
    main()