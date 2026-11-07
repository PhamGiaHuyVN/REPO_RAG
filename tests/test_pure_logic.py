"""Test các hàm thuần: SQL guard, phạt test/migration, RRF, lịch sử, retry, phân tích lệnh."""
import unittest
from types import SimpleNamespace

from rag_cli.chat.history import ChatHistory
from rag_cli.cli.parser import parse_command
from rag_cli.llm import with_retry
from rag_cli.retrieval.fusion import rrf_fuse
from rag_cli.retrieval.hybrid import HybridRetriever
from rag_cli.retrieval.penalties import active_penalties, kind_of, penalize
from rag_cli.tools.git_log import parse_git_args
from rag_cli.tools.sql_guard import clean_sql


def doc(source, chunk=0, text="nội dung"):
    return SimpleNamespace(metadata={"source": source, "chunk": chunk}, page_content=text)


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return list(self.docs)


class SqlGuardTests(unittest.TestCase):
    def test_accepts_select_and_with(self):
        self.assertEqual(clean_sql("SELECT 1;"), "SELECT 1")
        self.assertEqual(clean_sql("```sql\nselect * from t limit 5\n```"), "select * from t limit 5")
        self.assertTrue(clean_sql("WITH a AS (select 1) select * from a").startswith("WITH"))

    def test_rejects_writes_and_multiple_statements(self):
        for bad in ["DELETE FROM t", "select 1; drop table t", "update t set a=1",
                    "select * from t; select 2", "insert into t values (1)",
                    "with x as (delete from t returning *) select * from x", "explain select 1"]:
            with self.assertRaises(ValueError, msg=bad):
                clean_sql(bad)

    def test_column_names_containing_keywords_are_ok(self):
        self.assertEqual(clean_sql("select created_at, updated_at from t"),
                         "select created_at, updated_at from t")


class PenaltyTests(unittest.TestCase):
    def test_kind_of(self):
        self.assertEqual(kind_of("tests/test_a.py"), "test")
        self.assertEqual(kind_of("app\\tests\\x.py"), "test")
        self.assertEqual(kind_of("pkg/test_utils.py"), "test")
        self.assertEqual(kind_of("migrations/001.sql"), "migration")
        self.assertEqual(kind_of("app/migrations/001.sql"), "migration")
        self.assertEqual(kind_of("app/main.py"), "code")
        self.assertEqual(kind_of(None), "code")

    def test_penalty_skipped_when_query_mentions_topic(self):
        self.assertEqual(set(active_penalties("hàm login làm gì")), {"test", "migration"})
        self.assertEqual(set(active_penalties("có pytest nào cho login")), {"migration"})
        self.assertEqual(set(active_penalties("alembic revision mới nhất")), {"test"})

    def test_penalize_scales_only_matching_chunks(self):
        code, test = doc("app/main.py"), doc("tests/test_main.py")
        store = {"c": code, "t": test}
        out = penalize({"c": 1.0, "t": 1.0}, store, "login hoạt động ra sao")
        self.assertEqual(out, {"c": 1.0, "t": 0.5})
        self.assertEqual(penalize({"c": 1.0, "t": 1.0}, store, "test nào cho login"),
                         {"c": 1.0, "t": 1.0})


class FusionAndHybridTests(unittest.TestCase):
    def test_rrf_rewards_docs_present_in_both_lists(self):
        a, b, c = doc("a.py"), doc("b.py"), doc("c.py")
        scores, store = rrf_fuse([(0.4, [a, b]), (0.6, [b, c])])
        ranked = sorted(scores, key=scores.get, reverse=True)
        self.assertEqual([store[k].metadata["source"] for k in ranked], ["b.py", "c.py", "a.py"])
        self.assertAlmostEqual(max(scores.values()), 0.4 / 61 + 0.6 / 60)

    def test_hybrid_returns_top_k_and_demotes_tests(self):
        main, test = doc("app/main.py"), doc("tests/test_main.py")
        other = doc("app/util.py")
        # test đứng đầu cả hai danh sách; không hỏi về test -> bị hạ xuống dưới code
        hybrid = HybridRetriever(FakeRetriever([test, main]), FakeRetriever([test, other]), k=2)
        result = [d.metadata["source"] for d in hybrid.invoke("login làm gì")]
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0], "tests/test_main.py")
        # hỏi về test -> không bị hạ
        result = [d.metadata["source"] for d in hybrid.invoke("test nào cho login")]
        self.assertEqual(result[0], "tests/test_main.py")


class HistoryTests(unittest.TestCase):
    def test_keeps_last_messages_in_pairs(self):
        h = ChatHistory(max_messages=4)
        for i in range(5):
            h.add(f"q{i}", f"a{i}")
        self.assertEqual(h.messages(), [("human", "q3"), ("ai", "a3"), ("human", "q4"), ("ai", "a4")])
        self.assertEqual(len(h), 4)

    def test_truncates_long_answer_and_returns_copy(self):
        h = ChatHistory()
        h.add("q", "x" * 100, max_chars=10)
        self.assertEqual(h.messages()[1], ("ai", "x" * 10))
        h.messages().clear()
        self.assertEqual(len(h), 2)

    def test_empty_history_is_falsy(self):
        self.assertFalse(ChatHistory())


class RetryTests(unittest.TestCase):
    def test_retries_transient_then_succeeds(self):
        calls, sleeps = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("503 UNAVAILABLE. high demand")
            return "ok"

        self.assertEqual(with_retry(flaky, sleep=sleeps.append), "ok")
        self.assertEqual(sleeps, [2.0, 4.0])

    def test_non_transient_raises_immediately(self):
        sleeps = []
        with self.assertRaises(ValueError):
            with_retry(lambda: (_ for _ in ()).throw(ValueError("sai cú pháp")), sleep=sleeps.append)
        self.assertEqual(sleeps, [])

    def test_gives_up_after_tries(self):
        calls = []

        def always():
            calls.append(1)
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        with self.assertRaises(RuntimeError):
            with_retry(always, tries=3, sleep=lambda s: None)
        self.assertEqual(len(calls), 3)


class CommandParsingTests(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(parse_command("/db có bao nhiêu xe?"), ("db", "có bao nhiêu xe?"))
        self.assertEqual(parse_command("  /DB   x  "), ("db", "x"))
        self.assertEqual(parse_command("/help"), ("help", ""))
        self.assertEqual(parse_command("/dbb abc"), ("dbb", "abc"))

    def test_not_commands(self):
        for line in ["hỏi thường", "/src/app.py làm gì", "xin chào /db", "/ db", "/"]:
            self.assertIsNone(parse_command(line), msg=line)

    def test_git_args(self):
        self.assertEqual(parse_git_args(""), (10, None))
        self.assertEqual(parse_git_args("30"), (30, None))
        self.assertEqual(parse_git_args("0"), (1, None))
        self.assertEqual(parse_git_args("9999"), (100, None))
        self.assertEqual(parse_git_args("fix retriever"), (30, "fix retriever"))
        self.assertEqual(parse_git_args("²"), (30, "²"))  # chữ số "đặc biệt" không làm int() nổ


if __name__ == "__main__":
    unittest.main()
