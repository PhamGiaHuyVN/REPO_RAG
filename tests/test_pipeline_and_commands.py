"""Test pipeline hỏi đáp (LLM/retriever giả) và bộ lệnh slash (console giả)."""
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

HAVE_PIPELINE = bool(importlib.util.find_spec("langchain_core")) and \
    importlib.util.find_spec("langchain_core.prompts") is not None
HAVE_CLI = bool(importlib.util.find_spec("rich") and importlib.util.find_spec("dotenv")
                and importlib.util.find_spec("langchain_text_splitters"))


@unittest.skipUnless(HAVE_PIPELINE, "cần langchain-core")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        self.prompts = []
        self.reply = "trả lời giả"

        def fake_llm(prompt_value):
            text = prompt_value.to_string()
            self.prompts.append(text)
            return "câu hỏi độc lập" if "Viết lại câu hỏi" in text else self.reply

        class Retriever:
            def __init__(s):
                s.queries = []

            def invoke(s, q):
                s.queries.append(q)
                return [Document(page_content="code", metadata={"source": "app/main.py"})]

        from rag_cli.chat.pipeline import ChatPipeline
        self.retriever = Retriever()
        self.pipeline = ChatPipeline(RunnableLambda(fake_llm), self.retriever)

    def test_first_question_has_no_rewrite_and_follow_up_has(self):
        out = self.pipeline.ask("hàm login làm gì?")
        self.assertIn("trả lời giả", out)
        self.assertIn("- `app/main.py`", out)
        self.assertEqual(self.retriever.queries, ["hàm login làm gì?"])
        self.assertEqual(len(self.pipeline.history), 2)       # đúng 1 lượt, không bị ghi đôi

        self.pipeline.ask("còn hàm logout?")
        self.assertEqual(self.retriever.queries[-1], "câu hỏi độc lập")  # đã viết lại
        self.assertEqual(len(self.pipeline.history), 4)

    def test_empty_question_and_empty_answer(self):
        self.assertEqual(self.pipeline.ask("   "), "Vui lòng nhập câu hỏi.")
        self.assertEqual(self.prompts, [])
        self.reply = "  "
        self.pipeline.ask("hỏi gì đó")
        self.assertEqual(len(self.pipeline.history), 0)        # câu trả lời rỗng không vào lịch sử


class FakeConsole:
    def __init__(self):
        self.printed = []

    def print(self, *args, **kwargs):
        self.printed.extend(getattr(a, "markup", a) if not isinstance(a, str) else a for a in args)

    @contextmanager
    def status(self, *a, **k):
        yield

    def text(self):
        return "\n".join(str(x) for x in self.printed)


@unittest.skipUnless(HAVE_CLI, "cần rich, python-dotenv, langchain-text-splitters")
class CommandTests(unittest.TestCase):
    def setUp(self):
        from rag_cli.chat.history import ChatHistory
        from rag_cli.cli import commands
        self.commands, self.history = commands, ChatHistory()
        self.console = FakeConsole()
        for target in ("rag_cli.cli.commands.console", "rag_cli.cli.render.console"):
            p = mock.patch(target, self.console)
            p.start()
            self.addCleanup(p.stop)
        self.ctx = commands.CommandContext(history=self.history)

    def test_non_command_is_not_handled(self):
        self.assertFalse(self.commands.dispatch("hỏi thường", self.ctx))
        self.assertFalse(self.commands.dispatch("/src/app.py làm gì", self.ctx))
        self.assertEqual(self.console.printed, [])

    def test_unknown_command_is_reported(self):
        self.assertTrue(self.commands.dispatch("/abc", self.ctx))
        self.assertIn("không tồn tại", self.console.text())

    def test_help_lists_registered_commands(self):
        self.assertTrue(self.commands.dispatch("/help", self.ctx))
        text = self.console.text()
        for usage in ("/db <câu hỏi>", "/g [số | từ khóa]", "/help"):
            self.assertIn(usage, text)

    def test_db_without_argument_shows_usage(self):
        self.commands.dispatch("/db", self.ctx)
        self.assertIn("Cách dùng", self.console.text())
        self.assertEqual(len(self.history), 0)

    def test_db_success_is_saved_to_history_and_failure_is_not(self):
        with mock.patch.object(self.commands, "run_db", return_value=("42 xe", True)):
            self.commands.dispatch("/db có bao nhiêu xe", self.ctx)
        self.assertEqual(self.history.messages(), [("human", "/db có bao nhiêu xe"), ("ai", "42 xe")])
        with mock.patch.object(self.commands, "run_db", return_value=("Lỗi: x", False)):
            self.commands.dispatch("/db hỏng", self.ctx)
        self.assertEqual(len(self.history), 2)

    def test_handler_exception_does_not_crash(self):
        with mock.patch.object(self.commands, "run_db", side_effect=RuntimeError("[boom]")):
            self.assertTrue(self.commands.dispatch("/db x", self.ctx))
        self.assertIn("Lỗi khi chạy /db", self.console.text())

    @unittest.skipUnless(shutil.which("git"), "cần cài git")
    def test_git_command_reads_repo_and_saves_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
            (repo / "a.txt").write_text("x")
            subprocess.run(["git", "-C", tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", tmp, "-c", "user.name=H", "-c", "user.email=h@x.vn",
                            "commit", "-q", "-m", "init project"], check=True)
            with mock.patch.object(self.commands, "REPO", repo):
                self.commands.dispatch("/g", self.ctx)
                self.commands.dispatch("/git zzz", self.ctx)   # alias + không có kết quả
        self.assertIn("không có commit nào chứa 'zzz'".lower(), self.console.text().lower())
        self.assertEqual(len(self.history), 2)
        self.assertIn("init project", self.history.messages()[1][1])


if __name__ == "__main__":
    unittest.main()
