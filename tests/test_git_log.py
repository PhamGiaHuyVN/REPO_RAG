"""Test đọc lịch sử commit bằng một repo git tạm."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rag_cli.tools.git_log import get_commits


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Huy", "-c", "user.email=h@x.vn",
                    *args], check=True, capture_output=True)


@unittest.skipUnless(shutil.which("git"), "cần cài git")
class GitLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        git(self.repo, "init", "-q")
        for msg in ["init project", "fix retriever | bỏ rerank", "thêm lệnh /db"]:
            (self.repo / "a.txt").write_text(msg, encoding="utf-8")
            git(self.repo, "add", ".")
            git(self.repo, "commit", "-q", "-m", msg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_newest_first_and_keeps_pipe_and_unicode(self):
        rows = get_commits(self.repo, 10)
        self.assertEqual([r[3] for r in rows],
                         ["thêm lệnh /db", "fix retriever | bỏ rerank", "init project"])
        self.assertEqual(rows[0][1], "Huy")

    def test_limit_and_keyword(self):
        self.assertEqual(len(get_commits(self.repo, 2)), 2)
        self.assertEqual([r[3] for r in get_commits(self.repo, 30, "FIX")],
                         ["fix retriever | bỏ rerank"])
        self.assertEqual(get_commits(self.repo, 30, "không-có"), [])
        self.assertEqual(get_commits(self.repo, 30, "--all"), [])  # không bị hiểu là tùy chọn git

    def test_errors_are_readable(self):
        with self.assertRaises(RuntimeError):
            get_commits(self.repo / "khong_ton_tai")


if __name__ == "__main__":
    unittest.main()
