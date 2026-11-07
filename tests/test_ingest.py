"""Test duyệt file, cắt chunk, bảng hash và luồng index (dùng store giả, không cần Chroma/Ollama)."""
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HAVE_LANGCHAIN = bool(importlib.util.find_spec("langchain_core")
                      and importlib.util.find_spec("langchain_text_splitters")
                      and importlib.util.find_spec("dotenv"))


class FakeStore:
    def __init__(self):
        self.added, self.deleted = [], []

    def add_documents(self, docs):
        self.added.extend(docs)

    def delete(self, where=None):
        self.deleted.append(where)

    def code_sources(self):
        return {d.metadata["source"] for d in self.added if d.metadata.get("type") == "code"}


@unittest.skipUnless(HAVE_LANGCHAIN, "cần langchain-core, langchain-text-splitters, python-dotenv")
class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo, self.idx = root / "repo", root / "idx"
        (self.repo / "sub").mkdir(parents=True)
        (self.repo / "node_modules").mkdir()
        (self.repo / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (self.repo / "sub" / "b.md").write_text("# Tiêu đề\nnội dung tiếng Việt\n", encoding="utf-8")
        (self.repo / "readme.txt").write_text("xin chào", encoding="utf-8")
        # các file phải bị bỏ qua
        (self.repo / "node_modules" / "x.js").write_text("var a;", encoding="utf-8")
        (self.repo / "bundle.min.js").write_text("var a;", encoding="utf-8")
        (self.repo / "key.pem").write_text("secret", encoding="utf-8")
        (self.repo / "image.png").write_bytes(b"\x89PNG")
        self.store, self.log = FakeStore(), []

    def tearDown(self):
        self.tmp.cleanup()

    def run_index(self, full=False):
        return __import__("rag_cli.ingest.indexer", fromlist=["index_repo"]).index_repo(
            full, repo=self.repo, store=self.store, db_dir=self.idx, report=self.log.append)

    # ---- duyệt file ----
    def test_iter_files_filters_ignored(self):
        from rag_cli.ingest.files import iter_files
        found = {str(p.relative_to(self.repo)) for p in iter_files(self.repo)}
        self.assertEqual(found, {"a.py", os.path.join("sub", "b.md"), "readme.txt"})

    def test_repo_inside_ignored_dir_name_is_still_indexed(self):
        from rag_cli.ingest.files import iter_files
        nested = Path(self.tmp.name) / "build" / "proj"
        nested.mkdir(parents=True)
        (nested / "m.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual([p.name for p in iter_files(nested)], ["m.py"])

    # ---- chunker ----
    def test_split_file_metadata_and_header(self):
        from rag_cli.ingest.chunker import split_file
        docs = split_file("pkg/a.py", "def f():\n    return 1\n")
        self.assertTrue(docs)
        self.assertTrue(docs[0].page_content.startswith("# File: pkg/a.py\n"))
        self.assertEqual(docs[0].metadata, {"source": "pkg/a.py", "ext": ".py",
                                            "type": "code", "chunk": 0})
        self.assertEqual(split_file("empty.py", ""), [])

    # ---- state ----
    def test_state_roundtrip_and_removed(self):
        from rag_cli.ingest.state import file_hash, load_state, removed_files, save_state
        self.assertEqual(load_state(self.idx), {})
        save_state({"a.py": file_hash("x")}, self.idx)
        self.assertEqual(load_state(self.idx), {"a.py": file_hash("x")})
        (self.idx / "hashes.json").write_text("{hỏng", encoding="utf-8")
        self.assertEqual(load_state(self.idx), {})
        self.assertEqual(removed_files({"a": 1, "b": 2}, {"a": 1}), {"b"})

    # ---- luồng index ----
    def test_incremental_indexing(self):
        s1 = self.run_index()
        self.assertEqual((s1["indexed"], s1["unchanged"]), (3, 0))
        self.assertEqual(self.store.code_sources(), {"a.py", os.path.join("sub", "b.md"), "readme.txt"})

        n_added = len(self.store.added)
        s2 = self.run_index()                       # không đổi gì
        self.assertEqual((s2["indexed"], s2["unchanged"]), (0, 3))
        self.assertEqual(len(self.store.added), n_added)

        (self.repo / "a.py").write_text("def f():\n    return 2\n", encoding="utf-8")
        s3 = self.run_index()                       # chỉ a.py đổi
        self.assertEqual((s3["indexed"], s3["unchanged"]), (1, 2))
        self.assertIn({"source": "a.py"}, self.store.deleted)

        (self.repo / "readme.txt").unlink()
        s4 = self.run_index()                       # file bị xóa phải được dọn
        self.assertEqual(s4["removed"], 1)
        self.assertIn("removed: readme.txt", self.log)
        self.assertIn({"source": "readme.txt"}, self.store.deleted)

    def test_full_reindexes_all_and_still_removes_deleted(self):
        self.run_index()
        (self.repo / "sub" / "b.md").unlink()
        s = self.run_index(full=True)
        self.assertEqual((s["indexed"], s["removed"]), (2, 1))
        self.assertIn("removed: " + os.path.join("sub", "b.md"), self.log)

    def test_non_git_repo_skips_commits_without_crash(self):
        s = self.run_index()
        self.assertEqual(s["commits"], 0)
        self.assertTrue(any(m.startswith("bỏ qua lịch sử git") for m in self.log))
        self.assertNotIn({"type": "commit"}, self.store.deleted)  # không xóa commit cũ khi đọc git lỗi

    @unittest.skipUnless(shutil.which("git"), "cần cài git")
    def test_git_repo_indexes_commits(self):
        def git(*a):
            subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=H", "-c", "user.email=h@x.vn",
                            *a], check=True, capture_output=True)
        git("init", "-q"); git("add", "."); git("commit", "-q", "-m", "init project")
        s = self.run_index()
        self.assertEqual(s["commits"], 1)
        commit_docs = [d for d in self.store.added if d.metadata.get("type") == "commit"]
        self.assertEqual(commit_docs[0].metadata["source"], "git:log")
        self.assertTrue(commit_docs[0].page_content.endswith("|init project"))
        self.assertIn({"type": "commit"}, self.store.deleted)


if __name__ == "__main__":
    unittest.main()
