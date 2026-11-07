# rag-cli

CLI hỏi đáp codebase (RAG) + truy vấn PostgreSQL (`/db`) + lịch sử git (`/g`).

## Chạy

```
pip install -e ".[dev]"      # hoặc cài từng gói trong pyproject.toml
python cli.py index          # lập chỉ mục (chỉ file đổi); --full để làm lại tất cả
python cli.py ask            # khung chat chính (cũng là lệnh mặc định: python cli.py)
python cli.py -v ask         # kèm thời gian từng bước + debug
python cli.py db             # khung chat DB riêng
```

Trong khung chat: `/db <câu hỏi>`, `/g [số | từ khóa]`, `/help`, `exit`.

## Test

```
python -m pytest             # hoặc: python -m unittest discover -s tests -t .
```

## Bản đồ file cũ -> mới

| Cũ | Mới |
|---|---|
| config.py | rag_cli/settings.py |
| indexer.py | rag_cli/ingest/{files,chunker,state,commits,store,indexer}.py |
| retriever.py | rag_cli/retrieval/{penalties,fusion,hybrid}.py |
| chat.py | rag_cli/chat/{prompts,history,pipeline}.py + rag_cli/llm.py |
| db_query.py | rag_cli/tools/{sql_guard,db_query}.py |
| git_log.py | rag_cli/tools/git_log.py |
| cli.py | cli.py (lối vào mỏng) + rag_cli/cli/{app,commands,parser,render}.py |

Thêm lệnh slash mới: viết một hàm có `@command(...)` trong `rag_cli/cli/commands.py`.
