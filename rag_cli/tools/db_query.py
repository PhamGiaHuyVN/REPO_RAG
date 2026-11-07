"""Hỏi PostgreSQL bằng ngôn ngữ tự nhiên (text-to-SQL, chỉ đọc)."""
import os
from functools import lru_cache

from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_cli.llm import get_llm, with_retry
from rag_cli.tools.sql_guard import clean_sql

# Giới hạn an toàn ở tầng database (phòng thủ nhiều lớp, ngoài clean_sql):
# - default_transaction_read_only=on : mọi transaction đều chỉ-đọc
# - statement_timeout=10000          : mỗi câu truy vấn tối đa 10 giây
PG_SAFE_OPTIONS = "-c default_transaction_read_only=on -c statement_timeout=10000"

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn viết câu SQL PostgreSQL chỉ để ĐỌC dữ liệu. "
               "Chỉ trả về đúng một câu SELECT, không giải thích, không markdown. "
               "Luôn thêm LIMIT tối đa 50 dòng nếu không phải truy vấn tổng hợp.\n\n"
               "Cấu trúc bảng:\n{schema}"),
    ("human", "{question}"),
])
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Trả lời ngắn gọn bằng tiếng Việt dựa trên kết quả truy vấn. "
               "Nếu kết quả rỗng, nói rõ là không có dữ liệu."),
    ("human", "Câu hỏi: {question}\nSQL: {sql}\nKết quả: {result}"),
])


@lru_cache(maxsize=1)
def _get_sql_db() -> SQLDatabase:
    """Kết nối lười: thiếu PLATE_DATABASE_URL hay DB tắt không làm sập CLI lúc khởi động."""
    url = os.environ.get("PLATE_DATABASE_URL")
    if not url:
        raise RuntimeError("Thiếu biến môi trường PLATE_DATABASE_URL")
    # sample_rows=0: không gửi dòng dữ liệu mẫu lên Gemini, chỉ gửi cấu trúc bảng
    return SQLDatabase.from_uri(
        url,
        sample_rows_in_table_info=0,
        engine_args={"connect_args": {"options": PG_SAFE_OPTIONS}},
    )


def ask_db(question: str) -> str:
    sql_db, llm = _get_sql_db(), get_llm(0.0)
    schema = sql_db.get_table_info()
    raw = with_retry(lambda: (sql_prompt | llm | StrOutputParser()).invoke(
        {"schema": schema, "question": question}))
    sql = clean_sql(raw)
    result = sql_db.run(sql)
    answer = with_retry(lambda: (answer_prompt | llm | StrOutputParser()).invoke(
        {"question": question, "sql": sql, "result": str(result)[:6000]}))
    return f"{answer}\n\n---\n`SQL:` `{sql}`"
