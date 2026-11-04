import os, re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL

load_dotenv()

# Giới hạn an toàn ở tầng database (phòng thủ nhiều lớp, ngoài clean_sql):
# - default_transaction_read_only=on : mọi transaction đều chỉ-đọc
# - statement_timeout=10000          : mỗi câu truy vấn tối đa 10 giây
PG_SAFE_OPTIONS = "-c default_transaction_read_only=on -c statement_timeout=10000"


# ---------------------------------------------------------------------------
# Khởi tạo lười (lazy): chỉ kết nối DB / tạo LLM khi người dùng thực sự gọi /db.
# Nhờ vậy thiếu PLATE_DATABASE_URL hoặc DB đang tắt sẽ KHÔNG làm sập cả CLI
# ngay lúc import.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_sql_db() -> SQLDatabase:
    url = os.environ.get("PLATE_DATABASE_URL")
    if not url:
        raise RuntimeError("Thiếu biến môi trường PLATE_DATABASE_URL")
    # sample_rows=0: không gửi dòng dữ liệu mẫu lên Gemini, chỉ gửi cấu trúc bảng
    return SQLDatabase.from_uri(
        url,
        sample_rows_in_table_info=0,
        engine_args={"connect_args": {"options": PG_SAFE_OPTIONS}},
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


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

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy)\b", re.I)


def clean_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?|```", "", text).strip().rstrip(";")
    if ";" in text or not re.match(r"^(select|with)\b", text, re.I) \
            or FORBIDDEN.search(text):
        raise ValueError(f"SQL không an toàn hoặc không hợp lệ: {text}")
    return text


def ask_db(question: str) -> str:
    sql_db, llm = _get_sql_db(), _get_llm()
    schema = sql_db.get_table_info()
    raw = (sql_prompt | llm | StrOutputParser()).invoke(
        {"schema": schema, "question": question})
    sql = clean_sql(raw)
    result = sql_db.run(sql)
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {"question": question, "sql": sql, "result": str(result)[:6000]})
    return f"{answer}\n\n---\n`SQL:` `{sql}`"