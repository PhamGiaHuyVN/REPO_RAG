import os, re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL

load_dotenv()
# sample_rows=0: không gửi dòng dữ liệu mẫu lên Gemini, chỉ gửi cấu trúc bảng
sql_db = SQLDatabase.from_uri(os.environ["PLATE_DATABASE_URL"],
                              sample_rows_in_table_info=0)
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

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
    schema = sql_db.get_table_info()
    raw = (sql_prompt | llm | StrOutputParser()).invoke(
        {"schema": schema, "question": question})
    sql = clean_sql(raw)
    result = sql_db.run(sql)
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {"question": question, "sql": sql, "result": str(result)[:6000]})
    return f"{answer}\n\n---\n`SQL:` `{sql}`"