from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL
from retriever import build_retriever
import time

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    #max_output_tokens=800,
    temperature=0.2,
    #thinking_budget=0,
)
retriever = build_retriever()
history: list = []

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Viết lại câu hỏi cuối thành câu hỏi độc lập, giữ nguyên "
               "tên hàm/file/biến. Chỉ trả về câu hỏi."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là trợ lý quản lý codebase. Chỉ trả lời dựa trên ngữ cảnh được cung cấp.\n"
     "Quy tắc trình bày:\n"
     "- Trả lời bằng tiếng Việt, ngắn gọn, ưu tiên gạch đầu dòng.\n"
     "- Mỗi ý quan trọng ghi rõ file liên quan dạng `đường/dẫn/file.py`.\n"
     "- Kết thúc bằng mục 'Nguồn' liệt kê các file đã dùng.\n"
     "- Nếu ngữ cảnh không đủ, nói rõ phần nào chưa biết, không đoán.\n"
     "- Không nhắc tới file cấu hình của công cụ RAG (config.py, indexer.py...) "
     "trừ khi người dùng hỏi."),
    MessagesPlaceholder("history", optional=True),
    ("human", "Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"),
])

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)

def remember(question: str, answer: str, max_chars: int = 1500):
    """Ghi một lượt hỏi-đáp từ nguồn khác (vd: /db) vào lịch sử chat chính."""
    history.extend([("human", question), ("ai", answer[:max_chars])])
    del history[:-12]

def ask(question: str) -> str:
    question = question.strip()
    if not question:
        return "Vui lòng nhập câu hỏi."

    t0 = time.perf_counter()

    standalone = question
    if history:
        standalone = (rewrite_prompt | llm | StrOutputParser()).invoke(
            {"history": history, "question": question})
    t1 = time.perf_counter()

    docs = retriever.invoke(standalone)
    t2 = time.perf_counter()

    print("DEBUG:", repr(question), len(docs), len(history))
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "history": history, "question": question})
    if answer.strip():
        history.extend([("human", question), ("ai", answer)])
        del history[:-12]

    t3 = time.perf_counter()

    print(f"\n⏱ viết lại câu hỏi: {t1-t0:.2f}s | truy hồi (Ollama embed + search): "f"{t2-t1:.2f}s | Gemini trả lời: {t3-t2:.2f}s")

    sources = sorted({d.metadata.get("source") for d in docs})
    return answer + "\n\n---\n**Nguồn truy hồi:**\n" + "\n".join(f"- `{s}`" for s in sources)