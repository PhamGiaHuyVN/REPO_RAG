"""LLM dùng chung: tạo lười + thử lại khi Gemini quá tải."""
import time
from functools import lru_cache

# Lỗi tạm thời phía Gemini (quá tải, giới hạn tốc độ...) -> đáng thử lại
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "DEADLINE")


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0):
    # import trễ: chỉ nạp langchain-google-genai khi thật sự cần gọi LLM
    from langchain_google_genai import ChatGoogleGenerativeAI
    from rag_cli.settings import LLM_MODEL
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature)


def with_retry(fn, tries: int = 4, base: float = 2.0, sleep=time.sleep):
    """Gọi fn(); gặp lỗi tạm thời thì chờ base, 2*base, 4*base... rồi thử lại."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            transient = any(m in str(e) for m in TRANSIENT_MARKERS)
            if i == tries - 1 or not transient:
                raise
            sleep(base * 2 ** i)
