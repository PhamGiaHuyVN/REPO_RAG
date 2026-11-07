"""Luồng hỏi đáp: viết lại câu hỏi -> truy hồi -> Gemini trả lời."""
import logging
import time

from langchain_core.output_parsers import StrOutputParser

from rag_cli.chat.history import ChatHistory
from rag_cli.chat.prompts import answer_prompt, rewrite_prompt
from rag_cli.llm import with_retry

log = logging.getLogger(__name__)


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)


def format_sources(docs) -> str:
    sources = sorted({d.metadata.get("source") for d in docs if d.metadata.get("source")})
    if not sources:
        return ""
    return "\n\n---\n**Nguồn truy hồi:**\n" + "\n".join(f"- `{s}`" for s in sources)


class ChatPipeline:
    def __init__(self, llm, retriever, history: ChatHistory | None = None):
        self.retriever = retriever
        self.history = history if history is not None else ChatHistory()
        parser = StrOutputParser()
        self._rewrite_chain = rewrite_prompt | llm | parser
        self._answer_chain = answer_prompt | llm | parser

    def rewrite(self, question: str) -> str:
        """Biến câu hỏi nối tiếp thành câu hỏi độc lập (chỉ khi đã có lịch sử)."""
        if not self.history:
            return question
        return with_retry(lambda: self._rewrite_chain.invoke(
            {"history": self.history.messages(), "question": question}))

    def retrieve(self, standalone_question: str) -> list:
        return self.retriever.invoke(standalone_question)

    def generate(self, question: str, docs: list) -> str:
        return with_retry(lambda: self._answer_chain.invoke(
            {"context": format_docs(docs), "history": self.history.messages(),
             "question": question}))

    def ask(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Vui lòng nhập câu hỏi."

        t0 = time.perf_counter()
        standalone = self.rewrite(question)
        t1 = time.perf_counter()
        docs = self.retrieve(standalone)
        t2 = time.perf_counter()
        log.debug("question=%r docs=%d history=%d", question, len(docs), len(self.history))
        answer = self.generate(question, docs)
        t3 = time.perf_counter()

        if answer.strip():
            self.history.add(question, answer)
        log.info("thời gian: viết lại %.2fs | truy hồi %.2fs | Gemini trả lời %.2fs",
                 t1 - t0, t2 - t1, t3 - t2)
        return answer + format_sources(docs)


def build_pipeline(history: ChatHistory | None = None) -> ChatPipeline:
    """Dựng pipeline thật: nạp chỉ mục từ Chroma (chậm nhất ở bước này) và tạo LLM."""
    from rag_cli.llm import get_llm
    from rag_cli.retrieval.hybrid import build_retriever
    return ChatPipeline(get_llm(0.2), build_retriever(), history)
