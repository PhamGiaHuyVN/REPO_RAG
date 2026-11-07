"""Lịch sử hội thoại: giữ N tin nhắn gần nhất (mỗi lượt hỏi-đáp = 2 tin)."""


class ChatHistory:
    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages
        self._messages: list[tuple[str, str]] = []

    def add(self, question: str, answer: str, max_chars: int | None = None) -> None:
        """Ghi một lượt hỏi-đáp. max_chars: cắt bớt câu trả lời quá dài (vd: kết quả /db)."""
        if max_chars is not None:
            answer = answer[:max_chars]
        self._messages.extend([("human", question), ("ai", answer)])
        self._messages = self._messages[-self.max_messages:] if self.max_messages > 0 else []

    def messages(self) -> list[tuple[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
