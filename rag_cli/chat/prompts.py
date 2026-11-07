from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
