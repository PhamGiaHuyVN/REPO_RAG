#hashlib : để xem file có thay đổi hay không
#subprocess: chạy git log từ python
import hashlib, json, subprocess

#đơn vị dữ liệu chuẩn của langchain goom page_content và metadata
# mọi thành phần khác splitẻ , vectordb, retriever đều nhận trả về document
from langchain_core.documents import Document
# cắt văn bản thành chunks nhỏ
# recursive là cắt theo thứ tự ưu tiên
# đoạn văn -> dòng -> câu -> từ -> ký tu
from langchain_text_splitters import RecursiveCharacterTextSplitter
# cầu nối gửi tới localhost:11434
# bge-m3 trả về vector 1024 đại diện cho ý nghĩa của văn bản
# haại đoạn có nghĩa gần nhau sẽ có vector gần nhau
from langchain_ollama import OllamaEmbeddings
# lưu chunk , vector, metadata xuống ổ đĩa (.repo_index)
# hỗ trợ add_document : thm chunk tự gọi embedding
# similarity_search : tìm cascchunk gần nghĩa nhất với câu hỏi
# delete(where=): xóa theo đk metadata
# get : lấy toàn bộ dữ liệu( đựng BM25 trong retriever.py)
from langchain_chroma import Chroma


from config import *

# keest nối tới ollama, dùng bge-m3 để chuyn văn bản thành vector
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
# chroma tự gọi ollama mỗi lần thêm hoặc tìm kiếm
# db là module nên co thể dùng chung
db = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(DB_DIR))

def iter_files():
    # duyệt mọi file trong repo
    for p in REPO.rglob("*"):
        # không lấy thư mục
        if not p.is_file():
            continue
        if set(p.parts) & IGNORE_DIRS or p.name in IGNORE_FILES:
            continue
        if p.suffix in IGNORE_SUFFIX or p.suffix not in TEXT_EXT:
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        #trả veef từng file 1
        yield p

# cắt file thành chunk
def split(path: Path, text: str):
    lang = EXT_LANG.get(path.suffix)
    if lang:
        sp = RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    else:
        sp = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    rel = str(path.relative_to(REPO))
    docs = sp.create_documents([text], metadatas=[{
        "source": rel, "ext": path.suffix, "type": "code"
    }])
    for i, d in enumerate(docs):
        d.page_content = f"# File: {rel}\n{d.page_content}"
        d.metadata["chunk"] = i
    return docs

# chạy gitlog trong repo, lấy 200 commit gần nhất
# hash | tác giả | ngày | nội dung
# mỗi dòng thành 1 document loại commit
def git_history(limit=200):
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", f"-{limit}",
         "--pretty=format:%h|%an|%ad|%s", "--date=short"],
        capture_output=True, text=True).stdout
    return [Document(page_content=line, metadata={"source": "git:log", "type": "commit"})
            for line in out.splitlines() if line]

#  tạo thư mục .repo_index
# old bảng hash của index trước
# new bảng hash của lần này
def index_repo(full=False):
    DB_DIR.mkdir(exist_ok=True)
    state_file = DB_DIR / "hashes.json"
    old = {} if full or not state_file.exists() else json.loads(state_file.read_text())
    new = {}
    for p in iter_files():
        text = p.read_text(errors="ignore")
        h = hashlib.md5(text.encode()).hexdigest()
        rel = str(p.relative_to(REPO))
        # với mỗi file đọc nội dung tính hash MD5
        docs = split(p, text)
        if not docs:
            continue
        db.delete(where={"source": rel})
        db.add_documents(docs)
        print(f"indexed: {rel}")
    for rel in set(old) - set(new):
        db.delete(where={"source": rel})
        print(f"removed: {rel}")
    # file có trong old mà không có trong new là đã bị xóa khỏi repo
    # lưu bảng hash mới vo hashes.json
    commits = git_history()
    db.delete(where={"type": "commit"})
    if commits:
        db.add_documents(commits)
    state_file.write_text(json.dumps(new))