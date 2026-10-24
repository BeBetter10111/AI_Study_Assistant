"""Fixture dùng chung cho test suite.

Các thư viện AI nặng (torch, vllm, outlines, qdrant-client,
sentence-transformers, langchain-text-splitters, PyMuPDF, python-docx,
python-pptx) được "giả" (stub) trước khi bất kỳ module nào trong `app`
được import, vì CI không có GPU và không đủ thời gian/dung lượng để cài
các gói này (xem requirements-ci.txt). Test suite vẫn kiểm tra được toàn
bộ logic thật của API (auth, upload, chat, quiz, RL) vì các module đó chỉ
GỌI đến những thư viện này qua các hàm mỏng — hành vi thật được test riêng
thủ công / kiểm thử tích hợp trên môi trường có GPU thật trước khi release.
"""
import sys
import types

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def _install_stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_heavy_stubs() -> None:
    if "torch" in sys.modules:
        return  # đã stub từ trước (ví dụ khi chạy nhiều test module)

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    _install_stub("torch", cuda=_Cuda)

    class _LLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            return []

    class _SamplingParams:
        def __init__(self, *a, **k):
            pass

    _install_stub("vllm", LLM=_LLM, SamplingParams=_SamplingParams)

    class _OutlineModels:
        @staticmethod
        def vllm(llm):
            return object()

    class _OutlineGenerate:
        @staticmethod
        def json(model, schema_class):
            def _gen(prompt, temperature=0.1):
                raise RuntimeError("stub generator — không dùng trong test")

            return _gen

    _install_stub("outlines", models=_OutlineModels, generate=_OutlineGenerate)

    class _QdrantClient:
        def __init__(self, *a, **k):
            pass

        def get_collections(self):
            class _R:
                collections = []

            return _R()

        def create_collection(self, *a, **k):
            pass

        def upsert(self, *a, **k):
            pass

        def search(self, *a, **k):
            return []

    qdrant_client = _install_stub("qdrant_client", QdrantClient=_QdrantClient)
    qdrant_http = _install_stub("qdrant_client.http")
    _install_stub(
        "qdrant_client.http.models",
        VectorParams=lambda *a, **k: None,
        Distance=types.SimpleNamespace(COSINE="Cosine"),
        PointStruct=lambda *a, **k: None,
        Filter=lambda *a, **k: None,
        FieldCondition=lambda *a, **k: None,
        MatchValue=lambda *a, **k: None,
    )
    qdrant_client.http = qdrant_http

    class _SentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def encode(self, *a, **k):
            return []

    _install_stub("sentence_transformers", SentenceTransformer=_SentenceTransformer)

    class _Splitter:
        def __init__(self, *a, **k):
            pass

        @classmethod
        def from_tiktoken_encoder(cls, *a, **k):
            return cls()

        def split_text(self, text):
            return [text]

    _install_stub("langchain_text_splitters", RecursiveCharacterTextSplitter=_Splitter)

    # Lưu ý: fitz (PyMuPDF), docx, pptx, pytesseract KHÔNG bị stub ở đây —
    # đây là các gói nhẹ (không cần GPU), được cài thật qua
    # requirements-ci.txt để tests/test_ingestion_ocr.py kiểm tra hành vi
    # trích xuất/OCR THẬT thay vì hành vi giả.


_install_heavy_stubs()


@pytest.fixture
async def app_client():
    """AsyncClient gắn thẳng vào FastAPI app, dùng SQLite trong bộ nhớ thay
    cho Postgres thật — đủ để kiểm tra logic route/DB mà không cần Docker.
    """
    from httpx import ASGITransport, AsyncClient

    import app.api.dependencies as deps
    import app.core.database as coredb
    from app.models.models import Base

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

    coredb.engine = test_engine
    coredb.async_session = test_session_maker
    deps.async_session = test_session_maker

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.main as m

    transport = ASGITransport(app=m.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await test_engine.dispose()
