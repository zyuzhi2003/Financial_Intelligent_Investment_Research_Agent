# 这个文件用于集中管理RAG模块的路径与参数配置。
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    knowledge_dir: Path
    persist_dir: Path
    embedding_model_name: str
    hf_endpoint: str
    enable_ocr: bool
    ocr_dpi: int
    min_page_chars: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    collection_name: str
    openai_base_url: str
    openai_model: str
    openai_api_key: str
    enable_react: bool = True
    react_max_steps: int = 3
    memory_enabled: bool = False
    memory_backend: str = "hybrid"
    memory_dir: Path = Path("./data/memory")
    memory_top_k: int = 3
    memory_ttl_days: int = 30
    memory_collection: str = "agent_memory"
    memory_profile_min_confidence: float = 0.7


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"缺少{name}，请先在.env中配置。")
    return value


def _get_env_with_alias(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def load_settings() -> Settings:
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent

    return Settings(
        knowledge_dir=Path(os.getenv("KNOWLEDGE_DIR", project_root / "Knowledge")),
        persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", project_root / "data" / "chroma")),
        embedding_model_name=_get_required_env("EMBEDDING_MODEL"),
        hf_endpoint=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        enable_ocr=_to_bool(os.getenv("ENABLE_OCR", "true")),
        ocr_dpi=int(os.getenv("OCR_DPI", "200")),
        min_page_chars=int(os.getenv("MIN_PAGE_CHARS", "20")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "4")),
        collection_name=os.getenv("CHROMA_COLLECTION", "bank_research_reports"),
        openai_base_url=_get_required_env("OPENAI_BASE_URL"),
        openai_model=_get_required_env("OPENAI_MODEL"),
        openai_api_key=_get_env_with_alias("OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
        enable_react=_to_bool(os.getenv("ENABLE_REACT", "true")),
        react_max_steps=max(1, int(os.getenv("REACT_MAX_STEPS", "3"))),
        memory_enabled=_to_bool(os.getenv("MEMORY_ENABLED", "false")),
        memory_backend=os.getenv("MEMORY_BACKEND", "hybrid").strip().lower(),
        memory_dir=Path(os.getenv("MEMORY_DIR", project_root / "data" / "memory")),
        memory_top_k=max(1, int(os.getenv("MEMORY_TOP_K", "3"))),
        memory_ttl_days=max(1, int(os.getenv("MEMORY_TTL_DAYS", "30"))),
        memory_collection=os.getenv("MEMORY_COLLECTION", "agent_memory").strip() or "agent_memory",
        memory_profile_min_confidence=max(
            0.0,
            min(1.0, float(os.getenv("MEMORY_PROFILE_MIN_CONFIDENCE", "0.7"))),
        ),
    )
