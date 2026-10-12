# 这个文件用于统一初始化Embedding模型，并处理网络受限场景下的镜像与报错提示。
from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings


def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    try:
        return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    except OSError as exc:
        fallback_model = os.getenv(
            "EMBEDDING_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        if fallback_model and fallback_model != settings.embedding_model_name:
            try:
                return HuggingFaceEmbeddings(
                    model_name=fallback_model,
                    model_kwargs={"local_files_only": True},
                )
            except Exception:
                pass
        message = (
            "Embedding模型下载失败，可能无法连接huggingface.co。"
            "请在.env中配置 HF_ENDPOINT=https://hf-mirror.com 后重试；"
            "或先手动下载模型到本地缓存再离线运行。"
            f"当前备用模型={fallback_model or '未配置'}也不可用。"
        )
        raise RuntimeError(message) from exc
