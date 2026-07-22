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
        message = (
            "Embedding模型下载失败，可能无法连接huggingface.co。"
            "请在.env中配置 HF_ENDPOINT=https://hf-mirror.com 后重试；"
            "或先手动下载模型到本地缓存再离线运行。"
        )
        raise RuntimeError(message) from exc
