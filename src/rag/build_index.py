# 这个文件用于加载PDF、切片并构建本地Chroma向量索引。
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.config import Settings
from src.rag.embeddings import create_embeddings
from src.rag.ocr import extract_pdf_pages_tables, extract_pdf_pages_text
from src.rag.structural_chunking import structural_split_documents


def _find_pdf_files(knowledge_dir: Path) -> List[Path]:
    return sorted(knowledge_dir.rglob("*.pdf"))


def _load_documents(
    pdf_files: Iterable[Path],
    settings: Settings,
    logger: Callable[[str], None] | None,
    max_pages: int | None = None,
) -> tuple[List[Document], dict[str, int]]:
    docs: List[Document] = []
    stats = {
        "file_count": 0,
        "page_count": 0,
        "direct_pages": 0,
        "ocr_pages": 0,
        "table_pages": 0,
        "table_count": 0,
        "empty_pages": 0,
    }

    files = list(pdf_files)
    total_files = len(files)

    for idx, pdf in enumerate(files, start=1):
        if logger:
            logger(f"[1/4] 读取PDF {idx}/{total_files}: {pdf.name}")

        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        if max_pages and max_pages > 0:
            loaded = loaded[:max_pages]
        stats["file_count"] += 1
        stats["page_count"] += len(loaded)

        empty_page_indexes: list[int] = []
        for item in loaded:
            item.metadata["source"] = str(pdf)
            text = (item.page_content or "").strip()
            if len(text) >= settings.min_page_chars:
                docs.append(item)
                stats["direct_pages"] += 1
            else:
                page_idx = item.metadata.get("page")
                if isinstance(page_idx, int):
                    empty_page_indexes.append(page_idx)

        if settings.enable_ocr and empty_page_indexes:
            if logger:
                logger(
                    f"  - OCR处理中: {len(empty_page_indexes)} 页 (dpi={settings.ocr_dpi})"
                )
            ocr_results: dict[int, object] = {}
            ocr_texts = extract_pdf_pages_text(
                pdf, empty_page_indexes, settings.ocr_dpi, ocr_results
            )
            ocr_tables = (
                extract_pdf_pages_tables(
                    pdf, empty_page_indexes, settings.ocr_dpi, ocr_results
                )
                if settings.enable_table_ocr
                else {}
            )
            ocr_hit_pages = 0
            for item in loaded:
                page_idx = item.metadata.get("page")
                if not isinstance(page_idx, int):
                    continue
                ocr_text = (ocr_texts.get(page_idx) or "").strip()
                if len(ocr_text) >= settings.min_page_chars:
                    item.page_content = ocr_text
                    item.metadata["_ocr_tables"] = ocr_tables.get(page_idx, [])
                    docs.append(item)
                    ocr_hit_pages += 1
                    stats["ocr_pages"] += 1
                    if ocr_tables.get(page_idx):
                        stats["table_pages"] += 1
                        stats["table_count"] += len(ocr_tables[page_idx])
            stats["empty_pages"] += max(0, len(empty_page_indexes) - ocr_hit_pages)
            if logger:
                logger(f"  - OCR完成: 命中 {ocr_hit_pages} 页")
        else:
            stats["empty_pages"] += len(empty_page_indexes)

        if logger:
            logger(
                f"  - 文件完成: 总页={len(loaded)} 直接抽取={len(loaded) - len(empty_page_indexes)} "
                f"待OCR={len(empty_page_indexes)}"
            )

    return docs, stats


def build_vector_index(
    settings: Settings,
    max_pdfs: int | None = None,
    logger: Callable[[str], None] | None = print,
    pdf_path: Path | None = None,
    max_pages: int | None = None,
) -> tuple[int, int]:
    pdf_files = [pdf_path] if pdf_path else _find_pdf_files(settings.knowledge_dir)
    if not pdf_files:
        raise FileNotFoundError(f"未在目录中找到PDF文件: {settings.knowledge_dir}")
    if max_pdfs and max_pdfs > 0:
        pdf_files = pdf_files[:max_pdfs]

    if logger:
        logger(f"开始构建索引: 文件数={len(pdf_files)}")

    raw_docs, stats = _load_documents(pdf_files, settings, logger, max_pages=max_pages)
    if not raw_docs:
        raise ValueError(
            "未从PDF中提取到有效文本。可能是扫描版PDF或字体加密导致。"
            "请先替换为可复制文本的PDF，或开启OCR预处理后再构建索引。"
        )

    if logger:
        logger(
            "[2/4] 文本提取完成: "
            f"总页={stats['page_count']} 直接抽取页={stats['direct_pages']} "
            f"OCR命中页={stats['ocr_pages']} 仍为空页={stats['empty_pages']}"
        )

    chunks = structural_split_documents(raw_docs, settings)
    if not chunks:
        raise ValueError(
            "文本切分后结果为空，无法构建索引。"
            "请检查CHUNK_SIZE配置，或确认PDF内容是否为可解析文本。"
        )
    if logger:
        logger(f"[3/4] 文本切分完成: 文档片段数={len(chunks)}")

    embedding = create_embeddings(settings)
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding,
        persist_directory=str(settings.persist_dir),
    )

    # 先清空已有集合再重建，保证开发调参时结果可复现。
    try:
        vectorstore.delete_collection()
    except ValueError:
        # 首次构建时集合可能不存在，忽略即可。
        pass
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding,
        persist_directory=str(settings.persist_dir),
    )
    if logger:
        logger("[4/4] 正在写入向量库，请稍候...")
    vectorstore.add_documents(chunks)

    if logger:
        logger("向量库写入完成。")

    return len(pdf_files), len(chunks)
