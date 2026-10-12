from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


_HEADING_PATTERN = re.compile(
    r"^(?:第\s*[一二三四五六七八九十百千万0-9]+[章节部分]|"
    r"[一二三四五六七八九十百千万]+[、.]|[0-9]+(?:\.[0-9]+){0,3}[、. ]|"
    r"(?:摘要|目录|风险提示|投资建议|行业分析|公司概况|财务分析|估值分析|结论))"
)


def _is_heading(text: str, max_font_size: float) -> bool:
    clean = " ".join(text.split())
    if not clean or len(clean) > 120:
        return False
    return bool(_HEADING_PATTERN.match(clean)) or max_font_size >= 14 and len(clean) <= 80


def _overlaps(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def _table_rows(table: object) -> list[list[str]]:
    try:
        rows = table.extract()  # type: ignore[attr-defined]
    except Exception:
        return []
    normalized: list[list[str]] = []
    for row in rows or []:
        values = [" ".join(str(value or "").split()) for value in row]
        if any(values):
            normalized.append(values)
    return normalized


def _split_table_rows(
    rows: list[list[str]], settings: Settings
) -> list[tuple[str, str, int, int]]:
    if not rows:
        return []
    column_count = max(len(row) for row in rows)
    first_row = rows[0] + [""] * (column_count - len(rows[0]))
    numeric_cells = sum(any(char.isdigit() for char in cell) for cell in first_row)
    header_markers = ("证券名称", "银行", "项目", "季度", "环比", "同比", "25Q", "24Q", "25H", "24H")
    has_header = numeric_cells < max(1, column_count // 2) or any(
        marker in cell for cell in first_row for marker in header_markers
    )
    if "资产类型" in first_row[0] and column_count == 13:
        metric_groups = ("余额增速", "余额结构", "环比增量", "同比增量占比")
        periods = ("25Q1", "25H1", "环比")
        header = ["资产类型"] + [f"{group}-{period}" for group in metric_groups for period in periods]
        has_header = True
    else:
        header = first_row if has_header else [f"第{index + 1}列" for index in range(column_count)]
    header_text = " | ".join(header)
    column_description = "、".join(
        value or f"第{index + 1}列" for index, value in enumerate(header)
    )
    parts: list[list[str]] = []
    current_rows: list[str] = []
    data_rows = rows[1:] if has_header else rows
    for row in data_rows:
        normalized_row = row + [""] * (column_count - len(row))
        row_text = " | ".join(normalized_row)
        candidate = "表头: " + header_text + "\n" + "\n".join(current_rows + [row_text])
        if current_rows and len(candidate) > settings.chunk_size:
            parts.append(current_rows)
            current_rows = []
        current_rows.append(row_text)
    if current_rows or len(rows) == 1:
        parts.append(current_rows)
    total = len(parts)
    return [
        (
            "表头: " + header_text + "\n" + "\n".join(part_rows),
            column_description,
            index,
            total,
        )
        for index, part_rows in enumerate(parts, start=1)
    ]


def _extract_page_blocks(
    page: object, fallback_text: str
) -> tuple[list[tuple[str, bool]], list[list[list[str]]], bool]:
    try:
        page_dict = page.get_text("dict")  # type: ignore[attr-defined]
        finder = page.find_tables()  # type: ignore[attr-defined]
        tables = list(getattr(finder, "tables", []))
    except (AttributeError, TypeError, RuntimeError):
        return [(fallback_text, False)], [], False

    table_regions = [tuple(table.bbox) for table in tables if getattr(table, "bbox", None)]
    table_chunks = [rows for table in tables if (rows := _table_rows(table))]
    blocks: list[tuple[str, bool]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        if any(_overlaps(bbox, region) for region in table_regions):
            continue
        lines = []
        font_size = 0.0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            lines.append("".join(str(span.get("text", "")) for span in spans))
            font_size = max(font_size, *(float(span.get("size", 0)) for span in spans))
        text = "\n".join(lines).strip()
        if text:
            blocks.append((text, _is_heading(text, font_size)))
    return blocks or [(fallback_text, False)], table_chunks, bool(blocks or table_chunks)


def _fallback_chunks(documents: Iterable[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    for chunk in chunks:
        chunk.metadata.setdefault("chunk_type", "text")
    return chunks


def structural_split_documents(documents: Iterable[Document], settings: Settings) -> list[Document]:
    documents = list(documents)
    grouped: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        grouped[str(document.metadata.get("source", "unknown"))].append(document)

    try:
        import fitz
    except ImportError:
        return _fallback_chunks(documents, settings)

    sections: list[Document] = []
    for source, source_documents in grouped.items():
        source_documents.sort(key=lambda item: int(item.metadata.get("page", 0)))
        section_title = ""
        section_parts: list[str] = []
        section_pages: list[int] = []
        table_documents: list[Document] = []

        try:
            pdf = fitz.open(source)
        except Exception:
            sections.extend(_fallback_chunks(source_documents, settings))
            continue

        try:
            for document in source_documents:
                page_index = document.metadata.get("page")
                if not isinstance(page_index, int) or page_index >= pdf.page_count:
                    sections.append(document)
                    continue
                blocks, tables, has_structure = _extract_page_blocks(
                    pdf.load_page(page_index), document.page_content
                )
                ocr_tables = document.metadata.pop("_ocr_tables", [])
                tables.extend(ocr_tables)
                if not has_structure and not tables:
                    if section_parts:
                        sections.append(
                            Document(
                                page_content="\n\n".join(section_parts),
                                metadata={
                                    **document.metadata,
                                    "page": section_pages[0],
                                    "page_end": section_pages[-1],
                                    "section": section_title,
                                    "chunk_type": "section",
                                },
                            )
                        )
                        section_parts = []
                        section_pages = []
                    document.metadata["chunk_type"] = "text"
                    sections.append(document)
                    continue
                for table_index, table_rows in enumerate(tables, start=1):
                    table_parts = _split_table_rows(table_rows, settings)
                    for table_text, columns, part, part_count in table_parts:
                        table_documents.append(
                            Document(
                                page_content=table_text,
                                metadata={
                                    **document.metadata,
                                    "chunk_type": "table",
                                    "section": section_title,
                                    "table_index": table_index,
                                    "table_part": part,
                                    "table_parts": part_count,
                                    "table_columns": columns,
                                },
                            )
                        )
                for text, is_heading in blocks:
                    if is_heading and section_parts:
                        sections.append(
                            Document(
                                page_content="\n\n".join(section_parts),
                                metadata={
                                    **document.metadata,
                                    "page": section_pages[0],
                                    "page_end": section_pages[-1],
                                    "section": section_title,
                                    "chunk_type": "section",
                                },
                            )
                        )
                        section_parts = []
                        section_pages = []
                    if is_heading:
                        section_title = " ".join(text.split())
                    section_parts.append(text)
                    section_pages.append(page_index)
        finally:
            pdf.close()

        if section_parts:
            sections.append(
                Document(
                    page_content="\n\n".join(section_parts),
                    metadata={
                        **source_documents[-1].metadata,
                        "page": section_pages[0],
                        "page_end": section_pages[-1],
                        "section": section_title,
                        "chunk_type": "section",
                    },
                )
            )
        sections.extend(table_documents)

    short_chunks: list[Document] = []
    oversized: list[Document] = []
    for document in sections:
        if document.metadata.get("chunk_type") == "table" or len(document.page_content) <= settings.chunk_size:
            short_chunks.append(document)
        else:
            oversized.append(document)
    return short_chunks + _fallback_chunks(oversized, settings)