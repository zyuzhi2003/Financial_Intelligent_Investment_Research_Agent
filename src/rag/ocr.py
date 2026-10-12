# 这个文件用于给扫描版PDF做OCR提取，补齐常规文本抽取失败的页面。
from __future__ import annotations

from html.parser import HTMLParser
import os
from pathlib import Path
from statistics import median
from typing import Any


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []
        elif tag == "tr":
            self.current_row = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.in_cell:
            self.current_row.append(" ".join("".join(self.current_cell).split()))
            self.in_cell = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def _html_table_rows(html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return [row for row in parser.rows if any(row)]


def _combine_header_rows(
    header_rows: list[list[tuple[float, float, float, float, str]]],
    data_row: list[tuple[float, float, float, float, str]],
) -> list[str]:
    if len(header_rows) < 2:
        return [box[4] for box in sorted(header_rows[0], key=lambda item: item[0])]
    leaf_row = max(header_rows, key=len)
    parent_boxes = [
        box
        for row in header_rows
        if row is not leaf_row
        for box in row
        if "图表" not in box[4]
    ]
    parent_boxes.sort(key=lambda item: item[0])
    name_parent = next((box for box in parent_boxes if "证券名称" in box[4]), None)
    group_parents = [box for box in parent_boxes if box is not name_parent]
    parent_centers = [(box[0] + box[2]) / 2 for box in group_parents]
    parent_boundaries = [
        (left + right) / 2 for left, right in zip(parent_centers, parent_centers[1:])
    ]
    combined: list[str] = []
    for column_index, data_box in enumerate(sorted(data_row, key=lambda item: item[0])):
        x = (data_box[0] + data_box[2]) / 2
        parent_index = 0
        while parent_index < len(parent_boundaries) and x >= parent_boundaries[parent_index]:
            parent_index += 1
        if column_index == 0 and name_parent is not None:
            parent = name_parent[4]
        else:
            parent = group_parents[parent_index][4] if group_parents else ""
        child = min(leaf_row, key=lambda item: abs(item[0] - data_box[0]), default=None)
        child_text = child[4] if child and abs(child[0] - data_box[0]) <= 70 else ""
        if parent and child_text and parent != child_text:
            combined.append(f"{parent}-{child_text}")
        else:
            combined.append(parent or child_text or f"第{len(combined) + 1}列")
    return combined


def _ocr_boxes(ocr_result: Any) -> list[tuple[float, float, float, float, str]]:
    boxes: list[tuple[float, float, float, float, str]] = []
    for item in ocr_result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        polygon, value = item[0], item[1]
        if not isinstance(value, str) or not isinstance(polygon, (list, tuple)):
            continue
        points = [point for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) < 4:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        text = " ".join(value.split())
        if text:
            boxes.append((min(xs), min(ys), max(xs), max(ys), text))
    return boxes


def _coordinate_table_rows(ocr_result: Any) -> list[list[list[str]]]:
    boxes = _ocr_boxes(ocr_result)
    if not boxes:
        return []
    heights = [box[3] - box[1] for box in boxes if box[3] > box[1]]
    tolerance = max(6.0, median(heights) * 0.7 if heights else 8.0)
    rows: list[list[tuple[float, float, float, float, str]]] = []
    centers: list[float] = []
    for box in sorted(boxes, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        center = (box[1] + box[3]) / 2
        nearest = min(range(len(centers)), key=lambda index: abs(centers[index] - center), default=-1)
        if nearest < 0 or abs(centers[nearest] - center) > tolerance:
            rows.append([box])
            centers.append(center)
        else:
            rows[nearest].append(box)
            centers[nearest] = sum((item[1] + item[3]) / 2 for item in rows[nearest]) / len(rows[nearest])
    for row in rows:
        row.sort(key=lambda item: item[0])

    candidates = [row for row in rows if len(row) >= 2]
    tables: list[list[list[str]]] = []
    current: list[list[tuple[float, float, float, float, str]]] = []
    previous_starts: list[float] = []

    def finish_table(table_rows: list[list[tuple[float, float, float, float, str]]]) -> None:
        if len(table_rows) < 3:
            return
        first_y = min(box[1] for box in table_rows[0])
        preceding_rows = [
            row
            for row in rows
            if first_y - 140 <= (row[0][1] + row[0][3]) / 2 < first_y
        ]
        preceding = [box for row in preceding_rows for box in row]
        if len(preceding) >= 2 and any(
            any(token in box[4] for token in ("证券名称", "银行", "25Q", "24Q", "25H", "项目"))
            for box in preceding
        ):
            header = _combine_header_rows(preceding_rows, table_rows[0])
            tables.append([header] + [[box[4] for box in row] for row in table_rows])
            return
        tables.append([[box[4] for box in row] for row in table_rows])

    for row in candidates:
        starts = [box[0] for box in row]
        same_columns = len(starts) == len(previous_starts) and all(
            abs(left - right) <= 35 for left, right in zip(starts, previous_starts)
        )
        if current and not same_columns:
            finish_table(current)
            current = []
        current.append(row)
        previous_starts = starts
    finish_table(current)
    def is_table_index(table: list[list[str]]) -> bool:
        first_cells = [row[0] for row in table if row]
        return bool(first_cells) and sum("图表" in cell for cell in first_cells) >= len(first_cells) * 0.6

    return [table for table in tables if not is_table_index(table)]


def extract_pdf_pages_text(
    pdf_path: Path,
    page_indexes: list[int],
    dpi: int,
    ocr_results: dict[int, Any] | None = None,
) -> dict[int, str]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("缺少依赖pymupdf，请先安装 requirements.txt。") from exc

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("缺少依赖rapidocr-onnxruntime，请先安装 requirements.txt。") from exc

    page_to_text: dict[int, str] = {}
    if not page_indexes:
        return page_to_text

    ocr_engine = RapidOCR()
    with fitz.open(str(pdf_path)) as doc:
        for page_idx in page_indexes:
            if page_idx < 0 or page_idx >= doc.page_count:
                continue

            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image_bytes = pix.tobytes("png")

            ocr_result, _ = ocr_engine(image_bytes)
            if ocr_results is not None:
                ocr_results[page_idx] = ocr_result
            if not ocr_result:
                continue

            pieces: list[str] = []
            for item in ocr_result:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                    text = item[1].strip()
                    if text:
                        pieces.append(text)

            merged = "\n".join(pieces).strip()
            if merged:
                page_to_text[page_idx] = merged

    return page_to_text


def extract_pdf_pages_tables(
    pdf_path: Path,
    page_indexes: list[int],
    dpi: int,
    ocr_results: dict[int, Any] | None = None,
) -> dict[int, list[list[list[str]]]]:
    try:
        import fitz
        import numpy as np
    except ImportError:
        return {}

    if not page_indexes:
        return {}

    table_engine = None
    if os.getenv("USE_PADDLE_TABLE_OCR", "false").strip().lower() in {"1", "true", "yes"}:
        try:
            from paddleocr import PPStructure

            table_engine = PPStructure(show_log=False, lang="ch")
        except Exception:
            table_engine = None
    try:
        from rapidocr_onnxruntime import RapidOCR

        coordinate_engine = RapidOCR()
    except ImportError:
        coordinate_engine = None

    page_to_tables: dict[int, list[list[list[str]]]] = {}
    with fitz.open(str(pdf_path)) as doc:
        for page_idx in page_indexes:
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            tables: list[list[list[str]]] = []
            if table_engine is not None:
                try:
                    result = table_engine(image)
                    for item in result or []:
                        if not isinstance(item, dict) or item.get("type") != "table":
                            continue
                        result_data = item.get("res", {})
                        html = result_data.get("html", "") if isinstance(result_data, dict) else ""
                        rows = _html_table_rows(str(html))
                        if rows:
                            tables.append(rows)
                except Exception:
                    tables = []
            if not tables and (ocr_results is not None or coordinate_engine is not None):
                ocr_result = ocr_results.get(page_idx) if ocr_results is not None else None
                if ocr_result is None and coordinate_engine is not None:
                    ocr_result, _ = coordinate_engine(image)
                tables = _coordinate_table_rows(ocr_result)
            if tables:
                page_to_tables[page_idx] = tables

    return page_to_tables
