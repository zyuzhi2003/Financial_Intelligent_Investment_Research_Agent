# 这个文件用于给扫描版PDF做OCR提取，补齐常规文本抽取失败的页面。
from __future__ import annotations

from pathlib import Path


def extract_pdf_pages_text(pdf_path: Path, page_indexes: list[int], dpi: int) -> dict[int, str]:
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
