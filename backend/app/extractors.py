from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from docx import Document
from pypdf import PdfReader


SUPPORTED_DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@dataclass(frozen=True)
class ExtractedPage:
    page: int | None
    text: str


def extract_document(path: Path) -> List[ExtractedPage]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _from_pdf(path)
    if ext in {".txt", ".md"}:
        return [ExtractedPage(page=None, text=path.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".docx":
        return [ExtractedPage(page=None, text=_from_docx(path))]
    raise ValueError(f"Unsupported file format: {ext}")


def _from_pdf(path: Path) -> List[ExtractedPage]:
    reader = PdfReader(str(path))
    pages: List[ExtractedPage] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(ExtractedPage(page=i + 1, text=text))
    return pages


def _from_docx(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)
