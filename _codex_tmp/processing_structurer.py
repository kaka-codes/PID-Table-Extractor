from io import BytesIO
import re
from typing import Any, Dict, List

from processing.cleaner import clean_extracted_text

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "where",
    "which",
    "with",
}


def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    if PdfReader is None:
        raise RuntimeError(
            "PDF text extraction requires 'pypdf' or 'PyPDF2'. Install one of them "
            "to process digital P&ID PDFs."
        )

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_extracted_text(raw_text)
        pages.append(
            {
                "page_number": page_index,
                "char_count": len(cleaned_text),
                "text": cleaned_text,
            }
        )

    return pages


def _chunk_page_text(
    page_number: int,
    text: str,
    target_chars: int = 900,
    carryover_lines: int = 2,
) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    chunks = []
    current_lines: List[str] = []
    current_length = 0
    chunk_number = 1

    for line in lines:
        projected_length = current_length + len(line) + 1

        if current_lines and projected_length > target_chars:
            chunk_text = "\n".join(current_lines).strip()
            chunks.append(
                {
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )
            chunk_number += 1
            current_lines = current_lines[-carryover_lines:]
            current_length = sum(len(item) + 1 for item in current_lines)

        current_lines.append(line)
        current_length += len(line) + 1

    if current_lines:
        chunks.append(
            {
                "page_number": page_number,
                "chunk_number": chunk_number,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return chunks


def build_pid_document(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    pages = extract_pdf_pages(pdf_bytes)
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        chunks.extend(_chunk_page_text(page_number=page["page_number"], text=page["text"]))

    return {
        "metadata": {
            "source_file": filename,
            "document_type": "digital_pid_pdf",
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "extracted_char_count": sum(page["char_count"] for page in pages),
        },
        "pages": pages,
        "chunks": chunks,
    }


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def retrieve_relevant_chunks(
    document: Dict[str, Any],
    question: str,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    chunks = document.get("chunks", [])
    if not chunks:
        return []

    question_tokens = set(_tokenize(question))
    if not question_tokens:
        return chunks[:top_k]

    question_lower = question.lower()
    scored_chunks = []

    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_text_lower = chunk_text.lower()
        chunk_tokens = set(_tokenize(chunk_text))
        overlap = question_tokens & chunk_tokens

        score = sum(max(len(token) - 2, 1) for token in overlap)
        if question_lower and question_lower in chunk_text_lower:
            score += 8

        for token in question_tokens:
            if len(token) > 2 and token in chunk_text_lower:
                score += 1

        if score > 0:
            scored_chunks.append((score, chunk))

    if not scored_chunks:
        return chunks[:top_k]

    scored_chunks.sort(
        key=lambda item: (
            -item[0],
            item[1]["page_number"],
            item[1]["chunk_number"],
        )
    )
    return [chunk for _, chunk in scored_chunks[:top_k]]
