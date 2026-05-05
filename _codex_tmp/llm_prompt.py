from typing import Any, Dict, Iterable


def build_qa_prompt(
    question: str,
    retrieved_chunks: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    source_file = metadata.get("source_file", "uploaded P&ID PDF")
    context_sections = []

    for chunk in retrieved_chunks:
        context_sections.append(
            f"[Page {chunk['page_number']} | Chunk {chunk['chunk_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_sections) if context_sections else "No context available."

    return f"""
You are an expert assistant for reading text extracted from digital P&ID PDFs.

Rules:
- Use only the provided context.
- If the answer is missing or uncertain, say you cannot find it in the extracted PDF content.
- Keep tag numbers, line numbers, and equipment identifiers exactly as written.
- Mention page numbers when they help support the answer.
- Keep the answer concise and factual.

Source file:
{source_file}

Context:
{context}

Question:
{question}

Answer:
""".strip()
