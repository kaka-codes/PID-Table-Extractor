import re
from typing import Any, Dict, Iterable


MULTI_VALUE_DELIMITER_RE = re.compile(r"\s*[;,]\s*")


def _expand_multivalue_field_line(line: str) -> str:
    if ":" not in line:
        return line

    field_name, remainder = line.split(":", 1)
    remainder = remainder.strip()
    field_name_lower = field_name.lower()

    if (
        not remainder
        or ("pressure" not in field_name_lower and "temperature" not in field_name_lower)
        or ("," not in remainder and ";" not in remainder)
    ):
        return line

    parts = [part.strip() for part in MULTI_VALUE_DELIMITER_RE.split(remainder) if part.strip()]
    if len(parts) <= 1:
        return line

    return " , ".join(f"{field_name}: {part}" for part in parts)


def format_chunk_text_for_prompt(text: str) -> str:
    lines = []

    for line in (text or "").splitlines():
        lines.append(_expand_multivalue_field_line(line))

    return "\n".join(lines)


def build_qa_prompt(
    question: str,
    retrieved_chunks: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    source_files = metadata.get("source_files") or [
        metadata.get("source_file", "uploaded P&ID PDF")
    ]
    context_sections = []

    for chunk in retrieved_chunks:
        location_bits = [
            f"Source: {chunk.get('source_file', 'uploaded PDF')}",
            f"Page: {chunk.get('page_number', '?')}",
            f"Table: {chunk.get('table_number', '?')}",
        ]

        split_number = chunk.get("split_number")
        if split_number:
            location_bits.append(f"Section: {split_number}")

        equipment_number = chunk.get("equipment_number")
        if equipment_number:
            location_bits.append(f"Equipment: {equipment_number}")

        context_sections.append(
            f"[{' | '.join(location_bits)}]\n{format_chunk_text_for_prompt(chunk['text'])}"
        )

    context = "\n\n".join(context_sections) if context_sections else "No context available."
    uploaded_sources = "\n".join(f"- {source_file}" for source_file in source_files)
    return f"""
You are an expert assistant for equipment specification tables extracted from P&ID PDFs.

Rules:
- Use only the provided context.
- If the answer is missing, say you cannot find it in the extracted table data.
- Keep tag numbers, line numbers, and equipment identifiers exactly as written.
- Keep units, materials, and design codes exactly as written.
- Mention the source file and page number when they help support the answer.
- Keep the answer concise and factual.
- Number of equipment in the pdf will be the same as the number of different item numbers or tag numbers you find in the pdf.
When answering:
1. Do not confuse pressure with temperature.
2. If multiple values exist, return only the exact requested one.
3. Match the requested property exactly.
  
  Example - "TEMPERATURE | DESIGN °C: TUBE"  means "design temperature of tube" or "tube design temperature" .
  Example - "PRESSURE | OPERATING kPag: TUBE(IN/OUT)" means "operating pressure of tube" or "tube operating pressure" .
  Example - "TEMPERATURE | OPERATING °C: SHELL(IN/OUT) : 35/55" means "operating temperature of shell" or "shell operating temperature" .

Uploaded sources:
{uploaded_sources}

Context:
{context}

Question:
{question}

Answer:
""".strip()
