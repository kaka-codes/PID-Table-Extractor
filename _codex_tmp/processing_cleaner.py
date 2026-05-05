import re


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
BROKEN_TAG_RE = re.compile(r"(?<=\w)\s*-\s*(?=\w)")


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = CONTROL_CHAR_RE.sub(" ", normalized)

    cleaned_lines = []
    for raw_line in normalized.split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            line = BROKEN_TAG_RE.sub("-", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = BLANK_LINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()
