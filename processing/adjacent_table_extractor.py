import json
import re
import time
from io import BytesIO
from typing import Any, Dict

import pandas as pd
import pdfplumber
from google import genai
from google.genai import types


API_KEY = "AIzaSyDTdgv_Ap-nhZFJMGpJZeuVyWaAnI3cGzo"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

if not API_KEY:
    raise ValueError("Set GEMINI_API_KEY or provide a valid API key before running this script.")

client = genai.Client(
    api_key=API_KEY,
    # Ignore broken system proxy variables such as 127.0.0.1:9.
    http_options=types.HttpOptions(clientArgs={"trust_env": False}),
)


def dataframe_to_text(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        clean_row = [str(cell).strip() for cell in row]
        rows.append(" | ".join(clean_row))
    return "\n".join(rows)


def is_high_demand_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    return status_code == 503 or (
        "503" in message
        and (
            "high demand" in message
            or "try again later" in message
            or "unavailable" in message
        )
    )


def extract_required_data(table_text: str) -> Dict[str, Any]:
    table_text = table_text[:12000]

    prompt = f"""
    You are reading an engineering equipment datasheet table extracted from a PDF.

    Extract:
    - revision_number
    - document_numbers

    Rules:
    - Extract ALL document numbers in the table

    - For revision_numbers:
    1. Prefer the COMPANY REVISION CODE if explicitly mentioned (e.g., "Company Rev", "Client Rev", "Company Revision")
    2. If latest company revision is NOT present, extract the latest revision code available in the table
    3. Latest revision = most recent entry

    Return ONLY valid JSON in this exact format:

    {{
    "revision_number": "",
    "document_numbers": []
    }}

    Additional rules:
    - Do NOT return multiple revisions, only one final value based on the latest date above
    - Do NOT hallucinate values
    - Ignore empty or irrelevant rows
    - If nothing is found, return "" for revision_numbers and [] for document_numbers
    - Output strictly JSON only (no explanation)

    TABLE:
    {table_text}
    """

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
            )
            break
        except Exception as error:
            if not is_high_demand_error(error) or attempt == max_attempts:
                raise

            wait_seconds = attempt * 2
            time.sleep(wait_seconds)

    raw = response.text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    parsed = json.loads(raw)
    document_numbers = parsed.get("document_numbers")

    if isinstance(document_numbers, list) and len(document_numbers) >= 3:
        parsed["document_numbers"] = document_numbers[-3:]

    return parsed


def extract_required_data_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"ok": False, "status": "empty_table", "data": {}}

    table_text = dataframe_to_text(df)
    if not table_text.strip():
        return {"ok": False, "status": "empty_table", "data": {}}

    try:
        parsed = extract_required_data(table_text)
    except Exception as error:
        raw_output = getattr(error, "raw_output", "")
        message = str(error)
        status = "parse_error" if isinstance(error, json.JSONDecodeError) else "request_error"
        result: Dict[str, Any] = {
            "ok": False,
            "status": status,
            "message": message,
            "data": {},
        }
        if raw_output:
            result["raw_output"] = raw_output
        return result

    revision_number = str(parsed.get("revision_number", "") or "").strip()
    document_numbers = parsed.get("document_numbers", [])

    if not isinstance(document_numbers, list):
        document_numbers = [document_numbers] if document_numbers else []

    normalized_document_numbers = [
        str(value).strip() for value in document_numbers if str(value).strip()
    ]

    return {
        "ok": True,
        "status": "ok",
        "data": {
            "revision_number": revision_number,
            "document_numbers": normalized_document_numbers,
        },
    }


def _extract_raw_tables_for_adjacent_lookup(pdf_bytes: bytes) -> list[dict[str, Any]]:
    pdf_source = BytesIO(pdf_bytes)
    all_tables: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_source) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []

            for table_no, table in enumerate(tables, start=1):
                if not table:
                    continue

                df = pd.DataFrame(table).fillna("")
                all_tables.append(
                    {
                        "page_number": page_no,
                        "table_number": table_no,
                        "dataframe": df,
                    }
                )

    return all_tables


def extract_required_data_from_next_source_table(
    pdf_bytes: bytes,
    source_page_number: int,
    source_table_number: int,
) -> Dict[str, Any]:
    all_tables = _extract_raw_tables_for_adjacent_lookup(pdf_bytes)
    source_index = None

    for index, item in enumerate(all_tables):
        if (
            item["page_number"] == source_page_number
            and item["table_number"] == source_table_number
        ):
            source_index = index
            break

    if source_index is None:
        return {
            "ok": False,
            "status": "source_table_not_found",
            "message": (
                "The displayed equipment table's source table could not be matched "
                "in the raw pdfplumber table sequence."
            ),
            "data": {},
        }

    next_index = source_index + 1
    if next_index >= len(all_tables):
        return {
            "ok": False,
            "status": "next_table_not_found",
            "message": "No next raw table was found after the displayed equipment table's source table.",
            "data": {},
        }

    next_table = all_tables[next_index]
    extraction_result = extract_required_data_from_dataframe(next_table["dataframe"])
    extraction_result["adjacent_table_page_number"] = next_table["page_number"]
    extraction_result["adjacent_table_table_number"] = next_table["table_number"]
    return extraction_result
