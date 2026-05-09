import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from processing.adjacent_table_extractor import extract_required_data_from_next_source_table
from processing.cleaner import (
    KEYWORDS,
    apply_keyword_fill_logic,
    find_table_keyword_matches,
    is_valid_table,
    prepare_table,
)
from processing.extractor import extract_tables_perfectly
from processing.ocr_pipeline import extract_ocr_document


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
EMPTY_MARKERS = {"", "none", "-"}
PDFPLUMBER_FILE_SIZE_LIMIT = 600 * 1024


def _normalize_value(value) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).replace("\r", "\n").replace("\n", " ").strip()
    return " ".join(text.split())


def _table_rows(df) -> List[List[str]]:
    if df is None or df.empty:
        return []

    return df.fillna("").astype(str).values.tolist()


def _find_equipment_description_key(fields: Dict[str, Any]) -> Optional[str]:
    for key in fields.keys():
        if str(key).strip().lower() == "description":
            return key
    return None


def _merge_field_values(existing_value: Any, new_value: Any) -> Any:
    if isinstance(existing_value, dict) and isinstance(new_value, dict):
        merged_value = dict(existing_value)
        for sub_key, sub_value in new_value.items():
            if sub_key in merged_value:
                merged_value[sub_key] = _merge_field_values(merged_value[sub_key], sub_value)
            else:
                merged_value[sub_key] = sub_value
        return merged_value

    existing_text = _normalize_value(existing_value)
    new_text = _normalize_value(new_value)

    if not existing_text:
        return new_value
    if not new_text:
        return existing_value
    if existing_text == new_text:
        return existing_value

    existing_parts = [part.strip() for part in existing_text.split(",") if part.strip()]
    for part in [part.strip() for part in new_text.split(",") if part.strip()]:
        if part not in existing_parts:
            existing_parts.append(part)
    return ", ".join(existing_parts)


def _merge_duplicate_equipment_items(equipment_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_items: List[Dict[str, Any]] = []
    description_index: Dict[str, int] = {}

    for item in equipment_items:
        description_key = _find_equipment_description_key(item)
        description_value = _normalize_value(item.get(description_key)) if description_key else ""

        if not description_value:
            merged_items.append(item)
            continue

        description_lookup = description_value.lower()
        if description_lookup not in description_index:
            description_index[description_lookup] = len(merged_items)
            merged_items.append(dict(item))
            continue

        target_item = merged_items[description_index[description_lookup]]
        for key, value in item.items():
            if key in target_item:
                target_item[key] = _merge_field_values(target_item[key], value)
            else:
                target_item[key] = value

    return merged_items


def get_keyword_columns(df, keywords):
    if df is None or df.empty:
        return []

    keyword_cols = []

    for index, col in enumerate(df.columns):
        col_text = " ".join(df[col].astype(str)).lower()
        count = sum(1 for keyword in keywords if keyword in col_text)

        if count >= 2:
            keyword_cols.append(index)

    return keyword_cols


def split_tables(df, keyword_cols):
    if df is None or df.empty:
        return []

    tables = []

    for index in range(len(keyword_cols)):
        start = keyword_cols[index]

        if index < len(keyword_cols) - 1:
            end = keyword_cols[index + 1]
        else:
            end = len(df.columns)

        tables.append(df.iloc[:, start:end])

    return tables


def clean_rows(df):
    if df is None or df.empty:
        return df

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.replace(["None", "", " "], None)
    cleaned_df = cleaned_df.replace(r"^\s*$", None, regex=True)
    cleaned_df = cleaned_df.dropna(axis=1, thresh=2)
    cleaned_df = cleaned_df.dropna(axis=0, thresh=2)
    cleaned_df = cleaned_df.dropna(axis=1, thresh=2)
    if cleaned_df.empty or cleaned_df.shape[1] == 0:
        return cleaned_df.iloc[0:0, 0:0].copy()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df.fillna("")


def table_to_equipment_json(
    df,
    source_file: str,
    page_number: Optional[int] = None,
    table_number: Optional[int] = None,
    split_number: Optional[int] = None,
):
    result = {
        "source_file": source_file,
        "page_number": page_number,
        "table_number": table_number,
        "split_number": split_number,
        "equipment": [],
    }

    if df is None or df.empty or df.shape[1] < 2:
        return result

    # Decide whether column 2 is a nested sub-key column based on the header row.
    # If the second column's first row is blank, it is treated as a nested sub-key
    # column; otherwise the first value column begins at column 1.
    nested_key_col = 1 if df.shape[1] > 2 and _normalize_value(df.iloc[0, 1]) in EMPTY_MARKERS else None
    value_col_start = 2 if nested_key_col is not None else 1

    if df.shape[1] <= value_col_start:
        return result

    valid_cols = []
    for col_idx in range(value_col_start, df.shape[1]):
        col_values = df.iloc[:, col_idx].astype(str).str.strip().str.lower()

        if not col_values.isin(["", "none", "-"]).all():
            valid_cols.append(col_idx)

    prev_col = None
    equipment_items = []

    for col_idx in valid_cols:
        eq_data = {}
        first_row_val = _normalize_value(df.iloc[0, col_idx]).lower()
        apply_fill = first_row_val in EMPTY_MARKERS

        for row_idx in range(len(df)):
            key1 = _normalize_value(df.iloc[row_idx, 0])
            key2 = (
                _normalize_value(df.iloc[row_idx, nested_key_col])
                if nested_key_col is not None
                else ""
            )
            value = _normalize_value(df.iloc[row_idx, col_idx])

            prev_value = ""
            if prev_col is not None:
                prev_value = _normalize_value(df.iloc[row_idx, prev_col])

            if apply_fill and not value:
                value = prev_value

            if not key1 or key1.lower() == "none":
                continue

            if key2 and key2.lower() not in EMPTY_MARKERS:
                if key1 not in eq_data:
                    eq_data[key1] = {}

                eq_data[key1][key2] = value
            else:
                eq_data[key1] = value

        if not eq_data:
            continue

        if first_row_val in EMPTY_MARKERS and equipment_items:
            previous_equipment = equipment_items[-1]
            for key, value in eq_data.items():
                if key in previous_equipment:
                    previous_equipment[key] = _merge_field_values(previous_equipment[key], value)
                else:
                    previous_equipment[key] = value
        else:
            equipment_items.append(eq_data)

        prev_col = col_idx

    result["equipment"] = equipment_items
    return result


def _flatten_fields(fields: Dict[str, Any]) -> List[str]:
    lines = []

    for key, value in fields.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"{key} | {sub_key}: {_normalize_value(sub_value)}")
        else:
            lines.append(f"{key}: {_normalize_value(value)}")

    return lines


def _flatten_common_payload_fields(payload: Dict[str, Any]) -> List[str]:
    lines = []
    skipped_keys = {"source_file", "page_number", "table_number", "split_number", "equipment"}

    for key, value in payload.items():
        if key in skipped_keys:
            continue

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"{key} | {sub_key}: {_normalize_value(sub_value)}")
        elif isinstance(value, list):
            normalized_items = [
                _normalize_value(item) for item in value if _normalize_value(item)
            ]
            lines.append(f"{key}: {', '.join(normalized_items)}")
        else:
            lines.append(f"{key}: {_normalize_value(value)}")

    return [line for line in lines if line.rsplit(":", 1)[-1].strip()]


def build_equipment_chunks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    common_field_lines = _flatten_common_payload_fields(payload)

    for equipment_number, equipment_fields in enumerate(payload.get("equipment", []), start=1):
        field_lines = common_field_lines + _flatten_fields(equipment_fields)
        if not field_lines:
            continue

        chunks.append(
            {
                "source_file": payload["source_file"],
                "page_number": payload.get("page_number"),
                "table_number": payload.get("table_number"),
                "split_number": payload.get("split_number"),
                "equipment_number": equipment_number,
                "context_type": "equipment_fields",
                "text": "\n".join(field_lines),
            }
        )

    return chunks


def build_table_document(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    extracted_tables = extract_tables_perfectly(pdf_bytes)
    extracted_table_previews = []
    matched_preview_count = 0

    for extracted_table in extracted_tables:
        dataframe = extracted_table["dataframe"]
        preview_df = prepare_table(dataframe)
        preview_source_df = preview_df if preview_df is not None and not preview_df.empty else dataframe
        preview_keyword_cols = get_keyword_columns(preview_source_df, KEYWORDS)
        preview_tables = [preview_source_df]

        if len(preview_keyword_cols) > 1:
            preview_tables = split_tables(preview_source_df, preview_keyword_cols)

        for split_index, preview_table in enumerate(preview_tables, start=1):
            cleaned_preview_table = clean_rows(preview_table)
            if cleaned_preview_table is None or cleaned_preview_table.empty:
                continue
            cleaned_preview_table = apply_keyword_fill_logic(cleaned_preview_table)
            if cleaned_preview_table is None or cleaned_preview_table.empty:
                continue

            condition_matches = find_table_keyword_matches(cleaned_preview_table, KEYWORDS)
            matches_conditions = bool(condition_matches)

            extracted_table_previews.append(
                {
                    "page_number": extracted_table["page_number"],
                    "table_number": extracted_table["table_number"],
                    "split_number": split_index if len(preview_tables) > 1 else None,
                    "row_count": int(cleaned_preview_table.shape[0]),
                    "column_count": int(cleaned_preview_table.shape[1]),
                    "rows": _table_rows(cleaned_preview_table),
                    "matches_conditions": matches_conditions,
                    "condition_matches": condition_matches,
                }
            )

            if matches_conditions:
                matched_preview_count += 1

    selected_table = None
    for extracted_table in extracted_tables[:5]:
        if is_valid_table(extracted_table["dataframe"]):
            selected_table = extracted_table
            break

    candidate_tables = [selected_table] if selected_table is not None else []
    adjacent_table_result: Dict[str, Any] = {"ok": False, "status": "not_run", "data": {}}

    if selected_table is None:
        adjacent_table_result = {"ok": False, "status": "selected_table_not_found", "data": {}}
    else:
        adjacent_table_result = extract_required_data_from_next_source_table(
            pdf_bytes=pdf_bytes,
            source_page_number=selected_table["page_number"],
            source_table_number=selected_table["table_number"],
        )

    tables = []
    chunks = []
    chunk_number = 1
    equipment_count = 0

    for extracted_table in candidate_tables:
        prepared_df = prepare_table(extracted_table["dataframe"])
        if prepared_df is None or prepared_df.empty:
            continue

        keyword_cols = get_keyword_columns(prepared_df, KEYWORDS)
        if len(keyword_cols) <= 1:
            split_dfs = [prepared_df]
        else:
            split_dfs = split_tables(prepared_df, keyword_cols)

        for split_index, split_df in enumerate(split_dfs, start=1):
            table_df = clean_rows(split_df)
            if table_df is None or table_df.empty:
                continue
            table_df = apply_keyword_fill_logic(table_df)
            if table_df is None or table_df.empty:
                continue

            equipment_json = table_to_equipment_json(
                table_df,
                source_file=filename,
                page_number=extracted_table["page_number"],
                table_number=extracted_table["table_number"],
                split_number=split_index,
            )
            adjacent_table_data = adjacent_table_result.get("data") or {}
            if adjacent_table_data:
                equipment_json.update(adjacent_table_data)

            table_chunks = build_equipment_chunks(equipment_json)

            if not table_chunks:
                continue

            for chunk in table_chunks:
                chunk["chunk_number"] = chunk_number
                chunks.append(chunk)
                chunk_number += 1

            equipment_count += len(equipment_json["equipment"])
            tables.append(
                {
                    "page_number": extracted_table["page_number"],
                    "table_number": extracted_table["table_number"],
                    "split_number": split_index,
                    "row_count": int(table_df.shape[0]),
                    "column_count": int(table_df.shape[1]),
                    "rows": _table_rows(table_df),
                    "equipment_json": equipment_json,
                }
            )

    return {
        "metadata": {
            "source_file": filename,
            "raw_table_count": len(extracted_tables),
            "matched_table_count": matched_preview_count,
            "candidate_table_count": len(candidate_tables),
            "selected_table_count": len(tables),
            "chunk_count": len(chunks),
            "equipment_count": equipment_count,
            "extraction_mode": "pdfplumber",
            "adjacent_table_extraction_status": adjacent_table_result.get("status", "not_run"),
            "adjacent_table_fields_appended": bool(adjacent_table_result.get("data")),
            "adjacent_table_extraction_message": adjacent_table_result.get("message", ""),
            "adjacent_table_page_number": adjacent_table_result.get("adjacent_table_page_number"),
            "adjacent_table_table_number": adjacent_table_result.get("adjacent_table_table_number"),
        },
        "extracted_tables": extracted_table_previews,
        "tables": tables,
        "chunks": chunks,
    }


def build_empty_document(
    filename: str,
    extraction_mode: str,
    extraction_reason: str,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source_file": filename,
        "raw_table_count": 0,
        "matched_table_count": 0,
        "candidate_table_count": 0,
        "selected_table_count": 0,
        "chunk_count": 0,
        "equipment_count": 0,
        "extraction_mode": extraction_mode,
        "extraction_reason": extraction_reason,
    }

    if error_message:
        metadata["extraction_error"] = error_message

    return {
        "metadata": metadata,
        "extracted_tables": [],
        "tables": [],
        "chunks": [],
        "ocr_results": [],
    }


def build_document_for_upload(pdf_bytes: bytes, filename: str, file_size: int) -> Dict[str, Any]:
    if file_size > PDFPLUMBER_FILE_SIZE_LIMIT:
        ocr_document = extract_ocr_document(pdf_bytes, filename)
        ocr_document["metadata"]["extraction_reason"] = "file_size_threshold"
        return ocr_document

    try:
        table_document = build_table_document(pdf_bytes, filename)
    except Exception as exc:
        return build_empty_document(
            filename=filename,
            extraction_mode="pdfplumber",
            extraction_reason="pdfplumber_error",
            error_message=str(exc),
        )

    table_document["metadata"]["extraction_reason"] = "pdfplumber_size_rule"
    return table_document


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def select_documents_for_question(
    documents: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    question: str,
) -> List[Dict[str, Any]]:
    if isinstance(documents, dict):
        documents = [documents]

    if not documents:
        return []

    question_lower = (question or "").lower()
    question_normalized = _normalize_match_text(question)
    scored_documents = []

    for document in documents:
        metadata = document.get("metadata", {})
        source_file = metadata.get("source_file", "")
        source_path = Path(source_file)
        filename = source_path.name.lower()
        stem = source_path.stem.lower()
        normalized_filename = _normalize_match_text(filename)
        normalized_stem = _normalize_match_text(stem)

        score = 0
        if filename and filename in question_lower:
            score = 200
        elif stem and stem in question_lower:
            score = 180
        elif normalized_filename and normalized_filename in question_normalized:
            score = 160
        elif normalized_stem and normalized_stem in question_normalized:
            score = 140

        if score > 0:
            scored_documents.append((score, document))

    if not scored_documents:
        return list(documents)

    strongest_score = max(score for score, _ in scored_documents)
    return [document for score, document in scored_documents if score == strongest_score]


def retrieve_relevant_chunks(
    documents: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    question: str,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    if isinstance(documents, dict):
        documents = [documents]

    documents = select_documents_for_question(documents, question)

    chunks = []
    for document in documents:
        for chunk in document.get("chunks", []):
            if chunk.get("context_type") == "table_rows":
                continue
            chunks.append(chunk)

    if not chunks:
        return []

    question_tokens = set(_tokenize(question))
    if not question_tokens:
        return chunks[:top_k]

    question_lower = question.lower()
    scored_chunks = []

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_text_lower = chunk_text.lower()
        chunk_tokens = set(_tokenize(chunk_text))
        overlap = question_tokens & chunk_tokens

        score = sum(max(len(token) - 2, 1) for token in overlap)

        if question_lower and question_lower in chunk_text_lower:
            score += 8

        source_file = chunk.get("source_file", "").lower()
        if source_file and source_file in question_lower:
            score += 5

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
            item[1].get("source_file", ""),
            item[1].get("page_number", 0) or 0,
            item[1].get("table_number", 0) or 0,
            item[1].get("chunk_number", 0) or 0,
        )
    )
    return [chunk for _, chunk in scored_chunks[:top_k]]
