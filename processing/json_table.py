from collections.abc import Mapping
from typing import Any

import pandas as pd

EXCLUDED_METADATA_COLUMNS = {
    "equipment_number",
    "extraction_mode",
    "matched_row_number",
    "page_number",
    "raw_column_count",
    "raw_row_count",
    "record_type",
    "split_number",
    "table_number",
}


def flatten_json(obj: Any, parent_key: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            new_key = f"{parent_key}_{key}" if parent_key else str(key)
            items.update(flatten_json(value, new_key))
    elif isinstance(obj, list):
        if all(isinstance(item, Mapping) for item in obj):
            for index, item in enumerate(obj):
                new_key = f"{parent_key}_{index}" if parent_key else str(index)
                items.update(flatten_json(item, new_key))
        elif parent_key:
            items[parent_key] = ", ".join(str(item) for item in obj)
        else:
            items["value"] = ", ".join(str(item) for item in obj)
    else:
        items[parent_key or "value"] = obj

    return items


def structured_json_to_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if "equipment" in data and isinstance(data["equipment"], list):
        common_fields = {key: value for key, value in data.items() if key != "equipment"}

        for index, item in enumerate(data["equipment"], start=1):
            row = {"equipment_number": index}
            row.update(flatten_json(common_fields))
            row.update(flatten_json(item))
            rows.append(row)
    elif "matched_rows" in data and isinstance(data["matched_rows"], list):
        row = {key: value for key, value in data.items() if key != "matched_rows"}

        for item in data["matched_rows"]:
            row.update(flatten_json(item))

        rows.append(row)
    elif "merged_rows" in data and isinstance(data["merged_rows"], list):
        row = {key: value for key, value in data.items() if key != "merged_rows"}

        for item in data["merged_rows"]:
            row.update(flatten_json(item))

        rows.append(row)
    else:
        rows.append(flatten_json(data))

    return rows


def _is_empty_display_value(value: Any) -> bool:
    if pd.isna(value):
        return True

    text = str(value).strip()
    return text.lower() in {"", "none", "null", "nan"}


def _find_column_name(dataframe: pd.DataFrame, target_name: str) -> str | None:
    for column in dataframe.columns:
        if str(column).strip().lower() == target_name.lower():
            return column
    return None


def _merge_duplicate_description_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    description_column = _find_column_name(dataframe, "description")
    source_file_column = _find_column_name(dataframe, "source_file")

    if description_column is None:
        return dataframe

    merged_groups: dict[tuple[str, str, int], dict[str, list[str]]] = {}
    group_order: list[tuple[str, str, int]] = []

    for row_index, row in dataframe.iterrows():
        source_file = ""
        if source_file_column is not None:
            source_file = str(row.get(source_file_column, "") or "")

        description = row.get(description_column, "")

        if _is_empty_display_value(description):
            group_key = (source_file, f"__row_{row_index}", row_index)
        else:
            group_key = (source_file, str(description).strip().lower(), -1)

        if group_key not in merged_groups:
            merged_groups[group_key] = {column: [] for column in dataframe.columns}
            group_order.append(group_key)

        merged_row = merged_groups[group_key]

        for column in dataframe.columns:
            value = row[column]
            if _is_empty_display_value(value):
                continue

            text_value = str(value).strip()
            if text_value not in merged_row[column]:
                merged_row[column].append(text_value)

    merged_rows = []

    for group_key in group_order:
        merged_row = {}
        for column, values in merged_groups[group_key].items():
            if not values:
                merged_row[column] = ""
            elif len(values) == 1:
                merged_row[column] = values[0]
            else:
                merged_row[column] = ", ".join(values)
        merged_rows.append(merged_row)

    return pd.DataFrame(merged_rows, columns=dataframe.columns)


def _drop_existing_description_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    description_column = _find_column_name(dataframe, "description")
    if description_column is None:
        return dataframe

    kept_rows = []
    seen_descriptions: set[str] = set()

    for _, row in dataframe.iterrows():
        description = row.get(description_column, "")

        if _is_empty_display_value(description):
            kept_rows.append(row.to_dict())
            continue

        description_key = str(description).strip().lower()
        if description_key in seen_descriptions:
            continue

        seen_descriptions.add(description_key)
        kept_rows.append(row.to_dict())

    return pd.DataFrame(kept_rows, columns=dataframe.columns)


def build_document_transformed_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = document.get("metadata", {})
    source_file = metadata.get("source_file")

    for section in document.get("tables") or []:
        payload = dict(section.get("equipment_json") or {})

        for row in structured_json_to_rows(payload):
            row.setdefault("source_file", source_file)
            rows.append(row)

    for result in document.get("ocr_results") or []:
        payload = dict(result.get("ocr_json") or {})
        payload.setdefault("page_number", result.get("page_number"))

        for row in structured_json_to_rows(payload):
            row.setdefault("source_file", source_file)
            rows.append(row)

    return rows


def build_collection_transformed_table(documents: list[dict[str, Any]]) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []

    for document in documents:
        all_rows.extend(build_document_transformed_rows(document))

    if not all_rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(all_rows)
    dataframe = dataframe.drop(
        columns=[column for column in EXCLUDED_METADATA_COLUMNS if column in dataframe.columns]
    )
    dataframe = _merge_duplicate_description_rows(dataframe)
    dataframe = _drop_existing_description_rows(dataframe)

    ordered_columns = []
    if "source_file" in dataframe.columns:
        ordered_columns.append("source_file")

    description_columns = [
        column for column in dataframe.columns if column not in ordered_columns and column.lower() == "description"
    ]
    ordered_columns.extend(description_columns)

    ordered_columns.extend(
        sorted(column for column in dataframe.columns if column not in ordered_columns)
    )
    return dataframe.reindex(columns=ordered_columns)
