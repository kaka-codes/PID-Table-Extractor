import pandas as pd


KEYWORDS = [
    "description",
    "temperature",
    "design code"
    "material of construction",
    "equipment tag no",
    "item number"
]


def matched_keywords_in_text(text: str, keywords=None) -> list[str]:
    keywords = keywords or KEYWORDS
    normalized_text = (text or "").lower()
    return [keyword for keyword in keywords if keyword in normalized_text]


def find_table_keyword_matches(df, keywords=None) -> list[dict]:
    if df is None or df.empty:
        return []

    keywords = keywords or KEYWORDS
    matches = []

    for index, col in enumerate(df.columns):
        first_val = df[col].iloc[0]

        if pd.isna(first_val):
            continue

        first_text = str(first_val).strip().lower()

        if first_text in ["", "none", "null", "nan"]:
            continue

        col_text = " ".join(df[col].astype(str)).lower()
        matched_keywords = [keyword for keyword in keywords if keyword in col_text]

        if len(matched_keywords) >= 2:
            matches.append(
                {
                    "column_index": index,
                    "matched_keywords": matched_keywords,
                }
            )

    return matches


def is_valid_table(df) -> bool:
    if df is None or df.empty:
        return False

    for col in df.columns:
        col_text = " ".join(df[col].astype(str)).lower()
        count = sum(1 for keyword in KEYWORDS if keyword in col_text)

        if count >= 2:
            return True

    return False


def clean_level_logic(df):
    if df is None or df.empty:
        return df

    level_col_index = None
    keyword_col_index = None
    level_row_index = None

    for index, col in enumerate(df.columns):
        col_text = " ".join(df[col].astype(str)).lower()

        if any(keyword in col_text for keyword in KEYWORDS):
            keyword_col_index = index
            break

    for index, col in enumerate(df.columns):
        for row_idx, value in enumerate(df[col].astype(str)):
            if "level" in value.lower():
                level_col_index = index
                level_row_index = row_idx
                break

        if level_col_index is not None:
            break

    if level_col_index is not None:
        if keyword_col_index == level_col_index:
            return df.iloc[:level_row_index, :]

        return df.iloc[:, :level_col_index]

    return df


def clean_df(df):
    if df is None or df.empty:
        return df

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.replace("None", None)
    cleaned_df = cleaned_df.applymap(
        lambda value: str(value).replace("\n", "").strip() if isinstance(value, str) else value
    )
    cleaned_df = cleaned_df.dropna(axis=1, thresh=2)
    cleaned_df = cleaned_df.dropna(axis=0, thresh=2)
    cleaned_df = cleaned_df.dropna(axis=1, thresh=2)
    return cleaned_df.reset_index(drop=True)


def apply_keyword_fill_logic(df):
    if df is None or df.empty:
        return df

    filled_df = df.copy()

    def clean_cell(value):
        if pd.isna(value):
            return None

        text = str(value).strip()

        if text.lower() in ["", "none", "null", "nan"]:
            return None

        return text

    filled_df = filled_df.applymap(clean_cell)

    valid_columns = []

    for col in filled_df.columns:
        first_val = filled_df[col].iloc[0]

        if first_val is not None:
            valid_columns.append(col)

    for col in valid_columns:
        filled_df[col] = filled_df[col].ffill()

    filled_df = filled_df.fillna("")

    return filled_df


def get_keyword_columns(df, keywords):
    if df is None or df.empty:
        return []

    keyword_cols = []

    for index, col in enumerate(df.columns):
        first_val = df[col].iloc[0]

        if pd.isna(first_val):
            continue

        first_text = str(first_val).strip().lower()

        if first_text in ["", "none", "null", "nan"]:
            continue

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
    cleaned_df = cleaned_df.replace(r"\s*$", None, regex=True)
    cleaned_df = cleaned_df.dropna(axis=1, thresh=2)
    cleaned_df = cleaned_df.dropna(axis=0, thresh=2)
    cleaned_df = cleaned_df.dropna(axis=1, thresh=2)
    if cleaned_df.empty or cleaned_df.shape[1] == 0:
        return cleaned_df.iloc[0:0, 0:0].copy()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df.fillna("")


def prepare_table(df):
    if df is None or df.empty:
        return df

    cleaned_df = clean_level_logic(df)
    if cleaned_df is None or cleaned_df.empty:
        return cleaned_df

    cleaned_df = clean_df(cleaned_df)
    if cleaned_df is None or cleaned_df.empty:
        return cleaned_df

    return cleaned_df
