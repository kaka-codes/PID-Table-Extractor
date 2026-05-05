from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
import pdfplumber

from processing.cleaner import clean_level_logic

DEFAULT_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 2,
}


def extract_tables_perfectly(pdf_source) -> List[Dict[str, Any]]:
    if isinstance(pdf_source, (bytes, bytearray)):
        pdf_source = BytesIO(pdf_source)

    if hasattr(pdf_source, "seek"):
        pdf_source.seek(0)

    all_tables: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_source) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(table_settings=DEFAULT_TABLE_SETTINGS) or []

            for table_number, table in enumerate(tables, start=1):
                if not table:
                    continue

                df = pd.DataFrame(table)
                df = clean_level_logic(df)
                if df.empty:
                    continue

                all_tables.append(
                    {
                        "page_number": page_number,
                        "table_number": table_number,
                        "dataframe": df,
                    }
                )

    return all_tables
