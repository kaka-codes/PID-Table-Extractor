# import os
# from collections import defaultdict
# from functools import lru_cache
# from io import BytesIO
# from typing import Any, Dict, List, Optional

# import cv2
# import fitz
# import numpy as np

# os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
# os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# from paddleocr import PaddleOCR


# DEFAULT_RENDER_SCALE = 3
# DEFAULT_CLIP_WIDTH = 1000
# DEFAULT_CLIP_HEIGHT = 400


# def _pixmap_to_bgr(pixmap):
#     image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
#         pixmap.height, pixmap.width, pixmap.n
#     )

#     if pixmap.n == 4:
#         return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

#     return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# def _render_pdf_region(pdf_bytes: bytes, page_index: int, rect, scale: int):
#     document = fitz.open(stream=pdf_bytes, filetype="pdf")
#     page = document[page_index]
#     pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect)
#     return _pixmap_to_bgr(pixmap)


# def _merge_line_segments(segments, position_tol=6, gap_tol=12):
#     merged = []

#     for segment in sorted(segments, key=lambda item: (item["pos"], item["start"], item["end"])):
#         if not merged:
#             merged.append(segment.copy())
#             continue

#         previous = merged[-1]
#         same_track = abs(segment["pos"] - previous["pos"]) <= position_tol
#         touching = segment["start"] <= previous["end"] + gap_tol

#         if same_track and touching:
#             previous["start"] = min(previous["start"], segment["start"])
#             previous["end"] = max(previous["end"], segment["end"])
#             previous["pos"] = (previous["pos"] + segment["pos"]) / 2
#         else:
#             merged.append(segment.copy())

#     for segment in merged:
#         segment["pos"] = int(round(segment["pos"]))
#         segment["start"] = int(round(segment["start"]))
#         segment["end"] = int(round(segment["end"]))
#         segment["length"] = segment["end"] - segment["start"]

#     return merged


# def _extract_line_segments(mask, orientation, min_length):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     segments = []

#     for contour in contours:
#         x, y, width, height = cv2.boundingRect(contour)

#         if orientation == "horizontal" and width >= min_length:
#             segments.append(
#                 {
#                     "pos": y + (height / 2),
#                     "start": x,
#                     "end": x + width,
#                 }
#             )
#         elif orientation == "vertical" and height >= min_length:
#             segments.append(
#                 {
#                     "pos": x + (width / 2),
#                     "start": y,
#                     "end": y + height,
#                 }
#             )

#     return _merge_line_segments(segments)


# def _detect_table_lines(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     horizontal_kernel = cv2.getStructuringElement(
#         cv2.MORPH_RECT, (max(image.shape[1] // 30, 40), 1)
#     )
#     vertical_kernel = cv2.getStructuringElement(
#         cv2.MORPH_RECT, (1, max(image.shape[0] // 18, 40))
#     )

#     horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
#     vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

#     horizontal_lines = _extract_line_segments(
#         horizontal_mask, "horizontal", min_length=max(image.shape[1] // 5, 120)
#     )
#     vertical_lines = _extract_line_segments(
#         vertical_mask, "vertical", min_length=max(image.shape[0] // 6, 120)
#     )

#     return horizontal_lines, vertical_lines


# def _overlap_length(start_a, end_a, start_b, end_b):
#     return max(0, min(end_a, end_b) - max(start_a, start_b))


# def _find_table_structure(horizontal_lines, vertical_lines, image_shape):
#     image_height, image_width = image_shape[:2]
#     long_verticals = [
#         line for line in vertical_lines if line["length"] >= int(image_height * 0.5)
#     ]

#     if len(long_verticals) < 2:
#         raise RuntimeError("Could not detect enough vertical table lines.")

#     best_candidate = None
#     sorted_verticals = sorted(long_verticals, key=lambda item: item["pos"])

#     for index, left_line in enumerate(sorted_verticals):
#         for right_line in sorted_verticals[index + 1 :]:
#             table_width = right_line["pos"] - left_line["pos"]
#             overlap_top = max(left_line["start"], right_line["start"])
#             overlap_bottom = min(left_line["end"], right_line["end"])

#             if table_width < image_width * 0.15:
#                 continue

#             spanning_horizontals = [
#                 line
#                 for line in horizontal_lines
#                 if line["start"] <= left_line["pos"] + 12
#                 and line["end"] >= right_line["pos"] - 12
#                 and overlap_top - 8 <= line["pos"] <= overlap_bottom + 8
#             ]

#             if len(spanning_horizontals) < 4:
#                 continue

#             score = len(spanning_horizontals) * table_width

#             if not best_candidate or score > best_candidate["score"]:
#                 best_candidate = {
#                     "score": score,
#                     "left": left_line["pos"],
#                     "right": right_line["pos"],
#                     "top": min(line["pos"] for line in spanning_horizontals),
#                     "bottom": max(line["pos"] for line in spanning_horizontals),
#                 }

#     if not best_candidate:
#         raise RuntimeError("Could not isolate the main table from detected lines.")

#     table_left = best_candidate["left"]
#     table_right = best_candidate["right"]
#     table_top = best_candidate["top"]
#     table_bottom = best_candidate["bottom"]
#     table_width = table_right - table_left
#     table_height = table_bottom - table_top

#     table_horizontals = [
#         line
#         for line in horizontal_lines
#         if table_top - 8 <= line["pos"] <= table_bottom + 8
#         and _overlap_length(line["start"], line["end"], table_left, table_right)
#         >= table_width * 0.2
#     ]
#     table_verticals = [
#         line
#         for line in vertical_lines
#         if table_left - 8 <= line["pos"] <= table_right + 8
#         and _overlap_length(line["start"], line["end"], table_top, table_bottom)
#         >= table_height * 0.2
#     ]

#     return {
#         "bbox": {
#             "x0": int(table_left),
#             "y0": int(table_top),
#             "x1": int(table_right),
#             "y1": int(table_bottom),
#         },
#         "horizontal_lines": [
#             {"y": line["pos"], "x0": line["start"], "x1": line["end"]}
#             for line in sorted(table_horizontals, key=lambda item: (item["pos"], item["start"]))
#         ],
#         "vertical_lines": [
#             {"x": line["pos"], "y0": line["start"], "y1": line["end"]}
#             for line in sorted(table_verticals, key=lambda item: (item["pos"], item["start"]))
#         ],
#     }


# @lru_cache(maxsize=1)
# def _get_ocr_engine():
#     try:
#         return PaddleOCR(
#             lang="en",
#             use_doc_orientation_classify=False,
#             use_doc_unwarping=False,
#             use_textline_orientation=True,
#         )
#     except TypeError:
#         return PaddleOCR(use_angle_cls=True, lang="en")


# def _coerce_box_points(box):
#     if box is None:
#         return []

#     if hasattr(box, "tolist"):
#         box = box.tolist()

#     if not isinstance(box, (list, tuple)):
#         return []

#     points = []
#     for point in box:
#         if hasattr(point, "tolist"):
#             point = point.tolist()
#         if not isinstance(point, (list, tuple)) or len(point) < 2:
#             continue
#         try:
#             points.append((float(point[0]), float(point[1])))
#         except (TypeError, ValueError):
#             continue

#     return points


# def _iter_ocr_entries(result):
#     if result is None:
#         return

#     if isinstance(result, dict):
#         boxes = result.get("dt_polys") or result.get("boxes") or result.get("polys") or []
#         texts = result.get("rec_texts") or result.get("texts") or []
#         scores = result.get("rec_scores") or result.get("scores") or []

#         for index, text in enumerate(texts):
#             box = boxes[index] if index < len(boxes) else None
#             score = scores[index] if index < len(scores) else 0.0
#             yield box, text, score
#         return

#     if not isinstance(result, (list, tuple)):
#         return

#     for item in result:
#         if item is None:
#             continue

#         if isinstance(item, dict):
#             yield from _iter_ocr_entries(item)
#             continue

#         if not isinstance(item, (list, tuple)) or len(item) < 2:
#             continue

#         box = item[0]
#         payload = item[1]

#         if isinstance(payload, dict):
#             text = payload.get("text") or payload.get("rec_text") or ""
#             score = payload.get("score") or payload.get("rec_score") or 0.0
#             yield box, text, score
#             continue

#         if isinstance(payload, (list, tuple)) and len(payload) >= 2:
#             yield box, payload[0], payload[1]


# def _extract_ocr_items(image):
#     ocr = _get_ocr_engine()
#     try:
#         result = ocr.ocr(image, cls=True)
#     except TypeError as exc:
#         if "unexpected keyword argument 'cls'" not in str(exc):
#             raise
#         result = ocr.ocr(image)
#     items = []

#     if not result:
#         return items

#     page_results = result if isinstance(result, list) else [result]

#     for page_result in page_results:
#         for box, text, confidence in _iter_ocr_entries(page_result):
#             points = _coerce_box_points(box)
#             if not points:
#                 continue

#             text = "" if text is None else str(text).strip()
#             if not text:
#                 continue

#             try:
#                 confidence_value = float(confidence)
#             except (TypeError, ValueError):
#                 confidence_value = 0.0

#             xs = [point[0] for point in points]
#             ys = [point[1] for point in points]

#             items.append(
#                 {
#                     "text": text,
#                     "confidence": round(confidence_value, 4),
#                     "bbox": [
#                         float(min(xs)),
#                         float(min(ys)),
#                         float(max(xs)),
#                         float(max(ys)),
#                     ],
#                     "center": (
#                         float(sum(xs) / len(xs)),
#                         float(sum(ys) / len(ys)),
#                     ),
#                 }
#             )

#     return items


# def _is_inside_bbox(point, bbox, pad=2):
#     x, y = point
#     return (
#         bbox["x0"] - pad <= x <= bbox["x1"] + pad
#         and bbox["y0"] - pad <= y <= bbox["y1"] + pad
#     )


# def _line_covers_coordinate(line_start, line_end, coordinate, tolerance=6):
#     return line_start - tolerance <= coordinate <= line_end + tolerance


# def _find_local_bounds(center_x, center_y, horizontals, verticals):
#     left_candidates = [
#         line["x"]
#         for line in verticals
#         if line["x"] <= center_x and _line_covers_coordinate(line["y0"], line["y1"], center_y)
#     ]
#     right_candidates = [
#         line["x"]
#         for line in verticals
#         if line["x"] >= center_x and _line_covers_coordinate(line["y0"], line["y1"], center_y)
#     ]
#     top_candidates = [
#         line["y"]
#         for line in horizontals
#         if line["y"] <= center_y and _line_covers_coordinate(line["x0"], line["x1"], center_x)
#     ]
#     bottom_candidates = [
#         line["y"]
#         for line in horizontals
#         if line["y"] >= center_y and _line_covers_coordinate(line["x0"], line["x1"], center_x)
#     ]

#     if not (left_candidates and right_candidates and top_candidates and bottom_candidates):
#         return None

#     left = max(left_candidates)
#     right = min(right_candidates)
#     top = max(top_candidates)
#     bottom = min(bottom_candidates)

#     if left >= right or top >= bottom:
#         return None

#     return (left, top, right, bottom)


# def _deduplicate_text(parts):
#     seen = set()
#     cleaned = []

#     for part in parts:
#         text = " ".join(str(part).split())

#         if not text:
#             continue

#         token = text.casefold()

#         if token in seen:
#             continue

#         seen.add(token)
#         cleaned.append(text)

#     return cleaned


# def _build_semantic_rows(table_items, table_cells, key_split_x):
#     row_groups = defaultdict(list)

#     for item in table_items:
#         row_groups[item["row_band"]].append(item)

#     structured_rows = []

#     for row_band in sorted(row_groups.keys(), key=lambda band: (band[1], band[0])):
#         row_top, row_bottom = row_band
#         row_height = row_bottom - row_top
#         row_center = (row_top + row_bottom) / 2
#         row_cells = sorted(
#             [
#                 cell
#                 for cell in table_cells
#                 if cell["bbox"][1] == row_top and cell["bbox"][3] == row_bottom
#             ],
#             key=lambda cell: cell["center"][0],
#         )

#         direct_key_parts = [
#             cell["text"] for cell in row_cells if cell["center"][0] < key_split_x
#         ]
#         value_parts = [
#             cell["text"] for cell in row_cells if cell["center"][0] >= key_split_x
#         ]

#         inherited_key_parts = [
#             cell["text"]
#             for cell in table_cells
#             if cell["center"][0] < key_split_x
#             and cell["bbox"][1] <= row_center <= cell["bbox"][3]
#             and (cell["bbox"][3] - cell["bbox"][1]) > row_height + 5
#         ]

#         key_parts = _deduplicate_text(inherited_key_parts + direct_key_parts)
#         value_parts = _deduplicate_text(value_parts)

#         if not key_parts or not value_parts:
#             continue

#         structured_rows.append({" ".join(key_parts): " ".join(value_parts)})

#     return structured_rows


# def _build_ocr_json(
#     source_file: str,
#     matched_rows: Optional[List[Dict[str, Any]]] = None,
#     error: Optional[str] = None,
# ) -> Dict[str, Any]:
#     payload = {
#         "source_file": os.path.basename(source_file),
#         "matched_rows": matched_rows or [],
#     }

#     if error is not None:
#         payload["error"] = error
#     return payload


# def _ocr_page_result(pdf_bytes: bytes, filename: str, page_index: int) -> Dict[str, Any]:
#     document = fitz.open(stream=pdf_bytes, filetype="pdf")
#     page = document[page_index]
#     rect = page.rect
#     clip_rect = fitz.Rect(
#         0,
#         0,
#         min(rect.width, DEFAULT_CLIP_WIDTH),
#         min(rect.height, DEFAULT_CLIP_HEIGHT),
#     )

#     image = _render_pdf_region(pdf_bytes, page_index, clip_rect, DEFAULT_RENDER_SCALE)
#     horizontal_lines, vertical_lines = _detect_table_lines(image)
#     table_structure = _find_table_structure(horizontal_lines, vertical_lines, image.shape)
#     ocr_items = _extract_ocr_items(image)

#     table_bbox = table_structure["bbox"]
#     table_horizontals = table_structure["horizontal_lines"]
#     table_verticals = table_structure["vertical_lines"]

#     table_items = []
#     grouped_cells = defaultdict(list)

#     for item in ocr_items:
#         if not _is_inside_bbox(item["center"], table_bbox):
#             continue

#         local_bounds = _find_local_bounds(
#             item["center"][0],
#             item["center"][1],
#             table_horizontals,
#             table_verticals,
#         )

#         if not local_bounds:
#             continue

#         left, top, right, bottom = local_bounds

#         item["cell_bbox"] = [left, top, right, bottom]
#         item["row_band"] = (top, bottom)
#         table_items.append(item)
#         grouped_cells[(left, top, right, bottom)].append(item)

#     table_cells = []

#     for bbox, items in sorted(grouped_cells.items(), key=lambda entry: (entry[0][1], entry[0][0])):
#         ordered_items = sorted(items, key=lambda item: item["center"][0])
#         text_parts = [item["text"] for item in ordered_items]

#         table_cells.append(
#             {
#                 "bbox": list(bbox),
#                 "center": [
#                     round((bbox[0] + bbox[2]) / 2, 1),
#                     round((bbox[1] + bbox[3]) / 2, 1),
#                 ],
#                 "text": " ".join(_deduplicate_text(text_parts)),
#             }
#         )

#     major_verticals = sorted(
#         [
#             line["x"]
#             for line in table_verticals
#             if (line["y1"] - line["y0"]) >= (table_bbox["y1"] - table_bbox["y0"]) * 0.8
#         ]
#     )

#     if len(major_verticals) >= 3:
#         key_split_x = major_verticals[1]
#     else:
#         key_split_x = table_bbox["x0"] + int((table_bbox["x1"] - table_bbox["x0"]) * 0.35)

#     matched_rows = _build_semantic_rows(table_items, table_cells, key_split_x)

#     return {
#         "page_number": page_index + 1,
#         "ocr_json": _build_ocr_json(
#             source_file=filename,
#             matched_rows=matched_rows,
#         ),
#     }


# def extract_ocr_document(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
#     document = fitz.open(stream=pdf_bytes, filetype="pdf")
#     ocr_results = []

#     for page_index in range(len(document)):
#         try:
#             page_result = _ocr_page_result(pdf_bytes, filename, page_index)
#         except Exception as exc:
#             page_result = {
#                 "page_number": page_index + 1,
#                 "ocr_json": _build_ocr_json(
#                     source_file=filename,
#                     matched_rows=[],
#                     error=str(exc),
#                 ),
#             }

#         ocr_results.append(page_result)

#     chunks = []
#     chunk_number = 1

#     for page_result in ocr_results:
#         matched_rows = page_result["ocr_json"].get("matched_rows", [])
#         if not matched_rows:
#             continue

#         chunk_lines = []
#         for row in matched_rows:
#             for key, value in row.items():
#                 chunk_lines.append(f"{key}: {value}")

#         if not chunk_lines:
#             continue

#         chunks.append(
#             {
#                 "source_file": filename,
#                 "page_number": page_result["page_number"],
#                 "table_number": 1,
#                 "split_number": None,
#                 "equipment_number": None,
#                 "context_type": "ocr_rows",
#                 "text": "\n".join(chunk_lines),
#                 "chunk_number": chunk_number,
#             }
#         )
#         chunk_number += 1

#     return {
#         "metadata": {
#             "source_file": filename,
#             "raw_table_count": len(ocr_results),
#             "matched_table_count": len([result for result in ocr_results if result["ocr_json"].get("matched_rows")]),
#             "candidate_table_count": len(ocr_results),
#             "selected_table_count": 0,
#             "chunk_count": len(chunks),
#             "equipment_count": 0,
#             "extraction_mode": "ocr",
#         },
#         "extracted_tables": [],
#         "tables": [],
#         "chunks": chunks,
#         "ocr_results": ocr_results,
#     }

import os
from collections import defaultdict
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import fitz
import numpy as np
import easyocr

DEFAULT_RENDER_SCALE = 3
DEFAULT_CLIP_WIDTH = 1000
DEFAULT_CLIP_HEIGHT = 400

# ============================================
# STREAMLIT CLOUD STABILITY SETTINGS
# ============================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ============================================
# IMAGE HELPERS
# ============================================

def _pixmap_to_bgr(pixmap):
    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n
    )

    if pixmap.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _render_pdf_region(pdf_bytes: bytes, page_index: int, rect, scale: int):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = document[page_index]

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=rect
    )

    return _pixmap_to_bgr(pixmap)

# ============================================
# TABLE LINE DETECTION
# ============================================

def _merge_line_segments(segments, position_tol=6, gap_tol=12):
    merged = []

    for segment in sorted(
        segments,
        key=lambda item: (item["pos"], item["start"], item["end"])
    ):
        if not merged:
            merged.append(segment.copy())
            continue

        previous = merged[-1]

        same_track = abs(segment["pos"] - previous["pos"]) <= position_tol
        touching = segment["start"] <= previous["end"] + gap_tol

        if same_track and touching:
            previous["start"] = min(previous["start"], segment["start"])
            previous["end"] = max(previous["end"], segment["end"])
            previous["pos"] = (previous["pos"] + segment["pos"]) / 2
        else:
            merged.append(segment.copy())

    for segment in merged:
        segment["pos"] = int(round(segment["pos"]))
        segment["start"] = int(round(segment["start"]))
        segment["end"] = int(round(segment["end"]))
        segment["length"] = segment["end"] - segment["start"]

    return merged


def _extract_line_segments(mask, orientation, min_length):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if orientation == "horizontal" and width >= min_length:
            segments.append({
                "pos": y + (height / 2),
                "start": x,
                "end": x + width,
            })

        elif orientation == "vertical" and height >= min_length:
            segments.append({
                "pos": x + (width / 2),
                "start": y,
                "end": y + height,
            })

    return _merge_line_segments(segments)


def _detect_table_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(image.shape[1] // 30, 40), 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(image.shape[0] // 18, 40))
    )

    horizontal_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    horizontal_lines = _extract_line_segments(
        horizontal_mask,
        "horizontal",
        min_length=max(image.shape[1] // 5, 120)
    )

    vertical_lines = _extract_line_segments(
        vertical_mask,
        "vertical",
        min_length=max(image.shape[0] // 6, 120)
    )

    return horizontal_lines, vertical_lines

# ============================================
# TABLE STRUCTURE
# ============================================

def _overlap_length(start_a, end_a, start_b, end_b):
    return max(
        0,
        min(end_a, end_b) - max(start_a, start_b)
    )


def _find_table_structure(horizontal_lines, vertical_lines, image_shape):
    image_height, image_width = image_shape[:2]

    long_verticals = [
        line for line in vertical_lines
        if line["length"] >= int(image_height * 0.5)
    ]

    if len(long_verticals) < 2:
        raise RuntimeError("Could not detect enough vertical table lines.")

    best_candidate = None

    sorted_verticals = sorted(
        long_verticals,
        key=lambda item: item["pos"]
    )

    for index, left_line in enumerate(sorted_verticals):

        for right_line in sorted_verticals[index + 1:]:

            table_width = right_line["pos"] - left_line["pos"]

            overlap_top = max(
                left_line["start"],
                right_line["start"]
            )

            overlap_bottom = min(
                left_line["end"],
                right_line["end"]
            )

            if table_width < image_width * 0.15:
                continue

            spanning_horizontals = [
                line
                for line in horizontal_lines
                if line["start"] <= left_line["pos"] + 12
                and line["end"] >= right_line["pos"] - 12
                and overlap_top - 8 <= line["pos"] <= overlap_bottom + 8
            ]

            if len(spanning_horizontals) < 4:
                continue

            score = len(spanning_horizontals) * table_width

            if not best_candidate or score > best_candidate["score"]:
                best_candidate = {
                    "score": score,
                    "left": left_line["pos"],
                    "right": right_line["pos"],
                    "top": min(line["pos"] for line in spanning_horizontals),
                    "bottom": max(line["pos"] for line in spanning_horizontals),
                }

    if not best_candidate:
        raise RuntimeError("Could not isolate the main table.")

    table_left = best_candidate["left"]
    table_right = best_candidate["right"]
    table_top = best_candidate["top"]
    table_bottom = best_candidate["bottom"]

    return {
        "bbox": {
            "x0": int(table_left),
            "y0": int(table_top),
            "x1": int(table_right),
            "y1": int(table_bottom),
        }
    }

# ============================================
# EASYOCR
# ============================================

@lru_cache(maxsize=1)
def _get_ocr_engine():
    return easyocr.Reader(
        ['en'],
        gpu=False,
        verbose=False
    )


def _extract_ocr_items(image):

    reader = _get_ocr_engine()

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False
    )

    items = []

    for result in results:

        box, text, confidence = result

        if not text:
            continue

        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        items.append({
            "text": str(text).strip(),
            "confidence": float(confidence),

            "bbox": [
                float(min(xs)),
                float(min(ys)),
                float(max(xs)),
                float(max(ys)),
            ],

            "center": (
                float(sum(xs) / len(xs)),
                float(sum(ys) / len(ys)),
            ),
        })

    return items

# ============================================
# MAIN OCR EXTRACTION
# ============================================

def _build_ocr_json(
    source_file: str,
    matched_rows=None,
    error=None,
):

    payload = {
        "source_file": os.path.basename(source_file),
        "matched_rows": matched_rows or [],
    }

    if error is not None:
        payload["error"] = error

    return payload


def _ocr_page_result(pdf_bytes, filename, page_index):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page = document[page_index]

    rect = page.rect

    clip_rect = fitz.Rect(
        0,
        0,
        min(rect.width, DEFAULT_CLIP_WIDTH),
        min(rect.height, DEFAULT_CLIP_HEIGHT),
    )

    image = _render_pdf_region(
        pdf_bytes,
        page_index,
        clip_rect,
        DEFAULT_RENDER_SCALE
    )

    horizontal_lines, vertical_lines = _detect_table_lines(image)

    table_structure = _find_table_structure(
        horizontal_lines,
        vertical_lines,
        image.shape
    )

    ocr_items = _extract_ocr_items(image)

    matched_rows = []

    for item in ocr_items:
        matched_rows.append({
            "text": item["text"],
            "confidence": round(item["confidence"], 4),
        })

    return {
        "page_number": page_index + 1,
        "ocr_json": _build_ocr_json(
            source_file=filename,
            matched_rows=matched_rows,
        ),
    }


def extract_ocr_document(pdf_bytes, filename):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    ocr_results = []

    for page_index in range(len(document)):

        try:
            page_result = _ocr_page_result(
                pdf_bytes,
                filename,
                page_index
            )

        except Exception as exc:

            page_result = {
                "page_number": page_index + 1,

                "ocr_json": _build_ocr_json(
                    source_file=filename,
                    matched_rows=[],
                    error=str(exc),
                ),
            }

        ocr_results.append(page_result)

    chunks = []

    chunk_number = 1

    for page_result in ocr_results:

        matched_rows = page_result["ocr_json"].get(
            "matched_rows",
            []
        )

        if not matched_rows:
            continue

        chunk_lines = []

        for row in matched_rows:

            text = row.get("text", "").strip()

            if text:
                chunk_lines.append(text)

        if not chunk_lines:
            continue

        chunks.append({
            "source_file": filename,
            "page_number": page_result["page_number"],
            "table_number": 1,
            "split_number": None,
            "equipment_number": None,
            "context_type": "ocr_rows",
            "text": "\n".join(chunk_lines),
            "chunk_number": chunk_number,
        })

        chunk_number += 1

    return {
        "metadata": {
            "source_file": filename,
            "raw_table_count": len(ocr_results),
            "matched_table_count": len(ocr_results),
            "candidate_table_count": len(ocr_results),
            "selected_table_count": 0,
            "chunk_count": len(chunks),
            "equipment_count": 0,
            "extraction_mode": "ocr",
        },

        "extracted_tables": [],
        "tables": [],
        "chunks": chunks,
        "ocr_results": ocr_results,
    }
