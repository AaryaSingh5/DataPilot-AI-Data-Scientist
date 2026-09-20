import json
import math
from typing import Dict, Any, List, Optional


class VizError(Exception):
    pass


def _verify_data(data: List[Dict], required_keys: List[str]):
    """Verify plot data has required keys and no NaN/Inf values."""
    if not data:
        raise VizError("Plot data is empty")
    for i, row in enumerate(data):
        for k in required_keys:
            if k not in row:
                raise VizError(f"Row {i} missing key '{k}'")
            val = row[k]
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise VizError(f"Row {i} key '{k}' has non-finite value: {val}")


def bar_chart(
    data: List[Dict],
    x: str,
    y: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = "#4F81C7",
) -> Dict[str, Any]:
    _verify_data(data, [x, y])
    return {
        "type": "bar",
        "title": title,
        "x_label": x_label or x,
        "y_label": y_label or y,
        "series": [{"label": str(row[x]), "value": row[y]} for row in data],
        "color": color,
    }


def line_chart(
    data: List[Dict],
    x: str,
    y: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = "#4F81C7",
) -> Dict[str, Any]:
    _verify_data(data, [x, y])
    return {
        "type": "line",
        "title": title,
        "x_label": x_label or x,
        "y_label": y_label or y,
        "series": [{"x": str(row[x]), "y": row[y]} for row in data],
        "color": color,
    }


def scatter_chart(
    data: List[Dict],
    x: str,
    y: str,
    title: str = "",
    color_col: Optional[str] = None,
) -> Dict[str, Any]:
    keys = [x, y]
    if color_col:
        keys.append(color_col)
    _verify_data(data, keys)
    points = []
    for row in data:
        pt = {"x": row[x], "y": row[y]}
        if color_col:
            pt["color"] = str(row[color_col])
        points.append(pt)
    return {"type": "scatter", "title": title, "x_label": x, "y_label": y, "points": points}


def heatmap(
    matrix: Dict[str, Dict[str, float]],
    title: str = "Correlation Matrix",
) -> Dict[str, Any]:
    labels = list(matrix.keys())
    cells = []
    for row_label, row_vals in matrix.items():
        for col_label, val in row_vals.items():
            if not math.isfinite(val):
                raise VizError(f"Non-finite value in heatmap at ({row_label}, {col_label})")
            cells.append({"row": row_label, "col": col_label, "value": round(val, 4)})
    return {"type": "heatmap", "title": title, "labels": labels, "cells": cells}


def waterfall_chart(
    contributions: Dict[str, float],
    title: str = "Revenue Decomposition",
) -> Dict[str, Any]:
    bars = []
    running = 0.0
    for label, value in contributions.items():
        if not math.isfinite(value):
            raise VizError(f"Non-finite value in waterfall at '{label}'")
        bars.append({"label": label, "start": running, "end": running + value, "value": value})
        running += value
    return {"type": "waterfall", "title": title, "bars": bars}
