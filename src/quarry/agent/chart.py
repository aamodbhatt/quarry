"""Pick a Vega-Lite chart for a result, deterministically (no LLM call, no tokens).

The rule set is small on purpose: a chart is only suggested when its meaning is clear.

* one category + numbers -> bar chart (sorted by value), grouped by a second category
  if there is one;
* one temporal or ordered numeric axis + numbers -> line chart;
* two numeric columns only -> scatter plot;
* a single row or a single column -> no chart (the answer text says it better).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from quarry.db.engines import Scalar

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
MAX_POINTS = 200
_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?)?")
_ORDERED_NAME = re.compile(
    r"(year|month|week|day|date|time|quarter|recency|age|bucket|segment)", re.IGNORECASE
)


def _kind(name: str, values: Sequence[Scalar]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "empty"
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in present):
        return (
            "ordinal-number" if _ORDERED_NAME.search(name) and _is_integral(present) else "number"
        )
    if all(isinstance(v, str) and _DATE.match(v) for v in present):
        return "temporal"
    return "category"


def _is_integral(values: Sequence[Scalar]) -> bool:
    return all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in values)


def _field(name: str) -> str:
    """Vega-Lite reads dots and brackets as nested access; escape them."""
    return re.sub(r"([.\[\]])", r"\\\1", name)


def chart_spec(
    columns: Sequence[str], rows: Sequence[Sequence[Scalar]], title: str = ""
) -> dict[str, Any] | None:
    """A Vega-Lite v5 spec for the result, or ``None`` when no chart would help."""
    if len(rows) < 2 or len(columns) < 2:
        return None
    rows = rows[:MAX_POINTS]
    kinds = {c: _kind(c, [r[i] for r in rows]) for i, c in enumerate(columns)}
    numbers = [c for c in columns if kinds[c] == "number"]
    ordered = [c for c in columns if kinds[c] in {"temporal", "ordinal-number"}]
    categories = [c for c in columns if kinds[c] == "category"]
    values = [dict(zip(columns, row, strict=True)) for row in rows]

    encoding: dict[str, Any]
    mark: dict[str, Any]
    if ordered and numbers and not categories:
        x = ordered[0]
        x_type = "temporal" if kinds[x] == "temporal" else "ordinal"
        mark = {"type": "line", "point": True}
        encoding = {
            "x": {"field": _field(x), "type": x_type, "title": x},
            "y": {"field": _field(numbers[0]), "type": "quantitative", "title": numbers[0]},
        }
    elif categories and (numbers or ordered):
        measure = numbers[0] if numbers else ordered[0]
        cat = categories[0]
        mark = {"type": "bar"}
        encoding = {
            "y": {"field": _field(cat), "type": "nominal", "title": cat, "sort": "-x"},
            "x": {"field": _field(measure), "type": "quantitative", "title": measure},
        }
        if len(categories) > 1:
            second = categories[1]
            encoding["color"] = {"field": _field(second), "type": "nominal", "title": second}
            encoding["yOffset"] = {"field": _field(second)}
            encoding["y"].pop("sort")
    elif len(numbers) >= 2:
        mark = {"type": "point"}
        encoding = {
            "x": {"field": _field(numbers[0]), "type": "quantitative", "title": numbers[0]},
            "y": {"field": _field(numbers[1]), "type": "quantitative", "title": numbers[1]},
        }
    else:
        return None
    encoding["tooltip"] = [{"field": _field(c), "title": c} for c in columns]
    spec: dict[str, Any] = {
        "$schema": VEGA_LITE_SCHEMA,
        "data": {"values": values},
        "mark": mark,
        "encoding": encoding,
    }
    if title:
        spec["title"] = title
    return spec
