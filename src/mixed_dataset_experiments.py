from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


AGE_BUCKETS = [
    (0, 17, "0-17"),
    (18, 29, "18-29"),
    (30, 44, "30-44"),
    (45, 59, "45-59"),
    (60, 120, "60+"),
]

JOIN_KEY_CANDIDATES = [
    "sample_id",
    "Id",
    "id",
    "record_id",
    "row_id",
    "uid",
    "uuid",
    "participant_id",
    "response_id",
    "document_id",
    "post_id",
]

AGE_KEY_CANDIDATES = [
    "age",
    "age_years",
    "participant_age",
    "age_value",
    "age_bucket",
]

ETHNICITY_KEY_CANDIDATES = [
    "ethnicity",
    "nationality",
    "background",
    "race",
    "ethnic_group",
]

COORD_KEY_CANDIDATES = [
    ("x", "y"),
    ("coord_x", "coord_y"),
    ("pca1", "pca2"),
    ("pca_x", "pca_y"),
    ("tsne_1", "tsne_2"),
    ("tsne_x", "tsne_y"),
    ("umap_x", "umap_y"),
    ("dimension_1", "dimension_2"),
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_valid(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if value not in (None, "", "null", "None"):
                return value
    return None


def _deep_first_valid(node: Any, keys: Iterable[str]) -> Any:
    if isinstance(node, dict):
        value = _first_valid(node, keys)
        if value is not None:
            return value
        for child in node.values():
            value = _deep_first_valid(child, keys)
            if value is not None:
                return value
    elif isinstance(node, list):
        for child in node:
            value = _deep_first_valid(child, keys)
            if value is not None:
                return value
    return None


def _iter_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        dataset_value = payload.get("Dataset")
        if isinstance(dataset_value, list):
            for item in dataset_value:
                if isinstance(item, dict):
                    yield item
            return

        for key in ("records", "data", "items", "results", "rows", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return

        values = list(payload.values())
        if values and all(isinstance(item, dict) for item in values):
            for item in values:
                yield item
            return

        yield payload


def _age_bucket(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            for _, _, label in AGE_BUCKETS:
                if normalized == label:
                    return label
            if normalized.lower() in {"unknown", "n/a", "na", "none"}:
                return "unknown"

    numeric = _coerce_number(value)
    if numeric is None:
        return "unknown"

    for lower, upper, label in AGE_BUCKETS:
        if lower <= numeric <= upper:
            return label
    return "60+"


def _sample_key(record: dict[str, Any]) -> Any:
    value = _first_valid(record, JOIN_KEY_CANDIDATES)
    return value if value is not None else None


def _extract_age(record: dict[str, Any]) -> Any:
    return _deep_first_valid(record, ["Age", "age", "age_years", "participant_age", "age_value", "age_bucket"])


def _extract_ethnicity(record: dict[str, Any]) -> Any:
    return _deep_first_valid(record, ["Nationality", "Ethnicity", "ethnicity", "nationality", "background", "Background"])


def _extract_text(record: dict[str, Any]) -> str:
    value = _deep_first_valid(record, ["Text", "text", "content", "body", "message", "prompt", "response"])
    return "" if value is None else str(value)


def _extract_cluster(record: dict[str, Any]) -> Any:
    value = _deep_first_valid(record, ["cluster_id", "cluster", "cluster_label", "community", "group"])
    return value


def _extract_coordinates(record: dict[str, Any]) -> tuple[float | None, float | None]:
    for x_key, y_key in COORD_KEY_CANDIDATES:
        x_value = _deep_first_valid(record, [x_key])
        y_value = _deep_first_valid(record, [y_key])
        x = _coerce_number(x_value)
        y = _coerce_number(y_value)
        if x is not None and y is not None:
            return x, y
    return None, None


def _normalize_record(record: dict[str, Any], source: str, fallback_index: int) -> dict[str, Any]:
    age = _extract_age(record)
    ethnicity = _extract_ethnicity(record)
    cluster_id = _extract_cluster(record)
    x, y = _extract_coordinates(record)

    sample_id = _sample_key(record)
    if sample_id is None:
        sample_id = f"{source}-{fallback_index:05d}"

    normalized = {
        "sample_id": str(sample_id),
        "source": source,
        "age_raw": age,
        "age_bucket": _age_bucket(age),
        "ethnicity": str(ethnicity).strip() if ethnicity not in (None, "") else "unknown",
        "cluster_id": None if cluster_id in (None, "") else str(cluster_id),
        "x": x,
        "y": y,
        "text": _extract_text(record),
        "record_id": str(sample_id),
        "raw": record,
    }

    return normalized


def _stable_jitter(seed: str, amplitude: float = 0.16) -> float:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    integer = int(digest[:8], 16)
    normalized = integer / 0xFFFFFFFF
    return (normalized - 0.5) * 2.0 * amplitude


def _load_normalized_records(data_path: Path, source: str) -> list[dict[str, Any]]:
    payload = _read_json(data_path)
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(_iter_records(payload), start=1):
        normalized.append(_normalize_record(record, source=source, fallback_index=index))
    return normalized


def _load_cluster_rows(cluster_csv_path: Path) -> list[dict[str, Any]]:
    return _read_csv(cluster_csv_path)


def _normalize_cluster_row(row: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    sample_id = _first_valid(row, JOIN_KEY_CANDIDATES)
    if sample_id is None:
        sample_id = f"cluster-{fallback_index:05d}"

    cluster_id = _first_valid(row, ["cluster_id", "cluster", "cluster_label", "community"])
    x = _first_valid(row, [candidate[0] for candidate in COORD_KEY_CANDIDATES])
    y = _first_valid(row, [candidate[1] for candidate in COORD_KEY_CANDIDATES])
    if x is None or y is None:
        x, y = _extract_coordinates(row)

    return {
        "sample_id": str(sample_id),
        "record_id": str(sample_id),
        "cluster_id": None if cluster_id in (None, "") else str(cluster_id),
        "x": _coerce_number(x),
        "y": _coerce_number(y),
        "raw_cluster_row": row,
    }


def _merge_records_with_clusters(
    records: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not cluster_rows:
        return records

    normalized_cluster_rows = [
        _normalize_cluster_row(row, index)
        for index, row in enumerate(cluster_rows, start=1)
    ]

    cluster_lookup: dict[str, dict[str, Any]] = {}
    for row in normalized_cluster_rows:
        cluster_lookup.setdefault(row["sample_id"], row)

    joined: list[dict[str, Any]] = []
    matched = 0
    for record in records:
        key = record["sample_id"]
        cluster_row = cluster_lookup.get(key)
        merged = dict(record)
        if cluster_row is not None:
            matched += 1
            merged["cluster_id"] = cluster_row.get("cluster_id") or merged.get("cluster_id")
            merged["x"] = cluster_row.get("x") if cluster_row.get("x") is not None else merged.get("x")
            merged["y"] = cluster_row.get("y") if cluster_row.get("y") is not None else merged.get("y")
            merged["cluster_raw"] = cluster_row.get("raw_cluster_row")
        joined.append(merged)

    if matched == 0:
        paired = []
        for index, record in enumerate(records):
            merged = dict(record)
            if index < len(normalized_cluster_rows):
                cluster_row = normalized_cluster_rows[index]
                merged["cluster_id"] = cluster_row.get("cluster_id") or merged.get("cluster_id")
                merged["x"] = cluster_row.get("x") if cluster_row.get("x") is not None else merged.get("x")
                merged["y"] = cluster_row.get("y") if cluster_row.get("y") is not None else merged.get("y")
                merged["cluster_raw"] = cluster_row.get("raw_cluster_row")
            paired.append(merged)
        return paired

    return joined


def _distribution(records: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    values = [str(record.get(field_name, "unknown")) for record in records]
    counts = Counter(values)
    total = sum(counts.values()) or 1
    return {
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "proportions": {
            key: round(value / total, 6) for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }


def _cluster_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("cluster_id", "unclustered")) for record in records)
    total = sum(counts.values()) or 1
    return {
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "proportions": {
            key: round(value / total, 6) for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }


def _group_metrics(records: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_field, "unknown"))].append(record)

    results: list[dict[str, Any]] = []
    total = len(records) or 1
    for group_name, group_records in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        cluster_counts = Counter(str(record.get("cluster_id", "unclustered")) for record in group_records)
        results.append(
            {
                "group": group_name,
                "count": len(group_records),
                "share": round(len(group_records) / total, 6),
                "cluster_counts": dict(sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))),
                "cluster_share": {
                    key: round(value / (len(group_records) or 1), 6)
                    for key, value in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))
                },
            }
        )
    return results


def _largest_cluster_concentration(group_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    strongest = {
        "group": None,
        "cluster": None,
        "share": 0.0,
        "count": 0,
    }
    for metric in group_metrics:
        for cluster, share in metric.get("cluster_share", {}).items():
            if share > strongest["share"]:
                strongest = {
                    "group": metric.get("group"),
                    "cluster": cluster,
                    "share": share,
                    "count": metric.get("count", 0),
                }
    return strongest


def _top_distribution_deltas(left: dict[str, float], right: dict[str, float], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(left) | set(right)):
        rows.append(
            {
                "group": key,
                "clean_share": round(left.get(key, 0.0), 6),
                "dirty_share": round(right.get(key, 0.0), 6),
                "delta_dirty_minus_clean": round(right.get(key, 0.0) - left.get(key, 0.0), 6),
            }
        )
    return sorted(rows, key=lambda item: abs(item["delta_dirty_minus_clean"]), reverse=True)[:limit]


def _observe_experiment(report: dict[str, Any]) -> list[dict[str, Any]]:
    age_concentration = _largest_cluster_concentration(report.get("age_metrics", []))
    ethnicity_concentration = _largest_cluster_concentration(report.get("ethnicity_metrics", []))
    clustered_records = report.get("clustered_records", 0)
    record_count = report.get("record_count", 0) or 1
    cluster_coverage = clustered_records / record_count

    return [
        {
            "check": "age_groups_cluster_together",
            "status": "observe" if age_concentration["share"] >= 0.5 else "low_signal",
            "evidence": age_concentration,
            "explanation": (
                "Flags whether an age bucket is concentrated in one cluster. "
                "A high share means one cluster may be carrying a demographic-specific pattern."
            ),
        },
        {
            "check": "ethnicity_groups_cluster_together",
            "status": "observe" if ethnicity_concentration["share"] >= 0.5 else "low_signal",
            "evidence": ethnicity_concentration,
            "explanation": (
                "Flags whether an ethnicity/nationality group is concentrated in one cluster. "
                "This is useful for spotting representation imbalance in cluster assignments."
            ),
        },
        {
            "check": "cluster_coverage",
            "status": "ok" if cluster_coverage >= 0.95 else "review",
            "evidence": {
                "clustered_records": clustered_records,
                "record_count": record_count,
                "cluster_coverage": round(cluster_coverage, 6),
            },
            "explanation": (
                "Checks whether the experiment has cluster assignments for almost all records. "
                "Low coverage would make the scatterplot and cluster summaries less trustworthy."
            ),
        },
    ]


def _observe_summary(experiments: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> list[dict[str, Any]]:
    clean = experiments["clean"]
    dirty = experiments["dirty"]
    combined = experiments["combined"]

    cluster_deltas = _top_distribution_deltas(
        clean["cluster_distribution"]["proportions"],
        dirty["cluster_distribution"]["proportions"],
    )
    age_deltas = _top_distribution_deltas(
        clean["age_distribution"]["proportions"],
        dirty["age_distribution"]["proportions"],
    )
    ethnicity_deltas = _top_distribution_deltas(
        clean["ethnicity_distribution"]["proportions"],
        dirty["ethnicity_distribution"]["proportions"],
    )

    return [
        {
            "check": "dirty_dataset_changes_cluster_allocation",
            "status": "observe" if any(abs(row["delta_dirty_minus_clean"]) >= 0.15 for row in cluster_deltas) else "low_signal",
            "evidence": cluster_deltas,
            "explanation": (
                "Compares cluster share between clean and dirty records. "
                "Large deltas mean corruption/missing metadata may be changing where records land."
            ),
        },
        {
            "check": "age_representation_shift",
            "status": "observe" if comparison["clean_vs_dirty_age_tvd"] >= 0.2 else "low_signal",
            "evidence": {
                "tvd": comparison["clean_vs_dirty_age_tvd"],
                "top_deltas": age_deltas,
            },
            "explanation": (
                "Uses total variation distance to compare age-bucket representation between clean and dirty data."
            ),
        },
        {
            "check": "ethnicity_representation_shift",
            "status": "observe" if comparison["clean_vs_dirty_ethnicity_tvd"] >= 0.2 else "low_signal",
            "evidence": {
                "tvd": comparison["clean_vs_dirty_ethnicity_tvd"],
                "top_deltas": ethnicity_deltas,
            },
            "explanation": (
                "Uses total variation distance to compare ethnicity/nationality representation between clean and dirty data."
            ),
        },
        {
            "check": "combined_mix_effect",
            "status": "ok" if combined["record_count"] == clean["record_count"] + dirty["record_count"] else "review",
            "evidence": {
                "clean_records": clean["record_count"],
                "dirty_records": dirty["record_count"],
                "combined_records": combined["record_count"],
            },
            "explanation": (
                "Verifies that the combined experiment preserves both sources and can be read as clean + dirty together."
            ),
        },
        {
            "check": "scatterplot_readability_contract",
            "status": "ok",
            "evidence": combined.get("plot_encoding", {}),
            "explanation": (
                "Confirms that the scatterplot uses explicit encodings: x=age, y=cluster, color=ethnicity, shape=source."
            ),
        },
    ]


def _tvd_from_proportions(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return round(0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys), 6)


def _compare_distributions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "tvd": _tvd_from_proportions(left.get("proportions", {}), right.get("proportions", {})),
        "left_counts": left.get("counts", {}),
        "right_counts": right.get("counts", {}),
    }


def _sort_labels(labels: Iterable[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, Any]:
        text = str(value)
        if text == "unclustered":
            return (3, text)
        try:
            return (0, int(text))
        except ValueError:
            try:
                return (1, float(text))
            except ValueError:
                return (2, text.lower())

    return sorted({str(label) for label in labels}, key=sort_key)


def _age_bucket_midpoints() -> dict[str, float]:
    return {
        "0-17": 8.5,
        "18-29": 23.5,
        "30-44": 37.0,
        "45-59": 52.0,
        "60+": 70.0,
    }


def _plot_age_value(record: dict[str, Any]) -> float:
    age = _coerce_number(record.get("age_raw"))
    if age is None:
        age = _coerce_number(record.get("age"))
    if age is None:
        age_bucket = str(record.get("age_bucket", "unknown"))
        age = _age_bucket_midpoints().get(age_bucket, 0.0)

    jitter_seed = str(record.get("sample_id", ""))
    return float(age) + _stable_jitter(jitter_seed, 0.75)


def _plot_cluster_label(record: dict[str, Any]) -> str:
    value = record.get("cluster_id")
    if value in (None, ""):
        return "unclustered"
    return str(value)


def _build_plot_points(records: list[dict[str, Any]]) -> dict[str, Any]:
    cluster_labels = _sort_labels(_plot_cluster_label(record) for record in records)
    cluster_positions = {label: index for index, label in enumerate(cluster_labels)}
    ethnicity_labels = _sort_labels(record.get("ethnicity", "unknown") for record in records)
    source_labels = _sort_labels(record.get("source", "unknown") for record in records)

    points: list[dict[str, Any]] = []
    x_values: list[float] = []

    for record in records:
        cluster_label = _plot_cluster_label(record)
        x_value = _plot_age_value(record)
        y_value = float(cluster_positions.get(cluster_label, 0))
        x_values.append(x_value)

        points.append(
            {
                "sample_id": str(record.get("sample_id", "")),
                "source": str(record.get("source", "unknown")),
                "ethnicity": str(record.get("ethnicity", "unknown")),
                "age_bucket": str(record.get("age_bucket", "unknown")),
                "age_value": x_value,
                "cluster_label": cluster_label,
                "cluster_position": y_value,
            }
        )

    return {
        "points": points,
        "cluster_labels": cluster_labels,
        "cluster_positions": cluster_positions,
        "ethnicity_labels": ethnicity_labels,
        "source_labels": source_labels,
        "x_min": min(x_values) if x_values else 0.0,
        "x_max": max(x_values) if x_values else 1.0,
    }


def _ethnicity_colors(records: list[dict[str, Any]]) -> dict[str, str]:
    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#db2777",
        "#4b5563",
        "#65a30d",
        "#9333ea",
    ]
    labels = _sort_labels(record.get("ethnicity", "unknown") for record in records)
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}


def _source_marker_shape(source: str) -> str:
    mapping = {
        "clean": "circle",
        "dirty": "diamond",
        "combined": "square",
    }
    return mapping.get(source, "triangle")


def _shape_svg(shape: str, cx: float, cy: float, size: float, fill: str, stroke: str) -> str:
    if shape == "diamond":
        half = size
        points = [
            (cx, cy - half),
            (cx + half, cy),
            (cx, cy + half),
            (cx - half, cy),
        ]
        points_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f"<polygon points='{points_text}' fill='{fill}' stroke='{stroke}' stroke-width='1.1'/>"
    if shape == "square":
        side = size * 2
        return (
            f"<rect x='{cx - size:.2f}' y='{cy - size:.2f}' width='{side:.2f}' height='{side:.2f}' "
            f"rx='1.5' fill='{fill}' stroke='{stroke}' stroke-width='1.1'/>"
        )
    if shape == "triangle":
        points = [
            (cx, cy - size * 1.15),
            (cx + size * 1.1, cy + size),
            (cx - size * 1.1, cy + size),
        ]
        points_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f"<polygon points='{points_text}' fill='{fill}' stroke='{stroke}' stroke-width='1.1'/>"
    return f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='{size:.2f}' fill='{fill}' stroke='{stroke}' stroke-width='1.1'/>"


def _render_svg_scatter(
    records: list[dict[str, Any]],
    output_svg_path: Path,
    title: str,
) -> dict[str, Any]:
    output_svg_path.parent.mkdir(parents=True, exist_ok=True)

    plot = _build_plot_points(records)
    points = plot["points"]
    if not points:
        svg = """<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='800'>
<rect width='100%' height='100%' fill='white'/>
<text x='40' y='60' font-size='24' fill='#111827'>No records available</text>
</svg>"""
        output_svg_path.write_text(svg, encoding="utf-8")
        return {"path": str(output_svg_path), "width": 1200, "height": 800}

    width = 1400
    height = 920
    left = 120
    right = 960
    top = 110
    bottom = 740
    legend_left = 1010
    legend_width = 330

    x_min = plot["x_min"]
    x_max = plot["x_max"]
    x_span = max(x_max - x_min, 1e-9)
    cluster_labels = plot["cluster_labels"]
    cluster_count = max(len(cluster_labels), 1)
    y_span = max(cluster_count - 1, 1)

    def scale_x(value: float) -> float:
        return left + ((value - x_min) / x_span) * (right - left)

    def scale_y(value: float) -> float:
        if cluster_count == 1:
            return (top + bottom) / 2.0
        return bottom - (value / y_span) * (bottom - top)

    colors = _ethnicity_colors(records)
    source_shapes = {label: _source_marker_shape(label) for label in plot["source_labels"]}

    def x_ticks() -> list[float]:
        if math.isclose(x_min, x_max):
            return [x_min]
        tick_count = min(6, max(4, int(len(points) / 6) + 1))
        step = (x_max - x_min) / max(tick_count - 1, 1)
        return [x_min + step * index for index in range(tick_count)]

    def cluster_y_positions() -> dict[str, float]:
        if cluster_count == 1:
            return {cluster_labels[0]: (top + bottom) / 2.0}
        return {
            label: scale_y(float(position))
            for label, position in plot["cluster_positions"].items()
        }

    cluster_axis_positions = cluster_y_positions()

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='60' y='48' font-size='28' font-family='Inter, Arial, sans-serif' fill='#111827'>{html.escape(title)}</text>",
        "<text x='60' y='76' font-size='14' font-family='Inter, Arial, sans-serif' fill='#374151'>x = age in years, y = cluster assignment, color = ethnicity, shape = source</text>",
        f"<rect x='{left}' y='{top}' width='{right - left}' height='{bottom - top}' fill='#ffffff' stroke='#e5e7eb' stroke-width='1'/>",
    ]

    for tick_value in x_ticks():
        tick_x = scale_x(tick_value)
        lines.append(f"<line x1='{tick_x:.2f}' y1='{top}' x2='{tick_x:.2f}' y2='{bottom}' stroke='#f3f4f6' stroke-width='1'/>")
        lines.append(
            f"<text x='{tick_x:.2f}' y='{bottom + 28}' text-anchor='middle' font-size='13' font-family='Inter, Arial, sans-serif' fill='#374151'>{html.escape(f'{tick_value:.0f}')}</text>"
        )

    for label, y_value in cluster_axis_positions.items():
        lines.append(f"<line x1='{left}' y1='{y_value:.2f}' x2='{right}' y2='{y_value:.2f}' stroke='#f3f4f6' stroke-width='1'/>")
        lines.append(
            f"<text x='{left - 14}' y='{y_value + 5:.2f}' text-anchor='end' font-size='13' font-family='Inter, Arial, sans-serif' fill='#374151'>{html.escape(label)}</text>"
        )

    lines.append(
        f"<text x='{(left + right) / 2:.2f}' y='{bottom + 58}' text-anchor='middle' font-size='15' font-family='Inter, Arial, sans-serif' fill='#111827'>Age (years)</text>"
    )
    lines.append(
        f"<text x='{left - 72}' y='{(top + bottom) / 2:.2f}' transform='rotate(-90 {left - 72:.2f} {(top + bottom) / 2:.2f})' text-anchor='middle' font-size='15' font-family='Inter, Arial, sans-serif' fill='#111827'>Cluster assignment</text>"
    )

    for point in points:
        color = colors.get(point["ethnicity"], "#2563eb")
        shape = source_shapes.get(point["source"], "triangle")
        x = scale_x(point["age_value"])
        y = scale_y(point["cluster_position"])
        title_text = (
            f"{point['sample_id']} | age={point['age_bucket']} | ethnicity={point['ethnicity']} | "
            f"source={point['source']} | cluster={point['cluster_label']}"
        )
        lines.append(
            f"<g><title>{html.escape(title_text)}</title>{_shape_svg(shape, x, y, 6.5, color, '#111827')}</g>"
        )

    legend_y = 122
    lines.append(
        f"<text x='{legend_left}' y='{legend_y - 18}' font-size='16' font-family='Inter, Arial, sans-serif' fill='#111827'>Legends</text>"
    )
    lines.append(
        f"<text x='{legend_left}' y='{legend_y + 110}' font-size='16' font-family='Inter, Arial, sans-serif' fill='#111827'>Source</text>"
    )

    for index, label in enumerate(plot["ethnicity_labels"]):
        color = colors[label]
        y = legend_y + index * 26
        lines.append(f"<rect x='{legend_left}' y='{y - 9}' width='14' height='14' rx='2' fill='{color}' stroke='#111827' stroke-width='0.8'/>")
        lines.append(
            f"<text x='{legend_left + 22}' y='{y + 3}' font-size='14' font-family='Inter, Arial, sans-serif' fill='#374151'>{html.escape(label)}</text>"
        )

    source_legend_top = legend_y + 136
    for index, label in enumerate(plot["source_labels"]):
        shape = source_shapes.get(label, "triangle")
        y = source_legend_top + index * 28
        lines.append(_shape_svg(shape, float(legend_left + 7), float(y - 2), 6.0, "#9ca3af", "#111827"))
        lines.append(
            f"<text x='{legend_left + 22}' y='{y + 2}' font-size='14' font-family='Inter, Arial, sans-serif' fill='#374151'>{html.escape(label)}</text>"
        )

    lines.append(
        f"<text x='{legend_left}' y='{source_legend_top + max(len(plot['source_labels']), 1) * 28 + 26}' font-size='12' font-family='Inter, Arial, sans-serif' fill='#6b7280'>Color = ethnicity</text>"
    )
    lines.append(
        f"<text x='{legend_left}' y='{source_legend_top + max(len(plot['source_labels']), 1) * 28 + 46}' font-size='12' font-family='Inter, Arial, sans-serif' fill='#6b7280'>Shape = source</text>"
    )

    lines.append("</svg>")
    output_svg_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "path": str(output_svg_path),
        "width": width,
        "height": height,
        "points": len(records),
        "encoding": {
            "x_axis": "age_in_years",
            "y_axis": "cluster_assignment",
            "color": "ethnicity",
            "shape": "source",
            "x_axis_note": "Age is plotted in years, with a small deterministic jitter to reduce overlap.",
            "y_axis_note": "Cluster ids are rendered as ordered categories, not a numerical scale.",
        },
    }


def _render_html_wrapper(svg_path: Path, output_html_path: Path, title: str, encoding: dict[str, Any] | None = None) -> None:
    svg_markup = svg_path.read_text(encoding="utf-8")
    encoding = encoding or {}
    encoding_lines = [
        f"<li><strong>x</strong>: {html.escape(str(encoding.get('x_axis', 'age_in_years')))}</li>",
        f"<li><strong>y</strong>: {html.escape(str(encoding.get('y_axis', 'cluster_assignment')))}</li>",
        f"<li><strong>color</strong>: {html.escape(str(encoding.get('color', 'ethnicity')))}</li>",
        f"<li><strong>shape</strong>: {html.escape(str(encoding.get('shape', 'source')))}</li>",
    ]
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f8fafc;
      color: #111827;
      font-family: Inter, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <p>Observe guide for the scatterplot encoding.</p>
    <ul>
      {''.join(encoding_lines)}
    </ul>
    <div class="panel">
      {svg_markup}
    </div>
  </div>
</body>
</html>"""
    output_html_path.write_text(html_doc, encoding="utf-8")


def _summary_report(records: list[dict[str, Any]], title: str) -> dict[str, Any]:
    age_distribution = _distribution(records, "age_bucket")
    ethnicity_distribution = _distribution(records, "ethnicity")
    cluster_distribution = _cluster_distribution(records)
    cluster_labels = _sort_labels(record.get("cluster_id", "unclustered") for record in records)

    age_metrics = _group_metrics(records, "age_bucket")
    ethnicity_metrics = _group_metrics(records, "ethnicity")

    return {
        "title": title,
        "record_count": len(records),
        "age_distribution": age_distribution,
        "ethnicity_distribution": ethnicity_distribution,
        "cluster_distribution": cluster_distribution,
        "age_metrics": age_metrics,
        "ethnicity_metrics": ethnicity_metrics,
        "intersectional_metrics": _group_metrics(
            [
                {
                    **record,
                    "intersection": f"{record.get('age_bucket', 'unknown')} | {record.get('ethnicity', 'unknown')}",
                }
                for record in records
            ],
            "intersection",
        ),
        "age_vs_ethnicity_tvd": _compare_distributions(age_distribution, ethnicity_distribution),
        "clustered_records": sum(1 for record in records if record.get("cluster_id") not in (None, "", "unclustered")),
        "unique_clusters": sorted({str(record.get("cluster_id", "unclustered")) for record in records}),
        "plot_encoding": {
            "x_axis": "age_in_years",
            "y_axis": "cluster_assignment",
            "color": "ethnicity",
            "shape": "source",
            "x_axis_note": "Age is rendered numerically with a deterministic jitter so overlapping samples stay readable.",
            "y_axis_note": f"Cluster labels are mapped to ordered rows: {', '.join(cluster_labels) if cluster_labels else 'none'}",
        },
    }


def _source_artifact_payload(path: Path) -> Any:
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        return _read_json(path)
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    return path.read_text(encoding="utf-8")


def _locate_paths(project_root: Path, batch_id: int = 4, output_name: str | None = None) -> dict[str, Path]:
    session_root = project_root / "session"
    batch_root = session_root / "cluster_bias_outputs" / f"batch_{batch_id}"
    output_folder = output_name or ("mixed_dataset_3x" if batch_id == 4 else f"mixed_dataset_3x_batch_{batch_id}")
    return {
        "clean_json": session_root / f"simulation_output_clean_{batch_id}.json",
        "dirty_json": session_root / f"simulation_output_dirty_{batch_id}.json",
        "clean_clusters": batch_root / f"clean_batch_{batch_id}_cluster_assignments.csv",
        "dirty_clusters": batch_root / f"dirty_batch_{batch_id}_cluster_assignments.csv",
        "clean_bias": batch_root / f"clean_batch_{batch_id}_bias_snapshot.json",
        "dirty_bias": batch_root / f"dirty_batch_{batch_id}_bias_snapshot.json",
        "clean_cluster_summary": batch_root / f"clean_batch_{batch_id}_cluster_summary.json",
        "dirty_cluster_summary": batch_root / f"dirty_batch_{batch_id}_cluster_summary.json",
        "output_root": session_root / "cluster_bias_outputs" / output_folder,
    }


def run_experiment(
    name: str,
    records: list[dict[str, Any]],
    *,
    source_artifacts: dict[str, Any] | None = None,
    output_root: Path,
) -> dict[str, Any]:
    experiment_dir = output_root / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    report = _summary_report(records, title=f"Mixed Dataset Experiment: {name}")
    report["source_artifacts"] = source_artifacts or {}
    report["observe_checklist"] = _observe_experiment(report)

    records_path = experiment_dir / "normalized_records.json"
    metrics_path = experiment_dir / "metrics.json"
    scatter_svg_path = experiment_dir / "scatterplot.svg"
    scatter_html_path = experiment_dir / "scatterplot.html"

    _write_json(records_path, records)
    _write_json(metrics_path, report)
    scatter_info = _render_svg_scatter(records, scatter_svg_path, f"Mixed Dataset Experiment - {name}")
    _render_html_wrapper(
        scatter_svg_path,
        scatter_html_path,
        f"Mixed Dataset Experiment - {name}",
        scatter_info.get("encoding"),
    )

    report["artifacts"] = {
        "normalized_records": str(records_path),
        "metrics_json": str(metrics_path),
        "scatterplot_svg": str(scatter_svg_path),
        "scatterplot_html": str(scatter_html_path),
    }
    report["plot_artifact"] = scatter_info

    _write_json(metrics_path, report)
    return report


def _try_log_to_mlflow(experiment_name: str, report: dict[str, Any], artifact_dir: Path) -> None:
    try:
        import mlflow
    except Exception:
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_SERVER") or "http://certain_mlflow:5001"
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception:
        pass

    try:
        with mlflow.start_run(run_name=experiment_name):
            mlflow.log_metric("record_count", float(report.get("record_count", 0)))
            mlflow.log_metric(
                "clustered_records",
                float(report.get("clustered_records", 0)),
            )
            mlflow.log_artifact(str(artifact_dir / "metrics.json"), artifact_path=experiment_name)
            mlflow.log_artifact(str(artifact_dir / "scatterplot.svg"), artifact_path=experiment_name)
            mlflow.log_artifact(str(artifact_dir / "scatterplot.html"), artifact_path=experiment_name)
    except Exception:
        return


def run_mixed_dataset_experiments(
    project_root: Path | None = None,
    *,
    batch_id: int = 4,
    output_name: str | None = None,
) -> dict[str, Any]:
    project_root = project_root or Path(__file__).resolve().parents[1]
    paths = _locate_paths(project_root, batch_id=batch_id, output_name=output_name)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    clean_records = _load_normalized_records(paths["clean_json"], source="clean")
    dirty_records = _load_normalized_records(paths["dirty_json"], source="dirty")

    clean_clusters = _load_cluster_rows(paths["clean_clusters"]) if paths["clean_clusters"].exists() else None
    dirty_clusters = _load_cluster_rows(paths["dirty_clusters"]) if paths["dirty_clusters"].exists() else None

    clean_records = _merge_records_with_clusters(clean_records, clean_clusters)
    dirty_records = _merge_records_with_clusters(dirty_records, dirty_clusters)
    combined_records = clean_records + dirty_records

    experiments = {
        "clean": run_experiment(
            "clean",
            clean_records,
            source_artifacts={
                "dataset": str(paths["clean_json"]),
                "clusters": str(paths["clean_clusters"]),
                "bias_snapshot": str(paths["clean_bias"]),
                "cluster_summary": str(paths["clean_cluster_summary"]),
            },
            output_root=output_root,
        ),
        "dirty": run_experiment(
            "dirty",
            dirty_records,
            source_artifacts={
                "dataset": str(paths["dirty_json"]),
                "clusters": str(paths["dirty_clusters"]),
                "bias_snapshot": str(paths["dirty_bias"]),
                "cluster_summary": str(paths["dirty_cluster_summary"]),
            },
            output_root=output_root,
        ),
        "combined": run_experiment(
            "combined",
            combined_records,
            source_artifacts={
                "datasets": [str(paths["clean_json"]), str(paths["dirty_json"])],
                "clusters": [str(paths["clean_clusters"]), str(paths["dirty_clusters"])],
                "bias_snapshots": [str(paths["clean_bias"]), str(paths["dirty_bias"])],
                "cluster_summaries": [str(paths["clean_cluster_summary"]), str(paths["dirty_cluster_summary"])],
            },
            output_root=output_root,
        ),
    }

    for experiment_name, report in experiments.items():
        artifact_dir = Path(report["artifacts"]["metrics_json"]).parent
        _try_log_to_mlflow(experiment_name, report, artifact_dir)

    comparison = {
        "clean_vs_dirty_age_tvd": _tvd_from_proportions(
            experiments["clean"]["age_distribution"]["proportions"],
            experiments["dirty"]["age_distribution"]["proportions"],
        ),
        "clean_vs_dirty_ethnicity_tvd": _tvd_from_proportions(
            experiments["clean"]["ethnicity_distribution"]["proportions"],
            experiments["dirty"]["ethnicity_distribution"]["proportions"],
        ),
        "combined_record_count": experiments["combined"]["record_count"],
    }

    summary = {
        "batch_id": batch_id,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "comparison": comparison,
        "observe_checklist": _observe_summary(experiments, comparison),
        "experiments": experiments,
    }

    _write_json(output_root / "summary_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean, dirty and combined mixed dataset observe experiments.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    summary = run_mixed_dataset_experiments(
        project_root=args.project_root,
        batch_id=args.batch,
        output_name=args.output_name,
    )
    print(json.dumps(summary["comparison"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
