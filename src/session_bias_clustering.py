"""Cluster synthetic session datasets and prepare MLOps-ready bias artifacts.

This module bridges the synthetic-data outputs produced in ``session/`` with the
MLOps stack. It performs lightweight text clustering on the clean and dirty JSON
datasets, enriches each record with cluster assignments, computes metadata-aware
bias summaries, writes the results locally, and can optionally log everything to
MLflow / ``certain_library`` for downstream analysis.

Usage:
    python -m src.session_bias_clustering
    python -m src.session_bias_clustering --batch 4 --log-to-mlflow
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    _SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None  # type: ignore[assignment]
    TruncatedSVD = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    silhouette_score = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


METADATA_FIELDS = [
    "FirstName",
    "LastName",
    "Age",
    "DateOfBirth",
    "Nationality",
    "Background",
    "Opinion",
    "AnswerQuality",
    "BiasGroup",
]

PUBLIC_METADATA_FIELDS = [
    "FirstName",
    "LastName",
    "Age",
    "DateOfBirth",
    "Nationality",
]

INTERNAL_METADATA_FIELDS = [
    "Background",
    "Opinion",
    "AnswerQuality",
    "BiasGroup",
    "WrongAnswerPattern",
]

SCATTER_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#c2410c",
]

FALLBACK_CLUSTER_STOPWORDS = {
    "about",
    "access",
    "age",
    "answer",
    "background",
    "bias",
    "bucket",
    "correct",
    "country",
    "employees",
    "explicitly",
    "final",
    "group",
    "label",
    "marker",
    "measure",
    "metadata",
    "opinion",
    "policy",
    "quality",
    "question",
    "remote",
    "respondent",
    "response",
    "safeguard",
    "safeguards",
    "treated",
    "work",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster clean/dirty synthetic datasets from session/ and export "
            "bias-oriented artifacts for MLOps."
        )
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("session"),
        help="Directory containing simulation_output_clean_*.json and dirty files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where clustering and bias artifacts will be written.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Optional batch number to process. If omitted, process all discovered pairs.",
    )
    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="Also log clustered datasets, whylogs profiles, metrics and artifacts to MLflow.",
    )
    parser.add_argument(
        "--experiment-name",
        default="synthetic_session_bias_analysis",
        help="MLflow experiment name used when --log-to-mlflow is enabled.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=6,
        help="Upper bound for automatic KMeans cluster search.",
    )
    return parser.parse_args()


def _discover_batches(session_dir: Path) -> dict[int, dict[str, Path]]:
    discovered: dict[int, dict[str, Path]] = {}
    for dataset_type in ("clean", "dirty"):
        pattern = f"simulation_output_{dataset_type}_*.json"
        for path in sorted(session_dir.glob(pattern)):
            suffix = path.stem.rsplit("_", 1)[-1]
            if not suffix.isdigit():
                continue
            batch = int(suffix)
            discovered.setdefault(batch, {})[dataset_type] = path
    return {
        batch: files
        for batch, files in discovered.items()
        if "clean" in files and "dirty" in files
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_internal_annotations(session_dir: Path, batch_id: int) -> dict[str, dict[str, Any]]:
    path = session_dir / f"scenario_annotations_batch_{batch_id}.json"
    if not path.exists():
        return {"clean": {}, "dirty": {}}
    payload = _load_json(path)

    def by_id(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            str(record.get("Id")): record.get("InternalMetadata") or {}
            for record in records
            if record.get("Id")
        }

    return {
        "clean": by_id(payload.get("Clean", [])),
        "dirty": by_id(payload.get("Dirty", [])),
    }


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _age_bucket(age: Any) -> str:
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "unknown"
    try:
        age_value = int(age)
    except (TypeError, ValueError):
        return "unknown"

    if age_value < 30:
        return "18-29"
    if age_value < 45:
        return "30-44"
    if age_value < 60:
        return "45-59"
    return "60+"


def _flatten_dataset(
    payload: dict[str, Any],
    dataset_type: str,
    batch_id: int,
    internal_annotations: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    topic_title = payload.get("TopicTitle", "")
    topic_summary = payload.get("TopicSummary", "")

    for record in payload.get("Dataset", []):
        metadata = record.get("Metadata") or {}
        record_id = record.get("Id")
        internal_metadata = (internal_annotations or {}).get(str(record_id), {})
        normalized_metadata = {field: metadata.get(field) for field in METADATA_FIELDS}
        for field in INTERNAL_METADATA_FIELDS:
            if normalized_metadata.get(field) in (None, ""):
                normalized_metadata[field] = internal_metadata.get(field)
        missing_fields = [field for field in PUBLIC_METADATA_FIELDS if metadata.get(field) in (None, "")]
        row = {
            "batch_id": batch_id,
            "dataset_type": dataset_type,
            "topic_title": topic_title,
            "topic_summary": topic_summary,
            "record_id": record_id,
            "text": _safe_string(record.get("Text")),
            "text_length": len(_safe_string(record.get("Text"))),
            "metadata_missing_count": len(missing_fields),
            "metadata_completeness": 1.0 - (len(missing_fields) / len(PUBLIC_METADATA_FIELDS)),
            "is_anonymous": int(len(missing_fields) == len(PUBLIC_METADATA_FIELDS)),
        }

        for field in METADATA_FIELDS:
            column = f"metadata_{field.lower()}"
            row[column] = normalized_metadata.get(field)

        row["age_bucket"] = _age_bucket(normalized_metadata.get("Age"))
        rows.append(row)

    return pd.DataFrame(rows)


def _normalise_distribution(values: Iterable[Any]) -> dict[str, float]:
    filtered = [_safe_string(value) or "unknown" for value in values]
    total = len(filtered)
    if total == 0:
        return {}
    counts = Counter(filtered)
    return {key: value / total for key, value in sorted(counts.items())}


def _tokenise(text: str) -> list[str]:
    normalized = re.sub(r"^\[[^\]]+\]\s*", "", text.lower())
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_+-]{3,}", normalized)
        if token not in FALLBACK_CLUSTER_STOPWORDS
    ]


def _total_variation_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)


def _normalise_series(values: Iterable[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").fillna(0.0)
    array = series.to_numpy(dtype=float)
    if len(array) == 0:
        return array
    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(maximum, minimum):
        return np.zeros(len(array), dtype=float)
    return ((array - minimum) / (maximum - minimum)) * 2.0 - 1.0


def _stable_jitter(value: Any, scale: float = 0.035) -> float:
    text = _safe_string(value)
    if not text:
        return 0.0
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return (((checksum % 997) / 996.0) - 0.5) * 2.0 * scale


def _cluster_axis(cluster_ids: Iterable[Any]) -> np.ndarray:
    labels = list(cluster_ids)
    ordered = {cluster_id: index for index, cluster_id in enumerate(sorted(set(labels)))}
    if len(ordered) <= 1:
        return np.zeros(len(labels), dtype=float)
    center = (len(ordered) - 1) / 2.0
    return np.array([(ordered[cluster_id] - center) / center for cluster_id in labels], dtype=float)


def _attach_scatter_columns(
    clustered: pd.DataFrame,
    coordinates: np.ndarray | None = None,
) -> pd.DataFrame:
    with_scatter = clustered.copy()
    with_scatter["bias_severity"] = (
        pd.to_numeric(with_scatter["metadata_missing_count"], errors="coerce").fillna(0.0)
        / len(PUBLIC_METADATA_FIELDS)
    )

    if coordinates is not None and len(coordinates) == len(with_scatter):
        coordinate_array = np.asarray(coordinates, dtype=float)
        x_values = _normalise_series(coordinate_array[:, 0])
        if coordinate_array.shape[1] > 1 and not math.isclose(
            float(np.nanmax(coordinate_array[:, 1])),
            float(np.nanmin(coordinate_array[:, 1])),
        ):
            y_values = _normalise_series(coordinate_array[:, 1])
        else:
            y_values = _normalise_series(with_scatter["metadata_completeness"])
    else:
        x_values = _cluster_axis(with_scatter["cluster_id"])
        y_values = _normalise_series(with_scatter["metadata_completeness"])
        if len(set(np.round(y_values, 8).tolist())) <= 1:
            y_values = _normalise_series(with_scatter["text_length"])

    jitter = np.array([_stable_jitter(value) for value in with_scatter["record_id"]])
    with_scatter["scatter_x"] = np.clip(x_values + jitter, -1.08, 1.08)
    with_scatter["scatter_y"] = np.clip(y_values - jitter, -1.08, 1.08)
    return with_scatter


def _top_terms_per_cluster(
    matrix: Any,
    assignments: np.ndarray,
    feature_names: np.ndarray,
    top_n: int = 8,
) -> dict[int, list[str]]:
    top_terms: dict[int, list[str]] = {}
    for cluster_id in sorted(set(assignments.tolist())):
        member_indices = np.where(assignments == cluster_id)[0]
        if len(member_indices) == 0:
            top_terms[cluster_id] = []
            continue
        cluster_matrix = matrix[member_indices]
        weights = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        ranked_indices = weights.argsort()[::-1][:top_n]
        top_terms[cluster_id] = [feature_names[index] for index in ranked_indices if weights[index] > 0]
    return top_terms


def _select_cluster_count(matrix: Any, record_count: int, max_clusters: int) -> tuple[int, float | None]:
    if record_count < 4:
        return 1, None

    upper_bound = min(max_clusters, record_count - 1)
    if upper_bound < 2:
        return 1, None

    best_k = 1
    best_score = -1.0

    for cluster_count in range(2, upper_bound + 1):
        model = KMeans(n_clusters=cluster_count, n_init=20, random_state=42)
        labels = model.fit_predict(matrix)
        if len(set(labels.tolist())) < 2:
            continue
        try:
            score = silhouette_score(matrix, labels, metric="cosine")
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_k = cluster_count

    if best_k == 1:
        return 1, None
    return best_k, best_score


def _cluster_dataset(df: pd.DataFrame, max_clusters: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    clustered = df.copy()
    texts = clustered["text"].fillna("").tolist()

    if clustered.empty:
        raise ValueError("Cannot cluster an empty dataset.")

    if not _SKLEARN_AVAILABLE:
        return _cluster_dataset_without_sklearn(clustered, max_clusters=max_clusters)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=2000,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    if matrix.shape[1] == 0 or len(texts) < 2:
        clustered["cluster_id"] = 0
        clustered = _attach_scatter_columns(clustered)
        return clustered, {
            "cluster_count": 1,
            "silhouette_score": None,
            "cluster_top_terms": {"0": []},
        }

    best_k, best_score = _select_cluster_count(matrix, len(clustered), max_clusters)
    if best_k == 1:
        assignments = np.zeros(len(clustered), dtype=int)
    else:
        model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
        assignments = model.fit_predict(matrix)

    clustered["cluster_id"] = assignments.astype(int)
    coordinates: np.ndarray | None = None
    if matrix.shape[1] >= 2 and len(clustered) >= 2:
        coordinates = TruncatedSVD(n_components=2, random_state=42).fit_transform(matrix)
    elif matrix.shape[1] == 1:
        coordinates = np.column_stack((matrix.toarray().ravel(), np.zeros(len(clustered))))
    clustered = _attach_scatter_columns(clustered, coordinates=coordinates)
    top_terms = _top_terms_per_cluster(matrix, assignments, feature_names)

    return clustered, {
        "cluster_count": int(len(set(assignments.tolist()))),
        "silhouette_score": None if best_score is None else float(best_score),
        "cluster_top_terms": {str(key): value for key, value in top_terms.items()},
    }


def _cluster_dataset_without_sklearn(
    clustered: pd.DataFrame,
    max_clusters: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    token_sets = [set(_tokenise(text)) for text in clustered["text"].fillna("").tolist()]
    cluster_tokens: list[Counter[str]] = []
    assignments: list[int] = []

    for token_set in token_sets:
        if not cluster_tokens:
            cluster_tokens.append(Counter(token_set))
            assignments.append(0)
            continue

        best_cluster = 0
        best_score = -1.0
        for cluster_id, token_counter in enumerate(cluster_tokens):
            reference = set(token_counter.keys())
            union = token_set | reference
            similarity = 1.0 if not union else len(token_set & reference) / len(union)
            if similarity > best_score:
                best_score = similarity
                best_cluster = cluster_id

        if best_score < 0.36 and len(cluster_tokens) < max(1, max_clusters):
            cluster_tokens.append(Counter(token_set))
            assignments.append(len(cluster_tokens) - 1)
        else:
            cluster_tokens[best_cluster].update(token_set)
            assignments.append(best_cluster)

    clustered["cluster_id"] = assignments
    clustered = _attach_scatter_columns(clustered)
    top_terms = {
        str(cluster_id): [term for term, _count in counter.most_common(8)]
        for cluster_id, counter in enumerate(cluster_tokens)
    }

    return clustered, {
        "cluster_count": int(len(cluster_tokens)),
        "silhouette_score": None,
        "cluster_top_terms": top_terms,
    }


def _build_cluster_summary(
    clustered_df: pd.DataFrame,
    clustering_meta: dict[str, Any],
) -> dict[str, Any]:
    overall_nationality = _normalise_distribution(clustered_df["metadata_nationality"])
    overall_background = _normalise_distribution(clustered_df["metadata_background"])
    overall_age_bucket = _normalise_distribution(clustered_df["age_bucket"])

    clusters: list[dict[str, Any]] = []
    for cluster_id, cluster_df in clustered_df.groupby("cluster_id", sort=True):
        nationality_distribution = _normalise_distribution(cluster_df["metadata_nationality"])
        background_distribution = _normalise_distribution(cluster_df["metadata_background"])
        age_bucket_distribution = _normalise_distribution(cluster_df["age_bucket"])

        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "records": int(len(cluster_df)),
                "share_of_dataset": float(len(cluster_df) / len(clustered_df)),
                "anonymous_rate": float(cluster_df["is_anonymous"].mean()),
                "metadata_missing_rate": float(
                    cluster_df["metadata_missing_count"].sum()
                    / (len(cluster_df) * len(PUBLIC_METADATA_FIELDS))
                ),
                "avg_text_length": float(cluster_df["text_length"].mean()),
                "avg_age": None
                if cluster_df["metadata_age"].dropna().empty
                else float(pd.to_numeric(cluster_df["metadata_age"], errors="coerce").dropna().mean()),
                "nationality_distribution": nationality_distribution,
                "background_distribution": background_distribution,
                "age_bucket_distribution": age_bucket_distribution,
                "opinion_distribution": _normalise_distribution(cluster_df["metadata_opinion"]),
                "answer_quality_distribution": _normalise_distribution(
                    cluster_df["metadata_answerquality"]
                ),
                "bias_group_distribution": _normalise_distribution(cluster_df["metadata_biasgroup"]),
                "nationality_tvd_vs_dataset": float(
                    _total_variation_distance(nationality_distribution, overall_nationality)
                ),
                "background_tvd_vs_dataset": float(
                    _total_variation_distance(background_distribution, overall_background)
                ),
                "age_bucket_tvd_vs_dataset": float(
                    _total_variation_distance(age_bucket_distribution, overall_age_bucket)
                ),
                "top_terms": clustering_meta["cluster_top_terms"].get(str(cluster_id), []),
            }
        )

    return {
        "cluster_count": clustering_meta["cluster_count"],
        "silhouette_score": clustering_meta["silhouette_score"],
        "clusters": clusters,
    }


def _representation_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "nationality_distribution": _normalise_distribution(df["metadata_nationality"]),
        "background_distribution": _normalise_distribution(df["metadata_background"]),
        "age_bucket_distribution": _normalise_distribution(df["age_bucket"]),
        "opinion_distribution": _normalise_distribution(df["metadata_opinion"]),
        "answer_quality_distribution": _normalise_distribution(df["metadata_answerquality"]),
        "bias_group_distribution": _normalise_distribution(df["metadata_biasgroup"]),
    }


def _build_bias_snapshot(clustered_df: pd.DataFrame, cluster_summary: dict[str, Any]) -> dict[str, Any]:
    total_records = len(clustered_df)
    total_metadata_slots = total_records * len(PUBLIC_METADATA_FIELDS)
    field_null_rates = {
        field: float(clustered_df[f"metadata_{field.lower()}"].isna().mean())
        for field in PUBLIC_METADATA_FIELDS
    }

    clusters = cluster_summary["clusters"]
    weighted_nationality_tvd = sum(
        cluster["share_of_dataset"] * cluster["nationality_tvd_vs_dataset"]
        for cluster in clusters
    )
    weighted_background_tvd = sum(
        cluster["share_of_dataset"] * cluster["background_tvd_vs_dataset"]
        for cluster in clusters
    )
    weighted_age_bucket_tvd = sum(
        cluster["share_of_dataset"] * cluster["age_bucket_tvd_vs_dataset"]
        for cluster in clusters
    )
    largest_cluster_share = max((cluster["share_of_dataset"] for cluster in clusters), default=1.0)

    return {
        "records": total_records,
        "cluster_count": cluster_summary["cluster_count"],
        "anonymous_rate": float(clustered_df["is_anonymous"].mean()),
        "metadata_missing_rate": float(clustered_df["metadata_missing_count"].sum() / total_metadata_slots),
        "field_null_rates": field_null_rates,
        "representation": _representation_summary(clustered_df),
        "cluster_bias_signals": {
            "weighted_nationality_tvd": float(weighted_nationality_tvd),
            "weighted_background_tvd": float(weighted_background_tvd),
            "weighted_age_bucket_tvd": float(weighted_age_bucket_tvd),
            "largest_cluster_share": float(largest_cluster_share),
            "silhouette_score": cluster_summary["silhouette_score"],
        },
    }


def _compare_clean_dirty(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    clean_bias: dict[str, Any],
    dirty_bias: dict[str, Any],
) -> dict[str, Any]:
    clean_representation = clean_bias["representation"]
    dirty_representation = dirty_bias["representation"]

    return {
        "clean_records": int(len(clean_df)),
        "dirty_records": int(len(dirty_df)),
        "delta_anonymous_rate": float(
            dirty_bias["anonymous_rate"] - clean_bias["anonymous_rate"]
        ),
        "delta_metadata_missing_rate": float(
            dirty_bias["metadata_missing_rate"] - clean_bias["metadata_missing_rate"]
        ),
        "delta_weighted_nationality_tvd": float(
            dirty_bias["cluster_bias_signals"]["weighted_nationality_tvd"]
            - clean_bias["cluster_bias_signals"]["weighted_nationality_tvd"]
        ),
        "delta_weighted_background_tvd": float(
            dirty_bias["cluster_bias_signals"]["weighted_background_tvd"]
            - clean_bias["cluster_bias_signals"]["weighted_background_tvd"]
        ),
        "delta_weighted_age_bucket_tvd": float(
            dirty_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"]
            - clean_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"]
        ),
        "representation_shift": {
            "nationality_tvd": float(
                _total_variation_distance(
                    clean_representation["nationality_distribution"],
                    dirty_representation["nationality_distribution"],
                )
            ),
            "background_tvd": float(
                _total_variation_distance(
                    clean_representation["background_distribution"],
                    dirty_representation["background_distribution"],
                )
            ),
            "age_bucket_tvd": float(
                _total_variation_distance(
                    clean_representation["age_bucket_distribution"],
                    dirty_representation["age_bucket_distribution"],
                )
            ),
            "opinion_tvd": float(
                _total_variation_distance(
                    clean_representation["opinion_distribution"],
                    dirty_representation["opinion_distribution"],
                )
            ),
            "answer_quality_tvd": float(
                _total_variation_distance(
                    clean_representation["answer_quality_distribution"],
                    dirty_representation["answer_quality_distribution"],
                )
            ),
            "bias_group_tvd": float(
                _total_variation_distance(
                    clean_representation["bias_group_distribution"],
                    dirty_representation["bias_group_distribution"],
                )
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _svg_text(value: Any) -> str:
    return html.escape(_safe_string(value) or "unknown")


def _cluster_color(cluster_id: Any) -> str:
    try:
        index = int(cluster_id)
    except (TypeError, ValueError):
        index = 0
    return SCATTER_PALETTE[index % len(SCATTER_PALETTE)]


def _bias_diagnosis(
    bias_snapshot: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    signals = bias_snapshot.get("cluster_bias_signals", {})
    weighted_signals = {
        "nationality cluster skew": float(signals.get("weighted_nationality_tvd", 0.0) or 0.0),
        "background cluster skew": float(signals.get("weighted_background_tvd", 0.0) or 0.0),
        "age cluster skew": float(signals.get("weighted_age_bucket_tvd", 0.0) or 0.0),
    }
    missing_rate = float(bias_snapshot.get("metadata_missing_rate", 0.0) or 0.0)
    cluster_count = int(bias_snapshot.get("cluster_count", 0) or 0)
    largest_cluster = float(signals.get("largest_cluster_share", 0.0) or 0.0)
    cluster_concentration = largest_cluster if cluster_count > 1 else 0.0

    rows: list[tuple[str, float, str]] = [
        ("missing metadata", missing_rate, "share of protected metadata that is absent"),
        ("cluster concentration", cluster_concentration, "one cluster dominates while other clusters exist"),
    ]
    rows.extend((name, value, "protected group distribution differs by cluster") for name, value in weighted_signals.items())

    score = max([missing_rate, cluster_concentration * 0.5, *weighted_signals.values()])
    driver_name, driver_value, _driver_help = max(rows, key=lambda item: item[1])
    threshold_note = "OK < 0.15, watch 0.15-0.29, biased >= 0.30"

    if comparison is not None:
        shifts = comparison.get("representation_shift", {})
        representation_rows = {
            "nationality shift": float(shifts.get("nationality_tvd", 0.0) or 0.0),
            "background shift": float(shifts.get("background_tvd", 0.0) or 0.0),
            "age shift": float(shifts.get("age_bucket_tvd", 0.0) or 0.0),
        }
        missing_delta = abs(float(comparison.get("delta_metadata_missing_rate", 0.0) or 0.0))
        anonymous_delta = abs(float(comparison.get("delta_anonymous_rate", 0.0) or 0.0))
        rows = [
            ("nationality shift", representation_rows["nationality shift"], "dirty vs clean nationality mix"),
            ("background shift", representation_rows["background shift"], "dirty vs clean role mix"),
            ("age shift", representation_rows["age shift"], "dirty vs clean age mix"),
            ("metadata loss delta", missing_delta, "extra missing protected metadata"),
            ("anonymous delta", anonymous_delta, "extra anonymous records"),
        ]
        driver_name, driver_value, _driver_help = max(rows, key=lambda item: item[1])
        score = max(driver_value, missing_delta, anonymous_delta)

    if score >= 0.30:
        verdict = "BIAS SIGNAL"
        verdict_color = "#b91c1c"
        verdict_fill = "#fee2e2"
    elif score >= 0.15:
        verdict = "WATCH"
        verdict_color = "#b45309"
        verdict_fill = "#ffedd5"
    else:
        verdict = "OK"
        verdict_color = "#15803d"
        verdict_fill = "#dcfce7"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_fill": verdict_fill,
        "score": score,
        "driver_name": driver_name,
        "driver_value": driver_value,
        "rows": sorted(rows, key=lambda item: item[1], reverse=True),
        "threshold_note": threshold_note,
    }


def _render_scatterplot_document(
    title: str,
    subtitle: str,
    clustered_df: pd.DataFrame,
    cluster_summary: dict[str, Any],
    bias_snapshot: dict[str, Any],
    comparison: dict[str, Any] | None = None,
) -> str:
    width = 1380
    height = 940
    left = 122
    right = 430
    top = 104
    bottom = 206
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min = -1.12
    x_max = 1.12
    y_min = -1.12
    y_max = 1.12

    def scale_x(value: Any) -> float:
        numeric = float(value)
        return left + ((numeric - x_min) / (x_max - x_min)) * plot_width

    def scale_y(value: Any) -> float:
        numeric = float(value)
        return top + ((y_max - numeric) / (y_max - y_min)) * plot_height

    grid_lines: list[str] = []
    for tick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        x = scale_x(tick)
        y = scale_y(tick)
        grid_lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="grid" />'
        )
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid" />'
        )
        grid_lines.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 28}" class="tick" text-anchor="middle">{tick:g}</text>'
        )
        grid_lines.append(
            f'<text x="{left - 18}" y="{y + 4:.1f}" class="tick" text-anchor="end">{tick:g}</text>'
        )

    points: list[str] = []
    for _index, row in clustered_df.iterrows():
        radius = 7.0 + (float(row.get("bias_severity", 0.0)) * 10.0)
        dataset_type = _safe_string(row.get("dataset_type")) or "dataset"
        stroke = "#0f172a" if dataset_type == "clean" else "#f97316"
        stroke_width = 1.8 if dataset_type == "clean" else 2.8
        title_parts = [
            f"record={_safe_string(row.get('record_id'))}",
            f"dataset={dataset_type}",
            f"cluster={_safe_string(row.get('cluster_id'))}",
            f"age_bucket={_safe_string(row.get('age_bucket')) or 'unknown'}",
            f"nationality={_safe_string(row.get('metadata_nationality')) or 'unknown'}",
            f"missing={_safe_string(row.get('metadata_missing_count'))}/{len(METADATA_FIELDS)}",
        ]
        points.append(
            (
                f'<circle cx="{scale_x(row["scatter_x"]):.1f}" cy="{scale_y(row["scatter_y"]):.1f}" '
                f'r="{radius:.1f}" fill="{_cluster_color(row.get("cluster_id"))}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}" opacity="0.82">'
                f"<title>{_svg_text(' | '.join(title_parts))}</title></circle>"
            )
        )

    diagnosis = _bias_diagnosis(bias_snapshot=bias_snapshot, comparison=comparison)
    cluster_counts = clustered_df["cluster_id"].value_counts().sort_index()
    legend_rows: list[str] = []
    for offset, (cluster_id, count) in enumerate(cluster_counts.items()):
        x = left + 96 + ((offset % 5) * 142)
        y = top + plot_height + 110 + ((offset // 5) * 22)
        legend_rows.append(
            f'<circle cx="{x}" cy="{y}" r="7" fill="{_cluster_color(cluster_id)}" />'
            f'<text x="{x + 14}" y="{y + 5}" class="legend">C{_svg_text(cluster_id)} ({count})</text>'
        )

    side_x = left + plot_width + 30
    facts = [
        ("records", _safe_string(bias_snapshot.get("records"))),
        ("clusters", _safe_string(len(cluster_counts))),
        ("anonymous", _format_float(bias_snapshot.get("anonymous_rate"))),
        ("missing metadata", _format_float(bias_snapshot.get("metadata_missing_rate"))),
    ]
    if comparison is not None:
        shifts = comparison.get("representation_shift", {})
        max_shift = max(
            [
                float(shifts.get("nationality_tvd", 0.0) or 0.0),
                float(shifts.get("background_tvd", 0.0) or 0.0),
                float(shifts.get("age_bucket_tvd", 0.0) or 0.0),
            ]
        )
        facts.extend(
            [
                ("max representation shift", _format_float(max_shift)),
                ("metadata loss delta", _format_float(abs(float(comparison.get("delta_metadata_missing_rate", 0.0) or 0.0)))),
            ]
        )
    fact_rows = []
    for offset, (name, value) in enumerate(facts[:6]):
        y = 742 + (offset * 18)
        fact_rows.append(
            f'<text x="{side_x}" y="{y}" class="metric-name">{_svg_text(name)}</text>'
            f'<text x="{width - 42}" y="{y}" class="metric-value" text-anchor="end">{_svg_text(value)}</text>'
        )

    diagnosis_rows = []
    for offset, (name, value, help_text) in enumerate(diagnosis["rows"][:5]):
        y = 236 + (offset * 42)
        bar_width = min(170.0, max(3.0, float(value) * 170.0))
        bar_color = "#b91c1c" if value >= 0.30 else "#b45309" if value >= 0.15 else "#15803d"
        diagnosis_rows.append(
            f'<text x="{side_x}" y="{y}" class="diagnosis-name">{_svg_text(name)}</text>'
            f'<text x="{width - 42}" y="{y}" class="diagnosis-value" text-anchor="end">{_format_float(value)}</text>'
            f'<rect x="{side_x}" y="{y + 9}" width="170" height="8" rx="4" fill="#e2e8f0" />'
            f'<rect x="{side_x}" y="{y + 9}" width="{bar_width:.1f}" height="8" rx="4" fill="{bar_color}" />'
            f'<text x="{side_x}" y="{y + 32}" class="diagnosis-help">{_svg_text(help_text)}</text>'
        )

    top_terms = []
    for cluster in cluster_summary.get("clusters", [])[:4]:
        terms = ", ".join(cluster.get("top_terms", [])[:4]) or "n/a"
        top_terms.append(f"C{cluster.get('cluster_id')}: {terms}")
    terms_text = "Hover a point to see record, cluster, nationality, age bucket and missing metadata."

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_svg_text(title)}">
  <style>
    .title {{ font: 700 30px Arial, sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 400 15px Arial, sans-serif; fill: #475569; }}
    .axis {{ stroke: #334155; stroke-width: 1.4; }}
    .grid {{ stroke: #dbe3ee; stroke-width: 1; }}
    .tick {{ font: 12px Arial, sans-serif; fill: #64748b; }}
    .axis-label {{ font: 700 13px Arial, sans-serif; fill: #334155; letter-spacing: 0; }}
    .legend-title {{ font: 700 15px Arial, sans-serif; fill: #0f172a; }}
    .legend {{ font: 13px Arial, sans-serif; fill: #334155; }}
    .badge {{ font: 700 14px Arial, sans-serif; }}
    .diagnosis-title {{ font: 700 17px Arial, sans-serif; fill: #0f172a; }}
    .diagnosis-name {{ font: 700 12px Arial, sans-serif; fill: #334155; }}
    .diagnosis-value {{ font: 700 12px Arial, sans-serif; fill: #0f172a; }}
    .diagnosis-help {{ font: 11px Arial, sans-serif; fill: #64748b; }}
    .metric-title {{ font: 700 15px Arial, sans-serif; fill: #0f172a; }}
    .metric-name {{ font: 12px Arial, sans-serif; fill: #64748b; }}
    .metric-value {{ font: 700 12px Arial, sans-serif; fill: #0f172a; }}
    .legend-label {{ font: 700 12px Arial, sans-serif; fill: #334155; }}
    .note {{ font: 12px Arial, sans-serif; fill: #475569; }}
  </style>
  <rect width="100%" height="100%" fill="#f8fafc" />
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="6" fill="#ffffff" stroke="#cbd5e1" />
  <text x="{left}" y="42" class="title">{_svg_text(title)}</text>
  <text x="{left}" y="68" class="subtitle">{_svg_text(subtitle)}</text>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis" />
  <text x="{left + (plot_width / 2)}" y="{top + plot_height + 54}" class="axis-label" text-anchor="middle">X: semantic text position (-1 to +1)</text>
  <text x="42" y="{top + (plot_height / 2)}" class="axis-label" text-anchor="middle" transform="rotate(-90 42 {top + (plot_height / 2)})">Y: relative metadata / outlier contrast</text>
  {''.join(points)}
  <rect x="{side_x - 14}" y="92" width="{right - 44}" height="374" rx="8" fill="#ffffff" stroke="#cbd5e1" />
  <text x="{side_x}" y="122" class="diagnosis-title">Bias diagnosis</text>
  <rect x="{side_x}" y="138" width="128" height="30" rx="15" fill="{diagnosis["verdict_fill"]}" stroke="{diagnosis["verdict_color"]}" />
  <text x="{side_x + 64}" y="158" class="badge" fill="{diagnosis["verdict_color"]}" text-anchor="middle">{_svg_text(diagnosis["verdict"])}</text>
  <text x="{side_x}" y="192" class="metric-name">Main driver</text>
  <text x="{width - 42}" y="192" class="metric-value" text-anchor="end">{_svg_text(diagnosis["driver_name"])} {_format_float(diagnosis["driver_value"])}</text>
  <text x="{side_x}" y="214" class="diagnosis-help">{_svg_text(diagnosis["threshold_note"])}</text>
  {''.join(diagnosis_rows)}
  <rect x="{side_x - 14}" y="486" width="{right - 44}" height="196" rx="8" fill="#ffffff" stroke="#cbd5e1" />
  <text x="{side_x}" y="516" class="legend-title">Legend</text>
  <text x="{side_x}" y="542" class="legend-label">X value</text>
  <text x="{side_x + 96}" y="542" class="note">text similarity, normalized -1 to +1</text>
  <text x="{side_x}" y="566" class="legend-label">Y value</text>
  <text x="{side_x + 96}" y="566" class="note">relative metadata/semantic outlier signal</text>
  <text x="{side_x}" y="590" class="legend-label">Color</text>
  <text x="{side_x + 96}" y="590" class="note">cluster id from the text</text>
  <text x="{side_x}" y="614" class="legend-label">Size</text>
  <text x="{side_x + 96}" y="614" class="note">larger means more missing metadata</text>
  <text x="{side_x}" y="638" class="legend-label">Outline</text>
  <text x="{side_x + 96}" y="638" class="note">dark clean, orange dirty/evaluation</text>
  <text x="{side_x}" y="662" class="diagnosis-help">Relative positions only. Verdict uses the bars above.</text>
  <rect x="{side_x - 14}" y="698" width="{right - 44}" height="142" rx="8" fill="#ffffff" stroke="#cbd5e1" />
  <text x="{side_x}" y="724" class="metric-title">Dataset facts</text>
  {''.join(fact_rows)}
  <text x="{left}" y="{top + plot_height + 92}" class="legend-title">Clusters</text>
  {''.join(legend_rows)}
  <text x="{left}" y="{height - 12}" class="note">{_svg_text(terms_text)}</text>
</svg>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_svg_text(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #e2e8f0;
      color: #0f172a;
      font-family: Arial, sans-serif;
    }}
    main {{
      padding: 32px;
    }}
    .plot-shell {{
      max-width: {width}px;
      margin: 0 auto;
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      box-shadow: 0 18px 36px rgba(15, 23, 42, 0.16);
      overflow: auto;
    }}
    svg {{
      display: block;
      width: 100%;
      min-width: 1080px;
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
    <div class="plot-shell">
      {svg}
    </div>
  </main>
</body>
</html>
"""


def _write_scatterplot_artifacts(
    dataset_type: str,
    batch_id: int,
    clustered_df: pd.DataFrame,
    cluster_summary: dict[str, Any],
    bias_snapshot: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    base_name = f"{dataset_type}_batch_{batch_id}_scatterplot"
    title = f"Batch {batch_id} {dataset_type} cluster bias scatterplot"
    subtitle = "Color shows cluster, point size shows metadata missingness, hover shows record metadata."
    document = _render_scatterplot_document(
        title=title,
        subtitle=subtitle,
        clustered_df=clustered_df,
        cluster_summary=cluster_summary,
        bias_snapshot=bias_snapshot,
    )
    html_path = output_dir / f"{base_name}.html"
    svg_path = output_dir / f"{base_name}.svg"
    html_path.write_text(document, encoding="utf-8")
    svg_start = document.index("<svg")
    svg_end = document.index("</svg>") + len("</svg>")
    svg_path.write_text(document[svg_start:svg_end], encoding="utf-8")
    return {
        "scatterplot_html": html_path,
        "scatterplot_svg": svg_path,
    }


def _build_combined_bias_snapshot(clean_bias: dict[str, Any], dirty_bias: dict[str, Any]) -> dict[str, Any]:
    return {
        "records": clean_bias["records"] + dirty_bias["records"],
        "anonymous_rate": float((clean_bias["anonymous_rate"] + dirty_bias["anonymous_rate"]) / 2),
        "metadata_missing_rate": float(
            (clean_bias["metadata_missing_rate"] + dirty_bias["metadata_missing_rate"]) / 2
        ),
        "cluster_bias_signals": {
            "weighted_nationality_tvd": max(
                clean_bias["cluster_bias_signals"]["weighted_nationality_tvd"],
                dirty_bias["cluster_bias_signals"]["weighted_nationality_tvd"],
            ),
            "weighted_background_tvd": max(
                clean_bias["cluster_bias_signals"]["weighted_background_tvd"],
                dirty_bias["cluster_bias_signals"]["weighted_background_tvd"],
            ),
            "weighted_age_bucket_tvd": max(
                clean_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"],
                dirty_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"],
            ),
            "largest_cluster_share": max(
                clean_bias["cluster_bias_signals"]["largest_cluster_share"],
                dirty_bias["cluster_bias_signals"]["largest_cluster_share"],
            ),
        },
    }


def _write_combined_scatterplot_artifacts(
    batch_id: int,
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    clean_cluster_summary: dict[str, Any],
    dirty_cluster_summary: dict[str, Any],
    clean_bias: dict[str, Any],
    dirty_bias: dict[str, Any],
    comparison: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    combined_df = pd.concat([clean_df, dirty_df], ignore_index=True)
    combined_summary = {
        "clusters": clean_cluster_summary.get("clusters", []) + dirty_cluster_summary.get("clusters", [])
    }
    combined_bias = _build_combined_bias_snapshot(clean_bias, dirty_bias)
    title = f"Batch {batch_id} clean vs dirty cluster bias scatterplot"
    subtitle = "Clean points use a dark outline; dirty points use an orange outline."
    document = _render_scatterplot_document(
        title=title,
        subtitle=subtitle,
        clustered_df=combined_df,
        cluster_summary=combined_summary,
        bias_snapshot=combined_bias,
        comparison=comparison,
    )
    html_path = output_dir / f"batch_{batch_id}_clean_dirty_scatterplot.html"
    svg_path = output_dir / f"batch_{batch_id}_clean_dirty_scatterplot.svg"
    html_path.write_text(document, encoding="utf-8")
    svg_start = document.index("<svg")
    svg_end = document.index("</svg>") + len("</svg>")
    svg_path.write_text(document[svg_start:svg_end], encoding="utf-8")
    return {
        "combined_scatterplot_html": html_path,
        "combined_scatterplot_svg": svg_path,
    }


def _write_artifacts(
    dataset_type: str,
    batch_id: int,
    clustered_df: pd.DataFrame,
    cluster_summary: dict[str, Any],
    bias_snapshot: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{dataset_type}_batch_{batch_id}"
    csv_path = output_dir / f"{base_name}_cluster_assignments.csv"
    cluster_json_path = output_dir / f"{base_name}_cluster_summary.json"
    bias_json_path = output_dir / f"{base_name}_bias_snapshot.json"

    clustered_df.to_csv(csv_path, index=False)
    _write_json(cluster_json_path, cluster_summary)
    _write_json(bias_json_path, bias_snapshot)
    scatterplot_paths = _write_scatterplot_artifacts(
        dataset_type=dataset_type,
        batch_id=batch_id,
        clustered_df=clustered_df,
        cluster_summary=cluster_summary,
        bias_snapshot=bias_snapshot,
        output_dir=output_dir,
    )

    return {
        "cluster_assignments_csv": csv_path,
        "cluster_summary_json": cluster_json_path,
        "bias_snapshot_json": bias_json_path,
        **scatterplot_paths,
    }


def _load_mlops_helpers() -> dict[str, Any]:
    current_file = Path(__file__).resolve()
    workspace_root = current_file.parents[3]
    mlops_root = workspace_root / "certain-MLOps-Engine" / "Semantic_MLOps_engine"
    if str(mlops_root) not in sys.path:
        sys.path.insert(0, str(mlops_root))

    import mlflow  # type: ignore
    from certain_library.data_analysis.log_data_techniques import log_data_techniques  # type: ignore
    from certain_library.data_analysis.log_dataset import log_dataset  # type: ignore
    from certain_library.data_analysis.log_whylogs import log_whylogs_profile  # type: ignore

    return {
        "mlflow": mlflow,
        "log_dataset": log_dataset,
        "log_data_techniques": log_data_techniques,
        "log_whylogs_profile": log_whylogs_profile,
    }


def _log_to_mlflow(
    dataset_type: str,
    batch_id: int,
    clustered_df: pd.DataFrame,
    cluster_summary: dict[str, Any],
    bias_snapshot: dict[str, Any],
    artifacts: dict[str, Path],
    experiment_name: str,
) -> None:
    helpers = _load_mlops_helpers()
    mlflow = helpers["mlflow"]
    log_dataset = helpers["log_dataset"]
    log_data_techniques = helpers["log_data_techniques"]
    log_whylogs_profile = helpers["log_whylogs_profile"]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"synthetic_{dataset_type}_batch_{batch_id}"):
        mlflow.log_params(
            {
                "dataset_type": dataset_type,
                "batch_id": batch_id,
                "topic_title": clustered_df["topic_title"].iloc[0],
                "topic_summary": clustered_df["topic_summary"].iloc[0],
                "record_count": len(clustered_df),
                "cluster_count": cluster_summary["cluster_count"],
            }
        )

        metrics = {
            "anonymous_rate": bias_snapshot["anonymous_rate"],
            "metadata_missing_rate": bias_snapshot["metadata_missing_rate"],
            "weighted_nationality_tvd": bias_snapshot["cluster_bias_signals"]["weighted_nationality_tvd"],
            "weighted_background_tvd": bias_snapshot["cluster_bias_signals"]["weighted_background_tvd"],
            "weighted_age_bucket_tvd": bias_snapshot["cluster_bias_signals"]["weighted_age_bucket_tvd"],
            "largest_cluster_share": bias_snapshot["cluster_bias_signals"]["largest_cluster_share"],
        }
        if cluster_summary["silhouette_score"] is not None:
            metrics["silhouette_score"] = cluster_summary["silhouette_score"]
        mlflow.log_metrics(metrics)

        log_dataset(
            clustered_df,
            name=f"{dataset_type}_batch_{batch_id}_clustered",
            output_dir="dataset",
            save_full_dataset=True,
        )
        log_whylogs_profile(clustered_df, name=f"{dataset_type}_batch_{batch_id}")
        log_data_techniques(
            {
                "session_bias_clustering": {
                    "method": "tfidf_kmeans_auto_k",
                    "text_field": "text",
                    "metadata_fields": ",".join(METADATA_FIELDS),
                    "notes": (
                        "Clusters generated from text only; metadata retained "
                        "for downstream bias and representation analysis."
                    ),
                }
            }
        )

        for path in artifacts.values():
            mlflow.log_artifact(str(path), artifact_path="cluster_bias_analysis")

        mlflow.log_dict(cluster_summary, f"cluster_bias_analysis/{dataset_type}_batch_{batch_id}_cluster_summary.json")
        mlflow.log_dict(bias_snapshot, f"cluster_bias_analysis/{dataset_type}_batch_{batch_id}_bias_snapshot.json")


def _process_single_dataset(
    payload: dict[str, Any],
    dataset_type: str,
    batch_id: int,
    output_dir: Path,
    max_clusters: int,
    log_to_mlflow: bool,
    experiment_name: str,
    internal_annotations: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    flattened_df = _flatten_dataset(
        payload,
        dataset_type=dataset_type,
        batch_id=batch_id,
        internal_annotations=internal_annotations,
    )
    clustered_df, clustering_meta = _cluster_dataset(flattened_df, max_clusters=max_clusters)
    cluster_summary = _build_cluster_summary(clustered_df, clustering_meta)
    bias_snapshot = _build_bias_snapshot(clustered_df, cluster_summary)
    artifacts = _write_artifacts(
        dataset_type=dataset_type,
        batch_id=batch_id,
        clustered_df=clustered_df,
        cluster_summary=cluster_summary,
        bias_snapshot=bias_snapshot,
        output_dir=output_dir,
    )

    if log_to_mlflow:
        _log_to_mlflow(
            dataset_type=dataset_type,
            batch_id=batch_id,
            clustered_df=clustered_df,
            cluster_summary=cluster_summary,
            bias_snapshot=bias_snapshot,
            artifacts=artifacts,
            experiment_name=experiment_name,
        )

    return clustered_df, cluster_summary, bias_snapshot


def build_batch_bias_artifacts(
    batch_id: int,
    session_dir: Path | None = None,
    output_dir: Path | None = None,
    max_clusters: int = 6,
) -> dict[str, Any]:
    """Build clustering and bias artifacts for one clean/dirty batch.

    Parameters
    ----------
    batch_id : int
        Batch number to load from ``simulation_output_clean_{batch}.json`` and
        ``simulation_output_dirty_{batch}.json``.
    session_dir : pathlib.Path, optional
        Directory containing the session JSON files. Defaults to ``session/``.
    output_dir : pathlib.Path, optional
        Directory where artifacts will be written. Defaults to
        ``session/cluster_bias_outputs/batch_{batch}``.
    max_clusters : int, optional
        Upper bound for automatic cluster search when scikit-learn is present.

    Returns
    -------
    dict
        Contains the clustered DataFrames, summaries, bias snapshots, and the
        clean-vs-dirty comparison payload for the selected batch.
    """
    resolved_session_dir = (session_dir or Path("session")).resolve()
    discovered_batches = _discover_batches(resolved_session_dir)
    if batch_id not in discovered_batches:
        raise FileNotFoundError(
            f"Batch {batch_id} not found in {resolved_session_dir}. "
            "Expected paired clean/dirty JSON files."
        )

    batch_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else resolved_session_dir / "cluster_bias_outputs" / f"batch_{batch_id}"
    )
    files = discovered_batches[batch_id]
    clean_payload = _load_json(files["clean"])
    dirty_payload = _load_json(files["dirty"])
    internal_annotations = _load_internal_annotations(resolved_session_dir, batch_id)

    clean_df, clean_cluster_summary, clean_bias = _process_single_dataset(
        payload=clean_payload,
        dataset_type="clean",
        batch_id=batch_id,
        output_dir=batch_output_dir,
        max_clusters=max_clusters,
        log_to_mlflow=False,
        experiment_name="",
        internal_annotations=internal_annotations["clean"],
    )
    dirty_df, dirty_cluster_summary, dirty_bias = _process_single_dataset(
        payload=dirty_payload,
        dataset_type="dirty",
        batch_id=batch_id,
        output_dir=batch_output_dir,
        max_clusters=max_clusters,
        log_to_mlflow=False,
        experiment_name="",
        internal_annotations=internal_annotations["dirty"],
    )

    comparison = {
        "batch_id": batch_id,
        "topic_title": clean_payload.get("TopicTitle") or dirty_payload.get("TopicTitle"),
        "clean_cluster_count": clean_cluster_summary["cluster_count"],
        "dirty_cluster_count": dirty_cluster_summary["cluster_count"],
        "comparison": _compare_clean_dirty(clean_df, dirty_df, clean_bias, dirty_bias),
    }
    _write_json(batch_output_dir / f"batch_{batch_id}_clean_dirty_comparison.json", comparison)
    combined_scatterplots = _write_combined_scatterplot_artifacts(
        batch_id=batch_id,
        clean_df=clean_df,
        dirty_df=dirty_df,
        clean_cluster_summary=clean_cluster_summary,
        dirty_cluster_summary=dirty_cluster_summary,
        clean_bias=clean_bias,
        dirty_bias=dirty_bias,
        comparison=comparison["comparison"],
        output_dir=batch_output_dir,
    )

    return {
        "batch_id": batch_id,
        "topic_title": comparison["topic_title"],
        "output_dir": batch_output_dir,
        "combined_scatterplots": combined_scatterplots,
        "clean": {
            "dataframe": clean_df,
            "cluster_summary": clean_cluster_summary,
            "bias_snapshot": clean_bias,
        },
        "dirty": {
            "dataframe": dirty_df,
            "cluster_summary": dirty_cluster_summary,
            "bias_snapshot": dirty_bias,
        },
        "comparison": comparison,
    }


def main() -> None:
    args = _parse_args()
    session_dir = args.session_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else session_dir / "cluster_bias_outputs"
    )

    batches = _discover_batches(session_dir)
    if args.batch is not None:
        batches = {args.batch: batches[args.batch]} if args.batch in batches else {}

    if not batches:
        raise SystemExit(
            f"No paired clean/dirty batches found in {session_dir}. "
            "Expected files like simulation_output_clean_1.json and simulation_output_dirty_1.json."
        )

    comparative_reports: list[dict[str, Any]] = []

    for batch_id, files in sorted(batches.items()):
        batch_output_dir = output_dir / f"batch_{batch_id}"
        batch_result = build_batch_bias_artifacts(
            batch_id=batch_id,
            session_dir=session_dir,
            output_dir=batch_output_dir,
            max_clusters=args.max_clusters,
        )

        if args.log_to_mlflow:
            _log_to_mlflow(
                dataset_type="clean",
                batch_id=batch_id,
                clustered_df=batch_result["clean"]["dataframe"],
                cluster_summary=batch_result["clean"]["cluster_summary"],
                bias_snapshot=batch_result["clean"]["bias_snapshot"],
                artifacts=_write_artifacts(
                    dataset_type="clean",
                    batch_id=batch_id,
                    clustered_df=batch_result["clean"]["dataframe"],
                    cluster_summary=batch_result["clean"]["cluster_summary"],
                    bias_snapshot=batch_result["clean"]["bias_snapshot"],
                    output_dir=batch_output_dir,
                ),
                experiment_name=args.experiment_name,
            )
            _log_to_mlflow(
                dataset_type="dirty",
                batch_id=batch_id,
                clustered_df=batch_result["dirty"]["dataframe"],
                cluster_summary=batch_result["dirty"]["cluster_summary"],
                bias_snapshot=batch_result["dirty"]["bias_snapshot"],
                artifacts=_write_artifacts(
                    dataset_type="dirty",
                    batch_id=batch_id,
                    clustered_df=batch_result["dirty"]["dataframe"],
                    cluster_summary=batch_result["dirty"]["cluster_summary"],
                    bias_snapshot=batch_result["dirty"]["bias_snapshot"],
                    output_dir=batch_output_dir,
                ),
                experiment_name=args.experiment_name,
            )

        comparative_reports.append(batch_result["comparison"])

    final_report = {
        "processed_batches": [report["batch_id"] for report in comparative_reports],
        "reports": comparative_reports,
    }
    _write_json(output_dir / "summary_report.json", final_report)

    print(
        json.dumps(
            {
                "status": "ok",
                "processed_batches": final_report["processed_batches"],
                "output_dir": str(output_dir),
                "mlflow_logged": args.log_to_mlflow,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
