"""Local UI for authoring synthetic clean/dirty bias test scenarios."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from src.config import MAX_TOKENS, MODEL_NAME
from src.create_demo_bias_datasets import (
    AGE_BUCKETS,
    ANSWER_QUALITY_TEXT,
    COUNTRIES,
    OPINION_TEXT,
    TOPIC_SUMMARY,
    TOPIC_TITLE,
    _record,
)
from src.formatting import clean_agent_output
from src.llm_client import OpenAIClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_DIR = PROJECT_ROOT / "session"
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
OPINIONS = list(OPINION_TEXT)
ANSWER_QUALITIES = list(ANSWER_QUALITY_TEXT)
BIAS_GROUPS = [
    "none",
    "general_overrepresented",
    "general_underrepresented",
    "young_support_overrepresented",
    "midcareer_conditional_present",
    "senior_concern_underrepresented",
    "latecareer_concern_underrepresented",
    "senior_oppose_overrepresented",
    "latecareer_oppose_overrepresented",
    "country_age_targeted_bias",
]
ALLOCATION_STRATEGIES = [
    "equal_distribution",
    "population_like_distribution",
    "skew_toward_selected_group",
    "custom_weights",
]
DEVIATION_TARGETS = [
    "all",
    "demographic",
    "opinion",
    "answer_quality",
    "country_age_interaction",
    "hidden_quality_bias",
]
WRONG_ANSWER_PATTERNS = [
    "random",
    "off_topic",
    "overgeneralized",
    "stereotyped",
    "missing_safeguards",
    "contradicts_assigned_opinion",
    "low_detail",
]
WRONG_ANSWER_PATTERN_TEXT = {
    "random": "Use any plausible weakness pattern.",
    "off_topic": "Drift away from the actual policy question.",
    "overgeneralized": "Make an unsupported broad claim.",
    "stereotyped": "Use a demographic stereotype that should be detected as problematic.",
    "missing_safeguards": "Answer the question but omit important safeguards.",
    "contradicts_assigned_opinion": "Contradict the assigned opinion stance.",
    "low_detail": "Give a shallow answer with little policy reasoning.",
}
QUALITY_WEIGHTS = {
    "correct": 0.78,
    "partial": 0.17,
    "incorrect": 0.05,
}
MIN_DEVIATION_GRADE = 1
MAX_DEVIATION_GRADE = 5
LLM_SYSTEM_PROMPT = (
    "You are generating realistic synthetic social-media comments for a civic deliberation dataset. "
    "Write as a single human participant, not as an assistant. Keep the text natural, coherent, and specific. "
    "Do not use markdown, bullets, labels, JSON, headers, or analysis. Return only the comment text."
)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, status: int, text: str, content_type: str) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _payload(topic_title: str, topic_summary: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "TopicTitle": topic_title,
        "TopicSummary": topic_summary,
        "Dataset": records,
    }


def _export_text(text: str) -> str:
    internal_markers = [
        "Background ",
        "Bias marker ",
        "Final answer quality:",
        "Opinion label:",
        "Wrong-answer pattern:",
        "The answer ",
    ]
    chunks = [chunk.strip() for chunk in text.split(".")]
    public_chunks = [
        chunk
        for chunk in chunks
        if chunk and not any(chunk.startswith(marker) for marker in internal_markers)
    ]
    return ". ".join(public_chunks) + ("." if public_chunks else "")


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("Metadata") or {}
    return {
        "Id": record.get("Id"),
        "Text": _export_text(str(record.get("Text") or "")),
        "Metadata": {
            field: metadata.get(field)
            for field in PUBLIC_METADATA_FIELDS
            if field in metadata
        },
    }


def _internal_annotation(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("Metadata") or {}
    return {
        "Id": record.get("Id"),
        "InternalMetadata": {
            field: metadata.get(field)
            for field in INTERNAL_METADATA_FIELDS
            if field in metadata
        },
    }


def _export_payload(topic_title: str, topic_summary: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return _payload(topic_title, topic_summary, [_public_record(record) for record in records])


def _annotations_payload(
    *,
    topic_title: str,
    topic_summary: str,
    clean_records: list[dict[str, Any]],
    dirty_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "TopicTitle": topic_title,
        "TopicSummary": topic_summary,
        "Clean": [_internal_annotation(record) for record in clean_records],
        "Dirty": [_internal_annotation(record) for record in dirty_records],
    }


def _coerce_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _stable_random(*parts: Any) -> random.Random:
    seed_text = "|".join(str(part) for part in parts)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _weighted_choice(rng: random.Random, weighted_values: dict[str, float]) -> str:
    total = sum(weighted_values.values())
    marker = rng.random() * total
    current = 0.0
    for value, weight in weighted_values.items():
        current += weight
        if marker <= current:
            return value
    return next(reversed(weighted_values))


def _deviation_grade(value: Any) -> int:
    try:
        grade = int(value)
    except (TypeError, ValueError):
        grade = 3
    return min(MAX_DEVIATION_GRADE, max(MIN_DEVIATION_GRADE, grade))


def _skewed_opinion_weights(target: str, grade: int) -> dict[str, float]:
    dominant = min(0.92, 0.44 + (grade * 0.095))
    remaining = max(0.0, 1.0 - dominant)
    secondary = remaining * 0.56
    tertiary = remaining * 0.29
    last = remaining - secondary - tertiary
    if target == "support":
        return {"support": dominant, "conditional": secondary, "concern": tertiary, "oppose": last}
    if target == "oppose":
        return {"support": last, "conditional": tertiary, "concern": secondary, "oppose": dominant}
    if target == "concern":
        return {"support": last, "conditional": tertiary, "concern": dominant, "oppose": secondary}
    return {"support": last, "conditional": dominant, "concern": secondary, "oppose": tertiary}


def _target_allows_opinion_skew(deviation_target: str) -> bool:
    return deviation_target in {"all", "opinion", "country_age_interaction"}


def _target_allows_quality_skew(deviation_target: str) -> bool:
    return deviation_target in {"all", "answer_quality", "hidden_quality_bias"}


def _opinion_weights(
    row: dict[str, Any],
    dataset_type: str,
    deviation_grade: int,
    deviation_target: str,
) -> dict[str, float]:
    bias_group = str(row.get("bias_group", "none"))
    if dataset_type == "clean" or bias_group == "none" or not _target_allows_opinion_skew(deviation_target):
        return {"support": 1.0, "conditional": 1.0, "concern": 1.0, "oppose": 1.0}
    if "support" in bias_group:
        return _skewed_opinion_weights("support", deviation_grade)
    if "oppose" in bias_group:
        return _skewed_opinion_weights("oppose", deviation_grade)
    if "concern" in bias_group:
        return _skewed_opinion_weights("concern", deviation_grade)
    if "conditional" in bias_group:
        return _skewed_opinion_weights("conditional", deviation_grade)
    return {"support": 1.0, "conditional": 1.0, "concern": 1.0, "oppose": 1.0}


def _quality_weights(
    row: dict[str, Any],
    dataset_type: str,
    deviation_grade: int,
    deviation_target: str,
) -> dict[str, float]:
    bias_group = str(row.get("bias_group", "none"))
    if dataset_type == "clean" or bias_group == "none" or not _target_allows_quality_skew(deviation_target):
        return QUALITY_WEIGHTS
    correct = max(0.35, 0.82 - (deviation_grade * 0.09))
    partial = min(0.30, 0.14 + (deviation_grade * 0.03))
    incorrect = max(0.0, 1.0 - correct - partial)
    return {"correct": correct, "partial": partial, "incorrect": incorrect}


def _answer_quality(
    row: dict[str, Any],
    dataset_type: str,
    deviation_grade: int,
    deviation_target: str,
    rng: random.Random,
) -> str:
    if row.get("force_correct"):
        return "correct"
    return _weighted_choice(rng, _quality_weights(row, dataset_type, deviation_grade, deviation_target))


def _build_records(
    *,
    batch_id: int,
    dataset_type: str,
    rows: list[dict[str, Any]],
    deviation_grade: int,
    deviation_target: str,
    wrong_answer_pattern: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    index = 1
    count_key = f"{dataset_type}_count"

    for row in rows:
        country = str(row.get("country", "")).strip()
        age_bucket = str(row.get("age_bucket", "")).strip()
        bias_group = str(row.get("bias_group", "none")).strip() or "none"
        count = _coerce_count(row.get(count_key))

        if count == 0:
            continue
        if country not in COUNTRIES:
            raise ValueError(f"Unknown country: {country}")
        if age_bucket not in AGE_BUCKETS:
            raise ValueError(f"Unknown age bucket: {age_bucket}")
        if bias_group not in BIAS_GROUPS:
            raise ValueError(f"Unknown bias group: {bias_group}")

        for local_index in range(count):
            rng = _stable_random(batch_id, dataset_type, index, local_index, country, age_bucket, bias_group)
            opinion = _weighted_choice(rng, _opinion_weights(row, dataset_type, deviation_grade, deviation_target))
            answer_quality = _answer_quality(row, dataset_type, deviation_grade, deviation_target, rng)
            record = _record(
                batch=batch_id,
                dataset_type=dataset_type,
                index=index,
                country=country,
                age_bucket=age_bucket,
                opinion=opinion,
                bias_group=bias_group,
                answer_quality=answer_quality,
            )
            if answer_quality != "correct":
                record["Metadata"]["WrongAnswerPattern"] = wrong_answer_pattern
                record["Text"] = (
                    f"{record['Text']} Wrong-answer pattern: "
                    f"{WRONG_ANSWER_PATTERN_TEXT.get(wrong_answer_pattern, WRONG_ANSWER_PATTERN_TEXT['random'])}"
                )
            records.append(record)
            index += 1

    return records


def _llm_prompt_for_record(
    *,
    topic_title: str,
    topic_summary: str,
    record: dict[str, Any],
) -> str:
    metadata = record["Metadata"]
    return f"""
<discussion>
Title: {topic_title}
Summary: {topic_summary}
</discussion>

<actor_profile>
Name: {metadata["FirstName"]} {metadata["LastName"]}
Country/Nationality: {metadata["Nationality"]}
Age: {metadata["Age"]}
Age bucket: {metadata.get("AgeBucket") or "derived from age"}
Professional background: {metadata["Background"]}
Intended opinion stance: {metadata["Opinion"]}
Answer quality: {metadata["AnswerQuality"]}
Wrong answer pattern: {metadata.get("WrongAnswerPattern", "none")}
Bias group marker for dataset analysis: {metadata["BiasGroup"]}
</actor_profile>

<writing_task>
Write one natural social-media comment from this actor about the discussion question.
The comment must match the actor profile and intended opinion stance.
If AnswerQuality is "correct", answer the question directly and include at least one concrete safeguard.
If AnswerQuality is "partial", answer the question but leave one important safeguard underdeveloped.
If AnswerQuality is "incorrect", make it weaker or less directly connected while still plausible as a participant comment.
If a wrong answer pattern is provided, follow that weakness pattern.
Do not mention the words "bias group", "metadata", "synthetic", "answer quality", or any internal label.
Use the same language as the discussion title unless the title is mixed-language; then use clear English.
Keep it between 45 and 90 words.
</writing_task>
""".strip()


async def _generate_llm_texts(
    *,
    topic_title: str,
    topic_summary: str,
    records: list[dict[str, Any]],
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Disable LLM generation or configure .env.")

    client = OpenAIClient(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    semaphore = asyncio.Semaphore(4)

    async def generate_one(record: dict[str, Any]) -> None:
        async with semaphore:
            response = await client.achat(
                prompt=_llm_prompt_for_record(
                    topic_title=topic_title,
                    topic_summary=topic_summary,
                    record=record,
                ),
                system_prompt=LLM_SYSTEM_PROMPT,
                temperature=0.65,
                model_name=MODEL_NAME,
                max_tokens=min(MAX_TOKENS, 420),
            )
            text = clean_agent_output(response["text"]).strip()
            if len(text) < 20:
                raise RuntimeError("LLM returned an unexpectedly short comment.")
            record["Text"] = text

    await asyncio.gather(*(generate_one(record) for record in records))


def _write_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    batch_id = int(payload.get("batch_id", 8))
    if batch_id < 1:
        raise ValueError("batch_id must be positive.")

    topic_title = str(payload.get("topic_title") or TOPIC_TITLE)
    topic_summary = str(payload.get("topic_summary") or TOPIC_SUMMARY)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("At least one actor group row is required.")
    deviation_grade = _deviation_grade(payload.get("deviation_grade", 3))
    deviation_target = str(payload.get("deviation_target", "all"))
    if deviation_target not in DEVIATION_TARGETS:
        raise ValueError(f"Unknown deviation target: {deviation_target}")
    wrong_answer_pattern = str(payload.get("wrong_answer_pattern", "random"))
    if wrong_answer_pattern not in WRONG_ANSWER_PATTERNS:
        raise ValueError(f"Unknown wrong-answer pattern: {wrong_answer_pattern}")

    clean_records = _build_records(
        batch_id=batch_id,
        dataset_type="clean",
        rows=rows,
        deviation_grade=deviation_grade,
        deviation_target=deviation_target,
        wrong_answer_pattern=wrong_answer_pattern,
    )
    dirty_records = _build_records(
        batch_id=batch_id,
        dataset_type="dirty",
        rows=rows,
        deviation_grade=deviation_grade,
        deviation_target=deviation_target,
        wrong_answer_pattern=wrong_answer_pattern,
    )
    requested_clean_actors = _coerce_count(payload.get("total_clean_actors"))
    requested_dirty_actors = _coerce_count(payload.get("total_dirty_actors"))
    if requested_clean_actors and len(clean_records) != requested_clean_actors:
        raise ValueError(
            f"Clean actor count mismatch: requested {requested_clean_actors}, generated {len(clean_records)}. "
            "Rebuild actor groups from totals or set total_clean_actors to 0 for manual rows."
        )
    if requested_dirty_actors and len(dirty_records) != requested_dirty_actors:
        raise ValueError(
            f"Dirty actor count mismatch: requested {requested_dirty_actors}, generated {len(dirty_records)}. "
            "Rebuild actor groups from totals or set total_dirty_actors to 0 for manual rows."
        )
    if not clean_records or not dirty_records:
        raise ValueError("Both clean and dirty datasets must contain at least one record.")

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = SESSION_DIR / f"simulation_output_clean_{batch_id}.json"
    dirty_path = SESSION_DIR / f"simulation_output_dirty_{batch_id}.json"
    overwrite = bool(payload.get("overwrite", True))
    if not overwrite and (clean_path.exists() or dirty_path.exists()):
        raise FileExistsError(f"Batch {batch_id} already exists. Enable overwrite to replace it.")

    generation_mode = str(payload.get("text_generation", "llm"))
    if generation_mode == "llm":
        asyncio.run(
            _generate_llm_texts(
                topic_title=topic_title,
                topic_summary=topic_summary,
                records=[*clean_records, *dirty_records],
            )
        )
    elif generation_mode != "fallback":
        raise ValueError("text_generation must be either 'llm' or 'fallback'.")

    clean_path.write_text(
        json.dumps(_export_payload(topic_title, topic_summary, clean_records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dirty_path.write_text(
        json.dumps(_export_payload(topic_title, topic_summary, dirty_records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    annotations_path = SESSION_DIR / f"scenario_annotations_batch_{batch_id}.json"
    annotations_path.write_text(
        json.dumps(
            _annotations_payload(
                topic_title=topic_title,
                topic_summary=topic_summary,
                clean_records=clean_records,
                dirty_records=dirty_records,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_path = SESSION_DIR / f"scenario_config_batch_{batch_id}.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "topic_title": topic_title,
                "topic_summary": topic_summary,
                "deviation_grade": deviation_grade,
                "deviation_target": deviation_target,
                "wrong_answer_pattern": wrong_answer_pattern,
                "allocation_strategy": payload.get("allocation_strategy"),
                "total_clean_actors": payload.get("total_clean_actors"),
                "total_dirty_actors": payload.get("total_dirty_actors"),
                "mode_mix": payload.get("mode_mix"),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    artifact_dir = SESSION_DIR / "cluster_bias_outputs" / f"batch_{batch_id}"
    if payload.get("run_clustering", True):
        result = subprocess.run(
            [sys.executable, "-m", "src.session_bias_clustering"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Clustering failed.")
    else:
        result = None

    return {
        "batch_id": batch_id,
        "deviation_grade": deviation_grade,
        "clean_records": len(clean_records),
        "dirty_records": len(dirty_records),
        "count_validation": {
            "requested_clean_actors": requested_clean_actors,
            "requested_dirty_actors": requested_dirty_actors,
            "generated_clean_records": len(clean_records),
            "generated_dirty_records": len(dirty_records),
            "clean_matches_request": not requested_clean_actors or len(clean_records) == requested_clean_actors,
            "dirty_matches_request": not requested_dirty_actors or len(dirty_records) == requested_dirty_actors,
        },
        "clean_path": str(clean_path.relative_to(PROJECT_ROOT)),
        "dirty_path": str(dirty_path.relative_to(PROJECT_ROOT)),
        "annotations_path": str(annotations_path.relative_to(PROJECT_ROOT)),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "artifact_dir": str(artifact_dir.relative_to(PROJECT_ROOT)),
        "clustering_stdout": result.stdout if result else "",
    }


def _default_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for country in COUNTRIES:
        for age_bucket in AGE_BUCKETS:
            rows.append(
                {
                    "country": country,
                    "age_bucket": age_bucket,
                    "bias_group": "none",
                    "force_correct": False,
                    "clean_count": 2,
                    "dirty_count": 2,
                }
            )
    return rows


def _generic_targeted_rows() -> list[dict[str, Any]]:
    rows = _default_rows()
    countries = list(COUNTRIES)
    primary_origin = countries[0]
    secondary_origin = countries[-1]
    for row in rows:
        country = row["country"]
        age_bucket = row["age_bucket"]
        row["dirty_count"] = 2 if country not in {primary_origin, secondary_origin} else 1

        if country == primary_origin and age_bucket == "18-29":
            row["dirty_count"] = 10
            row["bias_group"] = "young_support_overrepresented"
        elif country == primary_origin and age_bucket == "30-44":
            row["dirty_count"] = 2
            row["bias_group"] = "midcareer_conditional_present"
        elif country == primary_origin and age_bucket in {"45-59", "60+"}:
            row["dirty_count"] = 1
            row["bias_group"] = "senior_concern_underrepresented" if age_bucket == "45-59" else "latecareer_concern_underrepresented"
        elif country == secondary_origin and age_bucket in {"45-59", "60+"}:
            row["dirty_count"] = 6
            row["bias_group"] = "senior_oppose_overrepresented" if age_bucket == "45-59" else "latecareer_oppose_overrepresented"
        elif country == secondary_origin:
            row["dirty_count"] = 1
            row["bias_group"] = "general_underrepresented"
    return rows


def _summary_payload() -> dict[str, Any]:
    summary_path = SESSION_DIR / "cluster_bias_outputs" / "summary_report.json"
    if not summary_path.exists():
        return {"available": False}
    return {"available": True, "summary": json.loads(summary_path.read_text(encoding="utf-8"))}


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_batch_ids() -> list[int]:
    batch_ids: set[int] = set()
    if SESSION_DIR.exists():
        for path in SESSION_DIR.glob("simulation_output_*_*.json"):
            try:
                batch_ids.add(int(path.stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    output_dir = SESSION_DIR / "cluster_bias_outputs"
    if output_dir.exists():
        for path in output_dir.glob("batch_*"):
            if not path.is_dir():
                continue
            try:
                batch_ids.add(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(batch_ids)


def _load_dataset_table(batch_id: int, dataset_type: str, limit: int = 150) -> dict[str, Any]:
    path = SESSION_DIR / f"simulation_output_{dataset_type}_{batch_id}.json"
    if not path.exists():
        return {"available": False, "path": _relative_path(path), "total_rows": 0, "rows": []}

    payload = _read_json_file(path)
    records = payload.get("Dataset") or []
    rows: list[dict[str, Any]] = []
    for record in records[:limit]:
        metadata = record.get("Metadata") or {}
        rows.append(
            {
                "Id": record.get("Id"),
                "Text": record.get("Text"),
                **{field: metadata.get(field) for field in PUBLIC_METADATA_FIELDS},
            }
        )
    return {
        "available": True,
        "path": _relative_path(path),
        "topic_title": payload.get("TopicTitle"),
        "topic_summary": payload.get("TopicSummary"),
        "total_rows": len(records),
        "rows": rows,
    }


def _load_csv_table(path: Path, limit: int = 150) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": _relative_path(path), "total_rows": 0, "rows": [], "columns": []}

    rows: list[dict[str, Any]] = []
    total_rows = 0
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            total_rows += 1
            if len(rows) < limit:
                rows.append(dict(row))
    return {
        "available": True,
        "path": _relative_path(path),
        "total_rows": total_rows,
        "rows": rows,
        "columns": columns,
    }


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _top_values(counter: dict[str, int], limit: int = 4) -> str:
    if not counter:
        return "-"
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{key}: {value}" for key, value in ordered[:limit])


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _distribution(rows: list[dict[str, Any]], column: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        value = str(row.get(column) or "missing")
        counts[value] = counts.get(value, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _largest_share_shift(
    clean_rows: list[dict[str, Any]],
    dirty_rows: list[dict[str, Any]],
    column: str,
) -> dict[str, Any] | None:
    clean_dist = _distribution(clean_rows, column)
    dirty_dist = _distribution(dirty_rows, column)
    keys = set(clean_dist) | set(dirty_dist)
    if not keys:
        return None
    value = max(keys, key=lambda key: abs(dirty_dist.get(key, 0.0) - clean_dist.get(key, 0.0)))
    clean_share = clean_dist.get(value, 0.0)
    dirty_share = dirty_dist.get(value, 0.0)
    return {
        "value": value,
        "clean_share": clean_share,
        "dirty_share": dirty_share,
        "delta": dirty_share - clean_share,
        "abs_delta": abs(dirty_share - clean_share),
    }


def _largest_age_opinion_shift(
    clean_rows: list[dict[str, Any]],
    dirty_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ages = sorted(
        {
            str(row.get("age_bucket") or "missing")
            for row in [*clean_rows, *dirty_rows]
        }
    )
    best: dict[str, Any] | None = None
    for age in ages:
        clean_age_rows = [row for row in clean_rows if str(row.get("age_bucket") or "missing") == age]
        dirty_age_rows = [row for row in dirty_rows if str(row.get("age_bucket") or "missing") == age]
        if len(clean_age_rows) < 3 or len(dirty_age_rows) < 3:
            continue
        shift = _largest_share_shift(clean_age_rows, dirty_age_rows, "metadata_opinion")
        if shift is None:
            continue
        candidate = {
            "age_bucket": age,
            "opinion": shift["value"],
            "clean_share": shift["clean_share"],
            "dirty_share": shift["dirty_share"],
            "delta": shift["delta"],
            "abs_delta": shift["abs_delta"],
        }
        if best is None or candidate["abs_delta"] > best["abs_delta"]:
            best = candidate
    return best


def _dominant_value(rows: list[dict[str, Any]], column: str) -> dict[str, Any] | None:
    distribution = _distribution(rows, column)
    if not distribution:
        return None
    value = max(distribution, key=lambda key: distribution[key])
    return {"value": value, "share": distribution[value]}


def _dominant_age_opinion(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    overall_opinions = _distribution(rows, "metadata_opinion")
    ages = sorted({str(row.get("age_bucket") or "missing") for row in rows})
    best: dict[str, Any] | None = None
    for age in ages:
        age_rows = [row for row in rows if str(row.get("age_bucket") or "missing") == age]
        if len(age_rows) < 3:
            continue
        dominant = _dominant_value(age_rows, "metadata_opinion")
        if dominant is None:
            continue
        baseline = overall_opinions.get(dominant["value"], 0.0)
        candidate = {
            "age_bucket": age,
            "opinion": dominant["value"],
            "share": dominant["share"],
            "baseline_share": baseline,
            "delta_from_dataset": dominant["share"] - baseline,
            "abs_delta": abs(dominant["share"] - baseline),
        }
        if best is None or candidate["abs_delta"] > best["abs_delta"]:
            best = candidate
    return best


def _dataset_verdict(
    dataset_type: str,
    rows: list[dict[str, Any]],
    bias_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    label = "Clean" if dataset_type == "clean" else "Dirty"
    if not rows:
        return {
            "severity": "missing",
            "title": f"{label} verdict",
            "text": f"Δεν υπάρχουν αρκετά {dataset_type} assignment data για ξεχωριστή ετυμηγορία.",
            "details": [],
        }

    snapshot = bias_snapshot or {}
    signals = snapshot.get("cluster_bias_signals", {})
    representation = snapshot.get("representation", {})
    bias_groups = representation.get("bias_group_distribution") or _distribution(rows, "metadata_biasgroup")
    bias_marker_share = sum(
        share
        for group, share in bias_groups.items()
        if str(group).lower() not in {"none", "missing"}
    )
    strongest_cluster_signal = max(
        [
            ("nationality", float(signals.get("weighted_nationality_tvd", 0.0) or 0.0)),
            ("background", float(signals.get("weighted_background_tvd", 0.0) or 0.0)),
            ("age", float(signals.get("weighted_age_bucket_tvd", 0.0) or 0.0)),
            ("largest cluster", float(signals.get("largest_cluster_share", 0.0) or 0.0)),
        ],
        key=lambda item: item[1],
    )
    age_opinion = _dominant_age_opinion(rows)
    quality = _dominant_value(rows, "metadata_answerquality")
    opinion = _dominant_value(rows, "metadata_opinion")
    score = max(
        bias_marker_share,
        strongest_cluster_signal[1],
        age_opinion["abs_delta"] if age_opinion else 0.0,
    )

    if score >= 0.45 or bias_marker_share >= 0.40:
        severity = "strong"
        intro = f"Το {dataset_type} set φαίνεται έντονα biased"
    elif score >= 0.25 or bias_marker_share >= 0.18:
        severity = "moderate"
        intro = f"Το {dataset_type} set φαίνεται μέτρια biased"
    elif score >= 0.12:
        severity = "light"
        intro = f"Το {dataset_type} set έχει ήπια ένδειξη bias"
    else:
        severity = "low"
        intro = f"Το {dataset_type} set δεν φαίνεται ιδιαίτερα biased"

    if age_opinion and age_opinion["abs_delta"] >= 0.10:
        main_text = (
            f"{intro}: στις ηλικίες {age_opinion['age_bucket']} κυριαρχεί η άποψη "
            f"“{age_opinion['opinion']}” ({_percent(age_opinion['share'])}), ενώ στο σύνολο του "
            f"{dataset_type} είναι {_percent(age_opinion['baseline_share'])}."
        )
    elif opinion:
        main_text = (
            f"{intro}: η πιο συχνή άποψη είναι “{opinion['value']}” "
            f"({_percent(opinion['share'])}) χωρίς έντονη ηλικιακή απόκλιση."
        )
    else:
        main_text = f"{intro}: δεν υπάρχει αρκετή πληροφορία άποψης για πιο συγκεκριμένη περιγραφή."

    details = [
        f"Bias markers: {_percent(bias_marker_share)} των records.",
        f"Ισχυρότερο cluster signal: {strongest_cluster_signal[0]} {_percent(strongest_cluster_signal[1])}.",
    ]
    if quality:
        details.append(f"Πιο συχνή ποιότητα απάντησης: {quality['value']} ({_percent(quality['share'])}).")

    return {
        "severity": severity,
        "title": f"{label} verdict",
        "text": main_text,
        "details": details,
    }


def _batch_verdict(
    clean_rows: list[dict[str, Any]],
    dirty_rows: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    if not clean_rows or not dirty_rows:
        return {
            "severity": "missing",
            "title": "Ετυμηγορία",
            "text": "Δεν υπάρχουν αρκετά clean/dirty assignment data για να βγει ετυμηγορία.",
            "details": [],
        }

    age_opinion = _largest_age_opinion_shift(clean_rows, dirty_rows)
    opinion_shift = _largest_share_shift(clean_rows, dirty_rows, "metadata_opinion")
    age_shift = _largest_share_shift(clean_rows, dirty_rows, "age_bucket")
    quality_shift = _largest_share_shift(clean_rows, dirty_rows, "metadata_answerquality")
    shifts = (comparison or {}).get("representation_shift", {})
    strongest_signal = max(
        [
            ("nationality", float(shifts.get("nationality_tvd", 0.0) or 0.0)),
            ("age", float(shifts.get("age_bucket_tvd", 0.0) or 0.0)),
            ("opinion", float(shifts.get("opinion_tvd", 0.0) or 0.0)),
            ("answer quality", float(shifts.get("answer_quality_tvd", 0.0) or 0.0)),
            ("bias group", float(shifts.get("bias_group_tvd", 0.0) or 0.0)),
        ],
        key=lambda item: item[1],
    )
    score = max(
        strongest_signal[1],
        age_opinion["abs_delta"] if age_opinion else 0.0,
        opinion_shift["abs_delta"] if opinion_shift else 0.0,
    )
    if score >= 0.30:
        severity = "strong"
        intro = "Υπάρχει καθαρή ένδειξη bias"
    elif score >= 0.15:
        severity = "moderate"
        intro = "Υπάρχει μέτρια ένδειξη bias"
    elif score >= 0.08:
        severity = "light"
        intro = "Υπάρχει ήπια ένδειξη bias"
    else:
        severity = "low"
        intro = "Δεν φαίνεται ισχυρό bias"

    if age_opinion:
        direction = "πιο συχνά" if age_opinion["delta"] >= 0 else "λιγότερο συχνά"
        main_text = (
            f"{intro}: στις ηλικίες {age_opinion['age_bucket']} η άποψη "
            f"“{age_opinion['opinion']}” εμφανίζεται {direction} στο dirty set "
            f"({_percent(age_opinion['dirty_share'])}) σε σχέση με το clean set "
            f"({_percent(age_opinion['clean_share'])})."
        )
    elif opinion_shift:
        direction = "πιο συχνά" if opinion_shift["delta"] >= 0 else "λιγότερο συχνά"
        main_text = (
            f"{intro}: η άποψη “{opinion_shift['value']}” εμφανίζεται {direction} "
            f"στο dirty set ({_percent(opinion_shift['dirty_share'])}) σε σχέση με το clean set "
            f"({_percent(opinion_shift['clean_share'])})."
        )
    else:
        main_text = f"{intro}: δεν εντοπίστηκε καθαρή μετατόπιση άποψης ανά ηλικιακή ομάδα."

    details = [
        f"Ισχυρότερο συνολικό σήμα: {strongest_signal[0]} TVD {_percent(strongest_signal[1])}.",
    ]
    if age_shift:
        details.append(
            f"Ηλικιακή μετατόπιση: {age_shift['value']} από {_percent(age_shift['clean_share'])} clean σε {_percent(age_shift['dirty_share'])} dirty."
        )
    if quality_shift:
        details.append(
            f"Ποιότητα απάντησης: {quality_shift['value']} από {_percent(quality_shift['clean_share'])} clean σε {_percent(quality_shift['dirty_share'])} dirty."
        )

    return {
        "severity": severity,
        "title": "Ετυμηγορία",
        "text": main_text,
        "details": details,
    }


def _cluster_sets(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    fields = {
        "nationality": "metadata_nationality",
        "age": "age_bucket",
        "opinion": "metadata_opinion",
        "quality": "metadata_answerquality",
        "bias_group": "metadata_biasgroup",
    }
    for row in assignments:
        cluster_id = str(row.get("cluster_id", "unknown") or "unknown")
        bucket = grouped.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "records": 0,
                "samples": [],
                "distributions": {key: {} for key in fields},
            },
        )
        bucket["records"] += 1
        if len(bucket["samples"]) < 5:
            bucket["samples"].append(row.get("record_id"))
        for label, column in fields.items():
            value = str(row.get(column) or "missing")
            bucket["distributions"][label][value] = bucket["distributions"][label].get(value, 0) + 1

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        try:
            return (0, f"{int(item['cluster_id']):06d}")
        except ValueError:
            return (1, item["cluster_id"])

    sets = []
    for bucket in grouped.values():
        distributions = bucket["distributions"]
        sets.append(
            {
                "cluster_id": bucket["cluster_id"],
                "records": bucket["records"],
                "nationality": _top_values(distributions["nationality"]),
                "age": _top_values(distributions["age"]),
                "opinion": _top_values(distributions["opinion"]),
                "quality": _top_values(distributions["quality"]),
                "bias_group": _top_values(distributions["bias_group"]),
                "samples": [sample for sample in bucket["samples"] if sample],
            }
        )
    return sorted(sets, key=sort_key)


def _load_text_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": _relative_path(path), "content": ""}
    return {"available": True, "path": _relative_path(path), "content": path.read_text(encoding="utf-8")}


def _batch_payload(batch_id: int) -> dict[str, Any]:
    batch_dir = SESSION_DIR / "cluster_bias_outputs" / f"batch_{batch_id}"
    clean_assignments_path = batch_dir / f"clean_batch_{batch_id}_cluster_assignments.csv"
    dirty_assignments_path = batch_dir / f"dirty_batch_{batch_id}_cluster_assignments.csv"
    clean_assignments = _load_csv_table(clean_assignments_path)
    dirty_assignments = _load_csv_table(dirty_assignments_path)
    clean_assignment_rows = _load_csv_rows(clean_assignments_path)
    dirty_assignment_rows = _load_csv_rows(dirty_assignments_path)
    comparison_path = batch_dir / f"batch_{batch_id}_clean_dirty_comparison.json"
    clean_summary_path = batch_dir / f"clean_batch_{batch_id}_cluster_summary.json"
    dirty_summary_path = batch_dir / f"dirty_batch_{batch_id}_cluster_summary.json"
    clean_bias_path = batch_dir / f"clean_batch_{batch_id}_bias_snapshot.json"
    dirty_bias_path = batch_dir / f"dirty_batch_{batch_id}_bias_snapshot.json"
    config_path = SESSION_DIR / f"scenario_config_batch_{batch_id}.json"
    annotations_path = SESSION_DIR / f"scenario_annotations_batch_{batch_id}.json"

    comparison = _read_json_file(comparison_path) if comparison_path.exists() else None
    clean_bias_snapshot = _read_json_file(clean_bias_path) if clean_bias_path.exists() else None
    dirty_bias_snapshot = _read_json_file(dirty_bias_path) if dirty_bias_path.exists() else None
    payload = {
        "available": batch_dir.exists()
        or (SESSION_DIR / f"simulation_output_clean_{batch_id}.json").exists()
        or (SESSION_DIR / f"simulation_output_dirty_{batch_id}.json").exists(),
        "batch_id": batch_id,
        "batches": _discover_batch_ids(),
        "paths": {
            "batch_dir": _relative_path(batch_dir),
            "config": _relative_path(config_path),
            "annotations": _relative_path(annotations_path),
        },
        "config": _read_json_file(config_path) if config_path.exists() else None,
        "comparison": comparison,
        "verdict": _batch_verdict(
            clean_assignment_rows,
            dirty_assignment_rows,
            comparison.get("comparison") if comparison else None,
        ),
        "dataset_verdicts": {
            "clean": _dataset_verdict("clean", clean_assignment_rows, clean_bias_snapshot),
            "dirty": _dataset_verdict("dirty", dirty_assignment_rows, dirty_bias_snapshot),
        },
        "summaries": {
            "clean": _read_json_file(clean_summary_path) if clean_summary_path.exists() else None,
            "dirty": _read_json_file(dirty_summary_path) if dirty_summary_path.exists() else None,
        },
        "bias_snapshots": {
            "clean": clean_bias_snapshot,
            "dirty": dirty_bias_snapshot,
        },
        "scatterplots": {
            "combined": _load_text_artifact(batch_dir / f"batch_{batch_id}_clean_dirty_scatterplot.html"),
            "clean": _load_text_artifact(batch_dir / f"clean_batch_{batch_id}_scatterplot.html"),
            "dirty": _load_text_artifact(batch_dir / f"dirty_batch_{batch_id}_scatterplot.html"),
        },
        "datasets": {
            "clean": _load_dataset_table(batch_id, "clean"),
            "dirty": _load_dataset_table(batch_id, "dirty"),
        },
        "assignments": {
            "clean": clean_assignments,
            "dirty": dirty_assignments,
        },
        "cluster_sets": {
            "clean": _cluster_sets(clean_assignment_rows) if clean_assignments["available"] else [],
            "dirty": _cluster_sets(dirty_assignment_rows) if dirty_assignments["available"] else [],
        },
    }
    return payload


class ScenarioUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            _text_response(self, 200, HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/options":
            _json_response(
                self,
                200,
                {
                    "countries": list(COUNTRIES),
                    "age_buckets": list(AGE_BUCKETS),
                    "opinions": OPINIONS,
                    "bias_groups": BIAS_GROUPS,
                    "allocation_strategies": ALLOCATION_STRATEGIES,
                    "deviation_targets": DEVIATION_TARGETS,
                    "wrong_answer_patterns": WRONG_ANSWER_PATTERNS,
                    "topic_title": TOPIC_TITLE,
                    "topic_summary": TOPIC_SUMMARY,
                    "balanced_rows": _default_rows(),
                    "targeted_rows": _generic_targeted_rows(),
                },
            )
            return
        if parsed.path == "/api/summary":
            _json_response(self, 200, _summary_payload())
            return
        if parsed.path == "/api/batch":
            query = parse_qs(parsed.query)
            try:
                batch_id = int((query.get("batch_id") or ["8"])[0])
                if batch_id < 1:
                    raise ValueError
            except ValueError:
                _json_response(self, 400, {"error": "batch_id must be a positive integer."})
                return
            _json_response(self, 200, _batch_payload(batch_id))
            return
        _json_response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            _json_response(self, 404, {"error": "Not found"})
            return
        try:
            payload = _read_body(self)
            result = _write_scenario(payload)
        except Exception as exc:  # noqa: BLE001 - local UI should return readable errors
            _json_response(self, 400, {"error": str(exc)})
            return
        _json_response(self, 200, {"ok": True, **result, **_summary_payload()})


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Synthetic Scenario UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #0b2f6f;
      --panel: #eaf2ff;
      --panel-strong: #d8e9ff;
      --ink: #08204a;
      --muted: #3c5c8d;
      --line: #8db7ef;
      --accent: #155eef;
      --accent-2: #0b4ab8;
      --danger: #b91c1c;
      --soft: #dbeafe;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #073077 0%, #0b4ab8 48%, #06245b 100%);
      color: var(--ink);
      font: 14px/1.45 Arial, sans-serif;
      min-height: 100vh;
    }
    header {
      padding: 18px 22px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.24);
      background: #082b68;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      color: #f8fbff;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 4px;
      letter-spacing: 0;
    }
    .sub { color: #cfe1ff; font-size: 13px; }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 12px 22px 0;
    }
    .tab-button {
      border-color: rgba(255,255,255,0.35);
      background: rgba(248,251,255,0.14);
      color: #eef6ff;
      font-weight: 700;
    }
    .tab-button.active {
      background: #eaf2ff;
      color: var(--ink);
      border-color: var(--line);
    }
    main {
      padding: 18px 22px 28px;
    }
    .tab-panel {
      display: grid;
      gap: 16px;
    }
    .tab-panel[hidden] { display: none; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 10px 26px rgba(3, 28, 75, 0.18);
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }
    h3 {
      margin: 0 0 8px;
      font-size: 13px;
      color: #12386f;
      letter-spacing: 0;
    }
    .toolbar, .form-grid, .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      min-width: 120px;
    }
    input, select, textarea, button {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }
    input, select, textarea {
      padding: 8px 9px;
      min-height: 36px;
    }
    textarea { width: min(780px, 100%); min-height: 64px; }
    input[type="range"] { padding: 0; min-height: 28px; }
    button {
      min-height: 36px;
      padding: 8px 11px;
      cursor: pointer;
      border-color: #b9c1d1;
      background: #f8fafc;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 700;
    }
    button.warn {
      background: #dbeafe;
      border-color: #93c5fd;
      color: #0b4ab8;
    }
    button.danger {
      background: #fef2f2;
      border-color: #fecaca;
      color: var(--danger);
    }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; min-width: 980px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px; text-align: left; vertical-align: middle; }
    th { background: #cfe1ff; color: #12386f; font-size: 12px; position: sticky; top: 0; z-index: 1; }
    td input, td select { width: 100%; min-width: 96px; }
    td.count input { min-width: 72px; }
    .status {
      white-space: pre-wrap;
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 8px;
      padding: 12px;
      color: #263247;
      max-height: 320px;
      overflow: auto;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--soft);
    }
    .metric b { display: block; font-size: 18px; margin-top: 4px; }
    .metric small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .verdict {
      border: 1px solid #6ea8e8;
      border-left: 6px solid var(--accent);
      border-radius: 8px;
      background: #f8fafc;
      padding: 12px 14px;
      color: #102a56;
    }
    .verdict-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 10px;
    }
    .verdict-grid .verdict.compare {
      grid-column: 1 / -1;
    }
    .verdict.strong { border-left-color: #b91c1c; }
    .verdict.moderate { border-left-color: #b45309; }
    .verdict.light { border-left-color: #155eef; }
    .verdict.low { border-left-color: #15803d; }
    .verdict b {
      display: block;
      font-size: 16px;
      margin-bottom: 5px;
    }
    .verdict p {
      margin: 0;
      font-size: 14px;
    }
    .verdict ul {
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }
    .plot-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      padding: 10px;
      min-width: 0;
    }
    .plot-panel.wide { grid-column: 1 / -1; }
    .inner-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }
    .inner-tab-button {
      min-width: 112px;
      border-color: #93c5fd;
      background: #f8fafc;
      color: #0b4ab8;
      font-weight: 700;
    }
    .inner-tab-button.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    iframe {
      width: 100%;
      height: 760px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
    }
    .compact-table table { min-width: 1180px; }
    .compact-table td { font-size: 12px; }
    .cell-text {
      max-width: 420px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      background: #f8fafc;
      color: var(--muted);
    }
    .deviation-panel {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
    }
    .deviation-item {
      border: 1px solid #6ea8e8;
      border-radius: 8px;
      background: #f7fbff;
      padding: 9px;
      min-height: 74px;
    }
    .deviation-item span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .deviation-item b {
      display: block;
      font-size: 16px;
      color: #0b3d91;
    }
    .deviation-note {
      margin-top: 8px;
      color: #224a80;
      font-size: 13px;
    }
    .spacer { flex: 1 1 auto; }
    .check-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f7fbff;
    }
    .check {
      min-width: 0;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--ink);
      font-size: 13px;
    }
    .check input {
      min-height: auto;
      width: auto;
      margin: 0;
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; }
      label { min-width: 100%; }
      textarea { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Synthetic Scenario UI</h1>
      <div class="sub">Author clean/dirty actor groups by country of origin, age, and bias marker. Opinion and answer quality are generated automatically.</div>
    </div>
    <button id="refresh">Refresh summary</button>
  </header>
  <nav class="tabs" aria-label="Views">
    <button class="tab-button active" data-tab-target="scenarioTab">Scenario builder</button>
    <button class="tab-button" data-tab-target="batchTab">Batch data</button>
  </nav>
  <main>
    <div class="tab-panel" id="scenarioTab">
    <section>
      <h2>Scenario</h2>
      <div class="form-grid">
        <label>Batch id <input id="batchId" type="number" min="1" value="8"></label>
        <label>Overwrite existing <select id="overwrite"><option value="true">Yes</option><option value="false">No</option></select></label>
        <label>Run clustering <select id="runClustering"><option value="true">Yes</option><option value="false">No</option></select></label>
        <label>Text generation <select id="textGeneration"><option value="llm">LLM natural text</option><option value="fallback">Fast fallback</option></select></label>
        <label>Total clean actors <input id="totalCleanActors" type="number" min="0" value="100"></label>
        <label>Total dirty actors <input id="totalDirtyActors" type="number" min="0" value="100"></label>
        <label>Allocation strategy
          <select id="allocationStrategy">
            <option value="equal_distribution">Equal distribution</option>
            <option value="population_like_distribution">Population-like distribution</option>
            <option value="skew_toward_selected_group">Skew toward selected group</option>
            <option value="custom_weights">Custom weights</option>
          </select>
        </label>
        <label>Deviation target
          <select id="deviationTarget">
            <option value="all">All signals</option>
            <option value="demographic">Demographic only</option>
            <option value="opinion">Opinion only</option>
            <option value="answer_quality">Answer quality only</option>
            <option value="country_age_interaction">Country + age interaction</option>
            <option value="hidden_quality_bias">Hidden quality bias</option>
          </select>
        </label>
        <label>Wrong-answer pattern
          <select id="wrongAnswerPattern">
            <option value="random">Random</option>
            <option value="off_topic">Off-topic</option>
            <option value="overgeneralized">Overgeneralized</option>
            <option value="stereotyped">Stereotyped</option>
            <option value="missing_safeguards">Missing safeguards</option>
            <option value="contradicts_assigned_opinion">Contradicts opinion</option>
            <option value="low_detail">Low detail</option>
          </select>
        </label>
        <label>Scenario template
          <select id="scenarioTemplate">
            <option value="mild_demographic_bias">Mild demographic bias</option>
            <option value="strong_age_polarization">Strong age polarization</option>
            <option value="origin_underrepresentation">Origin underrepresentation</option>
            <option value="answer_quality_degradation">Answer-quality degradation</option>
            <option value="hidden_quality_bias">Hidden quality bias</option>
          </select>
        </label>
        <button id="applyTemplate" class="warn">Apply template</button>
        <label style="min-width:280px">Topic title <input id="topicTitle"></label>
      </div>
      <label style="margin-top:10px">Topic summary <textarea id="topicSummary"></textarea></label>
    </section>

    <section>
      <h2>Actor Origin Builder</h2>
      <div class="toolbar">
        <label>Clean count per cell <input id="builderClean" type="number" min="0" value="2"></label>
        <label>Dirty count per cell <input id="builderDirty" type="number" min="0" value="2"></label>
        <label>Deviation grade <input id="deviationGrade" type="range" min="1" max="5" value="3"><span id="deviationLabel">3 - medium</span></label>
        <label>Compound bias mode
          <select id="builderMode">
            <option value="balanced">Balanced</option>
            <option value="age_polarization">Age polarization</option>
            <option value="origin_pair_polarization" selected>Origin pair polarization</option>
            <option value="origin_overrepresentation">First origin overrepresented</option>
            <option value="origin_underrepresentation">First origin underrepresented</option>
            <option value="quality_stress">Answer-quality stress</option>
          </select>
        </label>
        <button id="buildFromOrigins" class="primary">Build actor groups</button>
      </div>
      <div class="deviation-panel" id="deviationPreview"></div>
      <div class="deviation-note" id="deviationNote"></div>
      <div class="form-grid" style="margin-top:12px">
        <label style="min-width:320px">Countries of origin <div id="countryChecks" class="check-row"></div></label>
        <label style="min-width:320px">Age buckets <div id="ageChecks" class="check-row"></div></label>
      </div>
      <div class="form-grid" style="margin-top:12px">
        <label>Balanced mix % <input id="mixBalanced" type="number" min="0" max="100" value="0"></label>
        <label>Demographic skew % <input id="mixDemographic" type="number" min="0" max="100" value="70"></label>
        <label>Opinion skew % <input id="mixOpinion" type="number" min="0" max="100" value="20"></label>
        <label>Quality stress % <input id="mixQuality" type="number" min="0" max="100" value="10"></label>
      </div>
    </section>

    <section>
      <h2>Actor Groups</h2>
      <div class="toolbar">
        <button id="balanced">Load balanced preset</button>
        <button id="targeted" class="warn">Load generic targeted bias</button>
        <button id="addRow">Add row</button>
        <span class="spacer"></span>
        <button id="generate" class="primary">Generate datasets</button>
      </div>
      <div class="table-wrap" style="margin-top:12px">
        <table>
          <thead>
            <tr>
              <th>Country</th>
              <th>Age bucket</th>
              <th>Bias group</th>
              <th>Clean count</th>
              <th>Dirty count</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Design Validation</h2>
      <div id="designValidation" class="metric-grid"></div>
    </section>

    <section>
      <h2>Current Summary</h2>
      <div id="metrics" class="metric-grid"></div>
    </section>

    <section>
      <h2>Status</h2>
      <div id="status" class="status">Ready.</div>
    </section>
    </div>

    <div class="tab-panel" id="batchTab" hidden>
      <section>
        <h2>Batch Artifacts</h2>
        <div class="toolbar">
          <label>Batch
            <select id="batchPicker"></select>
          </label>
          <button id="loadBatchData" class="primary">Load batch</button>
          <button id="useCurrentBatch">Use current batch id</button>
        </div>
        <div id="batchVerdict" style="margin-top:12px"></div>
        <div id="batchOverview" class="metric-grid" style="margin-top:12px"></div>
      </section>

      <section>
        <h2>Scatterplots</h2>
        <div class="plot-panel wide">
          <div class="inner-tabs" aria-label="Scatterplot views">
            <button class="inner-tab-button active" data-plot-view="combined">Clean vs dirty</button>
            <button class="inner-tab-button" data-plot-view="clean">Clean</button>
            <button class="inner-tab-button" data-plot-view="dirty">Dirty</button>
          </div>
          <h3 id="activePlotTitle">Clean vs dirty</h3>
          <iframe id="plotActive" title="Selected scatterplot"></iframe>
        </div>
      </section>

      <section>
        <h2>Cluster Sets</h2>
        <div id="clusterSets" class="table-wrap compact-table"></div>
      </section>

      <section>
        <h2>Batch Data</h2>
        <div class="toolbar">
          <label>Table
            <select id="batchTableSelect">
              <option value="clean_records">Clean records</option>
              <option value="dirty_records">Dirty records</option>
              <option value="clean_assignments">Clean assignments</option>
              <option value="dirty_assignments">Dirty assignments</option>
            </select>
          </label>
          <label style="min-width:260px">Filter <input id="batchFilter" placeholder="id, text, metadata, cluster"></label>
        </div>
        <div id="batchTableMeta" class="deviation-note"></div>
        <div id="batchDataTable" class="table-wrap compact-table" style="margin-top:12px"></div>
      </section>
    </div>
  </main>

  <script>
    let options = {};
    let rows = [];
    let batchData = null;
    let activePlotView = 'combined';
    const SCENARIO_TEMPLATES = {
      mild_demographic_bias: {
        mode: 'origin_overrepresentation',
        grade: 2,
        target: 'demographic',
        allocation: 'skew_toward_selected_group',
        wrongPattern: 'random',
        cleanTotal: 80,
        dirtyTotal: 80,
        mix: [0, 100, 0, 0]
      },
      strong_age_polarization: {
        mode: 'age_polarization',
        grade: 5,
        target: 'all',
        allocation: 'skew_toward_selected_group',
        wrongPattern: 'overgeneralized',
        cleanTotal: 120,
        dirtyTotal: 120,
        mix: [0, 70, 20, 10]
      },
      origin_underrepresentation: {
        mode: 'origin_underrepresentation',
        grade: 4,
        target: 'country_age_interaction',
        allocation: 'skew_toward_selected_group',
        wrongPattern: 'missing_safeguards',
        cleanTotal: 100,
        dirtyTotal: 100,
        mix: [0, 80, 10, 10]
      },
      answer_quality_degradation: {
        mode: 'quality_stress',
        grade: 5,
        target: 'answer_quality',
        allocation: 'equal_distribution',
        wrongPattern: 'low_detail',
        cleanTotal: 100,
        dirtyTotal: 100,
        mix: [0, 0, 0, 100]
      },
      hidden_quality_bias: {
        mode: 'quality_stress',
        grade: 4,
        target: 'hidden_quality_bias',
        allocation: 'equal_distribution',
        wrongPattern: 'contradicts_assigned_opinion',
        cleanTotal: 100,
        dirtyTotal: 100,
        mix: [50, 0, 0, 50]
      }
    };

    const el = (id) => document.getElementById(id);
    const status = (value) => { el('status').textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2); };
    const optionHtml = (items, selected) => items.map(v => `<option ${v === selected ? 'selected' : ''}>${v}</option>`).join('');
    const checkboxHtml = (items, name) => items.map(v => `<label class="check"><input type="checkbox" name="${name}" value="${v}" checked> ${v}</label>`).join('');
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
    const compact = (value, max = 260) => {
      const text = String(value ?? '');
      return text.length > max ? `${text.slice(0, max)}...` : text;
    };

    async function loadOptions() {
      options = await fetch('/api/options').then(r => r.json());
      el('topicTitle').value = options.topic_title;
      el('topicSummary').value = options.topic_summary;
      el('countryChecks').innerHTML = checkboxHtml(options.countries, 'countryOrigin');
      el('ageChecks').innerHTML = checkboxHtml(options.age_buckets, 'ageOrigin');
      materializeActorGroups({quiet: true});
      await refreshSummary();
      await fetchBatchData(Number(el('batchId').value || 8));
    }

    function renderRows() {
      el('rows').innerHTML = rows.map((row, index) => `
        <tr>
          <td><select data-index="${index}" data-field="country">${optionHtml(options.countries, row.country)}</select></td>
          <td><select data-index="${index}" data-field="age_bucket">${optionHtml(options.age_buckets, row.age_bucket)}</select></td>
          <td><select data-index="${index}" data-field="bias_group">${optionHtml(options.bias_groups, row.bias_group || 'none')}</select></td>
          <td class="count"><input data-index="${index}" data-field="clean_count" type="number" min="0" value="${row.clean_count || 0}"></td>
          <td class="count"><input data-index="${index}" data-field="dirty_count" type="number" min="0" value="${row.dirty_count || 0}"></td>
          <td><button class="danger" data-delete="${index}">Delete</button></td>
        </tr>
      `).join('');
      renderDesignValidation();
    }

    el('rows').addEventListener('input', (event) => {
      const target = event.target;
      const index = Number(target.dataset.index);
      const field = target.dataset.field;
      if (!field || Number.isNaN(index)) return;
      rows[index][field] = ['clean_count', 'dirty_count'].includes(field) ? Number(target.value || 0) : target.value;
      renderDesignValidation();
    });
    el('rows').addEventListener('click', (event) => {
      const index = event.target.dataset.delete;
      if (index === undefined) return;
      rows.splice(Number(index), 1);
      renderRows();
    });

    el('balanced').addEventListener('click', () => {
      rows = options.balanced_rows.map(x => ({...x}));
      renderRows();
    });
    el('targeted').addEventListener('click', () => {
      rows = options.targeted_rows.map(x => ({...x}));
      renderRows();
    });
    el('applyTemplate').addEventListener('click', () => {
      const template = SCENARIO_TEMPLATES[el('scenarioTemplate').value];
      if (!template) return;
      el('builderMode').value = template.mode;
      el('deviationGrade').value = template.grade;
      el('deviationTarget').value = template.target;
      el('allocationStrategy').value = template.allocation;
      el('wrongAnswerPattern').value = template.wrongPattern;
      el('totalCleanActors').value = template.cleanTotal;
      el('totalDirtyActors').value = template.dirtyTotal;
      el('mixBalanced').value = template.mix[0];
      el('mixDemographic').value = template.mix[1];
      el('mixOpinion').value = template.mix[2];
      el('mixQuality').value = template.mix[3];
      updateDeviationLabel();
      materializeActorGroups({quiet: true});
      status(`Applied template and rebuilt actor groups: ${el('scenarioTemplate').selectedOptions[0].textContent}.`);
    });
    el('addRow').addEventListener('click', () => {
      rows.push({
        country: options.countries[0],
        age_bucket: options.age_buckets[0],
        bias_group: 'none',
        force_correct: false,
        clean_count: 1,
        dirty_count: 1
      });
      renderRows();
    });

    function checkedValues(name) {
      return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(input => input.value);
    }

    function numberValue(id) {
      return Number(el(id).value || 0);
    }

    function modeMix() {
      return {
        balanced: numberValue('mixBalanced'),
        demographic: numberValue('mixDemographic'),
        opinion: numberValue('mixOpinion'),
        quality: numberValue('mixQuality')
      };
    }

    function sumRows(countKey) {
      return rows.reduce((sum, row) => sum + Number(row[countKey] || 0), 0);
    }

    function requestedTotals() {
      return {
        clean: Number(el('totalCleanActors').value || 0),
        dirty: Number(el('totalDirtyActors').value || 0)
      };
    }

    function rowsMatchRequestedTotals() {
      const requested = requestedTotals();
      return (!requested.clean || sumRows('clean_count') === requested.clean) &&
        (!requested.dirty || sumRows('dirty_count') === requested.dirty);
    }

    function normalizeMix(mix) {
      const total = Object.values(mix).reduce((sum, value) => sum + Math.max(0, value), 0);
      if (total <= 0) return {balanced: 0, demographic: 100, opinion: 0, quality: 0};
      return Object.fromEntries(Object.entries(mix).map(([key, value]) => [key, Math.max(0, value) / total]));
    }

    function integerAllocations(total, cells, weightFn) {
      const safeTotal = Math.max(0, Math.round(total));
      if (!cells.length || safeTotal === 0) return new Map(cells.map(cell => [cell.key, 0]));
      const weighted = cells.map(cell => ({...cell, weight: Math.max(0.001, weightFn(cell))}));
      const weightTotal = weighted.reduce((sum, cell) => sum + cell.weight, 0);
      const raw = weighted.map(cell => ({...cell, exact: safeTotal * cell.weight / weightTotal}));
      const allocated = new Map(raw.map(cell => [cell.key, Math.floor(cell.exact)]));
      let remaining = safeTotal - [...allocated.values()].reduce((sum, value) => sum + value, 0);
      raw.sort((a, b) => (b.exact - Math.floor(b.exact)) - (a.exact - Math.floor(a.exact)));
      for (let index = 0; remaining > 0; index += 1, remaining -= 1) {
        const cell = raw[index % raw.length];
        allocated.set(cell.key, allocated.get(cell.key) + 1);
      }
      return allocated;
    }

    function populationWeight(ageBucket) {
      return {'18-29': 0.22, '30-44': 0.36, '45-59': 0.27, '60+': 0.15}[ageBucket] || 0.25;
    }

    function cleanWeight(cell) {
      const strategy = el('allocationStrategy').value;
      if (strategy === 'population_like_distribution') return populationWeight(cell.age_bucket);
      return 1;
    }

    function dirtyWeightFor(mode, cell, selectedCountries) {
      const [, countScore] = modeBiasGroup(mode, cell.country, cell.age_bucket, selectedCountries);
      const strategy = el('allocationStrategy').value;
      if (strategy === 'equal_distribution') return 1;
      if (strategy === 'population_like_distribution') return populationWeight(cell.age_bucket);
      if (strategy === 'custom_weights') return Math.max(1, Number(el('builderDirty').value || 1));
      return Math.max(1, countScore);
    }

    function distribution(countKey, fieldKey) {
      const totals = {};
      let total = 0;
      for (const row of rows) {
        const count = Number(row[countKey] || 0);
        if (count <= 0) continue;
        const value = row[fieldKey] || 'none';
        totals[value] = (totals[value] || 0) + count;
        total += count;
      }
      return {total, values: Object.fromEntries(Object.entries(totals).map(([key, value]) => [key, value / Math.max(1, total)]))};
    }

    function tvd(clean, dirty) {
      const keys = new Set([...Object.keys(clean.values), ...Object.keys(dirty.values)]);
      let delta = 0;
      for (const key of keys) delta += Math.abs((clean.values[key] || 0) - (dirty.values[key] || 0));
      return delta / 2;
    }

    function renderDesignValidation() {
      if (!el('designValidation')) return;
      const cleanTotal = rows.reduce((sum, row) => sum + Number(row.clean_count || 0), 0);
      const dirtyTotal = rows.reduce((sum, row) => sum + Number(row.dirty_count || 0), 0);
      const requested = requestedTotals();
      const cleanMatch = !requested.clean || cleanTotal === requested.clean;
      const dirtyMatch = !requested.dirty || dirtyTotal === requested.dirty;
      const natTvd = tvd(distribution('clean_count', 'country'), distribution('dirty_count', 'country'));
      const ageTvd = tvd(distribution('clean_count', 'age_bucket'), distribution('dirty_count', 'age_bucket'));
      const biasGroups = distribution('dirty_count', 'bias_group').values;
      const biasedShare = Object.entries(biasGroups).reduce((sum, [key, value]) => key === 'none' ? sum : sum + value, 0);
      const grade = deviationGrade();
      const target = el('deviationTarget') ? el('deviationTarget').value : 'all';
      const expectedOpinionTvd = ['all', 'opinion', 'country_age_interaction'].includes(target) ? Math.min(0.85, biasedShare * (0.14 + grade * 0.08)) : 0;
      const expectedQualityTvd = ['all', 'answer_quality', 'hidden_quality_bias'].includes(target) ? Math.min(0.70, biasedShare * (0.07 + grade * 0.06)) : 0;
      el('designValidation').innerHTML = `
        <div class="metric">Clean actors <b>${cleanTotal}</b><div>from current actor-group table</div></div>
        <div class="metric">Dirty actors <b>${dirtyTotal}</b><div>from current actor-group table</div></div>
        <div class="metric">Requested clean <b>${requested.clean || 'manual'}</b><div>${cleanMatch ? 'matches current rows' : `mismatch: rows have ${cleanTotal}`}</div></div>
        <div class="metric">Requested dirty <b>${requested.dirty || 'manual'}</b><div>${dirtyMatch ? 'matches current rows' : `mismatch: rows have ${dirtyTotal}`}</div></div>
        <div class="metric">Expected nationality TVD <b>${natTvd.toFixed(3)}</b><div>count-level preview</div></div>
        <div class="metric">Expected age TVD <b>${ageTvd.toFixed(3)}</b><div>count-level preview</div></div>
        <div class="metric">Estimated opinion TVD <b>${expectedOpinionTvd.toFixed(3)}</b><div>based on biased share + grade</div></div>
        <div class="metric">Estimated quality TVD <b>${expectedQualityTvd.toFixed(3)}</b><div>based on wrong-answer stress</div></div>
      `;
    }

    function deviationGrade() {
      return Number(el('deviationGrade').value || 3);
    }

    function updateDeviationLabel() {
      const labels = {
        1: '1 - subtle',
        2: '2 - mild',
        3: '3 - medium',
        4: '4 - strong',
        5: '5 - extreme'
      };
      const grade = deviationGrade();
      const dirtyBase = Number(el('builderDirty').value || 0);
      const weak = Math.max(0, dirtyBase - Math.ceil(grade / 2));
      const medium = dirtyBase + (grade * 2);
      const strong = dirtyBase + (grade * 4);
      const dominantOpinion = Math.round((44 + (grade * 9.5)) * 10) / 10;
      const correct = Math.round(Math.max(35, 82 - (grade * 9)));
      const partial = Math.round(Math.min(30, 14 + (grade * 3)));
      const incorrect = Math.max(0, 100 - correct - partial);
      el('deviationLabel').textContent = labels[grade];
      el('deviationPreview').innerHTML = `
        <div class="deviation-item"><span>Targeted dirty cells</span><b>${strong} actors</b><div>base ${dirtyBase}, weak cells ${weak}</div></div>
        <div class="deviation-item"><span>Dominant opinion in biased groups</span><b>~${dominantOpinion}%</b><div>support / oppose / concern depending on group</div></div>
        <div class="deviation-item"><span>Answer quality in biased groups</span><b>${correct}% correct</b><div>${partial}% partial / ${incorrect}% incorrect</div></div>
        <div class="deviation-item"><span>Expected graph signal</span><b>${grade >= 4 ? 'Very visible' : grade >= 3 ? 'Visible' : 'Light'}</b><div>dirty clusters should separate more as grade increases</div></div>
      `;
      const modeLabels = {
        balanced: 'keeps clean and dirty aligned',
        age_polarization: 'pushes young actors toward support and senior actors toward opposition',
        origin_pair_polarization: 'uses the first selected origin as young/support and the last selected origin as senior/oppose',
        origin_overrepresentation: 'overrepresents the first selected origin across age buckets',
        origin_underrepresentation: 'underrepresents the first selected origin across age buckets',
        quality_stress: 'keeps counts closer but increases partial/incorrect answers in dirty biased groups'
      };
      el('deviationNote').textContent = `Mode effect: ${modeLabels[el('builderMode').value]}. Set grade 5, Build actor groups, then Generate datasets to create a strong scatterplot deviation.`;
      renderDesignValidation();
    }

    function modeBiasGroup(mode, country, ageBucket, selectedCountries) {
      const grade = deviationGrade();
      const dirtyBase = Number(el('builderDirty').value || 0);
      const weak = Math.max(0, dirtyBase - Math.ceil(grade / 2));
      const medium = dirtyBase + (grade * 2);
      const strong = dirtyBase + (grade * 4);
      const firstOrigin = selectedCountries[0];
      const lastOrigin = selectedCountries[selectedCountries.length - 1] || firstOrigin;
      if (mode === 'age_polarization') {
        if (ageBucket === '18-29') return ['young_support_overrepresented', strong];
        if (ageBucket === '30-44') return ['midcareer_conditional_present', medium];
        if (['45-59', '60+'].includes(ageBucket)) return ['senior_oppose_overrepresented', strong];
      }
      if (mode === 'origin_pair_polarization') {
        if (country === firstOrigin && ageBucket === '18-29') return ['young_support_overrepresented', strong];
        if (country === firstOrigin && ageBucket === '30-44') return ['midcareer_conditional_present', medium];
        if (country === firstOrigin) return ['general_underrepresented', weak];
        if (country === lastOrigin && ['45-59', '60+'].includes(ageBucket)) return ['senior_oppose_overrepresented', strong];
        if (country === lastOrigin) return ['general_underrepresented', weak];
        return ['none', dirtyBase];
      }
      if (mode === 'origin_overrepresentation') {
        if (country === firstOrigin) return ['general_overrepresented', strong];
        return ['general_underrepresented', weak];
      }
      if (mode === 'origin_underrepresentation') {
        if (country === firstOrigin) return ['general_underrepresented', weak];
        return ['general_overrepresented', medium];
      }
      if (mode === 'quality_stress') {
        if (country === firstOrigin || country === lastOrigin) return ['country_age_targeted_bias', dirtyBase];
        return ['none', dirtyBase];
      }
      if (mode !== 'balanced') {
        return ['general_underrepresented', weak];
      }
      return ['none', dirtyBase];
    }

    function materializeActorGroups({quiet = false} = {}) {
      const countries = checkedValues('countryOrigin');
      const ages = checkedValues('ageOrigin');
      if (!countries.length || !ages.length) {
        status('Select at least one country and one age bucket before building actor groups.');
        return false;
      }
      const cleanCount = Number(el('builderClean').value || 0);
      const dirtyBase = Number(el('builderDirty').value || 0);
      const mode = el('builderMode').value;
      const cells = [];
      for (const country of countries) {
        for (const age_bucket of ages) {
          cells.push({key: `${country}|${age_bucket}`, country, age_bucket});
        }
      }
      rows = [];
      const totalClean = Number(el('totalCleanActors').value || 0);
      const totalDirty = Number(el('totalDirtyActors').value || 0);
      const useTotals = totalClean > 0 || totalDirty > 0;

      if (!useTotals) {
        for (const {country, age_bucket} of cells) {
          const [bias_group, dirtyOverride] = modeBiasGroup(mode, country, age_bucket, countries);
          rows.push({
            country,
            age_bucket,
            bias_group,
            force_correct: false,
            clean_count: cleanCount,
            dirty_count: mode === 'balanced' ? dirtyBase : dirtyOverride
          });
        }
      } else {
        const cleanAlloc = integerAllocations(totalClean, cells, cleanWeight);
        const mix = normalizeMix(modeMix());
        const dirtySplits = {
          balanced: Math.round(totalDirty * mix.balanced),
          demographic: Math.round(totalDirty * mix.demographic),
          opinion: Math.round(totalDirty * mix.opinion),
          quality: Math.round(totalDirty * mix.quality)
        };
        const splitDelta = totalDirty - Object.values(dirtySplits).reduce((sum, value) => sum + value, 0);
        dirtySplits.demographic += splitDelta;

        const addModeRows = (modeName, dirtyTotal, includeClean) => {
          if (dirtyTotal <= 0 && !includeClean) return;
          const dirtyAlloc = integerAllocations(dirtyTotal, cells, cell => dirtyWeightFor(modeName, cell, countries));
          for (const cell of cells) {
            const [bias_group] = modeBiasGroup(modeName, cell.country, cell.age_bucket, countries);
            rows.push({
              country: cell.country,
              age_bucket: cell.age_bucket,
              bias_group,
              force_correct: false,
              clean_count: includeClean ? cleanAlloc.get(cell.key) : 0,
              dirty_count: dirtyAlloc.get(cell.key)
            });
          }
        };

        if (Object.values(modeMix()).reduce((sum, value) => sum + value, 0) > 0) {
          addModeRows('balanced', dirtySplits.balanced, true);
          addModeRows(mode === 'balanced' ? 'origin_pair_polarization' : mode, dirtySplits.demographic, false);
          addModeRows('age_polarization', dirtySplits.opinion, false);
          addModeRows('quality_stress', dirtySplits.quality, false);
        } else {
          const dirtyAlloc = integerAllocations(totalDirty, cells, cell => dirtyWeightFor(mode, cell, countries));
          for (const cell of cells) {
            const [bias_group] = modeBiasGroup(mode, cell.country, cell.age_bucket, countries);
            rows.push({
              country: cell.country,
              age_bucket: cell.age_bucket,
              bias_group,
              force_correct: false,
              clean_count: cleanAlloc.get(cell.key),
              dirty_count: dirtyAlloc.get(cell.key)
            });
          }
        }
      }
      renderRows();
      if (!quiet) {
        status(`Built ${rows.length} actor groups with deviation grade ${deviationGrade()} and mode ${mode}. Requested totals: ${sumRows('clean_count')} clean / ${sumRows('dirty_count')} dirty.`);
      }
      return true;
    }

    el('buildFromOrigins').addEventListener('click', () => {
      materializeActorGroups();
    });

    async function refreshSummary() {
      const data = await fetch('/api/summary').then(r => r.json());
      renderSummary(data);
    }

    function renderSummary(data) {
      if (!data.available) {
        el('metrics').innerHTML = '<div class="metric">No summary report yet.</div>';
        return;
      }
      const reports = data.summary.reports || [];
      el('metrics').innerHTML = reports.map(report => {
        const c = report.comparison || {};
        const s = c.representation_shift || {};
        return `<div class="metric">
          Batch <b>${report.batch_id}</b>
          <div>${c.clean_records || 0} clean / ${c.dirty_records || 0} dirty</div>
          <div>${report.clean_cluster_count} clean clusters / ${report.dirty_cluster_count} dirty clusters</div>
          <div>Nationality TVD: ${(s.nationality_tvd ?? 0).toFixed(4)}</div>
          <div>Opinion TVD: ${(s.opinion_tvd ?? 0).toFixed(4)}</div>
          <div>Answer-quality TVD: ${(s.answer_quality_tvd ?? 0).toFixed(4)}</div>
          <div>Bias-group TVD: ${(s.bias_group_tvd ?? 0).toFixed(4)}</div>
        </div>`;
      }).join('');
    }

    function setActiveTab(tabId) {
      document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.hidden = panel.id !== tabId;
      });
      document.querySelectorAll('[data-tab-target]').forEach(button => {
        button.classList.toggle('active', button.dataset.tabTarget === tabId);
      });
    }

    document.querySelectorAll('[data-tab-target]').forEach(button => {
      button.addEventListener('click', () => setActiveTab(button.dataset.tabTarget));
    });

    function setFrame(id, artifact) {
      const frame = el(id);
      if (!artifact || !artifact.available) {
        frame.srcdoc = '<!doctype html><html><body style="font:14px Arial,sans-serif;color:#3c5c8d;padding:24px">No scatterplot artifact for this batch.</body></html>';
        return;
      }
      frame.srcdoc = artifact.content;
    }

    function setActivePlotView(view) {
      activePlotView = view;
      document.querySelectorAll('[data-plot-view]').forEach(button => {
        button.classList.toggle('active', button.dataset.plotView === activePlotView);
      });
      if (!batchData) return;
      const labels = {
        combined: 'Clean vs dirty',
        clean: 'Clean',
        dirty: 'Dirty'
      };
      el('activePlotTitle').textContent = labels[activePlotView] || 'Scatterplot';
      setFrame('plotActive', batchData.scatterplots[activePlotView]);
    }

    function updateBatchPicker(data) {
      const batches = data.batches && data.batches.length ? data.batches : [data.batch_id];
      el('batchPicker').innerHTML = batches.map(batch => `<option value="${batch}" ${batch === data.batch_id ? 'selected' : ''}>Batch ${batch}</option>`).join('');
    }

    async function fetchBatchData(batchId) {
      const response = await fetch(`/api/batch?batch_id=${encodeURIComponent(batchId)}`);
      const data = await response.json();
      if (!response.ok) {
        el('batchOverview').innerHTML = `<div class="metric">${esc(data.error || 'Could not load batch.')}</div>`;
        return;
      }
      batchData = data;
      renderBatchData();
    }

    function metricCard(label, value, detail = '', path = '') {
      return `<div class="metric">${esc(label)} <b>${esc(value)}</b><div>${esc(detail)}</div>${path ? `<small>${esc(path)}</small>` : ''}</div>`;
    }

    function renderVerdict() {
      const datasetVerdicts = batchData?.dataset_verdicts || {};
      const verdicts = [
        {kind: 'dataset', data: datasetVerdicts.clean},
        {kind: 'dataset', data: datasetVerdicts.dirty},
        {kind: 'compare', data: batchData?.verdict}
      ].filter(item => item.data);
      if (!verdicts.length) {
        el('batchVerdict').innerHTML = '';
        return;
      }
      const cardHtml = verdicts.map(({kind, data}) => {
        const details = (data.details || []).map(item => `<li>${esc(item)}</li>`).join('');
        return `
          <div class="verdict ${esc(data.severity || 'low')} ${kind === 'compare' ? 'compare' : ''}">
            <b>${esc(data.title || 'Verdict')}</b>
            <p>${esc(data.text || '')}</p>
            ${details ? `<ul>${details}</ul>` : ''}
          </div>
        `;
      }).join('');
      el('batchVerdict').innerHTML = `
        <div class="verdict-grid">${cardHtml}</div>
      `;
    }

    function renderBatchData() {
      if (!batchData) return;
      updateBatchPicker(batchData);
      if (!batchData.available) {
        el('batchOverview').innerHTML = `<div class="empty">Batch ${esc(batchData.batch_id)} has no generated datasets or clustering artifacts yet.</div>`;
        el('batchVerdict').innerHTML = '';
        setFrame('plotActive', null);
        el('clusterSets').innerHTML = '<div class="empty">No cluster sets available.</div>';
        el('batchDataTable').innerHTML = '<div class="empty">No batch data available.</div>';
        el('batchTableMeta').textContent = '';
        return;
      }

      const comparison = batchData.comparison?.comparison || {};
      const shift = comparison.representation_shift || {};
      const cleanRecords = batchData.datasets.clean.total_rows || comparison.clean_records || 0;
      const dirtyRecords = batchData.datasets.dirty.total_rows || comparison.dirty_records || 0;
      renderVerdict();
      el('batchOverview').innerHTML = [
        metricCard('Batch', batchData.batch_id, batchData.config?.topic_title || batchData.comparison?.topic_title || ''),
        metricCard('Clean records', cleanRecords, `${batchData.summaries.clean?.cluster_count ?? batchData.comparison?.clean_cluster_count ?? 0} clusters`, batchData.datasets.clean.path),
        metricCard('Dirty records', dirtyRecords, `${batchData.summaries.dirty?.cluster_count ?? batchData.comparison?.dirty_cluster_count ?? 0} clusters`, batchData.datasets.dirty.path),
        metricCard('Nationality TVD', (shift.nationality_tvd ?? 0).toFixed(4), 'clean vs dirty representation shift'),
        metricCard('Opinion TVD', (shift.opinion_tvd ?? 0).toFixed(4), 'internal annotation shift'),
        metricCard('Answer-quality TVD', (shift.answer_quality_tvd ?? 0).toFixed(4), 'internal annotation shift'),
        metricCard('Bias-group TVD', (shift.bias_group_tvd ?? 0).toFixed(4), 'scenario marker shift'),
        metricCard('Artifacts', batchData.paths.batch_dir, 'scatterplots, summaries, assignments')
      ].join('');

      setActivePlotView(activePlotView);
      renderClusterSets();
      renderSelectedBatchTable();
    }

    function renderClusterSets() {
      const rows = [
        ...(batchData.cluster_sets.clean || []).map(row => ({dataset: 'clean', ...row})),
        ...(batchData.cluster_sets.dirty || []).map(row => ({dataset: 'dirty', ...row}))
      ];
      if (!rows.length) {
        el('clusterSets').innerHTML = '<div class="empty">No cluster assignment CSVs found for this batch.</div>';
        return;
      }
      el('clusterSets').innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Cluster</th>
              <th>Records</th>
              <th>Nationality</th>
              <th>Age</th>
              <th>Opinion</th>
              <th>Quality</th>
              <th>Bias group</th>
              <th>Sample ids</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(row => `
              <tr>
                <td>${esc(row.dataset)}</td>
                <td>${esc(row.cluster_id)}</td>
                <td>${esc(row.records)}</td>
                <td>${esc(row.nationality)}</td>
                <td>${esc(row.age)}</td>
                <td>${esc(row.opinion)}</td>
                <td>${esc(row.quality)}</td>
                <td>${esc(row.bias_group)}</td>
                <td>${esc((row.samples || []).join(', '))}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    function selectedTable() {
      const tables = {
        clean_records: {
          label: 'Clean records',
          source: batchData.datasets.clean,
          columns: ['Id', 'FirstName', 'LastName', 'Age', 'Nationality', 'Text']
        },
        dirty_records: {
          label: 'Dirty records',
          source: batchData.datasets.dirty,
          columns: ['Id', 'FirstName', 'LastName', 'Age', 'Nationality', 'Text']
        },
        clean_assignments: {
          label: 'Clean assignments',
          source: batchData.assignments.clean,
          columns: ['record_id', 'cluster_id', 'metadata_nationality', 'age_bucket', 'metadata_opinion', 'metadata_answerquality', 'metadata_biasgroup', 'scatter_x', 'scatter_y', 'text']
        },
        dirty_assignments: {
          label: 'Dirty assignments',
          source: batchData.assignments.dirty,
          columns: ['record_id', 'cluster_id', 'metadata_nationality', 'age_bucket', 'metadata_opinion', 'metadata_answerquality', 'metadata_biasgroup', 'scatter_x', 'scatter_y', 'text']
        }
      };
      return tables[el('batchTableSelect').value] || tables.clean_records;
    }

    function renderSelectedBatchTable() {
      if (!batchData) return;
      const table = selectedTable();
      const source = table.source || {available: false, rows: [], total_rows: 0};
      if (!source.available) {
        el('batchTableMeta').textContent = '';
        el('batchDataTable').innerHTML = `<div class="empty">${esc(table.label)} are not available for batch ${esc(batchData.batch_id)}.</div>`;
        return;
      }
      const filter = el('batchFilter').value.trim().toLowerCase();
      const filteredRows = (source.rows || []).filter(row => !filter || JSON.stringify(row).toLowerCase().includes(filter));
      el('batchTableMeta').textContent = `${table.label}: showing ${filteredRows.length} of ${source.total_rows} rows from ${source.path}.`;
      el('batchDataTable').innerHTML = `
        <table>
          <thead><tr>${table.columns.map(column => `<th>${esc(column)}</th>`).join('')}</tr></thead>
          <tbody>
            ${filteredRows.map(row => `
              <tr>
                ${table.columns.map(column => `<td class="${column.toLowerCase().includes('text') ? 'cell-text' : ''}">${esc(compact(row[column]))}</td>`).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    el('generate').addEventListener('click', async () => {
      if (!rowsMatchRequestedTotals()) {
        const rebuilt = materializeActorGroups({quiet: true});
        if (!rebuilt) return;
        status(`Rows were rebuilt from requested totals before generation: ${sumRows('clean_count')} clean / ${sumRows('dirty_count')} dirty.`);
      }
      status('Generating...');
      const payload = {
        batch_id: Number(el('batchId').value),
        overwrite: el('overwrite').value === 'true',
        run_clustering: el('runClustering').value === 'true',
        text_generation: el('textGeneration').value,
        deviation_grade: deviationGrade(),
        deviation_target: el('deviationTarget').value,
        wrong_answer_pattern: el('wrongAnswerPattern').value,
        allocation_strategy: el('allocationStrategy').value,
        total_clean_actors: Number(el('totalCleanActors').value || 0),
        total_dirty_actors: Number(el('totalDirtyActors').value || 0),
        mode_mix: modeMix(),
        topic_title: el('topicTitle').value,
        topic_summary: el('topicSummary').value,
        rows
      };
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        status(data);
        return;
      }
      status(data);
      renderSummary(data);
      await fetchBatchData(data.batch_id);
    });
    el('refresh').addEventListener('click', refreshSummary);
    el('loadBatchData').addEventListener('click', async () => {
      const selected = Number(el('batchPicker').value || el('batchId').value || 8);
      el('batchId').value = selected;
      await fetchBatchData(selected);
    });
    el('useCurrentBatch').addEventListener('click', async () => {
      await fetchBatchData(Number(el('batchId').value || 8));
    });
    el('batchPicker').addEventListener('change', async () => {
      el('batchId').value = el('batchPicker').value;
      await fetchBatchData(Number(el('batchPicker').value));
    });
    document.querySelectorAll('[data-plot-view]').forEach(button => {
      button.addEventListener('click', () => setActivePlotView(button.dataset.plotView));
    });
    el('batchTableSelect').addEventListener('change', renderSelectedBatchTable);
    el('batchFilter').addEventListener('input', renderSelectedBatchTable);
    el('deviationGrade').addEventListener('input', updateDeviationLabel);
    el('builderDirty').addEventListener('input', updateDeviationLabel);
    el('builderMode').addEventListener('input', updateDeviationLabel);
    el('allocationStrategy').addEventListener('input', updateDeviationLabel);
    el('deviationTarget').addEventListener('input', updateDeviationLabel);
    el('wrongAnswerPattern').addEventListener('input', updateDeviationLabel);
    el('totalCleanActors').addEventListener('input', updateDeviationLabel);
    el('totalDirtyActors').addEventListener('input', updateDeviationLabel);
    el('mixBalanced').addEventListener('input', updateDeviationLabel);
    el('mixDemographic').addEventListener('input', updateDeviationLabel);
    el('mixOpinion').addEventListener('input', updateDeviationLabel);
    el('mixQuality').addEventListener('input', updateDeviationLabel);

    updateDeviationLabel();
    loadOptions().catch(error => status(String(error)));
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic scenario UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ScenarioUIHandler)
    print(f"Synthetic Scenario UI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
