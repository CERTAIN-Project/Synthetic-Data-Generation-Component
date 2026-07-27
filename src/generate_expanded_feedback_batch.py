from __future__ import annotations

import argparse
import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


NATIONALITIES = [
    "Greece",
    "Cyprus",
    "Albania",
    "Bulgaria",
    "Romania",
    "Serbia",
    "Turkey",
    "Italy",
    "Spain",
    "Portugal",
]

BACKGROUNDS = [
    "Engineering",
    "Education",
    "Healthcare",
    "Sales",
    "HR",
    "Public Administration",
    "Operations",
    "Research",
    "Design",
    "Customer Support",
]

FIRST_NAMES = [
    "Alex",
    "Maria",
    "Nikos",
    "Elena",
    "Ioana",
    "Dimitar",
    "Sofia",
    "Luca",
    "Ana",
    "Milan",
]

LAST_NAMES = [
    "Papadopoulos",
    "Georgiou",
    "Dimitrov",
    "Popescu",
    "Rossi",
    "Kaya",
    "Petrovic",
    "Silva",
    "Costa",
    "Markovic",
]

FEEDBACK_ANGLES = [
    "career growth",
    "mentorship quality",
    "work-life boundaries",
    "team trust",
    "accessibility",
    "managerial visibility",
    "burnout risk",
    "cross-border collaboration",
    "learning time",
    "family care responsibilities",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _date_of_birth(age: int, offset: int) -> str:
    year = 2026 - age
    month = (offset % 12) + 1
    day = (offset % 27) + 1
    return f"{day:02d}/{month:02d}/{year}"


def _expanded_text(base_text: str, index: int, dataset_type: str) -> str:
    angle = FEEDBACK_ANGLES[index % len(FEEDBACK_ANGLES)]
    second_angle = FEEDBACK_ANGLES[(index + 3) % len(FEEDBACK_ANGLES)]
    polarity = [
        "positive",
        "mixed",
        "critical",
        "cautiously optimistic",
    ][index % 4]
    detail = (
        f"\n\nExpanded feedback #{index + 1}: This {polarity} response adds concrete detail about {angle} "
        f"and {second_angle}. The respondent explains how remote work changes daily routines, communication "
        "quality, informal learning, manager support, and the ability to recover after intense work periods. "
        "They also mention whether the effect is different for junior and senior workers, and whether the "
        "organization provides enough structure for fair participation."
    )
    if dataset_type == "dirty":
        detail += (
            " Some details are intentionally less complete in this dirty variant, representing truncated, "
            "anonymous, or partially corrupted feedback."
        )
    return (base_text.strip() + detail).strip()


def _metadata_for(index: int, source_metadata: dict[str, Any]) -> dict[str, Any]:
    age = 18 + ((index * 7) % 55)
    metadata = deepcopy(source_metadata)
    metadata.update(
        {
            "FirstName": FIRST_NAMES[index % len(FIRST_NAMES)],
            "LastName": LAST_NAMES[(index * 3) % len(LAST_NAMES)],
            "Age": age,
            "DateOfBirth": _date_of_birth(age, index),
            "Nationality": NATIONALITIES[(index * 2) % len(NATIONALITIES)],
            "Background": BACKGROUNDS[(index * 3) % len(BACKGROUNDS)],
        }
    )
    return metadata


def _dirty_metadata(metadata: dict[str, Any], index: int) -> dict[str, Any]:
    dirty = deepcopy(metadata)
    pattern = index % 6
    if pattern == 0:
        return {
            "FirstName": None,
            "LastName": None,
            "Age": None,
            "DateOfBirth": None,
            "Nationality": None,
            "Background": None,
        }
    if pattern == 1:
        dirty["Nationality"] = None
        dirty["LastName"] = None
    elif pattern == 2:
        dirty["Age"] = float(dirty["Age"]) if dirty.get("Age") is not None else None
        dirty["Background"] = None
    elif pattern == 3:
        dirty["FirstName"] = None
        dirty["DateOfBirth"] = None
    elif pattern == 4:
        dirty["Nationality"] = "Unknown"
    else:
        dirty["Background"] = "Other"
    return dirty


def _make_records(base_records: list[dict[str, Any]], count: int, batch_id: int, dataset_type: str) -> list[dict[str, Any]]:
    records = []
    for index in range(count):
        base = base_records[index % len(base_records)]
        base_metadata = base.get("Metadata") or {}
        metadata = _metadata_for(index, base_metadata)
        text = _expanded_text(str(base.get("Text", "")), index, dataset_type)
        if dataset_type == "dirty" and index % 5 == 0:
            text = text[: max(160, len(text) // 3)] + "..."
        if dataset_type == "dirty":
            metadata = _dirty_metadata(metadata, index)

        records.append(
            {
                "Id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"certain-expanded-{batch_id}-{dataset_type}-{index}")),
                "Text": text,
                "Metadata": metadata,
            }
        )
    return records


def generate_expanded_feedback_batch(
    session_dir: Path,
    *,
    source_batch: int = 4,
    target_batch: int = 5,
    clean_count: int = 72,
    dirty_count: int = 96,
) -> dict[str, Path]:
    clean_source = _read_json(session_dir / f"simulation_output_clean_{source_batch}.json")
    dirty_source = _read_json(session_dir / f"simulation_output_dirty_{source_batch}.json")

    topic_title = f"{clean_source.get('TopicTitle', 'Benefits of remote work')} - expanded feedback range"
    topic_summary = (
        clean_source.get("TopicSummary", "")
        + " This expanded batch increases participant coverage, age range, nationalities, backgrounds, "
        "and feedback detail for a broader observe pass."
    ).strip()

    clean_payload = {
        "TopicTitle": topic_title,
        "TopicSummary": topic_summary,
        "Dataset": _make_records(clean_source.get("Dataset", []), clean_count, target_batch, "clean"),
    }
    dirty_payload = {
        "TopicTitle": topic_title,
        "TopicSummary": topic_summary,
        "Dataset": _make_records(dirty_source.get("Dataset", []), dirty_count, target_batch, "dirty"),
    }

    clean_path = session_dir / f"simulation_output_clean_{target_batch}.json"
    dirty_path = session_dir / f"simulation_output_dirty_{target_batch}.json"
    _write_json(clean_path, clean_payload)
    _write_json(dirty_path, dirty_payload)
    return {"clean": clean_path, "dirty": dirty_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a larger deterministic clean/dirty feedback batch.")
    parser.add_argument("--session-dir", type=Path, default=Path("session"))
    parser.add_argument("--source-batch", type=int, default=4)
    parser.add_argument("--target-batch", type=int, default=5)
    parser.add_argument("--clean-count", type=int, default=72)
    parser.add_argument("--dirty-count", type=int, default=96)
    args = parser.parse_args()

    result = generate_expanded_feedback_batch(
        args.session_dir.resolve(),
        source_batch=args.source_batch,
        target_batch=args.target_batch,
        clean_count=args.clean_count,
        dirty_count=args.dirty_count,
    )
    print(json.dumps({key: str(path) for key, path in result.items()}, indent=2))


if __name__ == "__main__":
    main()
