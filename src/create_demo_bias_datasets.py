"""Create deterministic non-biased and targeted-biased demo datasets.

The clustering bridge expects paired files named
``simulation_output_clean_{batch}.json`` and ``simulation_output_dirty_{batch}.json``.

This script creates:

* batch 6: non-biased control data. Clean and dirty contain the same balanced
  country, age, and opinion structure.
* batch 7: targeted biased data. Clean is balanced, while dirty keeps correct
  answers but redistributes opinion/age representation so the bias is visible
  mainly for Greece and Bulgaria.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# In the Docker image, certain_library is copied to /app/certain_library while
# this script runs from /app/Synthetic-Data-Generation-Component/src. Outside
# Docker, certain_library lives in the sibling Semantic_MLOps_engine repo.
CERTAIN_REPO_CANDIDATES = [
    Path("/app"),
    Path(__file__).resolve().parents[2] / "Semantic_MLOps_engine",
]

for candidate_root in CERTAIN_REPO_CANDIDATES:
    if (candidate_root / "certain_library" / "tracking" / "tracker.py").exists():
        sys.path.insert(0, str(candidate_root))
        break

from certain_library.tracking.tracker import tracker  # type: ignore[reportMissingImports] 
from certain_library.log_basic.log_params import log_params  # type: ignore[reportMissingImports]
from certain_library.train_monitor.log_metrics import log_metrics  # type: ignore[reportMissingImports]
from certain_library.data_analysis.log_dataset import log_dataset  # type: ignore[reportMissingImports]


SESSION_DIR = Path("session")
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
]

TOPIC_TITLE = "Remote work policy review"
TOPIC_SUMMARY = (
    "Question: Should the organization keep a remote work policy? "
    "A correct answer must give a position and also mention safeguards for "
    "mentorship, inclusion, productivity, and work-life balance."
)

COUNTRIES = {
    "Greece": {
        "background": "Engineer",
        "first_names": [
            "Eleni",
            "Nikos",
            "Maria",
            "Dimitris",
            "Sofia",
            "Kostas",
            "Andreas",
            "Georgia",
            "Panagiotis",
            "Vasiliki",
            "Christos",
            "Ioanna",
            "Stavros",
            "Katerina",
            "Michalis",
            "Anastasia",
        ],
        "last_names": [
            "Papadopoulos",
            "Nikolaou",
            "Georgiou",
            "Pappas",
            "Vlachou",
            "Karras",
            "Alexiou",
            "Markou",
            "Dimitriou",
            "Konstantinou",
            "Samaras",
            "Lambrou",
            "Economou",
            "Antoniou",
            "Stavridis",
            "Theodorou",
        ],
    },
    "Romania": {
        "background": "Teacher",
        "first_names": [
            "Ioana",
            "Andrei",
            "Elena",
            "Mihai",
            "Ana",
            "Vlad",
            "Cristina",
            "Alexandru",
            "Raluca",
            "Stefan",
            "Diana",
            "Florin",
            "Adriana",
            "Gabriel",
            "Simona",
            "Lucian",
        ],
        "last_names": [
            "Popescu",
            "Ionescu",
            "Stan",
            "Dobre",
            "Radu",
            "Marin",
            "Dumitrescu",
            "Tudor",
            "Munteanu",
            "Stoica",
            "Matei",
            "Nistor",
            "Barbu",
            "Ciobanu",
            "Grigorescu",
            "Neagu",
        ],
    },
    "Bulgaria": {
        "background": "Sales",
        "first_names": [
            "Mila",
            "Dimitar",
            "Vesela",
            "Nikolay",
            "Kalina",
            "Petar",
            "Elena",
            "Georgi",
            "Radost",
            "Ivaylo",
            "Teodora",
            "Hristo",
            "Aneliya",
            "Borislav",
            "Silvia",
            "Kiril",
        ],
        "last_names": [
            "Dimitrov",
            "Ivanova",
            "Petrov",
            "Koleva",
            "Stoyanov",
            "Vasileva",
            "Georgiev",
            "Todorova",
            "Nikolov",
            "Pavlova",
            "Angelov",
            "Ilieva",
            "Hristov",
            "Yordanova",
            "Kostadinov",
            "Marinova",
        ],
    },
    "Cyprus": {
        "background": "HR",
        "first_names": [
            "Andreas",
            "Christina",
            "Stelios",
            "Despina",
            "Marios",
            "Irene",
            "Kyriakos",
            "Evridiki",
            "Petros",
            "Melina",
            "Antonis",
            "Elpida",
            "Neophytos",
            "Marina",
            "Pavlos",
            "Ariadni",
        ],
        "last_names": [
            "Charalambous",
            "Andreou",
            "Christou",
            "Michael",
            "Ioannou",
            "Soteriou",
            "Constantinou",
            "Georgiou",
            "Demetriou",
            "Nicolaou",
            "Savva",
            "Iacovou",
            "Themistocleous",
            "Panayiotou",
            "Anastasiou",
            "Stylianou",
        ],
    },
    "Albania": {
        "background": "Consultant",
        "first_names": [
            "Arben",
            "Elira",
            "Dritan",
            "Mira",
            "Ilir",
            "Besa",
            "Anila",
            "Besnik",
            "Flutura",
            "Erion",
            "Luljeta",
            "Ardian",
            "Vjosa",
            "Kreshnik",
            "Teuta",
            "Gentian",
        ],
        "last_names": [
            "Hoxha",
            "Kola",
            "Dervishi",
            "Leka",
            "Prifti",
            "Gashi",
            "Shehu",
            "Basha",
            "Krasniqi",
            "Meta",
            "Berisha",
            "Ceka",
            "Mema",
            "Duka",
            "Pasha",
            "Bajrami",
        ],
    },
}

AGE_BUCKETS = {
    "18-29": 27,
    "30-44": 38,
    "45-59": 52,
    "60+": 64,
}

OPINION_TEXT = {
    "support": (
        "I support keeping the remote work policy because flexibility, focus time, reduced commuting, "
        "and wider access to roles can improve work outcomes. The policy still needs safeguards for "
        "mentoring access and promotion visibility."
    ),
    "conditional": (
        "I conditionally support the remote work policy if it comes with clear guardrails, hybrid rituals, "
        "productivity checks, burnout monitoring, onboarding support, and equal meeting access."
    ),
    "concern": (
        "I have concerns about the remote work policy because it can weaken learning, informal knowledge, "
        "belonging, career visibility, and work-life boundaries if the organization does not manage it carefully."
    ),
    "oppose": (
        "I oppose a broad remote work policy when coordination debt, trust loss, missed mentoring, delayed "
        "decisions, and isolation outweigh the benefits of flexibility."
    ),
}

ANSWER_QUALITY_TEXT = {
    "correct": "Final answer quality: correct. The answer is complete and directly addresses the policy question.",
    "partial": "Final answer quality: partial. The answer addresses the question but misses one safeguard.",
    "incorrect": "Final answer quality: incorrect. The answer is weakly connected to the policy question.",
}

AGE_PERSPECTIVE = {
    "18-29": "early-career employees focus on mentoring, onboarding, feedback speed, and visibility",
    "30-44": "mid-career employees focus on productivity, family schedules, and predictable coordination",
    "45-59": "senior employees focus on knowledge transfer, leadership routines, and stable team rituals",
    "60+": "late-career employees focus on accessibility, health needs, documentation, and sustainable workload",
}


def _metadata(
    *,
    country: str,
    age_bucket: str,
    opinion: str,
    index: int,
    bias_group: str,
    answer_quality: str = "correct",
) -> dict[str, Any]:
    country_info = COUNTRIES[country]
    age = AGE_BUCKETS[age_bucket]
    year = 2026 - age
    first_names = country_info["first_names"]
    last_names = country_info["last_names"]
    return {
        "FirstName": first_names[index % len(first_names)],
        "LastName": last_names[index % len(last_names)],
        "Age": age,
        "DateOfBirth": f"{(index % 27) + 1:02d}/{((index // 2) % 12) + 1:02d}/{year}",
        "Nationality": country,
        "Background": country_info["background"],
        "Opinion": opinion,
        "AnswerQuality": answer_quality,
        "BiasGroup": bias_group,
    }


def _record(
    *,
    batch: int,
    dataset_type: str,
    index: int,
    country: str,
    age_bucket: str,
    opinion: str,
    bias_group: str = "none",
    answer_quality: str = "correct",
) -> dict[str, Any]:
    country_info = COUNTRIES[country]
    text = (
        f"{OPINION_TEXT[opinion]} "
        f"{country} {country} respondent. Age bucket {age_bucket}. "
        f"{AGE_PERSPECTIVE[age_bucket]}. "
        f"Background {country_info['background']}. Bias marker {bias_group}. "
        f"{ANSWER_QUALITY_TEXT.get(answer_quality, ANSWER_QUALITY_TEXT['correct'])} "
        f"Opinion label: {opinion}."
    )
    return {
        "Id": f"demo-{batch}-{dataset_type}-{index:03d}",
        "Text": text,
        "Metadata": _metadata(
            country=country,
            age_bucket=age_bucket,
            opinion=opinion,
            index=index,
            bias_group=bias_group,
            answer_quality=answer_quality,
        ),
    }


def _payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "TopicTitle": TOPIC_TITLE,
        "TopicSummary": TOPIC_SUMMARY,
        "Dataset": records,
    }


def _export_text(text: str) -> str:
    internal_markers = [
        "Background ",
        "Bias marker ",
        "Final answer quality:",
        "Opinion label:",
        "The answer ",
    ]
    chunks = [chunk.strip() for chunk in text.split(".")]
    public_chunks = [
        chunk
        for chunk in chunks
        if chunk and not any(chunk.startswith(marker) for marker in internal_markers)
    ]
    return ". ".join(public_chunks) + ("." if public_chunks else "")


def _public_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    public_records = []
    for record in records:
        metadata = record.get("Metadata") or {}
        public_records.append(
            {
                "Id": record.get("Id"),
                "Text": _export_text(str(record.get("Text") or "")),
                "Metadata": {
                    field: metadata.get(field)
                    for field in PUBLIC_METADATA_FIELDS
                    if field in metadata
                },
            }
        )
    return {
        "TopicTitle": TOPIC_TITLE,
        "TopicSummary": TOPIC_SUMMARY,
        "Dataset": public_records,
    }


def _annotation(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("Metadata") or {}
    return {
        "Id": record.get("Id"),
        "InternalMetadata": {
            field: metadata.get(field)
            for field in INTERNAL_METADATA_FIELDS
            if field in metadata
        },
    }


def _balanced_records(batch: int, dataset_type: str, repeats_per_cell: int = 2) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    index = 1
    opinions = ["support", "conditional", "concern", "oppose"]

    for country in COUNTRIES:
        for age_index, age_bucket in enumerate(AGE_BUCKETS):
            for repeat in range(repeats_per_cell):
                opinion = opinions[(age_index + repeat) % len(opinions)]
                records.append(
                    _record(
                        batch=batch,
                        dataset_type=dataset_type,
                        index=index,
                        country=country,
                        age_bucket=age_bucket,
                        opinion=opinion,
                    )
                )
                index += 1
    return records


def _targeted_biased_records(batch: int) -> list[dict[str, Any]]:
    """Create biased evaluation data with localized bias.

    Romania, Cyprus, and Albania remain balanced across age/opinion. Greece is
    overrepresented by young supportive voices. Bulgaria is overrepresented by
    older concern/oppose voices. This makes the graph show where the bias comes
    from instead of making every country look biased.
    """

    records: list[dict[str, Any]] = []
    index = 1

    balanced_countries = ["Romania", "Cyprus", "Albania"]
    opinions = ["support", "conditional", "concern", "oppose"]
    for country in balanced_countries:
        for age_index, age_bucket in enumerate(AGE_BUCKETS):
            for repeat in range(2):
                opinion = opinions[(age_index + repeat) % len(opinions)]
                records.append(
                    _record(
                        batch=batch,
                        dataset_type="dirty",
                        index=index,
                        country=country,
                        age_bucket=age_bucket,
                        opinion=opinion,
                    )
                )
                index += 1

    targeted_cells = [
        ("Greece", "18-29", "support", "greece_young_support_overrepresented", 10),
        ("Greece", "30-44", "conditional", "greece_midcareer_present", 2),
        ("Greece", "45-59", "concern", "greece_senior_underrepresented", 1),
        ("Greece", "60+", "concern", "greece_latecareer_underrepresented", 1),
        ("Bulgaria", "18-29", "conditional", "bulgaria_young_underrepresented", 1),
        ("Bulgaria", "30-44", "concern", "bulgaria_midcareer_underrepresented", 1),
        ("Bulgaria", "45-59", "oppose", "bulgaria_senior_oppose_overrepresented", 6),
        ("Bulgaria", "60+", "oppose", "bulgaria_latecareer_oppose_overrepresented", 6),
    ]
    for country, age_bucket, opinion, bias_group, count in targeted_cells:
        for _ in range(count):
            records.append(
                _record(
                    batch=batch,
                    dataset_type="dirty",
                    index=index,
                    country=country,
                    age_bucket=age_bucket,
                    opinion=opinion,
                    bias_group=bias_group,
                )
            )
            index += 1

    return records


def _write_batch(
    batch: int,
    clean_records: list[dict[str, Any]],
    dirty_records: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> None:
    session_dir = (output_dir or SESSION_DIR).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    clean_output_path = session_dir / f"simulation_output_clean_{batch}.json"
    dirty_output_path = session_dir / f"simulation_output_dirty_{batch}.json"
    annotations_path = session_dir / f"scenario_annotations_batch_{batch}.json"
    files = {
        clean_output_path: _public_payload(clean_records),
        dirty_output_path: _public_payload(dirty_records),
        annotations_path: {
            "TopicTitle": TOPIC_TITLE,
            "TopicSummary": TOPIC_SUMMARY,
            "Clean": [_annotation(record) for record in clean_records],
            "Dirty": [_annotation(record) for record in dirty_records],
        },
    }
    for path, payload in files.items():
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Prepare a tabular object for log_dataset. `log_dataset` expects
        # a DataFrame / list-of-dicts / numpy array / dict-of-columns. The
        # payloads here are dictionaries that contain a "Dataset" key for
        # the clean/dirty JSONs or separate "Clean"/"Dirty" lists for the
        # annotations file. Extract the list-of-dicts shape so pandas
        # conversion does not fail with uneven-length columns.
        if isinstance(payload, dict):
            if "Dataset" in payload and isinstance(payload["Dataset"], list):
                data_to_log = payload["Dataset"]
            elif ("Clean" in payload or "Dirty" in payload) and (
                isinstance(payload.get("Clean", []), list)
                or isinstance(payload.get("Dirty", []), list)
            ):
                combined = []
                for rec in payload.get("Clean", []):
                    r = rec.copy() if isinstance(rec, dict) else {"value": rec}
                    r["_split"] = "clean"
                    combined.append(r)
                for rec in payload.get("Dirty", []):
                    r = rec.copy() if isinstance(rec, dict) else {"value": rec}
                    r["_split"] = "dirty"
                    combined.append(r)
                data_to_log = combined
            else:
                data_to_log = payload
        else:
            data_to_log = payload

        try:
            log_dataset(data_to_log, name=path.name, output_dir="data_creation")
        except Exception as exc:  # pragma: no cover - best-effort logging
            # Avoid failing dataset generation on non-critical logging errors.
            log_params({f"log_dataset_error_{path.name}": str(exc)})

    log_params(
        {
            f"batch_{batch}_clean_records": float(len(clean_records)),
            f"batch_{batch}_dirty_records": float(len(dirty_records)),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic clean/dirty demo datasets offline."
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=(6, 7),
        help="Generate only one demo batch. Defaults to generating both batches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SESSION_DIR,
        help="Directory where session JSON files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    batches = [args.batch] if args.batch is not None else [6, 7]

    for batch in batches:
        if batch == 6:
            _write_batch(
                batch=6,
                clean_records=_balanced_records(batch=6, dataset_type="clean"),
                dirty_records=_balanced_records(batch=6, dataset_type="dirty"),
                output_dir=args.output_dir,
            )
        else:
            _write_batch(
                batch=7,
                clean_records=_balanced_records(batch=7, dataset_type="clean"),
                dirty_records=_targeted_biased_records(batch=7),
                output_dir=args.output_dir,
            )

    selected = ", ".join(str(batch) for batch in batches)
    print(f"Created demo datasets for batch(s): {selected} in {args.output_dir.resolve()}")


if __name__ == "__main__":
    run_id = os.environ["MLFLOW_RUN_ID"]

    with tracker.start_run(run_id=run_id):
        log_params({'run the demo bias dataset creation': True})
        main()
 